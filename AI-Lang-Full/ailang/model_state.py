"""Portable model and optimizer persistence for AI-Lang.

The tensor engine intentionally keeps live graphs and Python objects in
memory. This module provides the durable boundary around them without using
pickle: models, records, maps, lists, tensors, and optimizer moments are
encoded as bounded JSON data and written through the atomic, integrity-checked
sleep checkpoint format.

``model_load`` can accept a template model. When one is supplied, record
values are reconstructed with the template's record type, so a package such as
``neural`` can load weights into its normal ``Net``/``Layer`` values. Without
a template, records are returned as maps and tensors are still live,
trainable ``Tensor`` objects.

Optimizer files preserve the step counter, hyperparameters, moments, and the
saved parameter values. Supplying fresh parameters to ``optimizer_load``
rebinds the optimizer to those parameters and validates every moment shape;
this is what makes a sleep/resume boundary safe after rebuilding a model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from .errors import VMError
from .training import load_checkpoint, save_checkpoint
from .values import MapKey, RecordValue, hashable_key, unmap_key

MODEL_KIND = "AILANG-MODEL-1"
OPTIMIZER_KIND = "AILANG-OPTIMIZER-1"
# Leave room below CPython's recursion limit for a clean VMError.
_MAX_DEPTH = 100
_MAX_ITEMS = 10_000_000


def _number(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VMError(f"{where}: expected a finite number")
    if isinstance(value, float) and not math.isfinite(value):
        raise VMError(f"{where}: expected a finite number")
    return value


def _encode(value, depth=0, active=None):
    """Encode runtime values using explicit tags for non-JSON structures."""
    if depth > _MAX_DEPTH:
        raise VMError("model state is nested too deeply")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _number(value, "model state")

    # Import lazily: model persistence must remain usable on a bare host and
    # must not make the standard library import the tensor engine eagerly.
    from . import accel
    from .autodiff import Tensor

    if isinstance(value, Tensor):
        return {
            "__tensor__": True,
            "shape": [int(x) for x in value.shape],
            "requires_grad": bool(value.requires_grad),
            "value": _encode(value.value(), depth + 1, active),
        }
    if accel.is_arr(value):
        return _encode(value.tolist(), depth + 1, active)

    if active is None:
        active = set()
    if isinstance(value, RecordValue):
        if id(value) in active:
            raise VMError("model state contains a cyclic record")
        active.add(id(value))
        try:
            return {
                "__record__": value.type_name,
                "fields": {
                    str(key): _encode(item, depth + 1, active)
                    for key, item in value.data.items()
                },
            }
        finally:
            active.remove(id(value))
    if isinstance(value, (list, tuple)):
        if id(value) in active:
            raise VMError("model state contains a cyclic list")
        if len(value) > _MAX_ITEMS:
            raise VMError("model state list is too large")
        active.add(id(value))
        try:
            return [_encode(item, depth + 1, active) for item in value]
        finally:
            active.remove(id(value))
    if isinstance(value, Mapping):
        if id(value) in active:
            raise VMError("model state contains a cyclic map")
        if len(value) > _MAX_ITEMS:
            raise VMError("model state map is too large")
        active.add(id(value))
        try:
            # A tagged entry list preserves non-text AI-Lang map keys. Plain
            # dictionaries remain easy to inspect after reading the JSON.
            if all(isinstance(unmap_key(key), str) for key in value):
                return {
                    str(unmap_key(key)): _encode(item, depth + 1, active)
                    for key, item in value.items()
                }
            return {
                "__map__": [
                    [_encode(unmap_key(key), depth + 1, active),
                     _encode(item, depth + 1, active)]
                    for key, item in value.items()
                ]
            }
        finally:
            active.remove(id(value))
    if isinstance(value, MapKey):
        return _encode(value.raw, depth + 1, active)
    raise VMError(f"model state cannot contain {type(value).__name__}")


def _template_at(template, index):
    if isinstance(template, list) and index < len(template):
        return template[index]
    return None


def _decode(value, template=None, depth=0):
    if depth > _MAX_DEPTH:
        raise VMError("model state is nested too deeply")
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise VMError("model state contains a non-finite number")
        return value
    if isinstance(value, list):
        if len(value) > _MAX_ITEMS:
            raise VMError("model state list is too large")
        return [_decode(item, _template_at(template, i), depth + 1)
                for i, item in enumerate(value)]
    if not isinstance(value, dict):
        raise VMError(f"model state contains {type(value).__name__}")

    if value.get("__tensor__") is True:
        required = {"__tensor__", "shape", "requires_grad", "value"}
        if set(value) != required or not isinstance(value["shape"], list):
            raise VMError("model state tensor marker is malformed")
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0
               for x in value["shape"]):
            raise VMError("model state tensor shape is malformed")
        if type(value["requires_grad"]) is not bool:
            raise VMError("model state tensor requires_grad is malformed")
        from .autodiff import Tensor
        tensor = Tensor.of(value["value"], value["requires_grad"])
        if tuple(value["shape"]) != tensor.shape:
            raise VMError("model state tensor shape does not match its values")
        return tensor

    if "__record__" in value:
        if set(value) != {"__record__", "fields"} or not isinstance(value["__record__"], str):
            raise VMError("model state record marker is malformed")
        fields = value["fields"]
        if not isinstance(fields, dict):
            raise VMError("model state record fields are malformed")
        if isinstance(template, RecordValue):
            if template.type_name != value["__record__"]:
                raise VMError(
                    f"model state record type {value['__record__']!r} does not fit "
                    f"template {template.type_name!r}"
                )
            expected = set(template.rtype.fields)
            if set(fields) != expected:
                raise VMError("model state record fields do not match its template")
            restored = {
                key: _decode(fields[key], template.data.get(key), depth + 1)
                for key in template.rtype.fields
            }
            return template.rtype(**restored)
        # A schema-free load is still useful for inspection and for models
        # represented as maps. Preserve the type marker under a private key.
        return {
            "__record_type__": value["__record__"],
            **{key: _decode(item, None, depth + 1)
               for key, item in fields.items()},
        }

    if "__map__" in value:
        if set(value) != {"__map__"} or not isinstance(value["__map__"], list):
            raise VMError("model state map marker is malformed")
        out = {}
        for pair in value["__map__"]:
            if not isinstance(pair, list) or len(pair) != 2:
                raise VMError("model state map entry is malformed")
            key = _decode(pair[0], None, depth + 1)
            try:
                key = hashable_key(key)
            except TypeError as exc:
                raise VMError(f"model state map key is invalid: {exc}") from None
            if key in out:
                raise VMError("model state contains duplicate map keys")
            out[key] = _decode(pair[1], None, depth + 1)
        return out

    # Plain text-key maps are the most common representation and are also
    # accepted for forward compatibility with older user-created checkpoints.
    return {
        key: _decode(item, template.get(key) if isinstance(template, dict) else None,
                     depth + 1)
        for key, item in value.items()
    }


def _kind(document, expected, operation):
    metadata = document.get("metadata", {})
    if not isinstance(metadata, dict) or metadata.get("kind") != expected:
        raise VMError(f"{operation}: file is not an {expected} checkpoint")


def _user_metadata(metadata):
    if metadata is None:
        return {}
    encoded = _encode(metadata)
    if not isinstance(encoded, (dict, list, str, int, float, bool)) and encoded is not None:
        raise VMError("persistence metadata is malformed")
    return encoded


def save_model(path, model, metadata=None, signing_key=None, key_id=""):
    """Atomically save a model's tensors and structure."""
    return save_checkpoint(
        path,
        {"__model__": _encode(model)},
        metadata={"kind": MODEL_KIND, "user": _user_metadata(metadata)},
        signing_key=signing_key,
        key_id=key_id,
    )


def load_model(path, template=None, signing_key=None, require_signature=False):
    """Load a model, optionally rebuilding record types from ``template``."""
    document = load_checkpoint(path, signing_key, bool(require_signature))
    _kind(document, MODEL_KIND, "model_load")
    state = document.get("state")
    if not isinstance(state, dict) or set(state) != {"__model__"}:
        raise VMError("model_load: model payload is malformed")
    return _decode(state["__model__"], template)


def save_optimizer(path, optimizer, metadata=None, signing_key=None, key_id=""):
    """Atomically save optimizer hyperparameters, moments, and step count."""
    return save_checkpoint(
        path,
        {"__optimizer__": _encode(optimizer)},
        metadata={"kind": OPTIMIZER_KIND, "user": _user_metadata(metadata)},
        signing_key=signing_key,
        key_id=key_id,
    )


def _fresh_params(params):
    from .autodiff import Tensor
    if isinstance(params, Tensor):
        return [params]
    if not isinstance(params, list):
        raise VMError("optimizer_load: params must be a Tensor or List of tensors")
    if any(not isinstance(item, Tensor) for item in params):
        raise VMError("optimizer_load: params must contain only tensors")
    return params


def _rebind_optimizer(optimizer, params):
    if not isinstance(optimizer, dict):
        raise VMError("optimizer_load: optimizer state is malformed")
    saved = optimizer.get("params")
    if not isinstance(saved, list) or len(saved) != len(params):
        raise VMError("optimizer_load: parameter count does not match the checkpoint")
    for index, (saved_parameter, parameter) in enumerate(zip(saved, params)):
        if (not hasattr(saved_parameter, "shape")
                or tuple(saved_parameter.shape) != tuple(parameter.shape)):
            raise VMError(
                f"optimizer_load: parameter {index} shape does not match the checkpoint"
            )
    optimizer["params"] = params
    for name in ("m", "v"):
        if name not in optimizer:
            continue
        moments = optimizer[name]
        if not isinstance(moments, list) or len(moments) != len(params):
            raise VMError(f"optimizer_load: {name} state does not match parameters")
        for index, (moment, parameter) in enumerate(zip(moments, params)):
            # A moment is stored as a plain list even when the original run
            # used numpy. Recreate the active backend's native buffer.
            if hasattr(moment, "value") and hasattr(moment, "shape"):
                moment = moment.value()
            if not isinstance(moment, list) or len(moment) != parameter.size:
                raise VMError(f"optimizer_load: {name}[{index}] shape does not match")
            from . import accel
            moments[index] = accel.asarr(moment) if accel.is_arr(parameter.data) else list(moment)
    return optimizer


def load_optimizer(path, params=None, signing_key=None, require_signature=False):
    """Load optimizer state and optionally bind it to fresh model parameters."""
    document = load_checkpoint(path, signing_key, bool(require_signature))
    _kind(document, OPTIMIZER_KIND, "optimizer_load")
    state = document.get("state")
    if not isinstance(state, dict) or set(state) != {"__optimizer__"}:
        raise VMError("optimizer_load: optimizer payload is malformed")
    optimizer = _decode(state["__optimizer__"])
    if params is not None:
        optimizer = _rebind_optimizer(optimizer, _fresh_params(params))
    return optimizer


__all__ = [
    "MODEL_KIND",
    "OPTIMIZER_KIND",
    "save_model",
    "load_model",
    "save_optimizer",
    "load_optimizer",
]
