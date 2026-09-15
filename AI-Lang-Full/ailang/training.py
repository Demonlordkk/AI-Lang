"""Portable, resumable training orchestration for AI-Lang.

This module deliberately does not replace the differentiable tensor engine.
It supplies the missing lifecycle around it: inspect the current device,
choose a conservative amount of work, persist a small verified checkpoint
after each chunk, and return control so a phone, laptop, server, or embedded
host can sleep and resume later.

The checkpoint contains user state and a cursor, not an open Tensor graph or
a Python callable. This keeps it portable, bounded, and safe to load. Tensor
leaves in state are rebuilt as fresh tensors on resume; model records and
optimizer parameter aliases remain the application's responsibility (the
``neural`` package handles that rebinding). The HMAC option authenticates
checkpoints when a host supplies a key; an untrusted program must still run in
an OS-level sandbox.
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import math
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Mapping

from .capabilities import require, resource_path
from .errors import VMError

CHECKPOINT_FORMAT = "AILANG-SLEEP-1"
_MAX_CHECKPOINT_BYTES = 256 * 1024 * 1024
# Keep this below CPython's recursion limit so malformed/deep state is
# rejected as an AI-Lang error instead of escaping as RecursionError.
_MAX_JSON_DEPTH = 100
_MAX_ITEMS = 10_000_000


def _runtime_path(path) -> str:
    try:
        path = os.fspath(path)
    except TypeError:
        raise VMError("checkpoint path must be Text") from None
    try:
        from .stdlib import script_dir
        base = script_dir()
    except (ImportError, AttributeError):
        base = None
    if base and not os.path.isabs(path):
        return os.path.join(base, path)
    return path


def _memory_available() -> int:
    """Best-effort available memory with conservative portable fallbacks.

    On Linux containers and some mobile runtimes, ``/proc/meminfo`` can report
    the host's capacity rather than the process's cgroup limit. Honor cgroup
    v2/v1 limits when they are visible so an adaptive batch cannot be sized
    for a much larger parent machine.
    """
    available = None
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as handle:
            values = {}
            for line in handle:
                key, _, raw = line.partition(":")
                if key in {"MemAvailable", "MemFree"}:
                    values[key] = int(raw.strip().split()[0]) * 1024
            if values.get("MemAvailable"):
                available = values["MemAvailable"]
            elif values.get("MemFree"):
                available = values["MemFree"]
    except (OSError, ValueError):
        pass
    if available is None:
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            size = os.sysconf("SC_PAGE_SIZE")
            if isinstance(pages, int) and isinstance(size, int) and pages > 0 and size > 0:
                available = pages * size
        except (AttributeError, OSError, ValueError):
            pass

    # A cgroup limit is a hard ceiling, not an estimate of free host memory.
    for limit_path, current_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
         "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        try:
            raw_limit = Path(limit_path).read_text(encoding="ascii").strip()
            if raw_limit == "max":
                continue
            limit = int(raw_limit)
            if limit <= 0 or limit >= (1 << 60):
                continue
            try:
                current = int(Path(current_path).read_text(encoding="ascii").strip())
            except (OSError, ValueError):
                current = 0
            constrained = max(1 * 1024 * 1024, limit - max(current, 0))
            available = constrained if available is None else min(available, constrained)
            break
        except (OSError, ValueError):
            continue

    if available is not None:
        return max(1 * 1024 * 1024, int(available))
    # A safe low-memory assumption is preferable to allocating the whole
    # machine on platforms that expose no memory counters.
    return 512 * 1024 * 1024


def device_profile() -> dict:
    """Describe safe training capacity without requiring third-party packages."""
    available = _memory_available()
    cores = os.cpu_count() or 1
    try:
        from . import accel
        backend = accel.info()
    except Exception:
        backend = "pure python"
    if available < 1 * 1024 * 1024 * 1024:
        tier = "small"
        default_item_bytes = 64 * 1024
    elif available < 8 * 1024 * 1024 * 1024:
        tier = "medium"
        default_item_bytes = 256 * 1024
    else:
        tier = "large"
        default_item_bytes = 1024 * 1024
    safe = max(1 * 1024 * 1024, int(available * 0.60))
    recommended = max(1, min(4096, safe // default_item_bytes))
    return {
        "platform": platform.system().lower() or "unknown",
        "python": platform.python_version(),
        "cores": int(cores),
        "memory_available": int(available),
        "memory_budget": int(safe),
        "tier": tier,
        "backend": backend,
        "recommended_batch": int(recommended),
        "checkpoint_format": CHECKPOINT_FORMAT,
    }


def batch_budget(item_bytes=0, requested=0) -> int:
    """Return a conservative batch size for this device and item estimate."""
    try:
        item_bytes = int(item_bytes)
        requested = int(requested)
    except (TypeError, ValueError, OverflowError):
        raise VMError("batch_budget: item_bytes and requested must be Ints") from None
    if item_bytes < 0 or requested < 0:
        raise VMError("batch_budget: item_bytes and requested must be non-negative")
    if item_bytes == 0:
        item_bytes = 256 * 1024
    budget = max(1, int(_memory_available() * 0.60))
    result = max(1, min(_MAX_ITEMS, budget // max(item_bytes, 1)))
    if requested:
        result = min(result, requested)
    return int(result)


def _safe(value, depth=0, active=None):
    """Convert AI-Lang values to bounded JSON data."""
    if depth > _MAX_JSON_DEPTH:
        raise VMError("checkpoint state is nested too deeply")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise VMError("checkpoint state contains a non-finite number")
        return value
    # Tensor values are copied as data plus shape. Reconstructing a trainable
    # Tensor is handled by ``restore_checkpoint_state`` on the next run, not
    # with pickle.
    if hasattr(value, "shape") and hasattr(value, "value") and hasattr(value, "requires_grad"):
        return {
            "__tensor__": True,
            "shape": [int(x) for x in value.shape],
            "requires_grad": bool(value.requires_grad),
            "value": _safe(value.value(), depth + 1, active),
        }
    # Optimizer moments may be numpy arrays on an accelerated host. Store
    # them as ordinary lists so the checkpoint remains portable to a phone or
    # a pure-Python host.
    if value.__class__.__name__ == "ndarray" and hasattr(value, "tolist"):
        return _safe(value.tolist(), depth + 1, active)
    if hasattr(value, "data") and value.__class__.__name__ == "RecordValue":
        value = value.data
    if active is None:
        active = set()
    if isinstance(value, (list, tuple)):
        if id(value) in active:
            raise VMError("checkpoint state contains a cyclic list")
        if len(value) > _MAX_ITEMS:
            raise VMError("checkpoint state list is too large")
        active.add(id(value))
        try:
            return [_safe(x, depth + 1, active) for x in value]
        finally:
            active.remove(id(value))
    if isinstance(value, Mapping):
        if id(value) in active:
            raise VMError("checkpoint state contains a cyclic map")
        if len(value) > _MAX_ITEMS:
            raise VMError("checkpoint state map is too large")
        active.add(id(value))
        try:
            out = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    key = str(key)
                if key in out:
                    raise VMError(
                        f"checkpoint state has duplicate map key after Text conversion: {key!r}"
                    )
                out[key] = _safe(item, depth + 1, active)
            return out
        finally:
            active.remove(id(value))
    raise VMError(f"checkpoint state cannot contain {type(value).__name__}")


def _json(value) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise VMError(f"checkpoint state is not JSON serializable: {exc}") from None


def _signed_payload(document: dict) -> bytes:
    # Both envelope hashes are derived fields. Excluding them keeps the
    # content digest stable when the outer checkpoint hash is added, while
    # the optional signature still covers the exact same canonical payload.
    return _json({k: v for k, v in document.items()
                  if k not in {"sha256", "signature", "checkpoint_sha256"}})


def _mac(payload: bytes, key: str, key_id: str) -> str:
    if not isinstance(key, str) or not key:
        raise VMError("checkpoint signing needs a non-empty text key")
    return hmac.new(key.encode("utf-8"), _json({"payload": payload.decode("utf-8"), "key_id": key_id}), hashlib.sha256).hexdigest()


def _validate_json(value, depth=0):
    if depth > _MAX_JSON_DEPTH:
        raise VMError("checkpoint JSON is nested too deeply")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise VMError("checkpoint JSON contains a non-finite number")
        return
    if isinstance(value, list):
        if len(value) > _MAX_ITEMS:
            raise VMError("checkpoint JSON list is too large")
        for item in value:
            _validate_json(item, depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > _MAX_ITEMS:
            raise VMError("checkpoint JSON map is too large")
        for key, item in value.items():
            if not isinstance(key, str):
                raise VMError("checkpoint JSON map key is not Text")
            _validate_json(item, depth + 1)
        return
    raise VMError(f"checkpoint JSON contains {type(value).__name__}")


def save_checkpoint(path, state, cursor=0, metrics=None, epoch=0,
                    metadata=None, signing_key=None, key_id="") -> dict:
    """Atomically persist a verified sleep/resume checkpoint."""
    path = _runtime_path(path)
    require("fs.write", resource_path(path), "sleep_save")
    try:
        cursor, epoch = int(cursor), int(epoch)
    except (TypeError, ValueError, OverflowError):
        raise VMError("sleep_save: cursor and epoch must be Ints") from None
    if cursor < 0 or epoch < 0:
        raise VMError("sleep_save: cursor and epoch must be non-negative")
    if metrics is None:
        metrics = {}
    if metadata is None:
        metadata = {}
    if not isinstance(metrics, Mapping) or not isinstance(metadata, Mapping):
        raise VMError("sleep_save: metrics and metadata must be Maps")
    document = {
        "format": CHECKPOINT_FORMAT,
        "state": _safe(state),
        "cursor": cursor,
        "epoch": epoch,
        "metrics": _safe(metrics),
        "metadata": _safe(metadata),
    }
    document["sha256"] = hashlib.sha256(_signed_payload(document)).hexdigest()
    if signing_key is not None:
        if not isinstance(key_id, str) or len(key_id) > 256:
            raise VMError("sleep_save: key_id must be Text of at most 256 characters")
        document["signature"] = {
            "alg": "hmac-sha256",
            "key_id": key_id,
            "sig": _mac(_signed_payload(document), signing_key, key_id),
        }
    document["checkpoint_sha256"] = hashlib.sha256(_json(document)).hexdigest()
    target = Path(path)
    if target.exists() and target.is_dir():
        raise VMError(f"sleep_save: path is a directory: {target}")
    try:
        encoded = json.dumps(document, indent=2, sort_keys=True,
                             ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    except (TypeError, ValueError, UnicodeError) as exc:
        raise VMError(f"sleep_save: checkpoint is not serializable: {exc}") from None
    if len(encoded) > _MAX_CHECKPOINT_BYTES:
        raise VMError("sleep_save: checkpoint is too large")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=target.parent,
                                        prefix=f".{target.name}.", suffix=".tmp",
                                        delete=False) as handle:
            temporary = Path(handle.name)
            # Checkpoints can contain credentials or private model weights;
            # do not leave the temporary file world-readable during the
            # replace window on platforms that support POSIX modes.
            try:
                os.chmod(temporary, 0o600)
            except OSError:
                pass
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
        # fsyncing the directory makes the rename durable after a sudden
        # power loss on POSIX filesystems. Some platforms reject directory
        # handles, where the file fsync above is still the best available
        # guarantee.
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except OSError as exc:
        raise VMError(f"sleep_save: cannot write '{target}': {exc}") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass
    return {
        "path": str(target),
        "cursor": cursor,
        "epoch": epoch,
        "checkpoint_sha256": document["checkpoint_sha256"],
        "signed": "signature" in document,
    }


def load_checkpoint(path, signing_key=None, require_signature=False) -> dict:
    """Load and authenticate a checkpoint before returning its state map."""
    target = Path(_runtime_path(path))
    require("fs.read", resource_path(target), "sleep_load")
    try:
        raw = target.read_bytes()
    except OSError as exc:
        raise VMError(f"sleep_load: cannot read '{target}': {exc}") from None
    if len(raw) > _MAX_CHECKPOINT_BYTES:
        raise VMError("sleep_load: checkpoint is too large")
    try:
        document = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda x: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {x}")
            ),
            object_pairs_hook=lambda pairs: _unique_pairs(pairs),
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise VMError(f"sleep_load: invalid checkpoint: {exc}") from None
    if not isinstance(document, dict):
        raise VMError("sleep_load: top level must be a Map")
    _validate_json(document)
    required = {"format", "state", "cursor", "epoch", "metrics", "metadata",
                "sha256", "checkpoint_sha256"}
    if set(document) - required - {"signature"} or not required.issubset(document):
        raise VMError("sleep_load: checkpoint schema is incomplete or has unknown fields")
    if document["format"] != CHECKPOINT_FORMAT:
        raise VMError(f"sleep_load: unsupported checkpoint format {document['format']!r}")
    if type(document["cursor"]) is not int or document["cursor"] < 0:
        raise VMError("sleep_load: cursor is malformed")
    if type(document["epoch"]) is not int or document["epoch"] < 0:
        raise VMError("sleep_load: epoch is malformed")
    if not isinstance(document["metrics"], dict) or not isinstance(document["metadata"], dict):
        raise VMError("sleep_load: metrics and metadata must be Maps")
    expected = hashlib.sha256(_signed_payload(document)).hexdigest()
    if not isinstance(document["sha256"], str) or not hmac.compare_digest(expected, document["sha256"]):
        raise VMError("sleep_load: checkpoint integrity check failed")
    complete_hash = hashlib.sha256(_json({k: v for k, v in document.items() if k != "checkpoint_sha256"})).hexdigest()
    if not isinstance(document["checkpoint_sha256"], str) or not hmac.compare_digest(complete_hash, document["checkpoint_sha256"]):
        raise VMError("sleep_load: checkpoint envelope is corrupt")
    signature = document.get("signature")
    if signature is None:
        if require_signature or signing_key is not None:
            raise VMError("sleep_load: a signed checkpoint is required")
    else:
        if (not isinstance(signature, dict)
                or set(signature) != {"alg", "key_id", "sig"}
                or signature.get("alg") != "hmac-sha256"
                or not isinstance(signature.get("key_id"), str)
                or len(signature["key_id"]) > 256
                or not isinstance(signature.get("sig"), str)):
            raise VMError("sleep_load: unsupported checkpoint signature")
        if signing_key is not None:
            expected = _mac(_signed_payload(document), signing_key, signature["key_id"])
            if not hmac.compare_digest(expected, signature["sig"]):
                raise VMError("sleep_load: checkpoint signature authentication failed")
        elif require_signature:
            raise VMError("sleep_load: verification key is required for this checkpoint")
    return document


def _unique_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate checkpoint key {key!r}")
        out[key] = value
    return out


def restore_checkpoint_state(value):
    """Rebuild portable tensor markers into live Tensor values.

    Primitive checkpoint data is returned unchanged. Record type identity is
    deliberately not guessed because a checkpoint can be loaded by a fresh
    process whose module schema differs; callers with a schema can use
    ``model_load(..., template)`` instead. This helper is sufficient for
    active-training state maps containing model tensors and optimizer moments.
    """
    try:
        from .model_state import _decode
        return _decode(value)
    except ImportError:
        return value


def _estimate(value, depth=0):
    """Bounded rough item-size estimate for adaptive batch selection."""
    if depth > 20:
        return 1024
    if value is None or isinstance(value, (bool, int, float)):
        return 32
    if isinstance(value, str):
        return min(len(value) + 49, 1 << 20)
    if isinstance(value, (list, tuple)):
        if not value:
            return 64
        return min(64 + sum(_estimate(x, depth + 1) for x in value[:64]) * max(1, len(value) // 64), 16 << 20)
    if isinstance(value, Mapping):
        return min(128 + sum(_estimate(k, depth + 1) + _estimate(v, depth + 1) for k, v in list(value.items())[:64]), 16 << 20)
    return 4096


def _call_step(step_fn, batch, state):
    # AI-Lang closures expose their real arity through the VM. inspect.signature
    # only sees Closure.__call__(*args, **kwargs), so consult ``arity`` first;
    # otherwise a perfectly valid one-argument source callback would receive
    # an erroneous state argument.
    arity = getattr(step_fn, "arity", None)
    if isinstance(arity, int):
        if arity == 1:
            return step_fn(batch)
        if arity == 2:
            return step_fn(batch, state)
        raise VMError("active_train: step_fn must accept batch or batch and state")
    try:
        parameters = inspect.signature(step_fn).parameters
        positional = [p for p in parameters.values()
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        has_varargs = any(p.kind == p.VAR_POSITIONAL for p in parameters.values())
    except (TypeError, ValueError):
        positional, has_varargs = [], True
    if has_varargs or len(positional) >= 2:
        return step_fn(batch, state)
    return step_fn(batch)


def active_train(data, step_fn, checkpoint_path, options=None) -> dict:
    """Train bounded chunks and sleep/resume from a verified cursor.

    ``step_fn`` receives ``(batch, state)`` (or just ``batch``) and may return
    ``{"state": next_state, "loss": value, ...}``.  The callback owns model
    math; this scheduler owns device-aware batching, progress, atomic saves,
    and resumption.  ``options`` supports ``batch_size``, ``epochs``,
    ``max_items``, ``max_seconds``, ``checkpoint_every``, ``reserved_bytes``
    (or ``model_bytes``), ``resume`` and an initial ``state``. An explicit
    batch size is still capped by the device budget.
    """
    if not isinstance(data, list):
        raise VMError("active_train: data must be a List")
    resolved_checkpoint = _runtime_path(checkpoint_path)
    require("fs.read", resource_path(resolved_checkpoint), "active_train")
    require("fs.write", resource_path(resolved_checkpoint), "active_train")
    if len(data) > _MAX_ITEMS:
        raise VMError("active_train: data is too large")
    if not callable(step_fn):
        raise VMError("active_train: step_fn must be a Function")
    if not isinstance(options, Mapping):
        options = {}
    try:
        requested_batch = int(options.get("batch_size", 0))
        epochs = int(options.get("epochs", 1))
        max_items = int(options.get("max_items", 0))
        checkpoint_every = int(options.get("checkpoint_every", 1))
        reserved_bytes = int(options.get(
            "reserved_bytes", options.get("model_bytes", 0)
        ))
        max_seconds = float(options.get("max_seconds", 0.0))
    except (TypeError, ValueError, OverflowError):
        raise VMError("active_train: numeric options are malformed") from None
    if requested_batch < 0 or epochs < 0 or max_items < 0 or reserved_bytes < 0:
        raise VMError(
            "active_train: batch_size, epochs, max_items, and reserved_bytes "
            "must be non-negative"
        )
    if checkpoint_every < 1:
        raise VMError("active_train: checkpoint_every must be at least 1")
    if max_seconds < 0 or not math.isfinite(max_seconds):
        raise VMError("active_train: max_seconds must be non-negative and finite")
    signing_key = options.get("signing_key")
    key_id = options.get("key_id", "")
    require_signature = options.get("require_signature", signing_key is not None)
    if signing_key is not None and (not isinstance(signing_key, str) or not signing_key):
        raise VMError("active_train: signing_key must be non-empty Text")
    if not isinstance(key_id, str) or len(key_id) > 256:
        raise VMError("active_train: key_id must be Text of at most 256 characters")
    if not isinstance(require_signature, bool):
        raise VMError("active_train: require_signature must be Bool")
    path = resolved_checkpoint
    resume = options.get("resume", True) is not False
    state = options.get("state")
    cursor = epoch = steps = processed = 0
    total_steps = total_items = 0
    metrics = {}
    if resume and Path(path).is_file():
        saved = load_checkpoint(path, signing_key, require_signature)
        state = restore_checkpoint_state(saved["state"])
        cursor = saved["cursor"]
        epoch = saved["epoch"]
        metrics = saved.get("metrics", {})
        previous_metadata = saved.get("metadata", {})
        try:
            total_steps = int(previous_metadata.get(
                "total_steps", previous_metadata.get("steps", 0)
            ))
            total_items = int(previous_metadata.get(
                "total_items", previous_metadata.get("items", 0)
            ))
        except (TypeError, ValueError, OverflowError):
            raise VMError("active_train: checkpoint progress metadata is malformed") from None
        if total_steps < 0 or total_items < 0:
            raise VMError("active_train: checkpoint progress metadata is negative")
    if cursor > len(data):
        raise VMError("active_train: checkpoint cursor is beyond the supplied data")
    item_bytes = _estimate(data[0]) if data else 1
    # Explicit requests are upper bounds, never a way to bypass the device
    # budget. Reserve caller-declared model/workspace bytes before sizing the
    # data batch; this matters on small phones and cgroup-limited containers.
    available_for_batch = max(1, _memory_available() - reserved_bytes)
    safe_batch = max(
        1,
        min(_MAX_ITEMS, int(available_for_batch * 0.60) // max(item_bytes, 1)),
    )
    batch_size = min(requested_batch or safe_batch, safe_batch)
    batch_size = max(1, min(batch_size, max(1, len(data) or 1)))
    started = time.monotonic()
    sleeping = False
    last_save = None
    # There is no meaningful batch for an empty dataset. Mark all requested
    # epochs complete instead of invoking the callback with an empty list.
    if not data:
        epoch = epochs
        cursor = 0
    while epoch < epochs:
        if cursor >= len(data):
            epoch += 1
            cursor = 0
            if epoch >= epochs:
                break
        if max_items and processed >= max_items:
            sleeping = True
            break
        if max_seconds and time.monotonic() - started >= max_seconds:
            sleeping = True
            break
        planned = batch_size
        if max_items:
            planned = min(planned, max_items - processed)
        end = min(len(data), cursor + max(1, planned))
        batch = data[cursor:end]
        result = _call_step(step_fn, batch, state)
        if isinstance(result, Mapping):
            if "state" in result:
                state = result["state"]
            updates = {}
            for key, value in result.items():
                key = str(key)
                if key == "state":
                    continue
                if key in updates:
                    raise VMError(
                        f"active_train: duplicate metric key after Text conversion: {key!r}"
                    )
                updates[key] = _safe(value)
            # Keep cumulative metrics that the callback does not update in a
            # particular batch; this makes a sleep/resume boundary retain the
            # last known metric set rather than silently dropping fields.
            metrics = dict(metrics)
            metrics.update(updates)
        else:
            state = result if state is None else state
            metrics = dict(metrics)
            metrics["result"] = _safe(result)
        cursor = end
        steps += 1
        processed += len(batch)
        if checkpoint_path and steps % checkpoint_every == 0:
            last_save = save_checkpoint(
                path, state, cursor, metrics, epoch,
                {
                    "batch_size": batch_size,
                    "steps": total_steps + steps,
                    "items": total_items + processed,
                    "window_steps": steps,
                    "window_items": processed,
                },
                signing_key, key_id,
            )
    complete = epoch >= epochs
    if not complete:
        sleeping = True
    if checkpoint_path and (last_save is None or complete or sleeping):
        last_save = save_checkpoint(
            path, state, cursor, metrics, epoch,
            {
                "batch_size": batch_size,
                "steps": total_steps + steps,
                "items": total_items + processed,
                "window_steps": steps,
                "window_items": processed,
            },
            signing_key, key_id,
        )
    return {
        "state": state,
        "cursor": cursor,
        "epoch": epoch,
        "epochs": epochs,
        "complete": complete,
        "sleeping": sleeping,
        "steps": steps,
        "items": processed,
        "total_steps": total_steps + steps,
        "total_items": total_items + processed,
        "batch_size": batch_size,
        "metrics": metrics,
        "checkpoint": last_save,
        "device": device_profile(),
    }


__all__ = [
    "CHECKPOINT_FORMAT",
    "device_profile",
    "batch_budget",
    "save_checkpoint",
    "load_checkpoint",
    "restore_checkpoint_state",
    "active_train",
]
