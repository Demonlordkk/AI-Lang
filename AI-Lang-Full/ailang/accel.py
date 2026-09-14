"""Optional host acceleration for AI-Lang's tensor engine.

AI-Lang's core depends on nothing but the host Python runtime, so it runs on
any device: a phone running Termux, a locked-down machine, a browser-hosted
server. The reference tensor engine is written from scratch in pure Python.

When the host *happens to* have numpy installed (Colab always does; a desktop
dev box usually does; Termux can `pkg install python-numpy`), the tensor
engine routes its heaviest operations -- matrix products, reductions,
softmax, losses -- through it for a large training speed-up. This is an
accelerator for the implementation, not a framework the user programs: the
AI-Lang source, the results and the diagnostics are identical either way.

`AILANG_NUMPY=0` forces the pure-Python path (useful for differential
testing); the default is auto-detection.
"""

from __future__ import annotations

import os

_np = None
_tried = False


def _try_import():
    global _np, _tried
    if _tried:
        return _np
    _tried = True
    if os.environ.get("AILANG_NUMPY", "1") == "0":
        return None
    try:
        # optional accelerator only; a bare host Python must keep working
        import numpy as m  # noqa: PLC0415 - deliberately lazy

        _np = m
    except ImportError:
        _np = None
    return _np


def have() -> bool:
    """True when the numpy acceleration path is active."""
    return _try_import() is not None


def np():
    """The numpy module, when available (raises if called without it)."""
    m = _try_import()
    if m is None:
        raise RuntimeError("numpy is not available")
    return m


def info() -> str:
    """Human readable name of the active engine, e.g. 'numpy 2.1.0'."""
    m = _try_import()
    return f"numpy {m.__version__}" if m is not None else "pure python"


def is_arr(x) -> bool:
    m = _np if _tried else _try_import()
    return m is not None and type(x) is m.ndarray


def asarr(x):
    """Coerce a flat list / array to a 1-D float64 numpy array."""
    m = np()
    if type(x) is m.ndarray:
        return x
    return m.asarray(x, dtype=m.float64)
