"""Foreign function interface: call C libraries from AI-Lang.

This is the escape hatch. Anything the standard library does not provide can
be reached through a shared library on the host, so a missing capability is a
binding away rather than a wall.

    let m := ffi_open("libm.so.6").
    let cos := ffi_fn(m, "cos", ["real"], "real").
    emit ffi_call(cos, [0.0]).

Types are named with AI-Lang words -- `int`, `real`, `text`, `bool`, `ptr`,
`void` -- not host spellings, so a binding reads the same on every platform.
The binding is checked when it is declared: an unknown symbol or a bad type
name fails at `ffi_fn`, not at the first call.

Safety: a foreign call can crash the process in ways the VM cannot catch, so
every entry point validates argument counts and types before crossing the
boundary, and the library search is explicit. There is no way to fabricate a
pointer from an integer.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import sys
import threading

from .errors import VMError

# AI-Lang type name -> ctypes type. Deliberately small and explicit.
_TYPES = {
    "void": None,
    "int": ctypes.c_long,
    "i32": ctypes.c_int32,
    "i64": ctypes.c_int64,
    "real": ctypes.c_double,
    "f32": ctypes.c_float,
    "bool": ctypes.c_bool,
    "text": ctypes.c_char_p,
    "ptr": ctypes.c_void_p,
    "byte": ctypes.c_ubyte,
}

_lock = threading.Lock()
_libs = {}
_next_id = [1]


class _Lib:
    __slots__ = ("handle", "path", "id")

    def __init__(self, handle, path, ident):
        self.handle = handle
        self.path = path
        self.id = ident


class _Fn:
    __slots__ = ("cfn", "argtypes", "restype", "name", "lib")

    def __init__(self, cfn, argtypes, restype, name, lib):
        self.cfn = cfn
        self.argtypes = argtypes
        self.restype = restype
        self.name = name
        self.lib = lib


def _resolve(name):
    """Find a shared library by name, path, or bare stem."""
    candidates = [name]
    # a bare name like "m" or "sqlite3" is looked up the platform's way
    found = ctypes.util.find_library(name)
    if found:
        candidates.append(found)
    if not name.startswith(("/", "./", "../")):
        stem = name
        for prefix in ("lib", ""):
            for suffix in (".so", ".so.6", ".dylib", ".dll"):
                candidates.append(f"{prefix}{stem}{suffix}")
    seen = set()
    errors = []
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        try:
            return ctypes.CDLL(cand), cand
        except OSError as e:
            errors.append(f"{cand}: {e}")
    raise VMError(
        f"ffi_open: cannot load library '{name}'. Tried:\n  "
        + "\n  ".join(errors[:4])
    )


def ffi_open(name):
    """Load a shared library and return a handle."""
    if not isinstance(name, str):
        raise VMError("ffi_open: library name must be Text")
    with _lock:
        if name in _libs:
            return _libs[name]
        handle, path = _resolve(name)
        ident = _next_id[0]
        _next_id[0] += 1
        lib = _Lib(handle, path, ident)
        _libs[name] = lib
        return lib


def _ctype(tname, where):
    key = str(tname).lower()
    if key not in _TYPES:
        known = ", ".join(sorted(k for k in _TYPES if k != "void"))
        raise VMError(f"{where}: unknown foreign type '{tname}'; use one of {known}")
    return _TYPES[key]


def ffi_fn(lib, symbol, argtypes=None, restype="void"):
    """Bind a symbol in a loaded library to a typed callable."""
    if not isinstance(lib, _Lib):
        raise VMError("ffi_fn: first argument must be a library from ffi_open")
    if not isinstance(symbol, str):
        raise VMError("ffi_fn: symbol name must be Text")
    argtypes = argtypes if argtypes is not None else []
    if not isinstance(argtypes, list):
        raise VMError("ffi_fn: argument types must be a List of Text")
    try:
        cfn = getattr(lib.handle, symbol)
    except AttributeError:
        raise VMError(
            f"ffi_fn: library '{lib.path}' has no symbol '{symbol}'"
        ) from None
    ats = [_ctype(a, "ffi_fn") for a in argtypes]
    if any(a is None for a in ats):
        raise VMError("ffi_fn: 'void' is not valid as an argument type")
    rt = _ctype(restype, "ffi_fn")
    cfn.argtypes = ats
    cfn.restype = rt
    return _Fn(cfn, [str(a).lower() for a in argtypes], str(restype).lower(), symbol, lib)


def _coerce(value, tname, fname, index):
    """Convert an AI-Lang value to the declared foreign type."""
    def bad(expected):
        got = type(value).__name__
        raise VMError(
            f"ffi_call: {fname} argument {index + 1} expects {expected}, got {got}"
        )

    if tname in ("int", "i32", "i64", "byte"):
        if type(value) is bool or not isinstance(value, int):
            bad(tname)
        return value
    if tname in ("real", "f32"):
        if type(value) is bool or not isinstance(value, (int, float)):
            bad(tname)
        return float(value)
    if tname == "bool":
        if type(value) is not bool:
            bad("bool")
        return value
    if tname == "text":
        if not isinstance(value, str):
            bad("text")
        return value.encode("utf-8")
    if tname == "ptr":
        if value is None:
            return None
        if isinstance(value, int) and type(value) is not bool:
            bad("ptr (pointers come from foreign calls, not integers)")
        return value
    raise VMError(f"ffi_call: unsupported argument type '{tname}'")


def ffi_call(fn, args=None):
    """Call a bound foreign function."""
    if not isinstance(fn, _Fn):
        raise VMError("ffi_call: first argument must be a function from ffi_fn")
    args = args if args is not None else []
    if not isinstance(args, list):
        raise VMError("ffi_call: arguments must be a List")
    if len(args) != len(fn.argtypes):
        raise VMError(
            f"ffi_call: {fn.name} expects {len(fn.argtypes)} argument(s), "
            f"got {len(args)}"
        )
    converted = [_coerce(v, t, fn.name, i) for i, (v, t) in enumerate(zip(args, fn.argtypes))]
    try:
        result = fn.cfn(*converted)
    except Exception as e:  # noqa: BLE001 - foreign code, anything can happen
        raise VMError(f"ffi_call: {fn.name} failed: {e}") from e
    if fn.restype == "void":
        return None
    if fn.restype == "text":
        return result.decode("utf-8", "replace") if result is not None else None
    if fn.restype == "bool":
        return bool(result)
    return result


def ffi_symbol(lib, symbol):
    """Report whether a library exports a symbol."""
    if not isinstance(lib, _Lib):
        raise VMError("ffi_symbol: first argument must be a library from ffi_open")
    return hasattr(lib.handle, str(symbol))


def ffi_info(lib):
    """Describe a loaded library."""
    if not isinstance(lib, _Lib):
        raise VMError("ffi_info: argument must be a library from ffi_open")
    return {"path": lib.path, "id": lib.id, "platform": sys.platform}
