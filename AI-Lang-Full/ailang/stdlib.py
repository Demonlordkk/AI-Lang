"""AI-Lang standard library.

Everything here is available without an import, which is central to the
language's brevity goal. Functions are deliberately total: they raise clear
AI-Lang errors instead of leaking Python exceptions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random as _random
import sys
import time
import urllib.error
import urllib.request
import uuid as _uuid
from concurrent.futures import ThreadPoolExecutor

from .errors import AILangRaise, VMError
from .values import Module, RecordValue, display, is_truthy, type_name


def _need_list(v, fname, argname="items"):
    if not isinstance(v, list):
        raise VMError(f"{fname}: {argname} must be a List, got {type_name(v)}")
    return v


def _need_text(v, fname, argname="text"):
    if not isinstance(v, str):
        raise VMError(f"{fname}: {argname} must be Text, got {type_name(v)}")
    return v


def _need_map(v, fname, argname="m"):
    if not isinstance(v, dict):
        raise VMError(f"{fname}: {argname} must be a Map, got {type_name(v)}")
    return v


def _need_int(v, fname, argname):
    if isinstance(v, bool) or not isinstance(v, int):
        raise VMError(f"{fname}: {argname} must be an Int, got {type_name(v)}")
    return v


def _need_fn(v, fname, argname="fn"):
    if not callable(v):
        raise VMError(f"{fname}: {argname} must be a function, got {type_name(v)}")
    return v


def _sort_key(v):
    """Total ordering key so sort() never explodes on mixed lists."""
    if v is None:
        return (0, 0)
    if isinstance(v, bool):
        return (1, int(v))
    if isinstance(v, (int, float)):
        return (2, v)
    if isinstance(v, str):
        return (3, v)
    return (4, display(v))


# ------------------------------------------------------------------ core
def _len(v):
    if isinstance(v, (list, str, dict)):
        return len(v)
    if isinstance(v, RecordValue):
        return len(v.data)
    raise VMError(f"len: cannot measure {type_name(v)}")


def _abs(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise VMError(f"abs: needs a number, got {type_name(v)}")
    return abs(v)


def _sum(items):
    items = _need_list(items, "sum")
    total = 0
    for x in items:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise VMError(f"sum: list must contain only numbers, found {type_name(x)}")
        total += x
    return total


def _to_int(v):
    from .vm import _convert

    return _convert("Int", v)


def _to_real(v):
    from .vm import _convert

    return _convert("Real", v)


def _sqrt(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise VMError(f"sqrt: needs a number, got {type_name(v)}")
    if v < 0:
        raise VMError("sqrt: cannot take the square root of a negative number")
    return math.sqrt(v)


def _assert(cond, message="assertion failed"):
    if not is_truthy(cond):
        raise AILangRaise(display(message))
    return None


# ------------------------------------------------------------------ text
def _split(text, sep=" "):
    _need_text(text, "split")
    if sep == "":
        return list(text)
    return text.split(sep)


def _format(template, values):
    """format("Hello {} you are {}", [name, age])"""
    _need_text(template, "format", "template")
    values = _need_list(values, "format", "values")
    out = []
    idx = 0
    i = 0
    while i < len(template):
        if template[i : i + 2] == "{}":
            if idx >= len(values):
                raise VMError(f"format: template needs more than {len(values)} value(s)")
            out.append(display(values[idx]))
            idx += 1
            i += 2
        else:
            out.append(template[i])
            i += 1
    return "".join(out)


def _contains(haystack, needle):
    if isinstance(haystack, str):
        return _need_text(needle, "contains", "needle") in haystack
    if isinstance(haystack, list):
        return any(_eq(x, needle) for x in haystack)
    if isinstance(haystack, dict):
        return needle in haystack
    raise VMError(f"contains: cannot search {type_name(haystack)}")


def _eq(a, b):
    from .vm import _equal

    return _equal(a, b)


def _index_of(items, value):
    if isinstance(items, str):
        return items.find(value)
    items = _need_list(items, "index_of")
    for i, x in enumerate(items):
        if _eq(x, value):
            return i
    return -1


# ------------------------------------------------------------------ lists
def _first(items):
    items = _need_list(items, "first")
    return items[0] if items else None


def _last(items):
    items = _need_list(items, "last")
    return items[-1] if items else None


def _push(items, value):
    return _need_list(items, "push") + [value]


def _slice(items, start, stop):
    if isinstance(items, str):
        return items[start:stop]
    items = _need_list(items, "slice")
    return items[start:stop]


def _reverse(items):
    if isinstance(items, str):
        return items[::-1]
    return list(reversed(_need_list(items, "reverse")))


def _sort(items):
    return sorted(_need_list(items, "sort"), key=_sort_key)


def _sort_by(items, key):
    items = _need_list(items, "sort_by")
    _need_fn(key, "sort_by", "key")
    return sorted(items, key=lambda x: _sort_key(key(x)))


def _map(items, fn):
    items = _need_list(items, "map")
    _need_fn(fn, "map")
    return [fn(x) for x in items]


def _filter(items, fn):
    items = _need_list(items, "filter")
    _need_fn(fn, "filter")
    return [x for x in items if is_truthy(fn(x))]


def _reduce(items, fn, initial):
    items = _need_list(items, "reduce")
    _need_fn(fn, "reduce")
    acc = initial
    for x in items:
        acc = fn(acc, x)
    return acc


def _any(items, fn):
    items = _need_list(items, "any")
    _need_fn(fn, "any")
    return any(is_truthy(fn(x)) for x in items)


def _all(items, fn):
    items = _need_list(items, "all")
    _need_fn(fn, "all")
    return all(is_truthy(fn(x)) for x in items)


def _find(items, fn):
    items = _need_list(items, "find")
    _need_fn(fn, "find")
    for x in items:
        if is_truthy(fn(x)):
            return x
    return None


def _count(items, fn):
    items = _need_list(items, "count")
    _need_fn(fn, "count")
    return sum(1 for x in items if is_truthy(fn(x)))


def _unique(items):
    items = _need_list(items, "unique")
    out = []
    for x in items:
        if not any(_eq(x, y) for y in out):
            out.append(x)
    return out


def _zip(a, b):
    return [[x, y] for x, y in zip(_need_list(a, "zip", "a"), _need_list(b, "zip", "b"))]


def _enumerate(items):
    return [[i, x] for i, x in enumerate(_need_list(items, "enumerate"))]


def _flatten(items):
    out = []
    for x in _need_list(items, "flatten"):
        if isinstance(x, list):
            out.extend(x)
        else:
            out.append(x)
    return out


def _concat(a, b):
    return _need_list(a, "concat", "a") + _need_list(b, "concat", "b")


def _range(n):
    return list(range(_need_int(n, "range", "n")))


def _range_from(start, stop, step=1):
    _need_int(start, "range_from", "start")
    _need_int(stop, "range_from", "stop")
    _need_int(step, "range_from", "step")
    if step == 0:
        raise VMError("range_from: step cannot be zero")
    return list(range(start, stop, step))


# ------------------------------------------------------------------ maps
def _keys(m):
    return list(_need_map(m, "keys").keys())


def _values(m):
    return list(_need_map(m, "values").values())


def _entries(m):
    return [[k, v] for k, v in _need_map(m, "entries").items()]


def _has(m, key):
    if isinstance(m, dict):
        return key in m
    if isinstance(m, list):
        return any(_eq(x, key) for x in m)
    if isinstance(m, RecordValue):
        return key in m.data
    raise VMError(f"has: cannot inspect {type_name(m)}")


def _get(m, key, default=None):
    if isinstance(m, dict):
        return m.get(key, default)
    if isinstance(m, list):
        if isinstance(key, int) and not isinstance(key, bool) and -len(m) <= key < len(m):
            return m[key]
        return default
    if isinstance(m, RecordValue):
        return m.data.get(key, default)
    raise VMError(f"get: cannot read from {type_name(m)}")


def _set(m, key, value):
    out = dict(_need_map(m, "set"))
    out[key] = value
    return out


def _remove(m, key):
    out = dict(_need_map(m, "remove"))
    out.pop(key, None)
    return out


def _merge(a, b):
    out = dict(_need_map(a, "merge", "a"))
    out.update(_need_map(b, "merge", "b"))
    return out


# ------------------------------------------------------------------ json
def _json_safe(v):
    if isinstance(v, RecordValue):
        return {k: _json_safe(x) for k, x in v.data.items()}
    if isinstance(v, list):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if callable(v):
        raise VMError("json_encode: cannot encode a function")
    return v


def _json_encode(v, indent=None):
    try:
        return json.dumps(_json_safe(v), ensure_ascii=False, sort_keys=True,
                          separators=(",", ":") if not indent else None,
                          indent=indent)
    except (TypeError, ValueError) as e:
        raise VMError(f"json_encode: {e}") from e


def _json_decode(text):
    _need_text(text, "json_decode")
    try:
        return json.loads(text)
    except ValueError as e:
        raise VMError(f"json_decode: invalid JSON: {e}") from e


# ------------------------------------------------------------------ io
def _read_file(path):
    _need_text(path, "read_file", "path")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise VMError(f"read_file: {e}") from e


def _write_file(path, content):
    _need_text(path, "write_file", "path")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(display(content))
        return None
    except OSError as e:
        raise VMError(f"write_file: {e}") from e


def _append_file(path, content):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(display(content))
        return None
    except OSError as e:
        raise VMError(f"append_file: {e}") from e


def _http_get(url, headers=None):
    return _http_request(url, "GET", None, headers)


def _http_post(url, body=None, headers=None):
    return _http_request(url, "POST", body, headers)


def _http_request(url, method, body, headers):
    _need_text(url, "http", "url")
    data = None
    hdrs = {"User-Agent": "AI-Lang/2.0"}
    if headers:
        hdrs.update({str(k): str(v) for k, v in _need_map(headers, "http", "headers").items()})
    if body is not None:
        if isinstance(body, (dict, list)):
            data = _json_encode(body).encode()
            hdrs.setdefault("Content-Type", "application/json")
        else:
            data = display(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8", "replace")
            return {"status": r.status, "body": raw, "headers": dict(r.headers)}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "body": e.read().decode("utf-8", "replace"), "headers": {}}
    except Exception as e:
        raise VMError(f"http {method.lower()}: {e}") from e


# ------------------------------------------------------------------ concurrency
_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=8)
    return _POOL


def _spawn(fn, *args):
    _need_fn(fn, "spawn")
    return _pool().submit(fn, *args)


def _await_all(tasks):
    tasks = _need_list(tasks, "await_all", "tasks")
    out = []
    for t in tasks:
        if hasattr(t, "result"):
            try:
                out.append(t.result(timeout=120))
            except Exception as e:
                raise VMError(f"await_all: task failed: {e}") from e
        else:
            out.append(t)
    return out


# ------------------------------------------------------------------ install
def build_globals(argv=None):
    env = {
        # core
        "len": _len,
        "type_of": type_name,
        "abs": _abs,
        "floor": lambda v: math.floor(v),
        "ceil": lambda v: math.ceil(v),
        "round": lambda v, d=0: round(v, d) if d else round(v),
        "sqrt": _sqrt,
        "pow": lambda a, b: float(a) ** float(b),
        "min": lambda *a: min(a[0], key=_sort_key) if len(a) == 1 and isinstance(a[0], list) else min(a, key=_sort_key),
        "max": lambda *a: max(a[0], key=_sort_key) if len(a) == 1 and isinstance(a[0], list) else max(a, key=_sort_key),
        "sum": _sum,
        "clock": time.time,
        "now": time.time,
        "sleep": lambda s: time.sleep(min(float(s), 30)),
        "print": lambda v="": (print(display(v)), None)[1],
        "assert": _assert,
        "is_nothing": lambda v: v is None,
        # conversions
        "str": display,
        "int": _to_int,
        "real": _to_real,
        "bool": is_truthy,
        # text
        "join": lambda items, sep="": sep.join(display(x) for x in _need_list(items, "join")),
        "split": _split,
        "upper": lambda t: _need_text(t, "upper").upper(),
        "lower": lambda t: _need_text(t, "lower").lower(),
        "trim": lambda t: _need_text(t, "trim").strip(),
        "replace": lambda t, o, n: _need_text(t, "replace").replace(o, n),
        "contains": _contains,
        "starts_with": lambda t, p: _need_text(t, "starts_with").startswith(p),
        "ends_with": lambda t, s: _need_text(t, "ends_with").endswith(s),
        "format": _format,
        "pad": lambda t, w: display(t).ljust(int(w)),
        "pad_left": lambda t, w: display(t).rjust(int(w)),
        "repeat_text": lambda t, n: _need_text(t, "repeat_text") * max(int(n), 0),
        "chars": lambda t: list(_need_text(t, "chars")),
        "code_of": lambda t: ord(_need_text(t, "code_of")[0]),
        "text_of": lambda c: chr(int(c)),
        "index_of": _index_of,
        # lists
        "range": _range,
        "range_from": _range_from,
        "first": _first,
        "last": _last,
        "push": _push,
        "concat": _concat,
        "slice": _slice,
        "reverse": _reverse,
        "sort": _sort,
        "sort_by": _sort_by,
        "map": _map,
        "filter": _filter,
        "reduce": _reduce,
        "any": _any,
        "all": _all,
        "find": _find,
        "count": _count,
        "unique": _unique,
        "zip": _zip,
        "enumerate": _enumerate,
        "flatten": _flatten,
        # maps
        "keys": _keys,
        "values": _values,
        "entries": _entries,
        "has": _has,
        "get": _get,
        "set": _set,
        "remove": _remove,
        "merge": _merge,
        # data
        "json_encode": _json_encode,
        "json_decode": _json_decode,
        "hash_text": lambda t: hashlib.sha256(_need_text(t, "hash_text").encode()).hexdigest(),
        "uuid": lambda: str(_uuid.uuid4()),
        "random": _random.random,
        "random_int": lambda a, b: _random.randint(int(a), int(b)),
        # io
        "read_file": _read_file,
        "write_file": _write_file,
        "append_file": _append_file,
        "env": lambda name: os.environ.get(str(name)),
        "args": lambda: list(argv or []),
        "input": lambda prompt="": input(display(prompt)),
        # net
        "http_get": _http_get,
        "http_post": _http_post,
        # concurrency
        "spawn": _spawn,
        "await_all": _await_all,
    }
    for name, fn in env.items():
        try:
            fn.ailang_name = name
        except AttributeError:
            pass
    return env


def install(env, argv=None):
    env.update(build_globals(argv))
    return env
