"""AI-Lang standard library.

Everything here is available without an import, which is central to the
language's brevity goal. Functions are deliberately total: they raise clear
AI-Lang errors instead of leaking Python exceptions.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import math
import os
import random as _random
import sys
import time
import urllib.error
import urllib.request
import queue as _queue
import threading as _threading
import uuid as _uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from .errors import AILangRaise, VMError
from .values import Module, RecordValue, display, hashable_key, is_truthy, type_name, unmap_key


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
def _len(value):
    if isinstance(value, (list, str, dict)):
        return len(value)
    if isinstance(value, RecordValue):
        return len(value.data)
    raise VMError(f"len: cannot measure {type_name(value)}")


def _abs(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VMError(f"abs: needs a number, got {type_name(value)}")
    return abs(value)


def _sum(items):
    items = _need_list(items, "sum")
    total = 0
    for x in items:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise VMError(f"sum: list must contain only numbers, found {type_name(x)}")
        total += x
    return total


def _to_int(value):
    from .vm import _convert

    return _convert("Int", value)


def _to_real(value):
    from .vm import _convert

    return _convert("Real", value)


def _sqrt(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VMError(f"sqrt: needs a number, got {type_name(value)}")
    if value < 0:
        raise VMError("sqrt: cannot take the square root of a negative number")
    return math.sqrt(value)


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
    """Pure append: returns a new list (O(n))."""
    return _need_list(items, "push") + [value]


def _append(items, value):
    """In-place append: O(1) amortised. Use in accumulation loops."""
    _need_list(items, "append").append(value)
    return items


def _extend(items, more):
    _need_list(items, "extend").extend(_need_list(more, "extend", "more"))
    return items


def _insert(items, index, value):
    items = _need_list(items, "insert")
    items.insert(_need_int(index, "insert", "index"), value)
    return items


def _pop_at(items, index=-1):
    items = _need_list(items, "pop")
    if not items:
        raise VMError("pop: list is empty")
    idx = _need_int(index, "pop", "index") if index != -1 else -1
    if not -len(items) <= idx < len(items):
        raise VMError(f"pop: index {idx} out of range for list of {len(items)}")
    return items.pop(idx)


def _clear(items):
    _need_list(items, "clear").clear()
    return items


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
    return [unmap_key(k) for k in _need_map(m, "keys").keys()]


def _values(m):
    return list(_need_map(m, "values").values())


def _entries(m):
    return [[unmap_key(k), v] for k, v in _need_map(m, "entries").items()]


def _map_key(key, fname):
    try:
        return hashable_key(key)
    except TypeError as e:
        raise VMError(f"{fname}: {e}") from None


def _has(m, key):
    if isinstance(m, dict):
        return _map_key(key, "has") in m
    if isinstance(m, list):
        return any(_eq(x, key) for x in m)
    if isinstance(m, RecordValue):
        return key in m.data
    raise VMError(f"has: cannot inspect {type_name(m)}")


def _get(m, key, default=None):
    if isinstance(m, dict):
        return m.get(_map_key(key, "get"), default)
    if isinstance(m, list):
        if isinstance(key, int) and not isinstance(key, bool) and -len(m) <= key < len(m):
            return m[key]
        return default
    if isinstance(m, RecordValue):
        return m.data.get(key, default)
    raise VMError(f"get: cannot read from {type_name(m)}")


def _set(m, key, value):
    out = dict(_need_map(m, "set"))
    out[_map_key(key, "set")] = value
    return out


def _remove(m, key):
    out = dict(_need_map(m, "remove"))
    out.pop(_map_key(key, "remove"), None)
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
        return {display(unmap_key(k)): _json_safe(x) for k, x in v.items()}
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
# Relative paths in a program resolve against the directory of the program
# that used them, not whatever directory the shell happened to be in. A
# script is therefore portable: it behaves the same from anywhere.
_SCRIPT_DIR = contextvars.ContextVar("ailang_script_dir", default=None)


def set_script_dir(path):
    """Set the base directory relative paths resolve against for this context."""
    _SCRIPT_DIR.set(os.fspath(path) if path is not None else None)


def script_dir():
    """Current script directory (the one relative paths resolve against)."""
    return _SCRIPT_DIR.get()


def _resolve(path):
    """Resolve a program-supplied path against the script directory.

    An absolute path is returned untouched. A relative path is taken from the
    script's own directory; if nothing is there but the file exists relative
    to the working directory, that is used instead so existing shell-relative
    invocations keep working.
    """
    base = _SCRIPT_DIR.get()
    if not isinstance(path, str) or base is None:
        return path
    if os.path.isabs(path):
        return path
    candidate = os.path.join(base, path)
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(path):
        return path
    # neither exists: report against the script directory, which is where the
    # author almost certainly meant the file to be
    return candidate


def _memo(fn):
    """Memoize a one-argument function: repeated calls with equal arguments
    return the cached result instead of recomputing. The cache is a normal
    dict, so keys must be hashable (numbers, text, lists)."""
    _need_fn(fn, "memo")
    cache = {}

    def wrapper(arg):
        key = _hashable_key(arg)
        if key in cache:
            return cache[key]
        val = fn(arg)
        cache[key] = val
        return val

    wrapper.ailang_name = "memo(" + getattr(fn, "ailang_name", getattr(fn, "__name__", "fn")) + ")"
    return wrapper


def _memo_rec(fn):
    """Memoize a recursive one-argument function: the wrapper is passed to
    the body as `self`, so recursive calls go through the cache.

        let fib := memo_rec(fn(self: Function, n: Int) -> Int:
            when n < 2:
                give n.
            done.
            give self(n - 1) + self(n - 2).
        done).
    """
    _need_fn(fn, "memo_rec")
    cache = {}

    def wrapper(arg):
        key = _hashable_key(arg)
        if key in cache:
            return cache[key]
        val = fn(wrapper, arg)
        cache[key] = val
        return val

    wrapper.ailang_name = "memo_rec(" + getattr(fn, "ailang_name", getattr(fn, "__name__", "fn")) + ")"
    return wrapper


def _exit(code=0):
    """Terminate the program with a status code (default 0)."""
    from .errors import ProcessExit

    raise ProcessExit(int(code))


def _read_line():
    """Read one line from standard input (no trailing newline); empty string
    at end of input. The building block for interactive CLI programs."""
    try:
        line = sys.stdin.readline()
    except (EOFError, OSError):
        return ""
    if line == "":
        return ""
    return line.rstrip("\n").rstrip("\r")


def _read_file(path):
    _need_text(path, "read_file", "path")
    try:
        with open(_resolve(path), "r", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise VMError(f"read_file: {e}") from e


def _write_file(path, content):
    _need_text(path, "write_file", "path")
    try:
        with open(_resolve(path), "w", encoding="utf-8") as f:
            f.write(display(content))
        return None
    except OSError as e:
        raise VMError(f"write_file: {e}") from e


def _append_file(path, content):
    try:
        with open(_resolve(path), "a", encoding="utf-8") as f:
            f.write(display(content))
        return None
    except OSError as e:
        raise VMError(f"append_file: {e}") from e


def _http_get(url, headers=None):
    return _http_request(url, "GET", None, headers)


def _http_post(url, body=None, headers=None):
    return _http_request(url, "POST", body, headers)


_MAX_HTTP_BODY = 16 * 1024 * 1024


def _http_request(url, method, body=None, headers=None, verify=True, timeout=30):
    _need_text(url, "http", "url")
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
    except ValueError as e:
        raise VMError(f"http: invalid URL: {e}") from None
    if scheme not in {"http", "https"} or not hostname:
        raise VMError("http: url must be an http:// or https:// URL with a host")
    if not isinstance(method, str) or not method or any(
        not (ch.isalpha() or ch in "-_") for ch in method
    ):
        raise VMError("http: method must be non-empty Text")
    if type(verify) is not bool:
        raise VMError("http: verify must be a Bool")
    try:
        timeout = float(timeout)
    except (TypeError, ValueError, OverflowError):
        raise VMError("http: timeout must be a finite positive number") from None
    if not math.isfinite(timeout) or timeout <= 0 or timeout > 24 * 60 * 60:
        raise VMError("http: timeout must be a finite positive number")

    data = None
    hdrs = {"User-Agent": "AI-Lang/2.10"}
    if headers is not None:
        supplied = _need_map(headers, "http", "headers")
        for key, value in supplied.items():
            if not isinstance(key, str) or not key or any(
                ord(ch) < 33 or ord(ch) > 126 or ch in ":()<>@,;\\\"/[]?={} \t"
                for ch in key
            ):
                raise VMError("http: header names must be valid Text tokens")
            text_value = str(value)
            if any(ord(ch) < 32 or ord(ch) == 127 for ch in text_value):
                raise VMError("http: header values cannot contain control characters")
            hdrs[key] = text_value
    if body is not None:
        if isinstance(body, (dict, list)):
            data = _json_encode(body).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
        elif isinstance(body, str):
            data = body.encode("utf-8")
        elif isinstance(body, (bytes, bytearray, memoryview)):
            data = bytes(body)
        else:
            raise VMError("http: body must be Text, bytes, a List, or a Map")
        if len(data) > _MAX_HTTP_BODY:
            raise VMError("http: request body is too large")

    ctx = None
    if scheme == "https" and not verify:
        import ssl

        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, data=data, method=method.upper(), headers=hdrs)

    def read_limited(stream):
        raw = stream.read(_MAX_HTTP_BODY + 1)
        if len(raw) > _MAX_HTTP_BODY:
            raise VMError("http: response body is too large")
        return raw.decode("utf-8", "replace")

    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as response:
            return {
                "status": response.status,
                "body": read_limited(response),
                "headers": dict(response.headers),
            }
    except urllib.error.HTTPError as e:
        return {"status": e.code, "body": read_limited(e), "headers": {}}
    except VMError:
        raise
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise VMError(f"http {method.lower()}: {e}") from None


# ------------------------------------------------------------------ concurrency
_POOL = None
_POOL_LOCK = _threading.Lock()


def _pool():
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                _POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="ailang")
    return _POOL


def _isolated_call(fn, args):
    """Invoke a language closure with a child VM execution context.

    The captured Environment is shared deliberately (mutexes/channels make
    shared mutable state explicit), but operand stacks, fuel counters, native
    helper caches, recursion depth and module state are private to the worker.
    """
    try:
        from .vm import Closure
    except ImportError:
        Closure = ()
    if isinstance(fn, Closure):
        child_vm = fn.vm.fork()
        fn = Closure(fn.code, fn.env, child_vm, fn.program)
    return fn(*args)


def _submit(fn, args):
    # ContextVars do not implicitly cross a ThreadPoolExecutor boundary.  The
    # copied context preserves the script directory for relative file/database
    # paths without reintroducing a process-global base directory.
    context = contextvars.copy_context()
    return _pool().submit(context.run, _isolated_call, fn, tuple(args))


def _spawn(fn, *args):
    _need_fn(fn, "spawn")
    return _submit(fn, args)


def _task_timeout(timeout, where):
    try:
        value = float(timeout)
    except (TypeError, ValueError, OverflowError):
        raise VMError(f"{where}: timeout must be a non-negative number") from None
    if not math.isfinite(value) or value < 0:
        raise VMError(f"{where}: timeout must be a non-negative finite number")
    return value


def _await_all(tasks):
    tasks = _need_list(tasks, "await_all", "tasks")
    out = []
    for t in tasks:
        if hasattr(t, "result") and callable(t.result):
            try:
                out.append(t.result(timeout=120))
            except Exception as e:
                raise VMError(f"await_all: task failed: {e}") from e
        else:
            raise VMError("await_all: every item must be a task returned by spawn")
    return out


def _await_task(task, timeout=120):
    """Wait for one spawn task and return its result."""
    if not hasattr(task, "result") or not callable(task.result):
        raise VMError("await: argument must be a task returned by spawn")
    try:
        return task.result(timeout=_task_timeout(timeout, "await"))
    except Exception as e:  # noqa: BLE001 - report the task's fault
        raise VMError(f"await: task failed: {e}") from e


# ------------------------------------------------------------------ locking
def _new_mutex():
    return {"__mutex__": _threading.Lock()}


def _as_mutex(m, name):
    if not isinstance(m, dict) or "__mutex__" not in m:
        raise VMError(f"{name}: argument must be a mutex from mutex()")
    return m["__mutex__"]


def _lock(m, timeout=30):
    lk = _as_mutex(m, "lock")
    if not lk.acquire(timeout=timeout):
        raise VMError("lock: timed out waiting for the mutex")
    return m


def _unlock(m):
    lk = _as_mutex(m, "unlock")
    try:
        lk.release()
    except RuntimeError:
        raise VMError("unlock: mutex is not locked") from None


# ------------------------------------------------------------------ channels
def _new_channel():
    return {"__queue__": _queue.Queue(), "closed": [False]}


def _as_channel(c, name):
    if not isinstance(c, dict) or "__queue__" not in c:
        raise VMError(f"{name}: argument must be a channel from channel()")
    return c


def _channel_send(c, value):
    c = _as_channel(c, "channel_send")
    if c["closed"][0]:
        raise VMError("channel_send: channel is closed")
    c["__queue__"].put(value)
    return None


def _channel_recv(c):
    """Blocking receive; returns nothing once the channel is closed and drained."""
    c = _as_channel(c, "channel_recv")
    q = c["__queue__"]
    while True:
        try:
            return q.get(timeout=0.05)
        except _queue.Empty:
            if c["closed"][0]:
                return None


def _channel_try_recv(c):
    c = _as_channel(c, "channel_try_recv")
    try:
        return c["__queue__"].get_nowait()
    except _queue.Empty:
        return None


def _channel_close(c):
    c = _as_channel(c, "channel_close")
    c["closed"][0] = True
    return None




# ------------------------------------------------------------------ numeric
def _dot(a, b):
    a = _need_list(a, "dot", "a"); b = _need_list(b, "dot", "b")
    if len(a) != len(b):
        raise VMError(f"dot: length mismatch {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def _vec_op(name, f):
    def op(a, b):
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                raise VMError(f"{name}: length mismatch {len(a)} vs {len(b)}")
            return [f(x, y) for x, y in zip(a, b)]
        if isinstance(a, list):
            return [f(x, b) for x in a]
        if isinstance(b, list):
            return [f(a, y) for y in b]
        return f(a, b)
    op.__name__ = name
    return op


def _mean(xs):
    xs = _need_list(xs, "mean")
    if not xs:
        raise VMError("mean: list is empty")
    return sum(xs) / len(xs)


def _variance(xs, sample=False):
    xs = _need_list(xs, "variance")
    n = len(xs)
    if n < 2:
        raise VMError("variance: need at least 2 values")
    m = sum(xs) / n
    return sum((x - m) ** 2 for x in xs) / ((n - 1) if sample else n)


def _stddev(xs, sample=False):
    return math.sqrt(_variance(xs, sample))


def _median(xs):
    xs = sorted(_need_list(xs, "median"))
    n = len(xs)
    if not n:
        raise VMError("median: list is empty")
    mid = n // 2
    return float(xs[mid]) if n % 2 else (xs[mid - 1] + xs[mid]) / 2


def _percentile(xs, q):
    xs = sorted(_need_list(xs, "percentile"))
    if not xs:
        raise VMError("percentile: list is empty")
    if not 0 <= q <= 100:
        raise VMError("percentile: q must be between 0 and 100")
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _normalize(xs):
    xs = _need_list(xs, "normalize")
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return [0.0] * len(xs)
    span = hi - lo
    return [(x - lo) / span for x in xs]


def _standardize(xs):
    xs = _need_list(xs, "standardize")
    if len(xs) < 2:
        raise VMError("standardize: need at least 2 values")
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
    if sd == 0:
        return [0.0] * len(xs)
    return [(x - m) / sd for x in xs]


def _correlation(a, b):
    a = _need_list(a, "correlation", "a"); b = _need_list(b, "correlation", "b")
    if len(a) != len(b):
        raise VMError(f"correlation: length mismatch {len(a)} vs {len(b)}")
    if len(a) < 2:
        raise VMError("correlation: need at least 2 points")
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


# ------------------------------------------------------------------ matrices
def _shape(m):
    if not isinstance(m, list):
        raise VMError("shape: needs a List")
    if m and isinstance(m[0], list):
        return [len(m), len(m[0])]
    return [len(m)]


def _check_rectangular(m, fname):
    if not m or not isinstance(m[0], list):
        return
    width = len(m[0])
    for i, row in enumerate(m):
        if not isinstance(row, list):
            raise VMError(f"{fname}: row {i} is not a list")
        if len(row) != width:
            raise VMError(
                f"{fname}: ragged matrix — row 0 has {width} columns, "
                f"row {i} has {len(row)}"
            )


def _matmul(a, b):
    a = _need_list(a, "matmul", "a"); b = _need_list(b, "matmul", "b")
    if not a or not b:
        raise VMError("matmul: empty matrix")
    if not isinstance(a[0], list) or not isinstance(b[0], list):
        raise VMError("matmul: both arguments must be 2-D lists")
    _check_rectangular(a, "matmul")
    _check_rectangular(b, "matmul")
    n, k, k2, m = len(a), len(a[0]), len(b), len(b[0])
    if k != k2:
        raise VMError(f"matmul: inner dimensions differ ({k} vs {k2})")
    bt = list(zip(*b))
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def _transpose(m):
    m = _need_list(m, "transpose")
    if not m:
        return []
    if not isinstance(m[0], list):
        return [[x] for x in m]
    _check_rectangular(m, "transpose")
    return [list(col) for col in zip(*m)]


def _identity(n):
    n = _need_int(n, "identity", "n")
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _zeros(rows, cols=None):
    r = _need_int(rows, "zeros", "rows")
    if cols is None:
        return [0.0] * r
    return [[0.0] * _need_int(cols, "zeros", "cols") for _ in range(r)]


# ------------------------------------------------------------------ ml
def _sigmoid(x):
    if isinstance(x, list):
        return [_sigmoid(v) for v in x]
    if x < -500:
        return 0.0
    if x > 500:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _relu(x):
    if isinstance(x, list):
        return [_relu(v) for v in x]
    return x if x > 0 else 0.0


def _tanh(x):
    if isinstance(x, list):
        return [_tanh(v) for v in x]
    return math.tanh(x)


def _softmax(xs):
    xs = _need_list(xs, "softmax")
    if not xs:
        return []
    mx = max(xs)
    exps = [math.exp(x - mx) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


def _argmax(xs):
    xs = _need_list(xs, "argmax")
    if not xs:
        raise VMError("argmax: list is empty")
    best, bi = xs[0], 0
    for i, v in enumerate(xs):
        if v > best:
            best, bi = v, i
    return bi


def _argmin(xs):
    xs = _need_list(xs, "argmin")
    if not xs:
        raise VMError("argmin: list is empty")
    best, bi = xs[0], 0
    for i, v in enumerate(xs):
        if v < best:
            best, bi = v, i
    return bi


def _mse(pred, actual):
    pred = _need_list(pred, "mse", "pred"); actual = _need_list(actual, "mse", "actual")
    if len(pred) != len(actual):
        raise VMError(f"mse: length mismatch {len(pred)} vs {len(actual)}")
    if not pred:
        raise VMError("mse: empty input")
    return sum((p - a) ** 2 for p, a in zip(pred, actual)) / len(pred)


def _mae(pred, actual):
    pred = _need_list(pred, "mae", "pred"); actual = _need_list(actual, "mae", "actual")
    if len(pred) != len(actual):
        raise VMError(f"mae: length mismatch {len(pred)} vs {len(actual)}")
    if not pred:
        raise VMError("mae: empty input")
    return sum(abs(p - a) for p, a in zip(pred, actual)) / len(pred)


def _cross_entropy(pred, actual, eps=1e-12):
    pred = _need_list(pred, "cross_entropy", "pred")
    actual = _need_list(actual, "cross_entropy", "actual")
    if len(pred) != len(actual):
        raise VMError("cross_entropy: length mismatch")
    for p in pred:
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0.0 <= p <= 1.0:
            raise VMError(
                f"cross_entropy: probabilities must be between 0 and 1, got {display(p)}"
            )
    return -sum(a * math.log(max(p, eps)) for p, a in zip(pred, actual))


def _accuracy(pred, actual):
    pred = _need_list(pred, "accuracy", "pred")
    actual = _need_list(actual, "accuracy", "actual")
    if len(pred) != len(actual):
        raise VMError("accuracy: length mismatch")
    if not pred:
        raise VMError("accuracy: empty input")
    return sum(1 for p, a in zip(pred, actual) if _eq(p, a)) / len(pred)


def _linear_fit(xs, ys):
    """Least-squares fit. Returns {slope, intercept, r2}."""
    xs = _need_list(xs, "linear_fit", "xs"); ys = _need_list(ys, "linear_fit", "ys")
    if len(xs) != len(ys):
        raise VMError("linear_fit: length mismatch")
    n = len(xs)
    if n < 2:
        raise VMError("linear_fit: need at least 2 points")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise VMError("linear_fit: all x values are identical")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot else 1.0
    return {"slope": slope, "intercept": intercept, "r2": r2}


def _train_test_split(items, ratio=0.8, seed=None):
    items = _need_list(items, "train_test_split")
    rng = _random.Random(seed) if seed is not None else _random
    shuffled = list(items)
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * float(ratio))
    return {"train": shuffled[:cut], "test": shuffled[cut:]}


def _shuffle(items, seed=None):
    out = list(_need_list(items, "shuffle"))
    (_random.Random(seed) if seed is not None else _random).shuffle(out)
    return out


def _sample(items, k, seed=None):
    items = _need_list(items, "sample")
    k = _need_int(k, "sample", "k")
    if k > len(items):
        raise VMError(f"sample: k={k} exceeds list length {len(items)}")
    rng = _random.Random(seed) if seed is not None else _random
    return rng.sample(items, k)


def _one_hot(index, size):
    i = _need_int(index, "one_hot", "index")
    n = _need_int(size, "one_hot", "size")
    if not 0 <= i < n:
        raise VMError(f"one_hot: index {i} out of range for size {n}")
    return [1.0 if j == i else 0.0 for j in range(n)]


def _bincount(items):
    out = {}
    for x in _need_list(items, "bincount"):
        k = _hashable_key(x)
        out[k] = out.get(k, 0) + 1
    return out


# ------------------------------------------------------------------ automation
def _list_dir(path="."):
    try:
        return sorted(os.listdir(_resolve(str(path))))
    except OSError as e:
        raise VMError(f"list_dir: {e}") from e


def _path_exists(path):
    return os.path.exists(_resolve(str(path)))


def _is_dir(path):
    return os.path.isdir(_resolve(str(path)))


def _make_dir(path):
    try:
        os.makedirs(_resolve(str(path)), exist_ok=True)
        return None
    except OSError as e:
        raise VMError(f"make_dir: {e}") from e


def _delete_file(path):
    try:
        os.remove(_resolve(str(path)))
        return None
    except OSError as e:
        raise VMError(f"delete_file: {e}") from e


def _find_files(root, suffix=""):
    out = []
    try:
        for base, _dirs, files in os.walk(_resolve(str(root))):
            for f in sorted(files):
                if not suffix or f.endswith(str(suffix)):
                    out.append(os.path.join(base, f))
    except OSError as e:
        raise VMError(f"find_files: {e}") from e
    return sorted(out)


def _read_lines(path):
    return [ln for ln in _read_file(path).split("\n")]


def _write_lines(path, lines):
    return _write_file(path, "\n".join(display(x) for x in _need_list(lines, "write_lines")))


def _parse_csv_rows(text, sep):
    """Split CSV text into rows of cells, honouring quoted fields.

    A quoted field may contain the separator, newlines and doubled quotes
    ("" inside a quoted field is a literal ").
    """
    rows = []
    row = []
    cell = []
    i = 0
    n = len(text)
    in_quotes = False
    while i < n:
        ch = text[i]
        if in_quotes:
            if ch == '"':
                if i + 1 < n and text[i + 1] == '"':
                    cell.append('"')
                    i += 2
                    continue
                in_quotes = False
                i += 1
                continue
            cell.append(ch)
            i += 1
            continue
        if ch == '"':
            in_quotes = True
            i += 1
        elif ch == sep:
            row.append("".join(cell))
            cell = []
            i += 1
        elif ch == "\n":
            row.append("".join(cell))
            rows.append(row)
            row = []
            cell = []
            i += 1
        elif ch == "\r":
            i += 1
        else:
            cell.append(ch)
            i += 1
    row.append("".join(cell))
    rows.append(row)
    out = []
    for r in rows:
        # a bare blank line is not a row
        if len(r) == 1 and r[0].strip() == "":
            continue
        out.append([c.strip() for c in r])
    return out


def _read_csv(path, sep=","):
    """Parse a CSV with a header row into a List of Maps."""
    text = _read_file(path)
    rows = _parse_csv_rows(text, sep)
    if not rows:
        return []
    header = rows[0]
    out = []
    for cells in rows[1:]:
        entry = {}
        for i, name in enumerate(header):
            entry[name] = _coerce(cells[i]) if i < len(cells) else None
        out.append(entry)
    return out


def _csv_escape(value, sep):
    s = display(value)
    if sep in s or '"' in s or "\n" in s or "\r" in s:
        return '"' + s.replace('"', '""') + '"'
    return s


def _write_csv(path, rows, sep=","):
    rows = _need_list(rows, "write_csv", "rows")
    if not rows:
        return _write_file(path, "")
    if not isinstance(rows[0], dict):
        raise VMError("write_csv: rows must be a List of Maps")
    header = list(rows[0].keys())
    lines = [sep.join(_csv_escape(h, sep) for h in header)]
    for r in rows:
        lines.append(sep.join(_csv_escape(r.get(h, ""), sep) for h in header))
    return _write_file(path, "\n".join(lines) + "\n")


def _coerce(text):
    """Best-effort scalar conversion used by read_csv."""
    t = text.strip()
    if t == "":
        return None
    low = t.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        if ("." in t) or ("e" in low):
            return float(t)
        return int(t)
    except ValueError:
        return t


def _run_command(cmd, timeout=60):
    """Run a shell command; returns {code, out, err}."""
    import subprocess

    try:
        r = subprocess.run(
            str(cmd), shell=True, capture_output=True, text=True, timeout=float(timeout)
        )
        return {"code": r.returncode, "out": r.stdout, "err": r.stderr}
    except subprocess.TimeoutExpired:
        raise VMError(f"run: command timed out after {timeout}s")
    except Exception as e:
        raise VMError(f"run: {e}") from e


def _timestamp(fmt="%Y-%m-%dT%H:%M:%S"):
    return time.strftime(str(fmt), time.localtime())


def _parallel_map(items, fn, workers=8):
    """Apply fn to every item concurrently, preserving order."""
    items = _need_list(items, "parallel_map")
    _need_fn(fn, "parallel_map")
    if not items:
        return []
    workers = int(workers)
    if workers < 1 or workers > 256:
        raise VMError("parallel_map: workers must be between 1 and 256")
    futures = [_submit(fn, (x,)) for x in items]
    out = []
    for f in futures:
        try:
            out.append(f.result(timeout=300))
        except Exception as e:
            raise VMError(f"parallel_map: {e}") from e
    return out


def _retry(fn, attempts=3, delay=0.5):
    """Call fn, retrying on failure with linear backoff."""
    _need_fn(fn, "retry")
    last = None
    for i in range(max(1, int(attempts))):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if i + 1 < int(attempts):
                time.sleep(float(delay) * (i + 1))
    raise VMError(f"retry: all {attempts} attempts failed: {last}")


def _timed(fn):
    """Run fn and report how long it took: {result, seconds}."""
    _need_fn(fn, "timed")
    t0 = time.perf_counter()
    r = fn()
    return {"result": r, "seconds": time.perf_counter() - t0}




# --------------------------------------------------------------- mathematics
def _asin(x):
    v = float(x)
    if not -1.0 <= v <= 1.0:
        raise VMError(f"asin: argument must be between -1 and 1, got {v}")
    return math.asin(v)


def _acos(x):
    v = float(x)
    if not -1.0 <= v <= 1.0:
        raise VMError(f"acos: argument must be between -1 and 1, got {v}")
    return math.acos(v)


def _log10(x):
    v = float(x)
    if v <= 0:
        raise VMError(f"log10: argument must be positive, got {v}")
    return math.log10(v)


def _log2(x):
    v = float(x)
    if v <= 0:
        raise VMError(f"log2: argument must be positive, got {v}")
    return math.log2(v)


def _sign(x):
    v = float(x)
    return 0 if v == 0 else (1 if v > 0 else -1)


def _clamp(value, low, high):
    if low > high:
        raise VMError(f"clamp: low ({low}) is above high ({high})")
    return max(low, min(high, value))


def _lcm(a, b):
    ia, ib = int(a), int(b)
    if ia == 0 or ib == 0:
        return 0
    return abs(ia * ib) // math.gcd(ia, ib)


def _factorial(n):
    i = int(n)
    if i < 0:
        raise VMError(f"factorial: argument must not be negative, got {i}")
    if i > 2000:
        raise VMError("factorial: argument too large (over 2000)")
    return math.factorial(i)


def _pow(base, exp):
    try:
        return float(base) ** float(exp)
    except (ValueError, OverflowError) as e:
        raise VMError(
            f"pow: {display(base)} ^ {display(exp)} has no real result"
        ) from e


# ------------------------------------------------------------------ graphics
def _gfx():
    from . import graphics

    return graphics


# ------------------------------------------------------------------- storage
def _db():
    from . import db

    return db


# ------------------------------------------------------------------- network
def _net():
    from . import net

    return net


# ----------------------------------------------------------------------- ffi
def _ffi():
    from . import ffi

    return ffi


# ------------------------------------------------------------------ autodiff
def _ad():
    from . import autodiff

    return autodiff


def _accel():
    from . import accel

    return accel


def _seed(n):
    """Fix the global random stream (random, randn, shuffle, sample, and
    numpy's global state when the numpy accelerator is active — dropout
    and unseeded randn draw from it, so a seeded program must be
    reproducible word for word with or without numpy)."""
    if n is None:
        return None
    _random.seed(int(n))
    try:
        from . import accel

        m = accel.np()
        if m is not None:
            m.random.seed(int(n))
    except Exception:
        pass
    return None


def _panic(msg):
    """Fatal, unrecoverable abort. Unlike an error, attempt/rescue cannot
    catch a panic — use it for states where continuing is never correct
    (corrupt configuration, missing credentials, impossible invariants)."""
    from .errors import Panic

    raise Panic(str(msg))


def _tensor(value, requires_grad=False):
    from .autodiff import Tensor

    return Tensor.of(value, bool(requires_grad))


def _param(value):
    """A tensor that accumulates gradients — the weights you want trained."""
    from .autodiff import Tensor

    return Tensor.of(value, True)


def _is_tensor(v):
    from .autodiff import Tensor

    return isinstance(v, Tensor)


def _value_of(t):
    from .autodiff import Tensor

    return t.value() if isinstance(t, Tensor) else t


def _grad_of(t):
    from .autodiff import Tensor

    if not isinstance(t, Tensor):
        raise VMError(f"grad_of: needs a tensor, got {type_name(t)}")
    g = t.grad_value()
    if g is None:
        raise VMError(
            "grad_of: no gradient yet — call backward(loss) before reading gradients"
        )
    return g


def _zero_grad(*tensors):
    from .autodiff import Tensor

    for t in tensors:
        if isinstance(t, list):
            for x in t:
                if isinstance(x, Tensor):
                    x.zero_grad()
        elif isinstance(t, Tensor):
            t.zero_grad()
    return None


def _param_list(params, fname):
    """Accept a tensor or a List of tensors; return the list."""
    from .autodiff import Tensor

    if isinstance(params, Tensor):
        return [params]
    ts = _need_list(params, fname, "params")
    for t in ts:
        if not isinstance(t, Tensor):
            raise VMError(f"{fname}: expected tensors, found {type_name(t)}")
    return ts


def _is_arr(v):
    from . import accel

    return accel.is_arr(v)


def _asarr(v):
    from . import accel

    return accel.asarr(v)


def _sgd_step(params, rate):
    """In-place gradient descent update over a list of params."""
    from .autodiff import Tensor

    rate = float(rate)
    for t in _param_list(params, "sgd_step"):
        if t.grad is None:
            continue
        d, g = t.data, t.grad
        if _is_arr(d):
            d -= rate * _asarr(g)
        else:
            for i in range(len(d)):
                d[i] -= rate * g[i]
    return None


def _momentum_state(params, rate=0.01, mu=0.9):
    """SGD with momentum: v <- mu*v + grad; param <- param - rate*v."""
    return {
        "t": 0,
        "v": [
            _asarr([0.0] * p.size) if _is_arr(p.data) else [0.0] * p.size
            for p in _param_list(params, "momentum")
        ],
        "params": _param_list(params, "momentum"),
        "rate": float(rate),
        "mu": float(mu),
    }


def _momentum_step(state, rate=None, mu=None):
    """One momentum update over the state created by `momentum`."""
    rate = state["rate"] if rate is None else float(rate)
    mu = state["mu"] if mu is None else float(mu)
    params = state["params"]
    for pi, p in enumerate(params):
        if p.grad is None:
            continue
        v, d, g = state["v"][pi], p.data, p.grad
        if _is_arr(d):
            # v <- mu*v + g  (decay the old velocity, THEN add the gradient)
            v *= mu
            v += _asarr(g)
            d -= rate * v
        else:
            for i in range(len(d)):
                v[i] = mu * v[i] + g[i]
                d[i] -= rate * v[i]
    return None


def _adam_state(params):
    from .autodiff import Tensor

    return {
        "t": 0,
        "m": [
            _asarr([0.0] * p.size) if _is_arr(p.data) else [0.0] * p.size
            for p in _param_list(params, "adam")
        ],
        "v": [
            _asarr([0.0] * p.size) if _is_arr(p.data) else [0.0] * p.size
            for p in _param_list(params, "adam")
        ],
        "params": _param_list(params, "adam"),
    }


def _adam_step(state, rate=0.01, b1=0.9, b2=0.999, eps=1e-8):
    """Adam update. Converges far faster than plain SGD on most models."""
    params = state["params"]
    state["t"] += 1
    t = state["t"]
    rate = float(rate)
    for pi, p in enumerate(params):
        if p.grad is None:
            continue
        m, v = state["m"][pi], state["v"][pi]
        d, g = p.data, p.grad
        if _is_arr(d):
            np_ = _accel().np()
            ga = _asarr(g)
            m = m * b1 + (1.0 - b1) * ga
            v = v * b2 + (1.0 - b2) * ga * ga
            mh = m / (1.0 - b1 ** t)
            vh = v / (1.0 - b2 ** t)
            d -= rate * mh / (np_.sqrt(vh) + eps)
            state["m"][pi], state["v"][pi] = m, v
        else:
            for i in range(len(d)):
                m[i] = b1 * m[i] + (1.0 - b1) * g[i]
                v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i]
                mh = m[i] / (1.0 - b1 ** t)
                vh = v[i] / (1.0 - b2 ** t)
                d[i] -= rate * mh / (math.sqrt(vh) + eps)
    return None


def _adamw_state(params, rate=0.001, wd=0.01):
    """Adam with decoupled weight decay — the modern default."""
    return {
        "t": 0,
        "m": [
            _asarr([0.0] * p.size) if _is_arr(p.data) else [0.0] * p.size
            for p in _param_list(params, "adamw")
        ],
        "v": [
            _asarr([0.0] * p.size) if _is_arr(p.data) else [0.0] * p.size
            for p in _param_list(params, "adamw")
        ],
        "params": _param_list(params, "adamw"),
        "rate": float(rate),
        "wd": float(wd),
    }


def _adamw_step(state, rate=None, wd=None, b1=0.9, b2=0.999, eps=1e-8):
    """One AdamW update: Adam step plus `wd * param` weight decay."""
    rate = state["rate"] if rate is None else float(rate)
    wd = state["wd"] if wd is None else float(wd)
    params = state["params"]
    state["t"] += 1
    t = state["t"]
    for pi, p in enumerate(params):
        if p.grad is None:
            continue
        m, v = state["m"][pi], state["v"][pi]
        d, g = p.data, p.grad
        if _is_arr(d):
            np_ = _accel().np()
            ga = _asarr(g)
            m = m * b1 + (1.0 - b1) * ga
            v = v * b2 + (1.0 - b2) * ga * ga
            mh = m / (1.0 - b1 ** t)
            vh = v / (1.0 - b2 ** t)
            d -= rate * (mh / (np_.sqrt(vh) + eps) + wd * d)
            state["m"][pi], state["v"][pi] = m, v
        else:
            for i in range(len(d)):
                m[i] = b1 * m[i] + (1.0 - b1) * g[i]
                v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i]
                mh = m[i] / (1.0 - b1 ** t)
                vh = v[i] / (1.0 - b2 ** t)
                d[i] -= rate * (mh / (math.sqrt(vh) + eps) + wd * d[i])
    return None


def _clip_grad(params, max_norm):
    """Scale every gradient so the global norm across `params` <= max_norm."""
    from .autodiff import Tensor

    max_norm = float(max_norm)
    ts = _param_list(params, "clip_grad")
    total = 0.0
    for t in ts:
        if t.grad is None:
            continue
        if _is_arr(t.data):
            total += float((t.grad ** 2).sum())
        else:
            total += sum(g * g for g in t.grad)
    total = math.sqrt(total)
    if total > max_norm:
        scale = max_norm / total
        for t in ts:
            if t.grad is None:
                continue
            if _is_arr(t.data):
                t.grad = t.grad * scale
            else:
                for i in range(len(t.grad)):
                    t.grad[i] *= scale
    return None


def _randn(rows, cols=None, scale=None, seed=None):
    """Gaussian init, scaled by 1/sqrt(fan_in) by default (Xavier-style)."""
    rng = _random.Random(seed) if seed is not None else _random
    r = _need_int(rows, "randn", "rows")
    if cols is None:
        sd = float(scale) if scale is not None else 1.0
        return [rng.gauss(0.0, sd) for _ in range(r)]
    c = _need_int(cols, "randn", "cols")
    sd = float(scale) if scale is not None else (1.0 / math.sqrt(max(r, 1)))
    return [[rng.gauss(0.0, sd) for _ in range(c)] for _ in range(r)]


# ------------------------------------------------------- production training
def _xavier(n_in, n_out, seed=None):
    """Glorot-uniform init: U(-limit, limit), limit = sqrt(6 / (fan_in+fan_out))."""
    n_in = _need_int(n_in, "xavier", "n_in"); n_out = _need_int(n_out, "xavier", "n_out")
    rng = _random.Random(seed) if seed is not None else _random
    limit = math.sqrt(6.0 / max(1, n_in + n_out))
    return [[rng.uniform(-limit, limit) for _ in range(n_out)] for _ in range(n_in)]


def _he_init(n_in, n_out, seed=None):
    """Kaiming-uniform init for ReLU nets: U(-limit, limit), limit = sqrt(6 / fan_in)."""
    n_in = _need_int(n_in, "he_init", "n_in"); n_out = _need_int(n_out, "he_init", "n_out")
    rng = _random.Random(seed) if seed is not None else _random
    limit = math.sqrt(6.0 / max(1, n_in))
    return [[rng.uniform(-limit, limit) for _ in range(n_out)] for _ in range(n_in)]


def _lr_step_decay(step, initial, factor, every):
    """Step decay: initial * factor ** (step // every)."""
    return float(initial) * (float(factor) ** (int(step) // int(every)))


def _lr_cosine(step, total, initial, floor=0.0):
    """Cosine annealing from `initial` down to `floor` over `total` steps."""
    import math as _m

    total = max(1, int(total))
    frac = min(int(step), total) / total
    return float(floor) + (float(initial) - float(floor)) * 0.5 * (1.0 + _m.cos(_m.pi * frac))


def _early_stop(patience):
    return {"best": None, "steps": 0, "patience": _need_int(patience, "early_stop", "patience")}


def _early_stop_step(state, val_loss):
    """One validation checkpoint. Returns {stop, best, improved, waiting}."""
    if not isinstance(state, dict) or "best" not in state:
        raise VMError("early_stop_step: first argument must come from early_stop()")
    val_loss = float(val_loss)
    improved = state["best"] is None or val_loss < state["best"]
    if improved:
        state["best"] = val_loss
        state["steps"] = 0
    else:
        state["steps"] += 1
    return {
        "stop": state["steps"] >= state["patience"],
        "best": state["best"],
        "improved": improved,
        "waiting": state["steps"],
    }


def _gradcheck(params, fn, eps=1e-6):
    """Compare analytical gradients against central numerical differences.

    `fn` is a function of no arguments giving the scalar loss graph. Returns
    the worst relative error over every parameter element — a healthy
    autodiff engine reports values around 1e-7..1e-10.
    """
    from .autodiff import Tensor, backward

    ts = [Tensor.of(p) if not isinstance(p, Tensor) else p for p in _need_list(params, "gradcheck")]
    for p in ts:
        if not p.requires_grad:
            raise VMError("gradcheck: every parameter must be a trainable param()")
    if not callable(fn):
        raise VMError("gradcheck: second argument must be a loss function")
    eps = float(eps)

    # analytical gradients
    for p in ts:
        p.zero_grad()
    loss = fn()
    backward(loss)
    ana = [list(p.grad) for p in ts]

    worst = 0.0
    for idx, p in enumerate(ts):
        data = p.data
        n = len(data)
        base = data[:] if isinstance(data, list) else [float(v) for v in data]
        for i in range(n):
            old = base[i]
            data[i] = old + eps
            fp = float(fn().value() if hasattr(fn(), "value") else fn())
            data[i] = old - eps
            fm = float(fn().value() if hasattr(fn(), "value") else fn())
            data[i] = old
            num = (fp - fm) / (2.0 * eps)
            a = float(ana[idx][i])
            denom = max(1.0, abs(num), abs(a))
            err = abs(num - a) / denom
            if err > worst:
                worst = err
    # leave the graph's gradients in place (the analytical ones)
    return float(worst)


def _f1(y_true, y_pred, threshold=0.5):
    """Binary F1 of predicted probabilities (or 0/1 labels) vs true labels."""
    y_true = _need_list(y_true, "f1", "y_true")
    y_pred = _need_list(y_pred, "f1", "y_pred")
    if len(y_true) != len(y_pred):
        raise VMError(f"f1: length mismatch {len(y_true)} vs {len(y_pred)}")
    tp = fp = fn_ = 0
    for t, v in zip(y_true, y_pred):
        pos, pred = (float(t) >= 0.5), (float(v) >= threshold)
        if pred and pos:
            tp += 1
        elif pred and not pos:
            fp += 1
        elif not pred and pos:
            fn_ += 1
    denom = 2 * tp + fp + fn_
    if denom == 0:
        return 0.0
    return (2.0 * tp) / denom


# ------------------------------------------------------------------ install

# --------------------------------------------------- collection operations
# Each of these replaces a loop and a temporary variable at the call site.
def _group_by(items, key):
    """Bucket items by a key function: group_by(people, \\p -> p.team)."""
    items = _need_list(items, "group_by")
    _need_fn(key, "group_by", "key")
    out = {}
    for x in items:
        out.setdefault(_hashable_key(key(x)), []).append(x)
    return out


def _count_by(items, key):
    """How many items fall in each bucket."""
    items = _need_list(items, "count_by")
    _need_fn(key, "count_by", "key")
    out = {}
    for x in items:
        k = _hashable_key(key(x))
        out[k] = out.get(k, 0) + 1
    return out


def _sum_by(items, key):
    """Total a numeric projection without an accumulator loop."""
    items = _need_list(items, "sum_by")
    _need_fn(key, "sum_by", "key")
    total = 0
    for x in items:
        v = key(x)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise VMError(f"sum_by: expected a number but got {type_name(v)}")
        total += v
    return total


def _max_by(items, key):
    items = _need_list(items, "max_by")
    _need_fn(key, "max_by", "key")
    if not items:
        raise VMError("max_by: the list is empty")
    return max(items, key=lambda x: _sort_key(key(x)))


def _min_by(items, key):
    items = _need_list(items, "min_by")
    _need_fn(key, "min_by", "key")
    if not items:
        raise VMError("min_by: the list is empty")
    return min(items, key=lambda x: _sort_key(key(x)))


def _chunk(items, size):
    """Split into fixed-size pieces: chunk([1,2,3,4,5], 2)."""
    items = _need_list(items, "chunk")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise VMError("chunk: size must be an Int of at least 1")
    return [items[i:i + size] for i in range(0, len(items), size)]


def _windows(items, size):
    """Every consecutive run of `size` items (sliding window)."""
    items = _need_list(items, "windows")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise VMError("windows: size must be an Int of at least 1")
    if size > len(items):
        return []
    return [items[i:i + size] for i in range(len(items) - size + 1)]


def _partition(items, fn):
    """Split into [matching, not_matching] in one pass."""
    items = _need_list(items, "partition")
    _need_fn(fn, "partition")
    yes, no = [], []
    for x in items:
        (yes if is_truthy(fn(x)) else no).append(x)
    return [yes, no]


def _take(items, n):
    items = _need_list(items, "take")
    if isinstance(n, bool) or not isinstance(n, int):
        raise VMError("take: count must be an Int")
    return items[:max(n, 0)]


def _drop(items, n):
    items = _need_list(items, "drop")
    if isinstance(n, bool) or not isinstance(n, int):
        raise VMError("drop: count must be an Int")
    return items[max(n, 0):]


def _take_while(items, fn):
    items = _need_list(items, "take_while")
    _need_fn(fn, "take_while")
    out = []
    for x in items:
        if not is_truthy(fn(x)):
            break
        out.append(x)
    return out


def _drop_while(items, fn):
    items = _need_list(items, "drop_while")
    _need_fn(fn, "drop_while")
    i = 0
    while i < len(items) and is_truthy(fn(items[i])):
        i += 1
    return items[i:]


def _zip_with(a, b, fn):
    """Combine two lists element-wise."""
    a = _need_list(a, "zip_with")
    b = _need_list(b, "zip_with")
    _need_fn(fn, "zip_with")
    if len(a) != len(b):
        raise VMError(f"zip_with: length mismatch {len(a)} vs {len(b)}")
    return [fn(x, y) for x, y in zip(a, b)]


def _flat_map(items, fn):
    """Map then flatten one level."""
    items = _need_list(items, "flat_map")
    _need_fn(fn, "flat_map")
    out = []
    for x in items:
        r = fn(x)
        out.extend(r) if isinstance(r, list) else out.append(r)
    return out


def _pluck(items, field):
    """Pull one field out of every record or map in a list."""
    items = _need_list(items, "pluck")
    name = str(field)
    out = []
    for x in items:
        if isinstance(x, dict):
            if name not in x:
                raise VMError(f"pluck: no key {display(name)}")
            out.append(x[name])
        else:
            out.append(_field_of(x, name))
    return out


def _field_of(obj, name):
    getter = getattr(obj, "get", None)
    if getter is not None:
        return getter(name)
    raise VMError(f"pluck: cannot read '{name}' from {type_name(obj)}")


def _hashable_key(v):
    try:
        return hashable_key(v)
    except TypeError as e:
        raise VMError(str(e)) from None


def _sort_desc(items):
    return sorted(_need_list(items, "sort_desc"), key=_sort_key, reverse=True)


def _index_where(items, fn):
    """First index matching a predicate, or -1."""
    items = _need_list(items, "index_where")
    _need_fn(fn, "index_where")
    for i, x in enumerate(items):
        if is_truthy(fn(x)):
            return i
    return -1


def _counts(items):
    """Frequency table of a list of values."""
    items = _need_list(items, "counts")
    out = {}
    for x in items:
        k = _hashable_key(x)
        out[k] = out.get(k, 0) + 1
    return out


# Public AI-Lang names for helpers whose concise implementation parameter
# names are not part of the language API. VM.call_value translates these
# aliases for named calls; positional calls remain untouched.
_PUBLIC_PARAM_ALIASES = {
    "type_of": {"value": "v"},
    "str": {"value": "v"},
    "bool": {"value": "v"},
    "is_tensor": {"value": "v"},
    "backward": {"loss": "t"},
    "t_reshape": {"shape": "s"},
    "t_slice": {"t": "a"},
    "t_gather": {"t": "a", "indices": "idx"},
    "t_clip": {"t": "a", "low": "lo", "high": "hi"},
    "where_t": {"a": "x", "b": "y"},
    "l2_norm_t": {"t": "a"},
    "bce_logits_t": {"target": "y"},
    "huber_t": {"pred": "p", "target": "y"},
    "mse_t": {"pred": "p", "target": "y"},
    "mae_t": {"pred": "p", "target": "y"},
    "bce_t": {"pred": "p", "target": "y"},
    "ce_t": {"logits": "p", "target": "y"},
    "json_encode": {"value": "v"},
    "canvas_size": {"canvas": "c"},
    "canvas_fill": {"canvas": "c"},
    "canvas_save": {"canvas": "c"},
    "pixel": {"canvas": "c"},
    "rect": {"canvas": "c"},
    "line": {"canvas": "c"},
    "circle": {"canvas": "c"},
    "text": {"canvas": "c"},
    "store_get": {"fallback": "default"},
    "panic": {"message": "msg"},
    "gradcheck": {"loss_fn": "fn"},
}


def _install_public_params(env):
    """Attach non-invasive keyword metadata to exported Python helpers."""
    import inspect

    for name, aliases in _PUBLIC_PARAM_ALIASES.items():
        fn = env.get(name)
        if fn is None or not callable(fn):
            continue
        try:
            signature = inspect.signature(fn)
            parameters = list(signature.parameters.values())
            actual = [
                p.name
                for p in parameters
                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            ]
            reverse = {real: public for public, real in aliases.items()}
            public = tuple(reverse.get(param, param) for param in actual)
            fn._ailang_param_aliases = dict(aliases)
            fn._ailang_public_params = public
            # inspect.signature honours __signature__, which keeps diagnostics
            # and third-party introspection aligned with the public language
            # API even when the Python helper uses terse local names.
            if len(actual) == len(public):
                translated = iter(public)
                fn.__signature__ = signature.replace(
                    parameters=[
                        p.replace(name=next(translated))
                        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
                        else p
                        for p in parameters
                    ]
                )
        except (AttributeError, TypeError, ValueError):
            pass


def build_globals(argv=None):
    env = {
        # core
        "len": _len,
        "type_of": type_name,
        "abs": _abs,
        "floor": lambda value: math.floor(value),
        "ceil": lambda value: math.ceil(value),
        "round": lambda value, digits=0: round(value, digits) if digits else round(value),
        "sqrt": _sqrt,
        "pow": _pow,
        "min": lambda *a: min(a[0], key=_sort_key) if len(a) == 1 and isinstance(a[0], list) else min(a, key=_sort_key),
        "max": lambda *a: max(a[0], key=_sort_key) if len(a) == 1 and isinstance(a[0], list) else max(a, key=_sort_key),
        "sum": _sum,
        "clock": time.time,
        "now": time.time,
        "sleep": lambda seconds: time.sleep(min(float(seconds), 30)),
        "print": lambda value="": (print(display(value)), None)[1],
        "assert": _assert,
        "is_nothing": lambda value: value is None,
        # conversions
        "str": display,
        "int": _to_int,
        "real": _to_real,
        "bool": is_truthy,
        # text
        "join": lambda items, sep="": sep.join(display(x) for x in _need_list(items, "join")),
        "split": _split,
        "upper": lambda text: _need_text(text, "upper").upper(),
        "lower": lambda text: _need_text(text, "lower").lower(),
        "trim": lambda text: _need_text(text, "trim").strip(),
        "replace": lambda text, old, new: _need_text(text, "replace").replace(old, new),
        "contains": _contains,
        "starts_with": lambda text, prefix: _need_text(text, "starts_with").startswith(prefix),
        "ends_with": lambda text, suffix: _need_text(text, "ends_with").endswith(suffix),
        "format": _format,
        "pad": lambda text, width: display(text).ljust(int(width)),
        "pad_left": lambda text, width: display(text).rjust(int(width)),
        "repeat_text": lambda text, times: _need_text(text, "repeat_text") * max(int(times), 0),
        "chars": lambda text: list(_need_text(text, "chars")),
        "code_of": lambda text: ord(_need_text(text, "code_of")[0]),
        "text_of": lambda code: chr(int(code)),
        "index_of": _index_of,
        # lists
        "range": _range,
        "range_from": _range_from,
        "first": _first,
        "last": _last,
        "push": _push,
        "append": _append,
        "extend": _extend,
        "insert": _insert,
        "pop": _pop_at,
        "clear": _clear,
        "concat": _concat,
        "slice": _slice,
        "reverse": _reverse,
        "sort": _sort,
        "sort_by": _sort_by,
        "sort_desc": _sort_desc,
        "group_by": _group_by,
        "count_by": _count_by,
        "sum_by": _sum_by,
        "max_by": _max_by,
        "min_by": _min_by,
        "chunk": _chunk,
        "windows": _windows,
        "partition": _partition,
        "take": _take,
        "drop": _drop,
        "take_while": _take_while,
        "drop_while": _drop_while,
        "zip_with": _zip_with,
        "flat_map": _flat_map,
        "pluck": _pluck,
        "index_where": _index_where,
        "counts": _counts,
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
        # numeric / statistics
        "dot": _dot,
        "vadd": _vec_op("vadd", lambda a, b: a + b),
        "vsub": _vec_op("vsub", lambda a, b: a - b),
        "vmul": _vec_op("vmul", lambda a, b: a * b),
        "vdiv": _vec_op("vdiv", lambda a, b: a / b if b else 0.0),
        "mean": _mean,
        "median": _median,
        "variance": _variance,
        "stddev": _stddev,
        "percentile": _percentile,
        "normalize": _normalize,
        "standardize": _standardize,
        "correlation": _correlation,
        # matrices
        "shape": _shape,
        "matmul": _matmul,
        "transpose": _transpose,
        "identity": _identity,
        "zeros": _zeros,
        # autodiff / tensors
        "tensor": _tensor,
        "param": _param,
        "is_tensor": _is_tensor,
        "value_of": _value_of,
        "grad_of": _grad_of,
        "zero_grad": _zero_grad,
        "backward": lambda t: _ad().backward(t),
        "sgd_step": _sgd_step,
        "adam": _adam_state,
        "adam_step": _adam_step,
        "randn": _randn,
        "shape_of": lambda t: list(_ad().T(t).shape),
        # tensor math (differentiable)
        "t_add": lambda a, b: _ad().add(a, b),
        "t_sub": lambda a, b: _ad().sub(a, b),
        "t_mul": lambda a, b: _ad().mul(a, b),
        "t_div": lambda a, b: _ad().div(a, b),
        "t_pow": lambda a, p: _ad().power(a, p),
        "t_neg": lambda a: _ad().neg(a),
        "t_exp": lambda a: _ad().t_exp(a),
        "t_log": lambda a: _ad().t_log(a),
        "t_sqrt": lambda a: _ad().t_sqrt(a),
        "t_abs": lambda a: _ad().t_abs(a),
        "t_sigmoid": lambda a: _ad().t_sigmoid(a),
        "t_relu": lambda a: _ad().t_relu(a),
        "t_tanh": lambda a: _ad().t_tanh(a),
        "t_softmax": lambda a: _ad().t_softmax(a),
        "t_matmul": lambda a, b: _ad().matmul(a, b),
        "t_matmul_bias": lambda a, b, c: _ad().matmul_bias(a, b, c),
        "ce_softmax_t": lambda logits, y: _ad().ce_softmax(logits, y),
        "t_dropout": lambda x, rate, training=True: _ad().dropout(x, rate, bool(training)),
        "t_transpose": lambda a: _ad().transpose(a),
        "t_reshape": lambda a, s: _ad().reshape(a, s),
        "t_slice": lambda a, start, stop=None, axis=0: _ad().t_slice(a, start, stop, axis),
        "t_gather": lambda a, idx: _ad().t_gather(a, idx),
        "t_concat": lambda a, b, axis=0: _ad().t_concat(a, b, axis),
        "t_clip": lambda a, lo, hi: _ad().t_clip(a, lo, hi),
        "where_t": lambda cond, x, y: _ad().where(cond, x, y),
        "l2_norm_t": lambda a: _ad().l2_norm(a),
        "bce_logits_t": lambda logits, y: _ad().bce_logits_loss(logits, y),
        "huber_t": lambda p, y, delta=1.0: _ad().huber_loss(p, y, delta),
        "l2_penalty_t": lambda *params: _ad().l2_penalty(list(params)),
        "ml_backend": lambda: _accel().info(),
        "momentum": _momentum_state,
        "momentum_step": _momentum_step,
        "adamw": _adamw_state,
        "adamw_step": _adamw_step,
        "clip_grad": _clip_grad,
        "seed": _seed,
        "panic": _panic,
        "sum_t": lambda a: _ad().t_sum(a),
        "mean_t": lambda a: _ad().t_mean(a),
        "mse_t": lambda p, y: _ad().mse_loss(p, y),
        "mae_t": lambda p, y: _ad().mae_loss(p, y),
        "bce_t": lambda p, y: _ad().bce_loss(p, y),
        "ce_t": lambda p, y: _ad().ce_loss(p, y),
        # production training
        "xavier": _xavier,
        "he_init": _he_init,
        "lr_step_decay": _lr_step_decay,
        "lr_cosine": _lr_cosine,
        "early_stop": _early_stop,
        "early_stop_step": _early_stop_step,
        "gradcheck": _gradcheck,
        "f1": _f1,
        # machine learning
        "sigmoid": _sigmoid,
        "relu": _relu,
        "tanh": _tanh,
        "softmax": _softmax,
        "argmax": _argmax,
        "argmin": _argmin,
        "mse": _mse,
        "mae": _mae,
        "cross_entropy": _cross_entropy,
        "accuracy": _accuracy,
        "linear_fit": _linear_fit,
        "train_test_split": _train_test_split,
        "shuffle": _shuffle,
        "sample": _sample,
        "one_hot": _one_hot,
        "bincount": _bincount,
        "exp": math.exp,
        "log": lambda x, base=None: math.log(x) if base is None else math.log(x, base),
        # data
        "json_encode": _json_encode,
        "read_line": _read_line,
        "exit": _exit,
        "memo": _memo,
        "memo_rec": _memo_rec,
        "json_decode": _json_decode,
        "hash_text": lambda text: hashlib.sha256(_need_text(text, "hash_text").encode()).hexdigest(),
        "uuid": lambda: str(_uuid.uuid4()),
        "random": _random.random,
        "random_int": lambda low, high: _random.randint(int(low), int(high)),
        # io
        "read_file": _read_file,
        "write_file": _write_file,
        "append_file": _append_file,
        "env": lambda name: os.environ.get(str(name)),
        # automation / filesystem
        "list_dir": _list_dir,
        "path_exists": _path_exists,
        "is_dir": _is_dir,
        "make_dir": _make_dir,
        "delete_file": _delete_file,
        "find_files": _find_files,
        "read_lines": _read_lines,
        "write_lines": _write_lines,
        "read_csv": _read_csv,
        "write_csv": _write_csv,
        "run": _run_command,
        # mathematics
        "pi": lambda: math.pi,
        "e": lambda: math.e,
        "sin": lambda x: math.sin(float(x)),
        "cos": lambda x: math.cos(float(x)),
        "tan": lambda x: math.tan(float(x)),
        "asin": _asin,
        "acos": _acos,
        "atan": lambda x: math.atan(float(x)),
        "atan2": lambda y, x: math.atan2(float(y), float(x)),
        "sinh": lambda x: math.sinh(float(x)),
        "cosh": lambda x: math.cosh(float(x)),
        "log10": _log10,
        "log2": _log2,
        "hypot": lambda a, b: math.hypot(float(a), float(b)),
        "degrees": lambda x: math.degrees(float(x)),
        "radians": lambda x: math.radians(float(x)),
        "sign": _sign,
        "clamp": _clamp,
        "gcd": lambda a, b: math.gcd(int(a), int(b)),
        "lcm": _lcm,
        "trunc": lambda x: math.trunc(float(x)),
        "is_finite": lambda x: math.isfinite(float(x)),
        "factorial": _factorial,
        # graphics
        "canvas": _gfx().canvas,
        "canvas_size": _gfx().canvas_size,
        "canvas_fill": _gfx().canvas_fill,
        "canvas_save": _gfx().canvas_save,
        "pixel": _gfx().pixel,
        "rect": _gfx().rect,
        "line": _gfx().line,
        "circle": _gfx().circle,
        "text": _gfx().text,
        "plot": _gfx().plot,
        # storage
        "db_open": _db().db_open,
        "db_exec": _db().db_exec,
        "db_query": _db().db_query,
        "db_one": _db().db_one,
        "db_many": _db().db_many,
        "db_transaction": _db().db_transaction,
        "db_tables": _db().db_tables,
        "db_close": _db().db_close,
        "store_open": _db().store_open,
        "store_put": _db().store_put,
        "store_get": _db().store_get,
        "store_delete": _db().store_delete,
        "store_keys": _db().store_keys,
        # networking
        "serve": _net().serve,
        "serve_stop": _net().serve_stop,
        "tcp_listen": _net().tcp_listen,
        "tcp_accept": _net().tcp_accept,
        "tcp_connect": _net().tcp_connect,
        "tcp_send": _net().tcp_send,
        "tcp_receive": _net().tcp_receive,
        "tcp_close": _net().tcp_close,
        # foreign function interface
        "ffi_open": _ffi().ffi_open,
        "ffi_fn": _ffi().ffi_fn,
        "ffi_call": _ffi().ffi_call,
        "ffi_symbol": _ffi().ffi_symbol,
        "ffi_info": _ffi().ffi_info,
        "timestamp": _timestamp,
        "parallel_map": _parallel_map,
        "retry": _retry,
        "timed": _timed,
        "args": lambda: list(argv or []),
        "script_dir": script_dir,
        "input": lambda prompt="": input(display(prompt)),
        # net
        "http_get": _http_get,
        "http_post": _http_post,
        "http_request": _http_request,
        # concurrency
        "spawn": _spawn,
        "await_all": _await_all,
        "await": _await_task,
        "mutex": _new_mutex,
        "lock": _lock,
        "unlock": _unlock,
        "channel": _new_channel,
        "channel_send": _channel_send,
        "channel_recv": _channel_recv,
        "channel_try_recv": _channel_try_recv,
        "channel_close": _channel_close,
    }
    _install_public_params(env)
    for name, fn in env.items():
        try:
            fn.ailang_name = name
        except AttributeError:
            pass
    return env


def install(env, argv=None):
    env.update(build_globals(argv))
    return env
