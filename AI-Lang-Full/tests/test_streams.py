"""Lazy streams, memoization and property-based testing (the v2.9 additions)."""

import io
import os
import sys
from contextlib import redirect_stdout

try:
    import pytest
except ImportError:  # no pytest installed (air-gapped): use the bundled shim
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    import _pytest_stub as pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from ailang.toolchain import run_source  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def out(source: str) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_source(source, "<test>", [ROOT])
    return buf.getvalue().strip()


def out_vm(source: str) -> str:
    old = os.environ.get("AILANG_NATIVE")
    os.environ["AILANG_NATIVE"] = "0"
    try:
        return out(source)
    finally:
        if old is None:
            os.environ.pop("AILANG_NATIVE", None)
        else:
            os.environ["AILANG_NATIVE"] = old


STREAM_FIB = r"""
use packages/stream as s.
var a := 0.
var b := 1.
let fib := s.stream(fn() -> Int:
    let tt := a.
    a <- b.
    b <- tt + b.
    give tt.
done).
"""


def test_stream_take_fibonacci():
    src = STREAM_FIB + "emit to Text(s.stream_take(fib, 10))."
    assert out(src) == "[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]"


def test_stream_is_lazy_and_stateful():
    # taking more continues from where the stream left off -- it is a
    # forward-only cursor, not a repeated sequence
    src = STREAM_FIB + """
let first := s.stream_take(fib, 10).
let next5 := s.stream_take(fib, 5).
emit to Text(next5).
"""
    assert out(src) == "[55, 89, 144, 233, 377]"


def test_stream_map_and_filter():
    src = STREAM_FIB + """
emit to Text(s.stream_take(s.stream_map(fib, \\x -> x * 2), 5)).
emit to Text(s.stream_take(s.stream_filter(fib, \\x -> x % 3 == 0), 4)).
"""
    assert out(src) == "[0, 2, 2, 4, 6]\n[21, 144, 987, 6765]"


def test_stream_zip_and_find():
    src = STREAM_FIB + """
emit to Text(s.stream_take(s.stream_zip(fib, fib), 3)).
let big := s.stream_find(fib, \\x -> x > 100 and x % 7 == 0).
emit to Text(big).
"""
    assert out(src) == "[[0, 1], [1, 2], [3, 5]]\n987"


def test_streams_agree_under_both_backends():
    src = STREAM_FIB + """
emit to Text(s.stream_take(s.stream_map(s.stream_filter(fib, \\x -> x % 2 == 0), \\x -> x + 1), 6)).
"""
    assert out(src) == out_vm(src)


def test_memo_caches_results():
    src = """
var calls := 0.
let f := memo(fn(x: Int) -> Int:
    calls <- calls + 1.
    give x * x.
done).
let r1 := f(9).
let r2 := f(9).
emit to Text([r1, r2, calls]).
"""
    assert out(src) == "[81, 81, 1]"


def test_memo_rec_solves_fib_fast():
    src = """
let fib := memo_rec(fn(self: Function, n: Int) -> Int:
    when n < 2:
        give n.
    done.
    give self(n - 1) + self(n - 2).
done).
emit to Text([fib(10), fib(20), fib(30)]).
"""
    assert out(src) == "[55, 6765, 832040]"
    assert out_vm(src) == out(src)


def test_prop_test_passes_valid_property():
    src = """
use packages/testing as t.
emit to Text(t.prop_test(\\x -> abs(x) >= 0, 100, 8)).
"""
    assert out(src) == "true"


def test_prop_test_reports_first_counterexample():
    src = """
use packages/testing as t.
attempt:
    t.prop_test(\\x -> abs(x) >= 1, 500, 42).
rescue e:
    emit e.message.
done.
"""
    got = out(src)
    assert got.startswith("counterexample: x =")


def test_prop_test_catches_raising_check():
    # x - x is always 0, so the check throws on the very first draw,
    # whatever the seed
    src = """
use packages/testing as t.
attempt:
    t.prop_test(\\x -> 100 / (x - x), 500, 3).
rescue e:
    emit e.message.
done.
"""
    got = out(src)
    assert got.startswith("check threw at x =")
