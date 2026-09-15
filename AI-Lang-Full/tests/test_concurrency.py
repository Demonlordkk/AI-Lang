"""Concurrency primitives: await(task), mutex/lock/unlock, channel send/recv.

These run real threads: spawn hands work to the shared thread pool, so the
primitives below are what an AI-Lang program uses to coordinate them.
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

try:
    import pytest
except ImportError:  # no pytest installed (air-gapped): use the bundled shim
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    import _pytest_stub as pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ailang.errors import AILangError  # noqa: E402
from ailang.toolchain import run_source  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def out(source: str) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_source(source, "<test>", [ROOT])
    return buf.getvalue().strip()


def fails(source: str, fragment: str = "") -> str:
    try:
        out(source)
    except AILangError as e:
        if fragment:
            assert fragment.lower() in str(e).lower(), f"expected {fragment!r} in {e!r}"
        return str(e)
    raise AssertionError(f"expected clean error for: {source!r}")


def test_await_returns_the_single_task_result():
    src = '''
fn work() -> Int:
    sleep(0.05).
    give 41 * 2 + 1.
done.
let t := spawn(work).
emit await(t).
let ts := [spawn(work), spawn(work)].
emit sum(map(ts, \\t -> await(t))).
'''
    assert out(src) == "83\n166"


def test_await_reports_a_task_fault():
    src = '''
fn boom() -> Real:
    give 1 / 0.
done.
let t := spawn(boom).
emit await(t).
'''
    fails(src, "task failed")


def test_await_rejects_non_tasks():
    fails('emit await(5).', "must be a task")


def test_mutex_serializes_shared_updates():
    src = '''
var total := 0.
let m := mutex().
fn bump() -> Void:
    var i := 0.
    while i < 500:
        lock(m).
        total <- total + 1.
        unlock(m).
        i <- i + 1.
    done.
done.
let a := spawn(bump).
let b := spawn(bump).
await_all([a, b]).
emit total.
'''
    assert out(src) == "1000"


def test_lock_timeout_is_a_clean_error():
    src = '''
let m := mutex().
lock(m).
fn waiter() -> Int:
    lock(m, 0.2).
    give 1.
done.
let t := spawn(waiter).
emit await(t).
'''
    fails(src, "timed out")


def test_mutex_argument_errors_are_clean():
    fails("lock(5).", "expects Map")
    fails("unlock(5).", "expects Map")
    # a worker thread can take the mutex held by the main thread
    src = '''
let m := mutex().
lock(m).
fn releaser() -> Void:
    sleep(0.2).
    unlock(m).
done.
fn waiter() -> Int:
    lock(m).
    give 7.
done.
let r := spawn(releaser).
let t := spawn(waiter).
await(r).
emit await(t).
'''
    assert out(src) == "7"


def test_channel_producer_consumer():
    src = '''
let ch := channel().
fn prod() -> Void:
    var i := 1.
    while i <= 5:
        channel_send(ch, i).
        i <- i + 1.
    done.
done.
let p := spawn(prod).
var s := 0.
var i := 0.
while i < 5:
    s <- s + channel_recv(ch).
    i <- i + 1.
done.
await(p).
emit s.
'''
    assert out(src) == "15"


def test_channel_close_drains_then_yields_nothing():
    src = '''
let ch := channel().
channel_send(ch, 1).
channel_send(ch, 2).
channel_close(ch).
emit channel_recv(ch).
emit channel_recv(ch).
emit is_nothing(channel_recv(ch)).
emit is_nothing(channel_try_recv(ch)).
'''
    assert out(src) == "1\n2\ntrue\ntrue"


def test_channel_try_recv_returns_nothing_when_empty():
    src = '''
let ch := channel().
emit is_nothing(channel_try_recv(ch)).
channel_close(ch).
'''
    assert out(src) == "true"


def test_channel_send_after_close_is_clean_error():
    src = '''
let ch := channel().
channel_close(ch).
channel_send(ch, "x").
'''
    fails(src, "closed")


def test_channel_argument_errors_are_clean():
    fails('channel_send(5, "x").', "expects Map")
    fails("channel_recv(5).", "expects Map")
    fails("channel_close(5).", "expects Map")
