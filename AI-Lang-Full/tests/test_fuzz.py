"""Fuzz and error-path tests.

Two guarantees, both enforced over many deterministic (seeded) programs:

1. **Equivalence** -- a random program must behave identically under the
   native backend and the bytecode VM: same stdout, same error message or a
   clean success on both.
2. **No raw leaks** -- no input, valid, malformed or pure garbage, may ever
   surface a raw Python exception (a ``Traceback``) to the user. Every
   failure must be a clean AI-Lang error.

All randomness is seeded from the test id, so any failure reproduces with
the same seed and the suite output is deterministic across runs.
"""
from __future__ import annotations

import os
import random
import string
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import io

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ailang.errors import AILangError  # noqa: E402
from ailang.toolchain import run_source  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BS = chr(92)  # backslash, built via chr() so this file stays escape-free

INTS = list(range(0, 12))
SMALL = list(range(-5, 6))
WORDS = ["alpha", "beta", "gamma", "delta", "xray"]


# ------------------------------------------------------------------ generator

def _ident(rng: random.Random) -> str:
    return "v" + "".join(rng.choices(string.ascii_lowercase, k=2))


def _gen_expr(rng: random.Random, depth: int, fns: list) -> str:
    """Random expression. `fns` is a list of (name, arity) user functions."""
    if depth <= 0:
        kind = rng.random()
        if kind < 0.4:
            return str(rng.choice(SMALL))
        if kind < 0.55:
            return str(rng.choice(INTS))
        if kind < 0.7:
            return str(round(rng.uniform(0, 9), 1))
        if kind < 0.85:
            return '"%s"' % rng.choice(WORDS)
        if kind < 0.95:
            return rng.choice(["true", "false"])
        n = rng.randint(0, 3)
        return "[%s]" % ", ".join(_gen_expr(rng, 0, fns) for _ in range(n))

    kind = rng.random()
    if kind < 0.30:
        return _gen_expr(rng, 0, fns)
    if kind < 0.45:
        # binary arithmetic
        op = rng.choice(["+", "-", "*", "%", "/"])
        a = _gen_expr(rng, depth - 1, fns)
        b = _gen_expr(rng, depth - 1, fns)
        if op == "/" and rng.random() < 0.12:
            b = "0"  # deliberate divide-by-zero: must stay clean on both
        return "(%s %s %s)" % (a, op, b)
    if kind < 0.55:
        a = _gen_expr(rng, depth - 1, fns)
        op = rng.choice(["<", ">", "<=", ">=", "==", "!="])
        return "(%s %s %s)" % (a, op, _gen_expr(rng, depth - 1, fns))
    if kind < 0.60:
        return "not %s" % _gen_expr(rng, depth - 1, fns)
    if kind < 0.64:
        return "-%s" % _gen_expr(rng, depth - 1, fns)
    if kind < 0.68:
        return "(%s and %s)" % (_gen_expr(rng, depth - 1, fns), _gen_expr(rng, depth - 1, fns))
    if kind < 0.72:
        return "(%s or %s)" % (_gen_expr(rng, depth - 1, fns), _gen_expr(rng, depth - 1, fns))
    if kind < 0.76:
        # membership over a small list
        inner = ", ".join(str(rng.choice(SMALL)) for _ in range(3))
        return "(%s in [%s])" % (_gen_expr(rng, depth - 1, fns), inner)
    if kind < 0.80:
        # coalesce: left side may fail, right side rescues
        left = rng.choice([
            _gen_expr(rng, depth - 1, fns) + " / 0",
            "get({}, \"%s\")" % rng.choice(WORDS),
            _gen_expr(rng, depth - 1, fns),
        ])
        return "(%s ?? %s)" % (left, _gen_expr(rng, 0, fns))
    if kind < 0.90:
        # builtin call
        b = rng.random()
        e = _gen_expr(rng, depth - 1, fns)
        if b < 0.2:
            return "abs(%s)" % e
        if b < 0.35:
            return "len(%s)" % _gen_expr(rng, depth - 1, fns)
        if b < 0.5:
            return rng.choice(["upper", "lower"]) + "(%s)" % e
        if b < 0.6:
            return "to Text(%s)" % e
        if b < 0.75:
            a2 = _gen_expr(rng, depth - 1, fns)
            return rng.choice(["min", "max"]) + "(%s, %s)" % (e, a2)
        if b < 0.9:
            n = rng.randint(0, 4)
            items = ", ".join(_gen_expr(rng, 0, fns) for _ in range(n))
            return "sum([%s])" % items
        return "sqrt(%d)" % rng.choice([0, 1, 2, 4, 9, 16])
    # user function call, sometimes with named arguments
    if not fns:
        return _gen_expr(rng, depth - 1, fns)
    name, arity = rng.choice(fns)
    params = [_gen_expr(rng, depth - 1, fns) for _ in range(arity)]
    if arity == 2 and rng.random() < 0.4:
        params = ["a: %s" % params[0], "b: %s" % params[1]]
        if rng.random() < 0.5:
            params.reverse()
    return "%s(%s)" % (name, ", ".join(params))


def _gen_fn(rng: random.Random, idx: int, fns: list) -> str:
    """A small function of 1-2 Int/Real args; body may use a counter loop."""
    arity = rng.choice([1, 2])
    names = ["a", "b"][:arity]
    types = [rng.choice(["Int", "Real"]) for _ in range(arity)]
    name = "f%d" % idx
    lines = []
    if rng.random() < 0.4 and arity >= 1:
        counter = _ident(rng)
        lines.append("    var %s := 0." % counter)
        lines.append("    repeat k in range(%d):" % rng.randint(1, 4))
        lines.append("        %s <- %s + 1." % (counter, counter))
        lines.append("    done.")
    lines.append("    give %s." % _gen_expr(rng, 2, fns))
    sig = ", ".join("%s: %s" % (n, t) for n, t in zip(names, types))
    return (
        "fn %s(%s) -> Any:\n%s\ndone.\n" % (name, sig, "\n".join(lines))
    ), (name, arity)


def rand_program(seed: int) -> str:
    rng = random.Random(seed)
    fns: list = []
    n_fns = rng.randint(1, 3)
    chunks = []
    for i in range(n_fns):
        fn_text, fn_spec = _gen_fn(rng, i, fns)
        fns.append(fn_spec)
        chunks.append(fn_text)
    for _ in range(rng.randint(1, 3)):
        chunks.append("emit %s.\n" % _gen_expr(rng, 3, fns))
    return "".join(chunks)


# ------------------------------------------------------------------ running

def _run_inproc(src: str, native: bool):
    """Run in-process; return (stdout, error-message-or-None)."""
    old = os.environ.get("AILANG_NATIVE")
    os.environ["AILANG_NATIVE"] = "1" if native else "0"
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            run_source(src, "<fuzz>", [str(ROOT)], check=True)
        return buf.getvalue().strip(), None
    except AILangError as e:
        return buf.getvalue().strip(), str(e)
    finally:
        if old is None:
            os.environ.pop("AILANG_NATIVE", None)
        else:
            os.environ["AILANG_NATIVE"] = old


def _run_subprocess(src: str, native: bool, tmp_path: Path):
    prog = tmp_path / "fz.al"
    prog.write_text(src, encoding="utf-8")
    env = dict(os.environ)
    env["AILANG_NATIVE"] = "1" if native else "0"
    r = subprocess.run(
        [sys.executable, str(ROOT / "ailang.py"), "run", str(prog)],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    )
    return r.stdout.strip(), r.returncode, r.stderr.strip()


# ------------------------------------------------------------------ the tests

SEEDS = list(range(24))


@pytest.mark.parametrize("seed", SEEDS)
def test_fuzzed_programs_agree_on_both_backends(seed, tmp_path):
    """The native backend must be indistinguishable from the VM on random
    programs: same stdout, and the same clean error (or success) message."""
    src = rand_program(seed)
    a_out, a_code, a_err = _run_subprocess(src, True, tmp_path)
    b_out, b_code, b_err = _run_subprocess(src, False, tmp_path)
    assert a_out == b_out, f"seed {seed}\nnative:\n{a_out}\nvm:\n{b_out}\n---\n{src}"
    assert a_code == b_code, f"seed {seed}: exit {a_code} vs {b_code}\n---\n{src}"
    assert a_err == b_err, f"seed {seed}\nnative:\n{a_err}\nvm:\n{b_err}\n---\n{src}"
    # whatever happened, it must be either a clean success or a clean error
    if a_code != 0:
        assert "Traceback" not in a_err, f"seed {seed}: raw Python leak:\n{a_err}\n---\n{src}"


def test_fuzzed_programs_never_leak_raw_exceptions():
    """In-process, every failure of a random program must be an AILangError;
    any other Python exception is a raw leak. 200 seeds x 2 backends."""
    for seed in range(200):
        src = rand_program(seed + 1000)
        for native in (True, False):
            old = os.environ.get("AILANG_NATIVE")
            os.environ["AILANG_NATIVE"] = "1" if native else "0"
            buf = io.StringIO()
            try:
                with redirect_stdout(buf):
                    run_source(src, "<fuzz>", [str(ROOT)], check=True)
            except AILangError:
                pass  # clean error, as required
            finally:
                if old is None:
                    os.environ.pop("AILANG_NATIVE", None)
                else:
                    os.environ["AILANG_NATIVE"] = old
            # reaching here without a non-AI-Lang exception is the guarantee


GARBAGE = [
    "",
    "   ",
    "\n\n\n",
    ";",
    ";;",
    "{",
    "{}}",
    "(((",
    ")))",
    "\\",
    "\\x",
    "fn",
    "fn fn fn",
    "fn f(:",
    "fn f(a:",
    "done.",
    "done.",
    "let",
    "let x",
    "let x := ",
    "1 2 3",
    "1 +",
    "+ 1",
    "a b",
    "\"unterminated",
    "emit",
    "emit .",
    "give 1.",
    "when:",
    "when 1:",
    "repeat in",
    "var x",
    "x <- ",
    "\u0000",
    "\u00ff",
    "a" * 5000,
    "fn" * 300,
    "1 " * 2000,
    "when true: emit 1.",
    "fn f(a: Int) -> Int:\n give a.\n",
    "record R:\n x Int.\ndone.",
]


@pytest.mark.parametrize("junk", GARBAGE, ids=lambda s: repr(s[:20]))
def test_garbage_input_is_always_clean(junk):
    """Garbage in must produce either a clean AI-Lang error or normal
    output -- never a raw Python traceback."""
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            run_source(junk, "<junk>", [str(ROOT)], check=True)
    except AILangError:
        pass  # clean
    # any non-AILangError exception propagates and fails the test


def test_random_binary_is_always_clean(tmp_path):
    rng = random.Random(777)
    for i in range(10):
        data = bytes(rng.randrange(256) for _ in range(rng.randint(1, 64)))
        p = tmp_path / ("junk%d.bin" % i)
        p.write_bytes(data)
        r = subprocess.run(
            [sys.executable, str(ROOT / "ailang.py"), "run", str(p)],
            capture_output=True,
            text=True,
            env=dict(os.environ),
            cwd=tmp_path,
        )
        assert "Traceback" not in r.stderr, f"raw leak on binary input: {r.stderr}"


ERROR_PATHS = [
    # (source, expected error fragment)
    ("fn f(a: Int) -> Int:\n    give a.\ndone.\nemit f(1, 2).", "2"),
    ("fn f(a: Int, b: Int) -> Int:\n    give a + b.\ndone.\nemit f(1).", "1"),
    ("emit min().", "1 argument"),
    ("emit max(a: 1, b: 2).", "positional arguments only"),
    ("emit round(bogus: 1).", "no parameter named 'bogus'"),
    ("emit 1 / 0.", "division by zero"),
    ("emit [1][5].", "index"),
    ("emit int(\"x\").", "cannot convert"),
    ("emit sqrt(-4).", "negative"),
    ("emit 1 + \"a\".", "cannot add Int and Text"),
    ("emit unknown_fn().", "unknown"),
    ("emit 5 |> max(a: 3).", "positional arguments only"),
    ("fn f(a: Int) -> Int:\n    give a.\ndone.\nfn f(b: Int) -> Int:\n    give b.\ndone.\nemit f(1).", "duplicate"),
    ("let x := 1.\nlet x := 2.\nemit x.", "already defined"),
    ("let x := 1.\nx <- 2.\nemit x.", "let binding"),
    ("emit q(1).", "undefined name"),
]


@pytest.mark.parametrize("src, frag", ERROR_PATHS, ids=lambda s: s[:30])
def test_error_paths_are_clean_and_descriptive(src, frag):
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_source(src, "<err>", [str(ROOT)], check=True)
        raise AssertionError(f"expected clean error, program ran: {buf.getvalue()!r}")
    except AILangError as e:
        assert "Traceback" not in str(e)
        msg = str(e)
        assert frag.lower() in msg.lower(), f"expected {frag!r} in {msg!r}"
    except Exception:
        raise AssertionError("non-AI-Lang exception leaked") from None


def test_parser_rejects_malformed_functions(tmp_path):
    for src in (
        "fn f(a: Int) -> Int:",
        "fn f(a: Int) -> Int:\n    give a +",
        "fn f(a: Int) -> Int:\n    give a.",
        "fn f(a: Int) -> Int:\n    when a > 0:",
        "fn f(a: Int) -> Int:\n    repeat i in",
    ):
        r = subprocess.run(
            [sys.executable, str(ROOT / "ailang.py"), "run",
             str(_write(tmp_path, src))],
            capture_output=True,
            text=True,
            env=dict(os.environ),
            cwd=tmp_path,
        )
        assert r.returncode != 0, f"malformed program must fail: {src!r}"
        assert "Traceback" not in r.stderr, f"raw leak: {r.stderr}\n---\n{src}"


def _write(tmp_path: Path, src: str) -> Path:
    p = tmp_path / "m.al"
    p.write_text(src, encoding="utf-8")
    return p
