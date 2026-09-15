"""Language debugging tools: panic(), line tracing, and disassembly."""
from __future__ import annotations

import io
import os
import subprocess
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

from ailang.errors import AILangError, Panic, ProcessExit  # noqa: E402
from ailang.toolchain import run_source  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def out(source: str, **kw) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_source(source, "<test>", [ROOT], **kw)
    return buf.getvalue().strip()


def cli(*args, env_extra=None, source=None, path=None):
    """Run the real CLI in a subprocess and return (rc, stdout, stderr)."""
    if path is None:
        path = ROOT / "_debug_cli_probe.al"
        if source is not None:
            path.write_text(source)
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run(
        [sys.executable, str(ROOT / "ailang.py"), *args, str(path)],
        capture_output=True, text=True, env=env, cwd=ROOT, timeout=120,
    )
    return r.returncode, r.stdout, r.stderr


# ---------------------------------------------------------------------- panic


def test_panic_is_uncatchable_by_rescue():
    src = '''attempt:
    let x := 1.
    panic("state is corrupt").
    let y := x.
rescue e:
    emit "caught: " + e.message.
done.
emit "unreached".
'''
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            run_source(src, "<test>", [ROOT])
        except Panic as e:
            assert "state is corrupt" in str(e)
        else:
            pytest.fail("panic must propagate out of attempt/rescue")
    assert "caught" not in buf.getvalue()
    assert "unreached" not in buf.getvalue()


def test_panic_escapes_function_boundaries():
    src = '''fn bad() -> Int:
    panic("inner").
    give 1.
done.
fn caller() -> Int:
    attempt:
        give bad().
    rescue e:
        emit "swallowed".
    done.
    give 0.
done.
emit caller().
'''
    try:
        out(src)
    except Panic:
        pass
    else:
        pytest.fail("panic must not be swallowed through a function call")


def test_panic_cli_reports_and_exits_one(tmp_path):
    src = 'emit "before".\npanic("boom: bad state").\nemit "after".\n'
    path = tmp_path / "p.al"
    path.write_text(src)
    rc, outp, errp = cli("run", path=path)
    assert rc == 1
    assert "before" in outp
    assert "after" not in outp
    assert f"ailang: panic: boom: bad state" in errp


def test_panic_type_checked():
    # registered in the type table: one argument, any type
    # a correct call reaches runtime (Panic), not a type error
    with pytest.raises(Panic):
        out('emit panic("x").')
    with pytest.raises(AILangError):
        out("panic().")
    with pytest.raises(AILangError):
        out("panic(1, 2).")
    # the message can be any type, e.g. an expression
    with pytest.raises(Panic):
        out('panic("a" + "b").')


def test_panic_distinct_from_rescueable_errors():
    # a normal error is still caught; only panic escapes
    src = '''attempt:
    let ok := attempt2().
rescue e:
    emit "caught normal".
done.
fn attempt2() -> Int:
    give to Int("not an int").
done.
emit "done".
'''
    try:
        result = out(src)
    except AILangError:
        pytest.fail(f"normal error must be rescueable: {src}")
    assert "caught normal" in result
    assert "done" in result


def test_exit_is_uncatchable_by_rescue_in_both_engines():
    # regression: the native backend's compiled rescue used to catch
    # BaseException, swallowing exit() (and panics) the interpreter lets through
    import os as _os

    for native in ("0", "1"):
        env = dict(_os.environ, AILANG_NATIVE=native)
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                run_source(
                    'attempt:\n    exit(3).\nrescue e:\n    emit "caught".\ndone.\n',
                    "<test>", [ROOT],
                )
        except ProcessExit as e:
            assert e.code == 3
        except Exception as e:
            pytest.fail(f"native={native}: exit must not be rescueable, got {e!r}")
        else:
            pytest.fail(f"native={native}: exit must propagate")
        assert "caught" not in buf.getvalue()


# --------------------------------------------------------------------- trace


TRACE_SRC = '''var s := 0.
var i := 0.
while i < 3:
    s <- s + i.
    i <- i + 1.
done.
emit s.
'''


def test_trace_prints_each_executed_line():
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_source(TRACE_SRC, "<test>", [ROOT], trace=True)
    text = buf.getvalue()
    # header line and every loop body line, once per iteration
    assert "  1 | var s := 0." in text
    assert "  4 |     s <- s + i." in text
    assert "  5 |     i <- i + 1." in text
    assert "  7 | emit s." in text
    # the body executed three times (once per traced iteration)
    assert text.count("s <- s + i.") == 3
    # the program's own output is unchanged
    assert text.strip().endswith("3")


def test_trace_matches_untraced_output():
    traced = io.StringIO()
    with redirect_stdout(traced):
        run_source(TRACE_SRC, "<test>", [ROOT], trace=True)
    plain = out(TRACE_SRC)
    # every plain-output line appears in the traced run
    for line in plain.splitlines():
        assert line in traced.getvalue()


def test_trace_subcommand_runs_and_prints(tmp_path):
    path = tmp_path / "t.al"
    path.write_text(TRACE_SRC)
    rc, outp, errp = cli("trace", path=path)
    assert rc == 0, errp
    assert "while i < 3:" in outp
    assert "s <- s + i." in outp
    assert outp.strip().endswith("3")


def test_run_flag_trace(tmp_path):
    path = tmp_path / "t2.al"
    path.write_text("emit 6 * 7.\n")
    r = subprocess.run(
        [sys.executable, str(ROOT / "ailang.py"), "run", str(path), "--trace"],
        capture_output=True, text=True, env=dict(os.environ), cwd=ROOT, timeout=60,
    )
    assert r.returncode == 0
    assert "1 | emit 6 * 7." in r.stdout
    assert r.stdout.strip().endswith("42")


# ----------------------------------------------------------------------- dis


DIS_SRC = '''fn add(a: Int, b: Int) -> Int:
    give a + b.
done.
emit add(2, 3).
'''


def test_dis_shows_functions_and_opcodes(tmp_path):
    path = tmp_path / "d.al"
    path.write_text(DIS_SRC)
    rc, outp, errp = cli("dis", path=path)
    assert rc == 0, errp
    assert "== <main>()" in outp
    assert "== add(a, b)" in outp
    assert "PUSH" in outp
    assert "HALT" in outp
    assert "PRINT" in outp
    # instruction counts and inline arguments are shown
    assert "6 instructions ==" in outp
    assert "5 instructions ==" in outp
    assert "PUSH 3" in outp
    assert "CALL 2" in outp


def test_dis_instruction_numbers_are_sequential(tmp_path):
    path = tmp_path / "d2.al"
    path.write_text("var x := 1.\nemit x.\n")
    rc, outp, errp = cli("dis", path=path)
    assert rc == 0, errp
    main = outp.split("== add")[0] if "add" in outp else outp
    nums = [int(l.strip().split(":")[0]) for l in main.splitlines()
            if l.strip() and l.strip()[0].isdigit() and ":" in l
            and "constants" not in l]
    assert nums == list(range(len(nums)))


def test_dis_reports_type_errors(tmp_path):
    path = tmp_path / "d3.al"
    path.write_text("let x: Int := \"nope\".\n")
    rc, outp, errp = cli("dis", path=path)
    assert rc != 0
    assert "type error" in (outp + errp).lower()


def test_dis_missing_file(tmp_path):
    rc, outp, errp = cli("dis", path=tmp_path / "nope.al")
    assert rc == 1
    assert "no such file" in errp
