"""Differential tests for the native backend.

The native backend compiles AI-Lang functions and top-level loops to host
bytecode. It is only correct if it is *indistinguishable* from the bytecode
VM, so almost every test here runs the same program twice -- once with the
backend enabled and once with `AILANG_NATIVE=0` -- and demands identical
output.

That equivalence is the whole safety argument: any program the backend gets
wrong shows up as a divergence.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

try:
    import pytest
except ImportError:  # no pytest installed (air-gapped): use the bundled shim
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    import _pytest_stub as pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = sorted((ROOT / "examples").rglob("*.al"))


def _run(src: str, native: bool, tmp_path: Path):
    prog = tmp_path / "p.al"
    prog.write_text(src, encoding="utf-8")
    env = dict(os.environ)
    env["AILANG_NATIVE"] = "1" if native else "0"
    r = subprocess.run(
        [sys.executable, str(ROOT / "ailang.py"), "run", str(prog)],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=300,
    )
    return r.stdout.strip(), r.stderr.strip(), r.returncode


def both(src: str, tmp_path: Path) -> str:
    """Run under both backends, assert they agree, return the shared output."""
    a_out, a_err, a_code = _run(src, True, tmp_path)
    b_out, b_err, b_code = _run(src, False, tmp_path)
    assert a_out == b_out, f"stdout diverged\nnative:\n{a_out}\nvm:\n{b_out}"
    assert a_code == b_code, f"exit code diverged: {a_code} vs {b_code}"
    # error text must match too, since `attempt` can observe it
    assert a_err == b_err, f"stderr diverged\nnative:\n{a_err}\nvm:\n{b_err}"
    return a_out


# ------------------------------------------------------------ arithmetic core


def test_int_arithmetic(tmp_path):
    src = (
        "fn f(a: Int, b: Int) -> Int:\n"
        "    give a + b * 2 - 1.\n"
        "done.\n"
        "emit f(3, 4).\n"
    )
    assert both(src, tmp_path) == "10"


def test_division_always_yields_real(tmp_path):
    src = "fn d(a: Int, b: Int) -> Real:\n    give a / b.\ndone.\nemit d(6, 3).\n"
    assert both(src, tmp_path) == "2.0"


def test_division_by_zero_matches_the_interpreter(tmp_path):
    src = "fn d(a: Int, b: Int) -> Real:\n    give a / b.\ndone.\nemit d(1, 0).\n"
    both(src, tmp_path)  # both must fail the same way


def test_text_concatenation_is_not_numeric_addition(tmp_path):
    """The guarded fast path must fall back cleanly for non-Int operands."""
    src = 'fn c(a: Text, b: Text) -> Text:\n    give a + b.\ndone.\nemit c("x", "y").\n'
    assert both(src, tmp_path) == "xy"


def test_real_arithmetic_falls_back(tmp_path):
    src = "fn f(a: Real, b: Real) -> Real:\n    give a + b.\ndone.\nemit f(1.5, 2.25).\n"
    assert both(src, tmp_path) == "3.75"


def test_bool_is_not_an_int(tmp_path):
    """AI-Lang does not treat true as 1; the fast path must not either."""
    src = "emit true == 1.\nemit 1 == true.\n"
    assert both(src, tmp_path) == "false\nfalse"


def test_mixed_int_real_comparison(tmp_path):
    src = "fn f(a: Int, b: Real) -> Bool:\n    give a < b.\ndone.\nemit f(1, 1.5).\n"
    assert both(src, tmp_path) == "true"


def test_equality_of_lists_and_maps(tmp_path):
    src = 'emit [1, 2] == [1, 2].\nemit {"a": 1} == {"a": 1}.\nemit [1] == [2].\n'
    assert both(src, tmp_path) == "true\ntrue\nfalse"


def test_modulo_and_negation(tmp_path):
    src = "fn f(a: Int) -> Int:\n    give -a % 3.\ndone.\nemit f(7).\nemit f(-7).\n"
    both(src, tmp_path)


# ----------------------------------------------------------------- truthiness


def test_only_false_and_nothing_are_falsy(tmp_path):
    src = (
        "fn t(v: Any) -> Text:\n"
        "    when v:\n"
        '        give "truthy".\n'
        "    done.\n"
        '    give "falsy".\n'
        "done.\n"
        "emit t(0).\nemit t(\"\").\nemit t([]).\nemit t(false).\nemit t(nothing).\n"
    )
    assert both(src, tmp_path) == "truthy\ntruthy\ntruthy\nfalsy\nfalsy"


def test_and_or_short_circuit(tmp_path):
    src = (
        "fn safe(c: Int, t: Int) -> Bool:\n"
        "    give c != 0 and t / c > 1.0.\n"
        "done.\n"
        "emit safe(0, 5).\nemit safe(2, 6).\n"
    )
    assert both(src, tmp_path) == "false\ntrue"


def test_not_operator(tmp_path):
    src = "fn n(v: Any) -> Bool:\n    give not v.\ndone.\nemit n(false).\nemit n(0).\n"
    assert both(src, tmp_path) == "true\nfalse"


# ------------------------------------------------------------ control flow


def test_recursion(tmp_path):
    src = (
        "fn fib(n: Int) -> Int:\n"
        "    when n < 2:\n"
        "        give n.\n"
        "    done.\n"
        "    give fib(n - 1) + fib(n - 2).\n"
        "done.\n"
        "emit fib(18).\n"
    )
    assert both(src, tmp_path) == "2584"


def test_mutual_recursion(tmp_path):
    src = (
        "fn is_even(n: Int) -> Bool:\n"
        "    when n == 0:\n"
        "        give true.\n"
        "    done.\n"
        "    give is_odd(n - 1).\n"
        "done.\n"
        "fn is_odd(n: Int) -> Bool:\n"
        "    when n == 0:\n"
        "        give false.\n"
        "    done.\n"
        "    give is_even(n - 1).\n"
        "done.\n"
        "emit is_even(10).\n"
    )
    assert both(src, tmp_path) == "true"


def test_loops_with_stop_and_next(tmp_path):
    src = (
        "var a := 0.\n"
        "repeat i in range(5):\n"
        "    when i == 3:\n"
        "        stop.\n"
        "    done.\n"
        "    a <- a + i.\n"
        "done.\n"
        "var b := 0.\n"
        "repeat i in range(5):\n"
        "    when i % 2 == 0:\n"
        "        next.\n"
        "    done.\n"
        "    b <- b + i.\n"
        "done.\n"
        'emit "{a} {b}".\n'
    )
    assert both(src, tmp_path) == "3 4"


def test_while_loop_carries_variables(tmp_path):
    src = (
        "var d := 0.\nvar k := 0.\n"
        "while k < 4:\n"
        "    d <- d + k.\n"
        "    k <- k + 1.\n"
        "done.\n"
        'emit "{d} {k}".\n'
    )
    assert both(src, tmp_path) == "6 4"


def test_loop_over_a_list(tmp_path):
    src = 'var s := "".\nrepeat w in ["a", "b"]:\n    s <- s + w.\ndone.\nemit s.\n'
    assert both(src, tmp_path) == "ab"


def test_loop_with_index(tmp_path):
    src = 'repeat w at i in ["a", "b"]:\n    emit "{i}:{w}".\ndone.\n'
    assert both(src, tmp_path) == "0:a\n1:b"


def test_nested_loops(tmp_path):
    src = (
        "var t := 0.\n"
        "repeat i in range(3):\n"
        "    repeat j in range(3):\n"
        "        t <- t + 1.\n"
        "    done.\n"
        "done.\n"
        "emit t.\n"
    )
    assert both(src, tmp_path) == "9"


# -------------------------------------------------------------- data & errors


def test_lists_maps_and_indexing(tmp_path):
    src = (
        "fn f() -> Any:\n"
        '    let m := {"a": [1, 2, 3]}.\n'
        "    give m.a[1].\n"
        "done.\n"
        "emit f().\n"
    )
    assert both(src, tmp_path) == "2"


def test_index_out_of_range_message(tmp_path):
    src = "fn f() -> Any:\n    let xs := [1].\n    give xs[9].\ndone.\nemit f().\n"
    both(src, tmp_path)


def test_negative_index(tmp_path):
    src = "fn f() -> Any:\n    give [1, 2, 3][-1].\ndone.\nemit f().\n"
    assert both(src, tmp_path) == "3"


def test_index_assignment(tmp_path):
    src = (
        "fn f() -> List:\n"
        "    var xs := [1, 2, 3].\n"
        "    xs[0] <- 99.\n"
        "    give xs.\n"
        "done.\n"
        "emit f().\n"
    )
    assert both(src, tmp_path) == "[99, 2, 3]"


def test_attempt_rescue_inside_a_function(tmp_path):
    src = (
        "fn f(n: Int) -> Text:\n"
        "    attempt:\n"
        "        when n == 0:\n"
        '            raise "bad".\n'
        "        done.\n"
        '        give "ok".\n'
        "    rescue e:\n"
        '        give "caught " + e.message.\n'
        "    done.\n"
        "done.\n"
        "emit f(1).\nemit f(0).\n"
    )
    assert both(src, tmp_path) == "ok\ncaught bad"


def test_rescue_catches_a_runtime_error(tmp_path):
    src = (
        "fn f() -> Text:\n"
        "    attempt:\n"
        "        let xs := [1].\n"
        '        give to Text(xs[9]).\n'
        "    rescue e:\n"
        '        give "recovered".\n'
        "    done.\n"
        "done.\n"
        "emit f().\n"
    )
    assert both(src, tmp_path) == "recovered"


def test_conversion(tmp_path):
    src = 'fn f(n: Int) -> Text:\n    give to Text(n) + "!".\ndone.\nemit f(7).\n'
    assert both(src, tmp_path) == "7!"


def test_interpolation_inside_a_native_function(tmp_path):
    src = 'fn f(n: Int) -> Text:\n    give "n={n * 2}".\ndone.\nemit f(4).\n'
    assert both(src, tmp_path) == "n=8"


# ------------------------------------------------------------------- closures


def test_closure_capturing_a_parameter_is_not_miscompiled(tmp_path):
    """Each closure must keep its own captured value, not the first one."""
    src = (
        "fn make_adder(n: Int) -> Function:\n"
        "    give \\x -> x + n.\n"
        "done.\n"
        "let add5 := make_adder(5).\n"
        "let add9 := make_adder(9).\n"
        "emit add5(10).\nemit add9(10).\n"
    )
    assert both(src, tmp_path) == "15\n19"


def test_closure_mutating_enclosing_state(tmp_path):
    src = (
        "fn make_counter() -> Function:\n"
        "    var total := 0.\n"
        "    give fn() -> Int:\n"
        "        total <- total + 1.\n"
        "        give total.\n"
        "    done.\n"
        "done.\n"
        "let tick := make_counter().\n"
        "emit tick().\nemit tick().\nemit tick().\n"
    )
    assert both(src, tmp_path) == "1\n2\n3"


def test_recursion_limit_still_reported(tmp_path):
    src = "fn f(n: Int) -> Int:\n    give f(n + 1).\ndone.\nemit f(0).\n"
    _, err, code = _run(src, True, tmp_path)
    assert code != 0
    assert "recursion limit" in err


# -------------------------------------------------------------- immutability


def test_let_binding_cannot_be_mutated_in_a_lifted_loop(tmp_path):
    src = "let x := 0.\nrepeat i in range(3):\n    x <- x + 1.\ndone.\nemit x.\n"
    _, err, code = _run(src, True, tmp_path)
    assert code != 0
    assert "let binding" in err


def test_parameters_stay_immutable(tmp_path):
    src = "fn f(a: Int) -> Int:\n    a <- 5.\n    give a.\ndone.\nemit f(1).\n"
    _, err, code = _run(src, True, tmp_path)
    assert code != 0


# ------------------------------------------------------- whole-example parity


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_examples_agree_under_both_backends(path):
    env_native = dict(os.environ, AILANG_NATIVE="1")
    env_vm = dict(os.environ, AILANG_NATIVE="0")
    cwd = path.parent
    a = subprocess.run(
        [sys.executable, str(ROOT / "ailang.py"), "run", str(path)],
        capture_output=True, text=True, env=env_native, cwd=cwd,
        timeout=300,
    )
    b = subprocess.run(
        [sys.executable, str(ROOT / "ailang.py"), "run", str(path)],
        capture_output=True, text=True, env=env_vm, cwd=cwd,
        timeout=300,
    )
    assert a.stdout == b.stdout, f"{path.name} diverged between backends"
    assert a.returncode == b.returncode


# ------------------------------------------------------------ backend control


def test_backend_can_be_disabled():
    from ailang import native

    os.environ["AILANG_NATIVE"] = "0"
    try:
        assert native.enabled() is False
    finally:
        os.environ.pop("AILANG_NATIVE", None)
    assert native.enabled() is True


def test_unsupported_function_is_declined_not_miscompiled():
    """A function the backend cannot model must return None, never bad code."""
    from ailang.native import try_compile
    from ailang.parser import Parser

    # `use` inside a function body is not modelled
    prog = Parser("fn f() -> Int:\n    give 1.\ndone.\n").program()
    fn = [s for s in prog.statements if type(s).__name__ == "Fn"][0]
    compiled = try_compile(fn, lambda n: None, "f", is_global=lambda n: True)
    assert compiled is not None
    assert compiled() == 1


# ------------------------------------------------- pipeline / operator coverage
#
# The native compiler lifts functions to bytecode, so operator precedence and
# the `|>` desugar must survive lifting exactly as they do in the VM.


def test_pipeline_inside_native_function(tmp_path):
    src = (
        "fn add(x: Int, n: Int) -> Int:\n"
        "    give x + n.\n"
        "done.\n"
        "fn half(x: Real) -> Real:\n"
        "    give x / 2.\n"
        "done.\n"
        "fn f(x: Int) -> Real:\n"
        "    give x |> add(10) |> half().\n"
        "done.\n"
        'emit to Text(f(7)) + " " + to Text(f(3)).\n'
    )
    assert both(src, tmp_path) == "8.5 6.5"


def test_pipeline_precedence_in_lifted_conditionals(tmp_path):
    src = (
        "fn sign(x: Int) -> Text:\n"
        "    when x > 0:\n"
        '        give "pos".\n'
        "    elif x < 0:\n"
        '        give "neg".\n'
        "    else:\n"
        '        give "zero".\n'
        "    done.\n"
        "done.\n"
        'emit sign(5 - 9) + sign(0) + sign(2 + 3).\n'
    )
    assert both(src, tmp_path) == "negzeropos"


def test_named_args_in_native_functions(tmp_path):
    src = (
        "fn add2(a: Int, b: Int) -> Int:\n"
        "    give a + b.\n"
        "done.\n"
        "fn f() -> Int:\n"
        "    give add2(a: 4, b: 5) + add2(b: 6, a: 1).\n"
        "done.\n"
        "emit f().\n"
    )
    assert both(src, tmp_path) == "16"


def test_coalesce_membership_and_pipe_in_loop(tmp_path):
    src = (
        "fn f(items: Any) -> Any:\n"
        "    var out := [0].\n"
        "    repeat x in items:\n"
        "        when x in [0.5, 1.5]:\n"
        "            out <- push(out, 1).\n"
        "        done.\n"
        "        when (x ?? 0.25) == 2.25:\n"
        "            out <- push(out, 2).\n"
        "        else:\n"
        "            out <- push(out, 0).\n"
        "        done.\n"
        "    done.\n"
        "    give out.\n"
        "done.\n"
        'emit f([0.5, 2.25, 3.0, 1.5]).\n'
    )
    assert both(src, tmp_path) == "[0, 1, 0, 2, 0, 1, 0]"


def test_closure_using_pipe_in_native_code(tmp_path):
    src = (
        "fn make_scale(factor: Real) -> Function:\n"
        "    give \\x -> x * factor.\n"
        "done.\n"
        "emit make_scale(3)(4.5).\n"
    )
    assert both(src, tmp_path) == "13.5"


def test_when_with_pipe_in_nested_loops(tmp_path):
    src = (
        "fn f(n: Int) -> Int:\n"
        "    var total := 0.\n"
        "    repeat i in range(n):\n"
        "        repeat j in range(n):\n"
        "            when (i * 10 + j) % 3 == 0:\n"
        "                total <- total + i + j.\n"
        "            done.\n"
        "        done.\n"
        "    done.\n"
        "    give total.\n"
        "done.\n"
        "emit f(5).\n"
    )
    assert both(src, tmp_path) == "30"


def test_string_escapes_and_interpolation_in_native(tmp_path):
    src = (
        "fn greet(name: Text) -> Text:\n"
        '    give "hi " + name + "\\n  \\"x\\".".\n'
        "done.\n"
        'emit greet("al").\n'
    )
    assert both(src, tmp_path) == 'hi al\n  "x".'

def test_records_in_native_functions(tmp_path):
    src = (
        "record Point:\n"
        "    x: Real.\n"
        "    y: Real.\n"
        "done.\n"
        "fn dist(p: Point) -> Real:\n"
        "    give (p.x * p.x + p.y * p.y) |> sqrt().\n"
        "done.\n"
        "emit dist(Point(3.0, 4.0)).\n"
    )
    assert both(src, tmp_path) == "5.0"
