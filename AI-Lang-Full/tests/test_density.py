"""Tests for the expressive-density features: text interpolation,
compound assignment, and the portable single-file bundle.

These are the "less code, more operations" surface. Every construct here
must lower to the same thing the long form produces -- no runtime cost and
no behavioural difference.
"""
from __future__ import annotations

import io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ailang.errors import AILangError
from ailang.parser import Parser
from ailang.toolchain import run_source

ROOT = Path(__file__).resolve().parent.parent


def out(src: str) -> str:
    """Run source, return trimmed stdout."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_source(src, "<test>", [ROOT], check=True)
    return buf.getvalue().strip()


def err(src: str) -> str:
    with pytest.raises(AILangError) as e:
        out(src)
    return str(e.value)


# ---------------------------------------------------------------- interpolation


def test_interpolation_simple_name():
    assert out('let name := "world".\nemit "hello {name}".') == "hello world"


def test_interpolation_converts_numbers():
    assert out('let n := 42.\nemit "n={n}".') == "n=42"


def test_interpolation_evaluates_expressions():
    assert out('let n := 5.\nemit "double={n * 2}".') == "double=10"


def test_interpolation_calls_functions():
    assert out('let xs := [1, 2, 3].\nemit "sum={sum(xs)}".') == "sum=6"


def test_interpolation_multiple_slots():
    src = 'let a := 1.\nlet b := 2.\nemit "{a} and {b} make {a + b}".'
    assert out(src) == "1 and 2 make 3"


def test_interpolation_of_real_and_bool():
    assert out('let x := 3.5.\nemit "x={x} big={x > 1.0}".') == "x=3.5 big=true"


def test_interpolation_escapes_double_braces():
    assert out('emit "braces {{literal}} ok".') == "braces {literal} ok"


def test_bare_braces_stay_literal_for_format():
    """format() uses bare {} placeholders -- interpolation must not eat them."""
    assert out('emit format("{} is {}", ["a", 1]).') == "a is 1"


def test_interpolation_lowers_to_concat_not_runtime_magic():
    """The interpolated form and the manual form compile to equivalent output."""
    interp = out('let n := 7.\nemit "v={n}".')
    manual = out('let n := 7.\nemit "v=" + to Text(n).')
    assert interp == manual == "v=7"


def test_interpolation_undefined_name_is_static_error():
    msg = err('emit "{nope}".')
    assert "nope" in msg


def test_interpolation_bad_expression_reports_clearly():
    msg = err('emit "{1 +}".')
    assert "interpolation" in msg.lower()


def test_interpolation_unterminated_brace():
    msg = err('emit "unclosed {x".')
    assert "unterminated" in msg.lower()


def test_empty_interpolation_is_literal():
    assert out('emit "a{}b".') == "a{}b"


# ------------------------------------------------------------ compound assign


def test_compound_add():
    assert out("var n := 10.\nn +<- 5.\nemit n.") == "15"


def test_compound_subtract():
    assert out("var n := 10.\nn -<- 3.\nemit n.") == "7"


def test_compound_multiply():
    assert out("var n := 6.\nn *<- 7.\nemit n.") == "42"


def test_compound_divide_yields_real():
    """`/` always yields Real in AI-Lang, so the target must already be Real."""
    assert out("var n := 6.0.\nn /<- 3.\nemit n.") == "2.0"


def test_compound_divide_on_int_is_rejected():
    """Consequence of the Real division rule: `/<-` cannot narrow Real to Int.

    This is intentional -- the compound form inherits the exact typing of its
    longhand `n <- n / 3`, with no special-casing.
    """
    msg = err("var n := 6.\nn /<- 3.\nemit n.")
    assert "Real" in msg and "Int" in msg


def test_compound_on_index_target():
    assert out("var xs := [1, 2, 3].\nxs[0] +<- 100.\nemit xs.") == "[101, 2, 3]"


def test_compound_in_loop_accumulator():
    src = "var total := 0.\nrepeat i in range(5):\n    total +<- i.\ndone.\nemit total."
    assert out(src) == "10"


def test_compound_equals_longhand():
    short = out("var n := 4.\nn *<- 3.\nemit n.")
    long = out("var n := 4.\nn <- n * 3.\nemit n.")
    assert short == long == "12"


def test_compound_rejects_immutable_binding():
    msg = err("let n := 1.\nn +<- 1.\nemit n.")
    assert "n" in msg


def test_compound_rejects_invalid_target():
    with pytest.raises(AILangError):
        Parser("5 +<- 1.").program()


def test_compound_type_checked():
    msg = err('var s := "a".\ns +<- 1.\nemit s.')
    assert "Text" in msg or "Int" in msg


# ------------------------------------------------------------------ portability


def test_no_external_dependencies():
    """AI-Lang must run on any device with a bare Python -- stdlib only."""
    import ast as pyast

    stdlib = sys.stdlib_module_names
    foreign = set()
    for path in sorted((ROOT / "ailang").glob("*.py")):
        tree = pyast.parse(path.read_text(encoding="utf-8"))
        for node in pyast.walk(tree):
            if isinstance(node, pyast.Import):
                for a in node.names:
                    foreign.add(a.name.split(".")[0])
            elif isinstance(node, pyast.ImportFrom) and node.level == 0 and node.module:
                foreign.add(node.module.split(".")[0])
    foreign -= set(stdlib) | {"ailang"}
    assert not foreign, f"non-stdlib imports found: {sorted(foreign)}"


def test_single_file_bundle_builds_and_runs(tmp_path):
    """The zipapp bundle is the zero-install distribution story."""
    build = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "make_bundle.py")],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    bundle = ROOT / "ailang-bundle.pyz"
    assert bundle.exists()

    prog = tmp_path / "p.al"
    prog.write_text('let n := 21.\nemit "bundled {n * 2}".\n', encoding="utf-8")
    run = subprocess.run(
        [sys.executable, str(bundle), "run", str(prog)],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    assert "bundled 42" in run.stdout


# ------------------------------------------- call-frame immutability (perf fix)


def test_parameters_remain_immutable():
    """Calls share a precomputed frozen parameter set; it must still enforce."""
    msg = err("fn h(a: Int) -> Int:\n    a <- 5.\n    give a.\ndone.\nemit h(1).")
    assert "a" in msg


def test_local_declaration_does_not_corrupt_shared_param_set():
    """Declaring locals in a frame must copy-on-write, never mutate the shared set."""
    src = (
        "fn f(a: Int, b: Int) -> Int:\n"
        "    var local := a + b.\n"
        "    local <- local * 2.\n"
        "    give local.\n"
        "done.\n"
        "emit f(1, 2).\n"
        "emit f(10, 20).\n"
        "emit f(1, 2).\n"
    )
    assert out(src) == "6\n60\n6"


def test_loop_variable_in_function_does_not_leak_across_calls():
    src = (
        "fn g(n: Int) -> Int:\n"
        "    var t := 0.\n"
        "    repeat i in range(n):\n"
        "        t +<- i.\n"
        "    done.\n"
        "    give t.\n"
        "done.\n"
        "emit g(3).\nemit g(3).\nemit g(5).\n"
    )
    assert out(src) == "3\n3\n10"


def test_rescue_binding_in_function_repeats_cleanly():
    src = (
        "fn g(x: Int) -> Text:\n"
        "    attempt:\n"
        "        let bad := 1 / 0.\n"
        '        give "no".\n'
        "    rescue e:\n"
        '        give "caught {x}".\n'
        "    done.\n"
        "done.\n"
        "emit g(7).\nemit g(8).\n"
    )
    assert out(src) == "caught 7\ncaught 8"


def test_keyword_arguments_still_work():
    src = (
        "fn area(w: Int, h: Int) -> Int:\n"
        "    give w * h.\n"
        "done.\n"
        "emit area(3, h: 4).\n"
    )
    assert out(src) == "12"


def test_wrong_arity_reports_clearly():
    msg = err("fn f(a: Int) -> Int:\n    give a.\ndone.\nemit f(1, 2).")
    assert "expects" in msg or "argument" in msg
