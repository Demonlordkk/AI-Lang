"""Tests for the v2.4 work: destructuring, `given`/`is`, membership,
collection operations, the rewritten formatter, script-relative paths, and
the bytecode peephole optimiser.
"""
from __future__ import annotations

import io
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

from ailang.errors import AILangError
from ailang.format import format_source
from ailang.parser import Parser
from ailang.toolchain import run_source

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = sorted((ROOT / "examples").rglob("*.al"))


def out(src: str) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_source(src, "<test>", [ROOT], check=True)
    return buf.getvalue().strip()


def err(src: str) -> str:
    with pytest.raises(AILangError) as e:
        out(src)
    return str(e.value)


# ------------------------------------------------------------- destructuring


def test_list_destructuring():
    assert out("let [a, b] := [10, 20].\nemit \"{a} {b}\".") == "10 20"


def test_map_destructuring():
    src = 'let p := {"name": "Ada", "age": 36}.\nlet {name, age} := p.\nemit "{name} {age}".'
    assert out(src) == "Ada 36"


def test_destructuring_into_mutable_bindings():
    assert out("var [x, y] := [1, 2].\nx <- x + 100.\nemit \"{x} {y}\".") == "101 2"


def test_destructuring_from_a_call():
    src = "fn pair() -> List:\n    give [7, 8].\ndone.\nlet [p, q] := pair().\nemit \"{p}/{q}\"."
    assert out(src) == "7/8"


def test_destructuring_evaluates_subject_once():
    """The subject goes into a hidden temporary, so no repeated side effects."""
    src = (
        "var calls := 0.\n"
        "fn make() -> List:\n"
        "    calls <- calls + 1.\n"
        "    give [1, 2, 3].\n"
        "done.\n"
        "let [a, b, c] := make().\n"
        'emit "{a}{b}{c} calls={calls}".\n'
    )
    assert out(src) == "123 calls=1"


def test_destructuring_inside_a_loop():
    src = (
        "repeat row in [[1, 2], [3, 4]]:\n"
        "    let [u, v] := row.\n"
        '    emit "{u}-{v}".\n'
        "done.\n"
    )
    assert out(src) == "1-2\n3-4"


def test_destructuring_too_few_elements_reports_clearly():
    assert "out of range" in err("let [a, b] := [1].\nemit a.")


def test_destructuring_missing_key_reports_clearly():
    assert "no key" in err('let {x} := {"y": 1}.\nemit x.')


def test_empty_destructuring_pattern_rejected():
    with pytest.raises(AILangError):
        Parser("let [] := [1].").program()


# ------------------------------------------------------------------- given/is


def test_given_selects_a_branch():
    src = (
        "fn describe(n: Int) -> Text:\n"
        "    given n:\n"
        "    is 0:\n"
        '        give "zero".\n'
        "    is 1, 2, 3:\n"
        '        give "small".\n'
        "    else:\n"
        '        give "big".\n'
        "    done.\n"
        "done.\n"
        "emit describe(0).\nemit describe(2).\nemit describe(99).\n"
    )
    assert out(src) == "zero\nsmall\nbig"


def test_given_on_text():
    src = (
        'let code := "b".\n'
        "given code:\n"
        'is "a":\n'
        '    emit "alpha".\n'
        'is "b":\n'
        '    emit "bravo".\n'
        "done.\n"
    )
    assert out(src) == "bravo"


def test_given_evaluates_subject_once():
    src = (
        "var n := 0.\n"
        "fn bump() -> Int:\n"
        "    n <- n + 1.\n"
        "    give 2.\n"
        "done.\n"
        "given bump():\n"
        "is 1:\n"
        '    emit "one".\n'
        "is 2:\n"
        '    emit "two".\n'
        "done.\n"
        'emit "calls={n}".\n'
    )
    assert out(src) == "two\ncalls=1"


def test_given_else_is_the_fallthrough():
    src = 'given 5:\nis 1:\n    emit "no".\nelse:\n    emit "fell".\ndone.\n'
    assert out(src) == "fell"


def test_given_with_no_match_and_no_else_does_nothing():
    assert out('given 9:\nis 1:\n    emit "no".\ndone.\nemit "after".') == "after"


# ---------------------------------------------------------------- membership


def test_membership_on_list():
    src = 'let names := ["Ada", "Grace"].\nemit "Ada" in names.\nemit "Bob" in names.'
    assert out(src) == "true\nfalse"


def test_not_in():
    assert out('emit "Bob" not in ["Ada"].') == "true"


def test_membership_on_map_checks_keys():
    assert out('let m := {"a": 1}.\nemit "a" in m.\nemit "z" in m.') == "true\nfalse"


def test_membership_composes_with_and():
    src = 'let ok := "Grace" in ["Grace"] and 1 in [1].\nemit ok.'
    assert out(src) == "true"


def test_repeat_in_still_parses():
    """`in` is also the loop keyword; adding the operator must not break it."""
    src = 'repeat n in ["a", "b"]:\n    emit n.\ndone.\n'
    assert out(src) == "a\nb"


def test_repeat_at_index_still_parses():
    src = 'repeat n at i in ["a", "b"]:\n    emit "{i}:{n}".\ndone.\n'
    assert out(src) == "0:a\n1:b"


# ------------------------------------------------------- collection builtins


def test_group_by():
    src = (
        'let xs := [{"t": "a", "v": 1}, {"t": "b", "v": 2}, {"t": "a", "v": 3}].\n'
        "let g := group_by(xs, \\x -> x.t).\n"
        "emit keys(g).\nemit len(g[\"a\"]).\n"
    )
    assert out(src) == '["a", "b"]\n2'


def test_count_by_and_counts():
    assert out('emit counts(["a", "b", "a"]).') == '{"a": 2, "b": 1}'
    assert out("emit count_by([1, 2, 3, 4], \\x -> x % 2).") == "{1: 2, 0: 2}"


def test_sum_by():
    assert out("emit sum_by([1, 2, 3], \\x -> x * 10).") == "60"


def test_sum_by_rejects_non_numbers():
    assert "number" in err('emit sum_by(["a"], \\x -> x).')


def test_max_by_and_min_by():
    src = 'let xs := [{"n": 1}, {"n": 9}, {"n": 5}].\nemit max_by(xs, \\x -> x.n).n.\nemit min_by(xs, \\x -> x.n).n.'
    assert out(src) == "9\n1"


def test_max_by_on_empty_list_is_an_error():
    assert "empty" in err("emit max_by([], \\x -> x).")


def test_chunk():
    assert out("emit chunk([1, 2, 3, 4, 5], 2).") == "[[1, 2], [3, 4], [5]]"


def test_chunk_rejects_zero_size():
    assert "at least 1" in err("emit chunk([1], 0).")


def test_windows():
    assert out("emit windows([1, 2, 3, 4], 2).") == "[[1, 2], [2, 3], [3, 4]]"


def test_windows_larger_than_input_is_empty():
    assert out("emit windows([1], 5).") == "[]"


def test_partition():
    assert out("emit partition([1, 2, 3, 4, 5], \\x -> x % 2 == 0).") == "[[2, 4], [1, 3, 5]]"


def test_take_and_drop():
    assert out("emit take([1, 2, 3], 2).\nemit drop([1, 2, 3], 1).") == "[1, 2]\n[2, 3]"


def test_take_and_drop_clamp():
    assert out("emit take([1], 99).\nemit drop([1], 99).") == "[1]\n[]"


def test_take_while_and_drop_while():
    src = "emit take_while([1, 2, 9, 3], \\x -> x < 5).\nemit drop_while([1, 2, 9, 3], \\x -> x < 5)."
    assert out(src) == "[1, 2]\n[9, 3]"


def test_zip_with():
    assert out("emit zip_with([1, 2], [10, 20], \\a, b -> a + b).") == "[11, 22]"


def test_flat_map():
    assert out("emit flat_map([[1, 2], [3]], \\x -> x).") == "[1, 2, 3]"


def test_pluck():
    assert out('emit pluck([{"n": 1}, {"n": 2}], "n").') == "[1, 2]"


def test_index_where_returns_minus_one_when_absent():
    assert out("emit index_where([1, 2], \\x -> x == 99).") == "-1"


def test_sort_desc():
    assert out("emit sort_desc([3, 1, 2]).") == "[3, 2, 1]"


def test_collection_ops_compose_in_a_pipeline():
    src = (
        "emit range(10)\n"
        "    |> filter(\\x -> x % 2 == 0)\n"
        "    |> chunk(2)\n"
        "    |> len().\n"
    )
    assert out(src) == "3"


def test_no_checker_drift():
    """Every builtin must have a type signature and vice versa."""
    from ailang.stdlib import build_globals
    from ailang.typecheck import BUILTIN_SIGS

    names = set(build_globals())
    sigs = set(BUILTIN_SIGS)
    assert not (names - sigs - {"true", "false", "nothing"}), "builtin without a signature"
    assert not (sigs - names), "signature without a builtin"


# ------------------------------------------------------------------ formatter


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_every_example_is_canonically_formatted(path):
    src = path.read_text(encoding="utf-8")
    assert format_source(src) == src, f"{path} is not canonically formatted"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_formatter_is_idempotent(path):
    once = format_source(path.read_text(encoding="utf-8"))
    assert format_source(once) == once


def test_formatter_reindents_messy_source():
    messy = "fn  f(a:Int)->Int:\ngive a+1.\ndone.\n"
    assert format_source(messy) == "fn f(a:Int)->Int:\n    give a+1.\ndone.\n"


def test_formatter_indents_bracket_continuations():
    src = "let xs := [1,\n2].\n"
    assert format_source(src) == "let xs := [1,\n    2].\n"


def test_formatter_keeps_record_field_named_done():
    """`done: Bool.` is a field, not a block end -- it must not dedent."""
    src = "record T:\n    done: Bool.\ndone.\n"
    assert format_source(src) == src


def test_formatter_handles_block_lambda_in_a_call():
    src = (
        "let r := map(xs, fn(t: Any) -> Any:\n"
        "    give t.\n"
        "done).\n"
    )
    assert format_source(src) == src


def test_formatter_preserves_intentional_alignment():
    src = 'let a  := [1, 2].\nlet bb := [3, 4].\n'
    assert format_source(src) == src


# ------------------------------------------------------- script-relative paths


def test_relative_paths_resolve_against_the_script(tmp_path):
    """A program must behave the same no matter which directory it runs from."""
    dest = tmp_path / "proj"
    dest.mkdir()
    (dest / "data.txt").write_text("hello", encoding="utf-8")
    prog = dest / "p.al"
    prog.write_text('emit read_file("data.txt").\n', encoding="utf-8")

    other = tmp_path / "elsewhere"
    other.mkdir()
    r = subprocess.run(
        [sys.executable, str(ROOT / "ailang.py"), "run", str(prog)],
        cwd=other,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "hello" in r.stdout


def test_cwd_relative_paths_still_work(tmp_path):
    """Back-compat: a file that only exists in the working directory is found."""
    prog = tmp_path / "w.al"
    prog.write_text('write_file("out.txt", "hi").\nemit read_file("out.txt").\n', encoding="utf-8")
    r = subprocess.run(
        [sys.executable, str(ROOT / "ailang.py"), "run", str(prog)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "hi" in r.stdout


def test_log_analyzer_runs_from_any_directory(tmp_path):
    prog = ROOT / "examples" / "log_analyzer" / "analyze.al"
    r = subprocess.run(
        [sys.executable, str(ROOT / "ailang.py"), "run", str(prog)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "Error rate" in r.stdout


# -------------------------------------------------------------- peephole pass


def test_peephole_preserves_jump_targets():
    """Fusing must never move an instruction that a jump lands on."""
    from ailang.opcodes import OPS
    from ailang.peephole import fuse

    code = [(OPS["LOAD"], "x"), (OPS["LOAD"], "y"), (OPS["JUMP"], 1)]
    assert fuse(code) == code  # index 1 is a target, so no fusion


def test_peephole_fuses_when_safe():
    from ailang.opcodes import OPS
    from ailang.peephole import fuse

    code = [(OPS["LOAD"], "x"), (OPS["LOAD"], "y"), (OPS["HALT"],)]
    fused = fuse(code)
    assert len(fused) == 2
    assert fused[0][0] == OPS["LOAD_LOAD"]


def test_fused_binary_operand_order():
    """`"a" + name` must not become `name + "a"`."""
    assert out('let name := "Ada".\nemit "Hello, " + name.') == "Hello, Ada"


def test_fused_comparison_operand_order():
    assert out("let n := 5.\nemit 1 < n.\nemit 9 < n.") == "true\nfalse"


def test_fused_index_operand_order():
    assert out("let i := 1.\nlet xs := [10, 20, 30].\nemit xs[i].") == "20"


# --------------------------------------------------------------- loop scoping


def test_let_inside_a_loop_rebinds_each_iteration():
    src = "repeat i in range(3):\n    let d := i * 2.\n    emit d.\ndone.\n"
    assert out(src) == "0\n2\n4"


def test_let_inside_a_while_loop():
    src = "var k := 0.\nwhile k < 3:\n    let m := k * 10.\n    emit m.\n    k +<- 1.\ndone.\n"
    assert out(src) == "0\n10\n20"


def test_rescue_inside_a_loop_repeats_cleanly():
    src = (
        "repeat i in range(3):\n"
        "    attempt:\n"
        "        when i == 1:\n"
        '            raise "boom".\n'
        "        done.\n"
        '        emit "ok {i}".\n'
        "    rescue e:\n"
        '        emit "caught {i}".\n'
        "    done.\n"
        "done.\n"
    )
    assert out(src) == "ok 0\ncaught 1\nok 2"


def test_nested_loop_declaring_inner_binding():
    src = (
        "repeat i in range(2):\n"
        "    repeat j in range(2):\n"
        "        let p := i * 10 + j.\n"
        "        emit p.\n"
        "    done.\n"
        "done.\n"
    )
    assert out(src) == "0\n1\n10\n11"


def test_loop_without_declarations_still_isolates_the_loop_variable():
    src = "var t := 0.\nrepeat i in range(4):\n    t +<- i.\ndone.\nemit t."
    assert out(src) == "6"
