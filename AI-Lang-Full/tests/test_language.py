"""Comprehensive AI-Lang conformance suite.

Run with:  python3 -m pytest tests/ -q      (or)   python3 tests/test_language.py
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ailang.errors import AILangError, CheckError, ParseError, VMError  # noqa: E402
from ailang.toolchain import run_file, run_source  # noqa: E402


def out(source: str, check=True) -> str:
    """Run source, return captured stdout."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_source(source, "<test>", [ROOT], check=check)
    return buf.getvalue().strip()


def fails(source: str, fragment: str = "", check=True):
    try:
        out(source, check=check)
    except AILangError as e:
        if fragment:
            assert fragment.lower() in str(e).lower(), f"expected {fragment!r} in {e!r}"
        return str(e)
    raise AssertionError(f"expected failure for:\n{source}")


# ---------------------------------------------------------------- regressions
def test_zero_arg_call_does_not_corrupt_stack():
    # stack[-0:] used to wipe the whole operand stack
    assert out("fn f() -> Int:\n give 7.\ndone.\nemit f().") == "7"


def test_zero_arg_call_preserves_pending_values():
    src = """
fn one() -> Int:
    give 1.
done.
let a := 10.
emit a + one().
emit a.
"""
    assert out(src) == "11\n10"


def test_empty_list_does_not_wipe_stack():
    assert out("let a := 1.\nemit [].\nemit a.") == "[]\n1"


def test_empty_map_does_not_wipe_stack():
    assert out("let a := 5.\nemit {}.\nemit a.") == "{}\n5"


def test_local_shadows_global():
    src = """
let x := 1.
fn f() -> Int:
    let x := 5.
    give x.
done.
emit f().
emit x.
"""
    assert out(src) == "5\n1"


def test_nested_function_resolves():
    src = """
fn outer() -> Int:
    fn inner() -> Int:
        give 42.
    done.
    give inner().
done.
emit outer().
"""
    assert out(src) == "42"


def test_closure_captures_parameter():
    src = r"""
fn make_adder(n: Int) -> Function:
    give \x -> x + n.
done.
let add5 := make_adder(5).
let add9 := make_adder(9).
emit add5(10).
emit add9(10).
"""
    assert out(src) == "15\n19"


def test_and_short_circuits():
    src = """
let x := 0.
when x != 0 and 10 / x > 1:
    emit "unreachable".
else:
    emit "safe".
done.
"""
    assert out(src) == "safe"


def test_or_short_circuits():
    src = """
let xs := [].
when len(xs) == 0 or xs[0] > 5:
    emit "safe".
done.
"""
    assert out(src) == "safe"


def test_immutability_enforced():
    fails("let a := 1.\na <- 2.", "let binding")


def test_var_is_mutable():
    assert out("var a := 1.\na <- 2.\nemit a.") == "2"


# ------------------------------------------------------------------- literals
def test_number_forms():
    assert out("emit 1_000_000.") == "1000000"
    assert out("emit 1.5.") == "1.5"
    assert out("emit 2e3.") == "2000.0"


def test_int_followed_by_terminator():
    assert out("emit 5.") == "5"
    assert out("let x := 42.\nemit x.") == "42"


def test_text_escapes():
    assert out(r'emit "a\tb".') == "a\tb"
    assert out(r'emit "q\"q".') == 'q"q'
    assert out(r'emit "\u0041".') == "A"


def test_nothing_and_truthiness():
    assert out("emit nothing.") == "nothing"
    assert out('when 0: emit "zero is truthy". done.') == "zero is truthy"
    assert out('when not nothing: emit "nothing is falsy". done.') == "nothing is falsy"


# ---------------------------------------------------------------- control flow
def test_elif_chain():
    src = """
fn grade(n: Int) -> Text:
    when n >= 90:
        give "A".
    elif n >= 80:
        give "B".
    elif n >= 70:
        give "C".
    else:
        give "F".
    done.
done.
emit grade(95).
emit grade(85).
emit grade(75).
emit grade(10).
"""
    assert out(src) == "A\nB\nC\nF"


def test_while_with_stop_and_next():
    src = """
var i := 0.
while true:
    i <- i + 1.
    when i == 3:
        next.
    done.
    when i > 5:
        stop.
    done.
    emit i.
done.
"""
    assert out(src) == "1\n2\n4\n5"


def test_repeat_with_index():
    src = """
repeat ch at i in ["a", "b", "c"]:
    emit to Text(i) + ":" + ch.
done.
"""
    assert out(src) == "0:a\n1:b\n2:c"


def test_repeat_over_text_and_map():
    assert out('repeat c in "hi":\n emit c.\ndone.') == "h\ni"
    assert out('repeat k in {"a": 1}:\n emit k.\ndone.') == "a"


def test_nested_loops_with_stop():
    src = """
repeat i in [1, 2, 3]:
    repeat j in [1, 2, 3]:
        when j == 2:
            stop.
        done.
        emit to Text(i) + "-" + to Text(j).
    done.
done.
"""
    assert out(src) == "1-1\n2-1\n3-1"


# ------------------------------------------------------------------ functions
def test_recursion():
    src = """
fn fact(n: Int) -> Int:
    when n <= 1:
        give 1.
    done.
    give n * fact(n - 1).
done.
emit fact(10).
"""
    assert out(src) == "3628800"


def test_mutual_recursion_via_hoisting():
    src = """
fn is_even(n: Int) -> Bool:
    when n == 0:
        give true.
    done.
    give is_odd(n - 1).
done.
fn is_odd(n: Int) -> Bool:
    when n == 0:
        give false.
    done.
    give is_even(n - 1).
done.
emit is_even(10).
emit is_odd(7).
"""
    assert out(src) == "true\ntrue"


def test_named_arguments():
    src = """
fn greet(name: Text, greeting: Text) -> Text:
    give greeting + ", " + name.
done.
emit greet(name: "Ada", greeting: "Hello").
emit greet("Bob", "Hi").
"""
    assert out(src) == "Hello, Ada\nHi, Bob"


def test_lambda_forms():
    assert out("let f := \\x -> x * 2.\nemit f(21).") == "42"
    src = """
let g := fn(a: Int, b: Int) -> Int:
    give a + b.
done.
emit g(2, 3).
"""
    assert out(src) == "5"


def test_higher_order_pipeline():
    src = r"""
emit [1, 2, 3, 4, 5, 6]
    |> filter(\x -> x % 2 == 0)
    |> map(\x -> x * x)
    |> sum().
"""
    assert out(src) == "56"


def test_pipeline_works_in_every_position():
    """The pipe must behave identically in every syntactic position, on both
    engines (the 'pipeline precedence bug' regression matrix)."""
    src = r"""
fn double(x: Int) -> Int:
    give x * 2.
done.
fn is_big(x: Int) -> Bool:
    give x > 10.
done.
fn add2(a: Int, n: Int) -> Int:
    give a + n.
done.
record Point:
    x: Int.
    y: Int.
done.
fn f1(x: Int) -> Int:
    give x |> double().
done.
emit f1(3).
let y := 5 |> double().
emit y.
var z := 1.
z <- 4 |> double().
emit z.
emit to Text((1 |> Point(9)).y).
1 |> print().
emit "v={3 |> double()}".
emit sum(map(range(4), \x -> x |> double())).
let g := double.
emit 5 |> g().
emit 5 |> add2(n: 3).
when 12 |> is_big():
    emit "big".
done.
emit 1 |> double() |> double() |> double().
emit 5 |> double.
"""
    expected = '6\n10\n8\n9\n1\nv=6\n12\n10\n8\nbig\n8\n10'
    assert out(src) == expected


def test_pipeline_is_lowest_precedence():
    """`a |> f(b)` binds as `f(a, b)` — the pipe is looser than every other
    operator, and chained pipes apply left to right."""
    assert out("emit 1 - 6 |> abs().") == "5"
    assert out("emit 2 * 3 |> abs().") == "6"
    src = """
fn add3(x: Int, n: Int) -> Int:
    give x + n.
done.
fn double(x: Int) -> Int:
    give x * 2.
done.
emit 10 |> add3(4) |> double().
"""
    assert out(src) == "28"
    assert out('emit "ab" + "cd" |> upper().') == "ABCD"


def test_pipeline_comparisons_wrap_the_chain():
    """Comparison operators apply to the *result* of the whole pipe chain:
    `xs |> f() |> g() == 6` is `((xs |> f()) |> g()) == 6`, never a call on
    a Bool. The old parser swallowed `== 6` as the last pipe's right side,
    which type-checked as 'value of type Bool is not callable'."""
    src = """
let xs := [1, 2, 3, 4, 5].
emit xs |> filter(\\x -> x > 1) |> sum() == 14.
emit xs |> filter(\\x -> x > 1) |> sum() != 14.
emit xs |> sum() < 15.
emit xs |> sum() >= 15.
emit 3 in xs |> filter(\\x -> x > 1).
emit 9 not in xs |> filter(\\x -> x > 1).
emit (xs |> sum()) == 15.
emit xs |> map(\\x -> x * 2) |> max() > 9.
"""
    assert out(src) == "true\nfalse\nfalse\ntrue\ntrue\ntrue\ntrue\ntrue"


def test_pipeline_precedence_is_the_same_on_both_backends():
    import os

    src = """
let xs := [1, 2, 3, 4, 5].
emit xs |> filter(\\x -> x > 1) |> sum() == 14.
emit xs |> sum() <= 15.
emit 1 - 6 |> abs().
"""
    old = os.environ.get("AILANG_NATIVE")
    os.environ["AILANG_NATIVE"] = "0"
    try:
        assert out(src) == "true\ntrue\n5"
    finally:
        if old is None:
            os.environ.pop("AILANG_NATIVE", None)
        else:
            os.environ["AILANG_NATIVE"] = old
    assert out(src) == "true\ntrue\n5"


def test_named_args_to_builtins():
    """Named arguments must work for builtins with the documented names, and
    the checker's parameter names must never drift from the implementations
    (which used to leak raw Python TypeErrors, e.g. `5 |> max(a: 3)`)."""
    assert out("emit round(value: 2.567, digits: 2).") == "2.57"
    assert out('emit upper(text: "hi").') == "HI"
    assert out("emit pow(base: 2, exp: 10).") == "1024.0"
    assert out('emit repeat_text(text: "ab", times: 3).') == "ababab"
    # variadic builtins: named args rejected at check time, cleanly
    fails("emit max(a: 1, b: 2).", "positional arguments only")
    fails("emit 5 |> max(a: 3).", "positional arguments only")
    # unknown names rejected at check time
    fails("emit round(bogus: 1).", "no parameter named 'bogus'")
    # ...and at runtime when the checker is skipped — never a raw Python error
    try:
        out("emit 5 |> max(a: 3).", check=False)
        assert False, "expected clean runtime error"
    except AILangError as e:
        assert "raw" not in str(e) and "unexpected keyword" not in str(e)
        assert "max" in str(e)


def test_run_file_restores_script_dir(tmp_path):
    """`run_file` must not leak its script directory into the next program
    in the same process (a cumulative-state bug that made relative paths in
    later in-process runs resolve against an earlier example's folder)."""
    from ailang.stdlib import script_dir

    d = tmp_path / "prog"
    d.mkdir()
    (d / "data.txt").write_text("hello")
    (d / "main.al").write_text('emit read_file("data.txt").')
    prev = script_dir()
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_file(str(d / "main.al"))
    assert script_dir() == prev
    assert buf.getvalue().strip() == "hello"


def test_builtin_signatures_align_with_implementations():
    """Structural guard: every parameter name the checker advertises for a
    builtin must exist on the real Python implementation (or the implementation
    must be variadic, in which case the checker must reject named args).
    This is what keeps `x |> f(a: 1)` from ever reaching a raw Python call."""
    import inspect

    from ailang import stdlib
    from ailang.typecheck import TypeChecker

    g = stdlib.build_globals([])
    tc = TypeChecker()
    for name, sig in tc.functions.items():
        fn = g.get(name)
        if fn is None:
            continue
        try:
            psig = list(inspect.signature(fn).parameters.values())
        except (TypeError, ValueError):
            continue
        if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in psig):
            assert sig.variadic, f"{name}: impl is variadic, checker is not"
            continue
        real = {p.name for p in psig}
        for pname, _ in sig.params:
            assert pname in real, (
                f"{name}: checker accepts named '{pname}' but impl has {sorted(real)}"
            )


def test_arity_error_is_static():
    fails("fn f(a: Int) -> Int:\n give a.\ndone.\nemit f(1, 2).", "expects 1")


def test_deep_recursion_reports_cleanly():
    src = """
fn down(n: Int) -> Int:
    give down(n + 1).
done.
emit down(0).
"""
    fails(src, "recursion limit")


# -------------------------------------------------------------------- records
def test_record_construct_and_read():
    src = """
record Point:
    x: Real.
    y: Real.
done.
let p := Point(3.0, 4.0).
emit p.x.
emit sqrt(p.x * p.x + p.y * p.y).
"""
    assert out(src) == "3.0\n5.0"


def test_record_named_fields_and_mutation():
    src = """
record User:
    name: Text.
    age: Int.
done.
var u := User(name: "Ada", age: 36).
u.age <- 37.
emit u.age.
emit u.
"""
    assert out(src) == "37\nUser(name: Ada, age: 37)"


def test_record_missing_field_errors():
    fails("record P:\n x: Int.\n y: Int.\ndone.\nlet p := P(1).", "expects 2")


def test_unknown_field_errors():
    fails("record P:\n x: Int.\ndone.\nlet p := P(1).\nemit p.z.", "no field")


# -------------------------------------------------------------- collections
def test_list_operations():
    assert out("emit sort([3, 1, 2]).") == "[1, 2, 3]"
    assert out("emit reverse([1, 2, 3]).") == "[3, 2, 1]"
    assert out("emit unique([1, 1, 2, 2, 3]).") == "[1, 2, 3]"
    assert out("emit concat([1], [2]).") == "[1, 2]"
    assert out("emit flatten([[1, 2], [3]]).") == "[1, 2, 3]"
    assert out("emit slice([1,2,3,4], 1, 3).") == "[2, 3]"


def test_map_operations():
    src = """
let m := {"a": 1, "b": 2}.
emit keys(m).
emit values(m).
emit has(m, "a").
emit get(m, "z", 0).
emit merge(m, {"c": 3}).
"""
    assert out(src) == '["a", "b"]\n[1, 2]\ntrue\n0\n{"a": 1, "b": 2, "c": 3}'


def test_reduce_and_sort_by():
    src = r"""
emit reduce([1, 2, 3, 4], \a, b -> a + b, 0).
emit sort_by(["ccc", "a", "bb"], \s -> len(s)).
"""
    assert out(src) == '10\n["a", "bb", "ccc"]'


def test_index_out_of_range_message():
    fails("let xs := [1, 2].\nemit xs[9].", "out of range")


def test_negative_index():
    assert out("emit [1, 2, 3][-1].") == "3"


def test_text_helpers():
    src = """
emit upper("abc").
emit split("a,b,c", ",").
emit join(["a", "b"], "-").
emit replace("aXa", "X", "-").
emit format("{} is {}", ["x", 1]).
"""
    assert out(src) == 'ABC\n["a", "b", "c"]\na-b\na-a\nx is 1'


# ------------------------------------------------------------ error handling
def test_attempt_rescue_catches_runtime_error():
    src = """
attempt:
    let x := 1 / 0.
    emit x.
rescue e:
    emit "caught " + e.message.
done.
"""
    assert out(src) == "caught division by zero"


def test_raise_and_rescue_value():
    src = """
attempt:
    raise "custom failure".
rescue e:
    emit e.message.
done.
"""
    assert out(src) == "custom failure"


def test_rescue_restores_stack():
    src = """
let base := 100.
attempt:
    raise "x".
rescue e:
    emit "handled".
done.
emit base.
"""
    assert out(src) == "handled\n100"


def test_attempt_inside_function():
    src = """
fn safe_div(a: Int, b: Int) -> Text:
    attempt:
        give to Text(a / b).
    rescue e:
        give "undefined".
    done.
done.
emit safe_div(6, 3).
emit safe_div(1, 0).
"""
    assert out(src) == "2.0\nundefined"


def test_assert_builtin():
    assert out('assert(1 == 1, "ok").\nemit "passed".') == "passed"
    fails('assert(false, "boom").', "boom")


# ------------------------------------------------------------- static checking
def test_undefined_name_is_static():
    fails("emit missing_thing.", "undefined name")


def test_type_mismatch_is_static():
    fails('let n: Int := "text".', "cannot bind")


def test_text_plus_int_rejected():
    fails('emit "a" + 1.', "convert")


def test_give_outside_function_rejected():
    fails("give 1.", "only valid inside a function")


def test_stop_outside_loop_rejected():
    fails("stop.", "only valid inside a loop")


def test_missing_return_rejected():
    fails("fn f(n: Int) -> Int:\n emit n.\ndone.", "without 'give'")


def test_duplicate_binding_rejected():
    fails("let a := 1.\nlet a := 2.", "already defined")


def test_no_check_mode_still_runs():
    assert out('emit "a" + to Text(1).', check=False) == "a1"


# --------------------------------------------------------------- diagnostics
def test_parse_error_has_position():
    try:
        out("let x := .")
    except ParseError as e:
        assert e.line == 1 and e.col > 0
        assert "expression" in str(e)
    else:
        raise AssertionError("expected ParseError")


def test_error_render_includes_caret():
    try:
        out("let a := 1.\nemit unknown_name.")
    except AILangError as e:
        text = e.render("let a := 1.\nemit unknown_name.", "demo.al")
        assert "demo.al:2" in text
        assert "^" in text
    else:
        raise AssertionError("expected an error")


# -------------------------------------------------------------------- numbers
def test_division_always_real():
    assert out("emit 6 / 3.") == "2.0"


def test_integer_arithmetic_stays_int():
    assert out("emit 2 + 3 * 4.") == "14"


def test_modulo_by_zero():
    fails("emit 5 % 0.", "modulo by zero")


def test_precedence_and_parens():
    assert out("emit (2 + 3) * 4.") == "20"
    assert out("emit 10 - 2 - 3.") == "5"


def test_comparison_chain_via_and():
    assert out("let x := 5.\nemit x > 1 and x < 10.") == "true"


def test_equality_is_type_strict():
    assert out('emit 1 == "1".') == "false"
    assert out("emit 1 == 1.0.") == "true"
    assert out("emit true == 1.") == "false"


# ------------------------------------------------------------------ coalescing
def test_coalesce_operator():
    assert out('emit get({"a": 1}, "z", nothing) ?? 99.') == "99"
    assert out('emit get({"a": 1}, "a", nothing) ?? 99.') == "1"


# --------------------------------------------------------------------- output
def test_display_formats():
    assert out("emit true.") == "true"
    assert out("emit 1.0.") == "1.0"
    assert out('emit ["a", 1, true].') == '["a", 1, true]'
    assert out("emit nothing.") == "nothing"

# ------------------------------------------------------------- loop scoping
def test_let_inside_repeat_body():
    assert out("repeat i in [1,2,3]:\n let sq := i * i.\n emit sq.\ndone.") == "1\n4\n9"


def test_let_inside_while_body():
    src = """
var i := 0.
while i < 3:
    let d := i * 10.
    emit d.
    i <- i + 1.
done.
"""
    assert out(src) == "0\n10\n20"


def test_next_unwinds_iteration_scope():
    src = """
repeat i in [1, 2, 3, 4]:
    when i % 2 == 0:
        next.
    done.
    let v := i * 100.
    emit v.
done.
"""
    assert out(src) == "100\n300"


def test_stop_unwinds_iteration_scope():
    src = """
repeat i in [1, 2, 3]:
    let v := i.
    when i == 2:
        stop.
    done.
    emit v.
done.
emit "after".
"""
    assert out(src) == "1\nafter"


def test_nested_loop_scopes():
    src = """
repeat i in [1, 2]:
    let outer := i.
    repeat j in [10, 20]:
        let inner := outer + j.
        emit inner.
    done.
done.
"""
    assert out(src) == "11\n21\n12\n22"


def test_builtin_can_be_shadowed():
    assert out("var count := 0.\ncount <- count + 5.\nemit count.") == "5"


def test_optional_builtin_arguments():
    assert out("emit range_from(1, 5, 2).") == "[1, 3]"
    assert out("emit range_from(1, 4).") == "[1, 2, 3]"
    assert out('emit join(["a","b"]).') == "ab"


def test_closure_counter_keeps_state():
    src = """
fn make_counter() -> Function:
    var total := 0.
    give fn() -> Int:
        total <- total + 1.
        give total.
    done.
done.
let tick := make_counter().
emit tick().
emit tick().
let other := make_counter().
emit other().
"""
    assert out(src) == "1\n2\n1"


def test_module_import(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        (base / "lib").mkdir()
        (base / "lib" / "util.al").write_text(
            "fn triple(n: Int) -> Int:\n give n * 3.\ndone.\n", encoding="utf-8"
        )
        main = "use lib/util as util.\nemit util.triple(14).\n"
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_source(main, str(base / "main.al"), [base])
        assert buf.getvalue().strip() == "42"


def test_circular_import_detected():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        (base / "a.al").write_text("use b as b.\nlet x := 1.\n", encoding="utf-8")
        (base / "b.al").write_text("use a as a.\nlet y := 2.\n", encoding="utf-8")
        try:
            run_source("use a as a.\nemit 1.", str(base / "m.al"), [base])
        except AILangError as e:
            assert "circular" in str(e).lower()
        else:
            raise AssertionError("expected circular import error")


# ------------------------------------------------------------------ bytecode
def test_bytecode_roundtrip_is_deterministic():
    import tempfile
    from ailang.bytecode import artifact, load, write
    from ailang.toolchain import compile_source

    src = 'fn f(a: Int) -> Int:\n give a * 2.\ndone.\nemit f(21).'
    p1 = compile_source(src)
    p2 = compile_source(src)
    assert artifact(p1, src)["artifact_sha256"] == artifact(p2, src)["artifact_sha256"]

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "out.albc.json"
        write(p1, path, src)
        restored = load(path)
        assert restored.main.code == p1.main.code
        assert set(restored.functions) == set(p1.functions)


def test_built_artifact_executes():
    import tempfile
    from ailang.bytecode import load, write
    from ailang.stdlib import build_globals
    from ailang.toolchain import compile_source
    from ailang.vm import VM

    src = 'fn f(a: Int) -> Int:\n give a * 2.\ndone.\nemit f(21).'
    program = compile_source(src)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "a.albc.json"
        write(program, path, src)
        buf = io.StringIO()
        with redirect_stdout(buf):
            VM(build_globals()).run(load(path))
        assert buf.getvalue().strip() == "42"


def test_build_twice_is_byte_identical(tmp_path):
    """`ailang build` twice on the same source must produce byte-identical
    artifact files (reproducible, cacheable builds)."""
    from ailang.bytecode import write
    from ailang.toolchain import compile_source

    src = 'fn f(a: Int) -> Int:\n give a * 2.\ndone.\nemit f(21).'
    program = compile_source(src)
    p1 = tmp_path / "a.albc.json"
    p2 = tmp_path / "b.albc.json"
    write(program, p1, src)
    write(program, p2, src)
    assert p1.read_bytes() == p2.read_bytes()


def test_bytecode_read_rejects_corrupt_and_foreign_artifacts():
    """Strict AILBC-3 validation: truncated, malformed, foreign, and tampered
    artifacts must be rejected with a clear ValueError - never a
    KeyError/TypeError/JSONDecodeError leaking through."""
    import json
    import tempfile
    from ailang.bytecode import artifact, read, write
    from ailang.toolchain import compile_source

    src = 'fn f(a: Int) -> Int:\n give a * 2.\ndone.\nemit f(21).'
    program = compile_source(src)

    def make(corrupt):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.albc.json"
            obj = artifact(program, src)
            corrupt(obj, path)
            try:
                read(path)
            except ValueError as e:
                return str(e)
            raise AssertionError("corrupt artifact was accepted")

    def truncated(obj, path):
        full = json.dumps(obj, indent=2) + "\n"
        path.write_text(full[: len(full) // 2], encoding="utf-8")

    def non_object(obj, path):
        path.write_text("[1, 2, 3]", encoding="utf-8")

    def wrong_format(obj, path):
        obj["format"] = "AILBC-9"
        path.write_text(json.dumps(obj), encoding="utf-8")

    def wrong_language(obj, path):
        obj["language"] = "SomeOtherLang"
        path.write_text(json.dumps(obj), encoding="utf-8")

    def wrong_version(obj, path):
        obj["version"] = "0.0.1"
        path.write_text(json.dumps(obj), encoding="utf-8")

    def missing_main(obj, path):
        del obj["main"]
        path.write_text(json.dumps(obj), encoding="utf-8")

    def tampered(obj, path):
        obj["main"]["name"] = "evil"  # artifact_sha256 still covers the old name
        path.write_text(json.dumps(obj), encoding="utf-8")

    assert "corrupt bytecode artifact" in make(truncated)
    assert "top level is not an object" in make(non_object)
    assert "unsupported bytecode format" in make(wrong_format)
    assert "foreign bytecode artifact" in make(wrong_language)
    assert "built by" in make(wrong_version)
    assert "malformed 'main'" in make(missing_main)
    assert "integrity check" in make(tampered)


# -------------------------------------------------------------- formatter
def test_formatter_is_idempotent():
    from ailang.format import format_source

    messy = "fn  f( a:Int )->Int:\ngive a+1.\ndone.\nemit f(1).\n"
    once = format_source(messy)
    assert format_source(once) == once


def test_formatter_indents_blocks():
    from ailang.format import format_source

    src = "fn f() -> Int:\ngive 1.\ndone.\n"
    assert format_source(src) == "fn f() -> Int:\n    give 1.\ndone.\n"


# ------------------------------------------------- keyword field regressions
def test_done_is_a_valid_field_name():
    src = """
record Task:
    title: Text.
    done: Bool.
done.
let t := Task(title: "x", done: true).
emit t.done.
"""
    assert out(src) == "true"


def test_keyword_field_on_map():
    assert out('let m := {"done": true}.\nemit m.done.') == "true"


def test_module_closure_resolves(tmp=None):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        (base / "m.al").write_text(
            "fn doubled(xs: List) -> List:\n"
            "    give map(xs, \\x -> x * 2).\n"
            "done.\n",
            encoding="utf-8",
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_source("use m as m.\nemit m.doubled([1, 2, 3]).", str(base / "main.al"), [base])
        assert buf.getvalue().strip() == "[2, 4, 6]"


def test_broken_module_reports_error():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        (base / "bad.al").write_text("fn oops(\n", encoding="utf-8")
        try:
            run_source("use bad as bad.\nemit 1.", str(base / "main.al"), [base])
        except AILangError as e:
            assert "bad" in str(e)
        else:
            raise AssertionError("a broken module must not be silently ignored")


# --------------------------------------------- stdlib / checker consistency
def test_every_runtime_builtin_is_known_to_the_checker():
    """A function callable at runtime must not be rejected statically."""
    from ailang.stdlib import build_globals
    from ailang.typecheck import BUILTIN_SIGS

    runtime = set(build_globals())
    declared = set(BUILTIN_SIGS)
    missing = sorted(runtime - declared)
    assert not missing, f"callable but rejected by the checker: {missing}"


def test_every_declared_builtin_exists_at_runtime():
    from ailang.stdlib import build_globals
    from ailang.typecheck import BUILTIN_SIGS

    runtime = set(build_globals())
    declared = set(BUILTIN_SIGS)
    phantom = sorted(declared - runtime)
    assert not phantom, f"declared by the checker but missing at runtime: {phantom}"


def test_builtin_signature_arity_matches_implementation():
    """Catch signatures that promise more/fewer params than the function takes."""
    import inspect

    from ailang.stdlib import build_globals
    from ailang.typecheck import BUILTIN_SIGS

    g = build_globals()
    problems = []
    for name, spec in BUILTIN_SIGS.items():
        fn = g.get(name)
        if fn is None or not callable(fn):
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        if any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values()):
            continue
        declared = len(spec[0])
        optional = spec[2] if len(spec) > 2 else 0
        required = declared - optional
        real_total = len(sig.parameters)
        real_required = sum(1 for p in sig.parameters.values() if p.default is p.empty)
        if required < real_required or declared > real_total:
            problems.append(
                f"{name}: checker accepts {required}-{declared}, "
                f"implementation takes {real_required}-{real_total}"
            )
    assert not problems, "signature mismatches:\n  " + "\n  ".join(problems)


def test_append_file_is_callable():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "log.txt"
        src = f'''
write_file("{path}", "a").
append_file("{path}", "b").
emit read_file("{path}").
'''
        assert out(src) == "ab"


def test_pad_left_is_callable():
    # `out()` strips surrounding whitespace, so anchor both sides
    assert out('emit "|" + pad_left("7", 3) + "|".') == "|  7|"
    assert out('emit "|" + pad("7", 3) + "|".') == "|7  |"


# ------------------------------------------------ v0.1 specification syntax
def test_original_spec_syntax_still_runs():
    """Every construct in docs/05_syntax.md must keep working."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        (base / "math.al").write_text(
            "fn square(n: Int) -> Int:\n    give n * n.\ndone.\n", encoding="utf-8"
        )
        src = """
use math.

let score := 42.
var counter := 42.
counter <- 43.
emit "hello".

when score > 40:
    emit "high".
else:
    emit "low".
done.

fn add(a: Int, b: Int) -> Int:
    give a + b.
done.

let values := [1, 2, 3].
repeat item in values:
    emit item.
done.

record Point:
    x: Real.
    y: Real.
done.

let real_count := 7.9.
let count := to Int(real_count).
emit count.
emit add(1, 2).
emit math.square(5).
"""
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_source(src, str(base / "main.al"), [base])
        assert buf.getvalue().split() == [
            "hello", "high", "1", "2", "3", "7", "3", "25"
        ]


def test_bare_use_binds_module_name():
    """`use math.` (no alias) must bind `math`, per the v0.1 spec."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        (base / "helper.al").write_text(
            "fn twice(n: Int) -> Int:\n    give n * 2.\ndone.\n", encoding="utf-8"
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_source("use helper.\nemit helper.twice(21).", str(base / "m.al"), [base])
        assert buf.getvalue().strip() == "42"


def _run_all():
    """Standalone runner: python3 tests/test_language.py -- no pytest needed.

    Injects the `tmp_path` and `monkeypatch` fixtures that pytest would
    provide, skips parametrization placeholders, and enforces a per-test
    timeout so a bad test can never hang the run.
    """
    import inspect
    import shutil
    import signal
    import tempfile
    import time

    mod = sys.modules[__name__]
    tests = sorted(n for n in dir(mod) if n.startswith("test_"))

    class _Timeout(Exception):
        pass

    def _on_alarm(signum, frame):  # noqa: ARG001
        raise _Timeout("exceeded 120s")

    class _MonkeyPatch:
        def __init__(self):
            self._undo = []

        def setattr(self, target, name, value):
            old = getattr(target, name, object())
            had = old is not object()
            setattr(target, name, value)
            self._undo.append((target, name, old, had))

        def undo(self):
            while self._undo:
                target, name, old, had = self._undo.pop()
                if had:
                    setattr(target, name, old)
                else:
                    try:
                        delattr(target, name)
                    except AttributeError:
                        pass

    signal.signal(signal.SIGALRM, _on_alarm)
    passed = failed = 0
    failures = []
    for name in tests:
        fn = getattr(mod, name)
        if not callable(fn) or not getattr(fn, "__name__", "").startswith("test_"):
            continue  # parametrization placeholder
        params = inspect.signature(fn).parameters
        tmp = tempfile.mkdtemp(prefix="ailang_test_") if "tmp_path" in params else None
        monkey = _MonkeyPatch() if "monkeypatch" in params else None
        fixtures = {}
        if tmp:
            from pathlib import Path as _P
            fixtures["tmp_path"] = _P(tmp)
        if monkey:
            fixtures["monkeypatch"] = monkey
        signal.alarm(120)
        try:
            fn(**fixtures)
            passed += 1
        except BaseException as e:  # noqa: BLE001
            failed += 1
            failures.append((name, e))
        finally:
            signal.alarm(0)
            if monkey:
                monkey.undo()
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)
    for name, e in failures:
        print(f"FAIL {name}: {e}")
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
    return 1 if failed else 0


# --------------------------------------------------- record names as types


def test_record_name_works_as_a_parameter_type():
    assert out("""record Point:
    x: Int.
    y: Int.
done.
fn shift(p: Point, by: Int) -> Point:
    give Point(p.x + by, p.y + by).
done.
emit shift(Point(1, 2), 10).x.
""") == "11"


def test_record_name_works_as_a_return_type():
    assert out("""record Box:
    v: Int.
done.
fn wrap(n: Int) -> Box:
    give Box(n).
done.
emit wrap(7).v.
""") == "7"


def test_record_field_may_be_another_record():
    assert out("""record Inner:
    n: Int.
done.
record Outer:
    inner: Inner.
done.
let o := Outer(Inner(5)).
emit o.inner.n.
""") == "5"


def test_unknown_type_name_is_rejected():
    fails("fn f(p: Nonexistent) -> Int:\n    give 1.\ndone.\nemit f(1).\n",
          "unknown type 'Nonexistent'")


def test_misspelled_record_type_is_suggested():
    fails("""record Point:
    x: Int.
done.
fn f(p: Poimt) -> Int:
    give p.x.
done.
emit f(Point(1)).
""", "Point")


def test_builtin_type_names_still_work():
    assert out('fn f(xs: List, n: Int, s: Text) -> Bool:\n'
               '    give len(xs) > n and len(s) > 0.\n'
               'done.\nemit f([1, 2], 1, "a").\n') == "true"

if __name__ == "__main__":
    raise SystemExit(_run_all())
