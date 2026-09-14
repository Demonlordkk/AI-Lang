"""Tests for the numeric/ML layer, automation layer, and compiler fast paths."""

from __future__ import annotations

import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ailang.errors import AILangError  # noqa: E402
from ailang.toolchain import run_source  # noqa: E402


def out(source: str, check=True) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_source(source, "<test>", [ROOT], check=check)
    return buf.getvalue().strip()


def fails(source: str, fragment: str = ""):
    try:
        out(source)
    except AILangError as e:
        if fragment:
            assert fragment.lower() in str(e).lower(), f"expected {fragment!r} in {e!r}"
        return str(e)
    raise AssertionError(f"expected failure for:\n{source}")


# ------------------------------------------------------- numeric / statistics
def test_statistics_match_reference_values():
    src = """
let xs := [2, 4, 4, 4, 5, 5, 7, 9].
emit mean(xs).
emit stddev(xs).
emit median(xs).
emit variance(xs).
"""
    assert out(src) == "5.0\n2.0\n4.5\n4.0"


def test_percentile():
    assert out("emit percentile([1, 2, 3, 4], 50).") == "2.5"
    assert out("emit percentile([1, 2, 3, 4], 0).") == "1.0"
    assert out("emit percentile([1, 2, 3, 4], 100).") == "4.0"


def test_linear_fit_recovers_exact_line():
    src = """
let f := linear_fit([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]).
emit round(f.slope, 6).
emit round(f.intercept, 6).
emit round(f.r2, 6).
"""
    assert out(src) == "2.0\n0.0\n1.0"


def test_correlation_of_perfect_line():
    assert out("emit round(correlation([1, 2, 3], [2, 4, 6]), 6).") == "1.0"
    assert out("emit round(correlation([1, 2, 3], [6, 4, 2]), 6).") == "-1.0"


def test_vector_and_matrix_ops():
    assert out("emit dot([1, 2, 3], [4, 5, 6]).") == "32"
    assert out("emit vadd([1, 2], [10, 20]).") == "[11, 22]"
    assert out("emit vsub([10, 20], [1, 2]).") == "[9, 18]"
    assert out("emit vmul([1, 2, 3], 2).") == "[2, 4, 6]"
    assert out("emit matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]).") == "[[19, 22], [43, 50]]"
    assert out("emit transpose([[1, 2, 3], [4, 5, 6]]).") == "[[1, 4], [2, 5], [3, 6]]"
    assert out("emit shape([[1, 2, 3], [4, 5, 6]]).") == "[2, 3]"
    assert out("emit identity(2).") == "[[1.0, 0.0], [0.0, 1.0]]"


def test_matmul_dimension_mismatch_errors():
    fails("emit matmul([[1, 2]], [[1, 2]]).", "inner dimensions")


def test_activations():
    assert out("emit sigmoid(0).") == "0.5"
    assert out("emit relu(-3).") == "0.0"
    assert out("emit relu(2.5).") == "2.5"
    assert out("emit argmax([0.1, 0.7, 0.2]).") == "1"
    assert out("emit argmin([0.1, 0.7, 0.2]).") == "0"


def test_sigmoid_saturates_without_overflow():
    assert out("emit sigmoid(1000).") == "1.0"
    assert out("emit sigmoid(-1000).") == "0.0"


def test_softmax_sums_to_one():
    assert out("emit round(sum(softmax([1.0, 2.0, 3.0])), 9).") == "1.0"
    assert out("emit round(sum(softmax([500.0, 501.0])), 9).") == "1.0"


def test_losses_and_accuracy():
    assert out("emit round(mse([1, 2, 3], [1, 2, 4]), 6).") == "0.333333"
    assert out("emit round(mae([1, 2, 3], [1, 2, 4]), 6).") == "0.333333"
    assert out("emit accuracy([1, 0, 1, 1], [1, 0, 0, 1]).") == "0.75"


def test_one_hot_and_bincount():
    assert out("emit one_hot(2, 4).") == "[0.0, 0.0, 1.0, 0.0]"
    assert out('emit bincount(["a", "b", "a"]).') == '{"a": 2, "b": 1}'


def test_normalize_and_standardize():
    assert out("emit normalize([0, 5, 10]).") == "[0.0, 0.5, 1.0]"
    assert out("emit round(mean(standardize([1, 2, 3, 4])), 9).") == "0.0"


def test_numeric_errors_are_clear():
    fails("emit mean([]).", "empty")
    fails("emit dot([1, 2], [1]).", "length mismatch")
    fails("emit one_hot(9, 3).", "out of range")


def test_train_test_split_partitions_completely():
    src = """
let s := train_test_split(range(10), 0.7, 42).
emit len(s.train).
emit len(s.test).
emit len(s.train) + len(s.test).
"""
    assert out(src) == "7\n3\n10"


def test_shuffle_with_seed_is_reproducible():
    src = """
emit shuffle(range(8), 7) == shuffle(range(8), 7).
emit len(shuffle(range(8), 7)).
"""
    assert out(src) == "true\n8"


# --------------------------------------------------------------- in-place lists
def test_append_mutates_in_place():
    src = """
var xs := [].
repeat i in range(5):
    append(xs, i).
done.
emit xs.
emit len(xs).
"""
    assert out(src) == "[0, 1, 2, 3, 4]\n5"


def test_push_stays_pure():
    src = """
let a := [1, 2].
let b := push(a, 3).
emit a.
emit b.
"""
    assert out(src) == "[1, 2]\n[1, 2, 3]"


def test_extend_insert_pop_clear():
    assert out("var x := [1].\nextend(x, [2, 3]).\nemit x.") == "[1, 2, 3]"
    assert out("var x := [1, 3].\ninsert(x, 1, 2).\nemit x.") == "[1, 2, 3]"
    assert out("var x := [1, 2, 3].\nemit pop(x).\nemit x.") == "3\n[1, 2]"
    assert out("var x := [1, 2].\nclear(x).\nemit x.") == "[]"


def test_pop_on_empty_list_errors():
    fails("var x := [].\nemit pop(x).", "empty")


# ------------------------------------------------------------------ automation
def test_csv_roundtrip_with_type_inference():
    with tempfile.TemporaryDirectory() as d:
        src_csv = Path(d) / "in.csv"
        src_csv.write_text("name,score,active\nAda,90,true\nAlan,85,false\n", encoding="utf-8")
        outp = Path(d) / "out.csv"
        src = f"""
let rows := read_csv("{src_csv}").
emit len(rows).
emit rows[0].name.
emit rows[0].score + 10.
emit type_of(rows[0].active).
write_csv("{outp}", rows).
emit len(read_csv("{outp}")).
"""
        assert out(src) == "2\nAda\n100\nBool\n2"


def test_filesystem_helpers():
    with tempfile.TemporaryDirectory() as d:
        src = f"""
make_dir("{d}/sub").
write_file("{d}/sub/a.txt", "hello").
emit path_exists("{d}/sub/a.txt").
emit is_dir("{d}/sub").
emit len(find_files("{d}", ".txt")).
delete_file("{d}/sub/a.txt").
emit path_exists("{d}/sub/a.txt").
"""
        assert out(src) == "true\ntrue\n1\nfalse"


def test_read_write_lines():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "l.txt"
        src = f"""
write_lines("{path}", ["a", "b", "c"]).
emit len(read_lines("{path}")).
emit read_lines("{path}")[1].
"""
        assert out(src) == "3\nb"


def test_run_shell_command():
    src = """
let r := run("echo hi").
emit trim(r.out).
emit r.code.
"""
    assert out(src) == "hi\n0"


def test_parallel_map_preserves_order():
    assert out("emit parallel_map(range(6), \\x -> x * x).") == "[0, 1, 4, 9, 16, 25]"


def test_retry_eventually_succeeds():
    src = """
var tries := 0.
let r := retry(fn() -> Int:
    tries <- tries + 1.
    when tries < 3:
        raise "not yet".
    done.
    give tries.
done, 5, 0.01).
emit r.
"""
    assert out(src) == "3"


def test_retry_gives_up_and_reports():
    src = """
let r := retry(fn() -> Int:
    raise "always fails".
done, 2, 0.01).
emit r.
"""
    fails(src, "attempts failed")


def test_timed_reports_result():
    src = """
let t := timed(fn() -> Int:
    give 6 * 7.
done).
emit t.result.
"""
    assert out(src) == "42"


# --------------------------------------------------------- compiler fast paths
def test_function_hoisting_at_runtime():
    """A forward reference must work at runtime, not just pass the checker."""
    src = """
emit helper(6).
fn helper(n: Int) -> Int:
    give n * 7.
done.
"""
    assert out(src) == "42"


def test_forward_reference_inside_function():
    src = """
fn outer() -> Int:
    give inner().
    fn inner() -> Int:
        give 5.
    done.
done.
emit outer().
"""
    assert out(src) == "5"


def test_range_loop_fast_path_matches_semantics():
    assert out("var t := 0.\nrepeat i in range(5):\n t <- t + i.\ndone.\nemit t.") == "10"


def test_range_loop_with_stop_and_next():
    src = """
var seen := [].
repeat i in range(10):
    when i % 2 == 0:
        next.
    done.
    when i > 6:
        stop.
    done.
    append(seen, i).
done.
emit seen.
"""
    assert out(src) == "[1, 3, 5]"


def test_inc_fast_respects_immutability():
    fails("let i := 0.\ni <- i + 1.", "let binding")


def test_inc_fast_handles_non_int():
    assert out("var x := 1.5.\nx <- x + 1.\nemit x.") == "2.5"


def test_specialised_arithmetic_matches_generic():
    src = """
emit 7 + 3.
emit 7 - 3.
emit 7 * 3.
emit 7 / 2.
emit 7 % 3.
emit 7 > 3.
emit 7 < 3.
emit 2.5 + 1.
emit "a" + "b".
emit [1] + [2].
"""
    assert out(src) == "10\n4\n21\n3.5\n1\ntrue\nfalse\n3.5\nab\n[1, 2]"


def test_specialised_ops_still_type_check():
    """Fast-path opcodes must not bypass static checking."""
    fails('emit "a" - "b".', "needs numbers")
    fails("emit true * 3.", "needs numbers")


def test_ml_example_trains_to_perfect_accuracy():
    """The shipped logistic regression must converge."""
    path = ROOT / "examples" / "ml" / "logistic.al"
    if not path.is_file():
        return
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_source(path.read_text(encoding="utf-8"), str(path), [path.parent, ROOT])
    assert "accuracy: 1.0" in buf.getvalue()


def _run_all():
    mod = sys.modules[__name__]
    tests = sorted(n for n in dir(mod) if n.startswith("test_"))
    passed = failed = 0
    failures = []
    for name in tests:
        try:
            getattr(mod, name)()
            passed += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            failures.append((name, e))
    for name, e in failures:
        print(f"FAIL {name}: {e}")
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
