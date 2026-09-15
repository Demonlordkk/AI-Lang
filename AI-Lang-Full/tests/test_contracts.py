"""Tests for `needs` / `ensures` function contracts."""
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run(src, tmp_path, native="1", contracts="1"):
    p = tmp_path / "p.al"
    p.write_text(src, encoding="utf-8")
    # explicit, so an ambient AILANG_CONTRACTS cannot change the outcome
    env = dict(os.environ, AILANG_NATIVE=native, AILANG_CONTRACTS=contracts)
    r = subprocess.run(
        [sys.executable, str(ROOT / "ailang.py"), "run", str(p)],
        capture_output=True, text=True, env=env, cwd=tmp_path,
        timeout=120,
    )
    return r.stdout + r.stderr


WITHDRAW = """fn withdraw(balance: Real, amount: Real) -> Real:
    needs amount > 0.
    needs amount <= balance.
    ensures result >= 0.
    give balance - amount.
done.
"""

CLAMP = """fn clamp(x: Int, lo: Int, hi: Int) -> Int:
    needs lo <= hi.
    ensures result >= lo.
    ensures result <= hi.
    when x < lo:
        give lo.
    elif x > hi:
        give hi.
    done.
    give x.
done.
"""


# ------------------------------------------------------------------- passing


def test_satisfied_contract_is_invisible(tmp_path):
    assert run(WITHDRAW + "emit withdraw(100.0, 30.0).", tmp_path).strip() == "70.0"


def test_every_branch_satisfies_its_postcondition(tmp_path):
    out = run(CLAMP + "emit clamp(5, 1, 10).\nemit clamp(-2, 1, 10).\nemit clamp(99, 1, 10).", tmp_path)
    assert out.split() == ["5", "1", "10"]


def test_recursion_checks_each_call(tmp_path):
    src = """fn fact(n: Int) -> Int:
    needs n >= 0.
    ensures result >= 1.
    when n <= 1:
        give 1.
    done.
    give n * fact(n - 1).
done.
emit fact(5).
emit fact(0).
"""
    assert run(src, tmp_path).split() == ["120", "1"]


# ------------------------------------------------------------------- failing


def test_precondition_failure_names_function_and_condition(tmp_path):
    out = run(WITHDRAW + "emit withdraw(100.0, -5.0).", tmp_path)
    assert "withdraw: precondition failed: amount > 0" in out


def test_second_precondition_is_checked_too(tmp_path):
    out = run(WITHDRAW + "emit withdraw(10.0, 50.0).", tmp_path)
    assert "amount <= balance" in out


def test_postcondition_failure_is_reported(tmp_path):
    src = """fn bad(x: Int) -> Int:
    ensures result >= 0.
    give x.
done.
emit bad(-3).
"""
    out = run(src, tmp_path)
    assert "bad: postcondition failed: result >= 0" in out


def test_postcondition_applies_to_a_nested_give(tmp_path):
    src = """fn bad(x: Int) -> Int:
    ensures result >= 0.
    when x < 0:
        give x.
    done.
    give 1.
done.
emit bad(-3).
"""
    assert "postcondition failed" in run(src, tmp_path)


def test_postcondition_is_enforced_on_an_early_give_in_every_branch(tmp_path):
    src = """fn branchy(x: Int) -> Int:
    ensures result >= 100.
    when x > 0:
        give x * 10.
    else:
        give x * 10.
    done.
done.
emit branchy(9).
"""
    assert "postcondition failed" in run(src, tmp_path)


def test_postcondition_is_enforced_on_a_give_inside_a_loop(tmp_path):
    src = """fn looped(x: Int) -> Int:
    ensures result >= 100.
    var i := 0.
    while i < x:
        i <- i + 1.
        give i.
    done.
    give 999.
done.
emit looped(3).
"""
    assert "postcondition failed" in run(src, tmp_path)


def test_postcondition_is_enforced_on_a_give_inside_a_rescue(tmp_path):
    src = """fn rescued(x: Int) -> Int:
    ensures result >= 100.
    attempt:
        give 5.
    rescue e:
        give 5.
    done.
done.
emit rescued(1).
"""
    assert "postcondition failed" in run(src, tmp_path)


def test_postcondition_on_all_branches_returning_function_is_legal(tmp_path):
    """A function whose final statement is a when/else (or attempt) where
    every path gives must not get a dead `result := Nothing` check - that
    used to turn a valid typed postcondition into a spurious type error."""
    src = """fn branchy(x: Int) -> Int:
    ensures result >= 0.
    when x > 0:
        give x.
    else:
        give 0.
    done.
done.
emit branchy(7).
emit branchy(-2).
"""
    assert run(src, tmp_path).strip() == "7\n0"


def test_rescue_shares_the_attempt_scope(tmp_path):
    """The rescue clause shares the enclosing scope: redeclaring a name the
    attempt body already bound is a duplicate-declaration error - and both
    backends report it identically (the native backend used to silently
    rebind, which diverged from the interpreter)."""
    src = """fn f(x: Int) -> Int:
    attempt:
        let r := 1.
        raise "boom".
        give r + 1.
    rescue e:
        let r := 2.
        give r + 10.
    done.
done.
emit f(1).
"""
    for native in ("1", "0"):
        out = run(src, tmp_path, native=native)
        assert "already defined in this scope" in out, (native, out)


def test_rescue_may_use_names_from_the_attempt_body(tmp_path):
    """...while simply using a name from the attempt body is legal (the
    common pattern: `let args := c.parse(...)` in the body, used after)."""
    src = """fn f(x: Int) -> Int:
    attempt:
        let r := 5.
        give r * 2.
    rescue e:
        give 0.
    done.
done.
emit f(1).
"""
    for native in ("1", "0"):
        assert run(src, tmp_path, native=native).strip() == "10"


def test_precondition_failure_points_at_the_contract_line(tmp_path):
    out = run(WITHDRAW + "emit withdraw(100.0, -5.0).", tmp_path)
    assert ":2:" in out


def test_a_failed_contract_stops_the_program(tmp_path):
    out = run(WITHDRAW + 'emit withdraw(1.0, -1.0).\nemit "unreachable".', tmp_path)
    assert "unreachable" not in out


# ------------------------------------------------------------ interaction


def test_contract_failure_can_be_rescued(tmp_path):
    src = """fn safe(x: Int) -> Int:
    needs x > 0.
    give x * 2.
done.
attempt:
    emit safe(-1).
rescue err:
    emit "caught".
done.
emit safe(21).
"""
    out = run(src, tmp_path)
    assert "caught" in out and "42" in out


@pytest.mark.parametrize("native", ["0", "1"])
def test_both_backends_agree(tmp_path, native):
    out = run(WITHDRAW + "emit withdraw(100.0, -5.0).", tmp_path, native=native)
    assert "withdraw: precondition failed: amount > 0" in out


@pytest.mark.parametrize("native", ["0", "1"])
def test_passing_contract_same_result_on_both_backends(tmp_path, native):
    out = run(CLAMP + "emit clamp(99, 1, 10).", tmp_path, native=native)
    assert out.strip() == "10"


# ---------------------------------------------------------------- stripping


def test_contracts_can_be_stripped(tmp_path):
    """AILANG_CONTRACTS=0 removes the checks entirely."""
    out = run(WITHDRAW + "emit withdraw(100.0, -5.0).", tmp_path, contracts="0")
    assert out.strip() == "105.0"


def test_stripping_does_not_change_valid_results(tmp_path):
    a = run(CLAMP + "emit clamp(5, 1, 10).", tmp_path, contracts="1")
    b = run(CLAMP + "emit clamp(5, 1, 10).", tmp_path, contracts="0")
    assert a == b


# ------------------------------------------------------------------ parsing


def test_contracts_may_reference_several_parameters(tmp_path):
    src = """fn span(lo: Int, hi: Int) -> Int:
    needs hi > lo and lo >= 0.
    give hi - lo.
done.
emit span(2, 9).
"""
    assert run(src, tmp_path).strip() == "7"


def test_contract_on_a_function_without_arguments(tmp_path):
    src = """fn pi() -> Real:
    ensures result > 3.0.
    give 3.14159.
done.
emit pi().
"""
    assert "3.14" in run(src, tmp_path)


def test_condition_text_is_quoted_as_written(tmp_path):
    src = """fn f(xs: List) -> Int:
    needs len(xs) > 0.
    give xs[0].
done.
emit f([]).
"""
    assert "len(xs) > 0" in run(src, tmp_path)


def test_undefined_name_in_a_contract_is_a_static_error(tmp_path):
    src = """fn f(x: Int) -> Int:
    needs x > limitt.
    give x.
done.
emit f(1).
"""
    out = run(src, tmp_path)
    assert "undefined name" in out
