"""Production ML toolkit: fused ops, dropout, init, schedules, early
stopping, gradcheck, f1 — with engine-parity checks where the math runs."""
from __future__ import annotations

import io
import os
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


def out(source: str, native=None) -> str:
    old = os.environ.get("AILANG_NATIVE")
    if native is not None:
        os.environ["AILANG_NATIVE"] = native
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_source(source, "<test>", [ROOT])
        return buf.getvalue().strip()
    finally:
        if old is None:
            os.environ.pop("AILANG_NATIVE", None)
        else:
            os.environ["AILANG_NATIVE"] = old


def out_subprocess(source: str, env_extra=None) -> str:
    """Run in a fresh process (tensor engines cache at first use)."""
    import subprocess

    code = (
        "import io, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from contextlib import redirect_stdout\n"
        "from ailang.toolchain import run_source\n"
        f"src = {source!r}\n"
        "buf = io.StringIO()\n"
        "with redirect_stdout(buf):\n"
        "    run_source(src, '<sub>', [__import__('pathlib').Path.cwd()])\n"
        "print(buf.getvalue().strip())\n"
    )
    env = dict(os.environ)
    env.pop("AILANG_NATIVE", None)
    env.update(env_extra or {})
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120
    )
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def fails(source: str, fragment: str = "") -> str:
    try:
        out(source)
    except AILangError as e:
        if fragment:
            assert fragment.lower() in str(e).lower(), f"expected {fragment!r} in {e!r}"
        return str(e)
    raise AssertionError(f"expected clean error for: {source!r}")


# ------------------------------------------------------------------ fused ops
def test_matmul_bias_matches_matmul_plus_add():
    src = '''
let a := tensor([[1.0, 2.0], [3.0, 4.0]]).
let w := param([[0.5, 1.5, -1.0], [2.0, 0.0, 1.0]]).
let b := param([1.0, 0.0, -2.0]).
let r1 := t_add(t_matmul(a, w), b).
let r2 := t_matmul_bias(a, w, b).
emit value_of(r1) == value_of(r2).
'''
    assert out(src) == "true"


def test_matmul_bias_gradients_match():
    src = '''
let a := tensor([[1.0, 2.0], [3.0, 4.0]]).
let w1 := param([[0.5, 1.5], [2.0, 0.0]]).
let b1 := param([1.0, -2.0]).
let w2 := param([[0.5, 1.5], [2.0, 0.0]]).
let b2 := param([1.0, -2.0]).
let l1 := sum_t(t_add(t_matmul(a, w1), b1)).
let l2 := sum_t(t_matmul_bias(a, w2, b2)).
backward(l1).
backward(l2).
let g1 := grad_of(w1).
let g2 := grad_of(w2).
var d := 0.0.
repeat i in range(len(g1)):
    repeat j in range(len(g1[0])):
        d <- max(d, abs(g1[i][j] - g2[i][j])).
    done.
done.
emit d < 1e-12.
let b1g := grad_of(b1).
let b2g := grad_of(b2).
var db := 0.0.
repeat i in range(len(b1g)):
    db <- max(db, abs(b1g[i] - b2g[i])).
done.
emit db < 1e-12.
'''
    assert out(src) == "true\ntrue"


def test_ce_softmax_matches_separate_softmax_and_ce():
    src = '''
let x := tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.2]]).
let y := tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]).
# ce_t takes LOGITS (it softmaxes internally); the fused op takes the same
let r1 := ce_t(x, y).
let r2 := ce_softmax_t(x, y).
emit round(value_of(r1), 9) == round(value_of(r2), 9).
'''
    assert out(src) == "true"


def test_ce_softmax_gradients_match():
    src = '''
let x1 := param([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.2]]).
let x2 := param([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.2]]).
let y := tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]).
backward(ce_t(x1, y)).
backward(ce_softmax_t(x2, y)).
let g1 := grad_of(x1).
let g2 := grad_of(x2).
var d := 0.0.
repeat i in range(len(g1)):
    repeat j in range(len(g1[0])):
        d <- max(d, abs(g1[i][j] - g2[i][j])).
    done.
done.
emit d < 1e-12.
'''
    assert out(src) == "true"


def test_fused_ops_agree_under_both_engines():
    src = '''
let a := tensor([[1.0, 2.0, 0.5], [3.0, 4.0, 1.5]]).
let w := param([[0.1, 0.9], [0.3, 0.7], [0.2, 0.8]]).
let b := param([0.0, 1.0]).
let h := t_relu(t_matmul_bias(a, w, b)).
let y := tensor([[1.0, 0.0], [0.0, 1.0]]).
let w2 := param([[0.05, -0.1], [0.1, 0.95]]).
let b2 := param(zeros(2)).
emit round(value_of(ce_softmax_t(t_matmul_bias(h, w2, b2), y)), 6).
'''
    assert out(src, native="1") == out(src, native="0")


# ------------------------------------------------------------------ dropout
def test_dropout_is_identity_when_off():
    src = '''
let x := tensor([[1.0, 2.0], [3.0, 4.0]]).
emit value_of(t_dropout(x, 0.5, false)).
emit value_of(t_dropout(x, 0.0, true)).
'''
    assert out(src) == "[[1.0, 2.0], [3.0, 4.0]]\n[[1.0, 2.0], [3.0, 4.0]]"


def test_dropout_scales_kept_units():
    """Inverted dropout: kept units are scaled by 1/(1-rate), so the
    expected value of the output equals the input."""
    src = '''
var s := 0.0.
var i := 0.
while i < 400:
    let x := tensor([[2.0]]).
    s <- s + value_of(t_dropout(x, 0.2, true))[0][0].
    i <- i + 1.
done.
emit round(s / 400.0, 2).
'''
    v = out(src)
    assert 1.8 < float(v) < 2.2, v  # E[out] = 2.0


def test_dropout_rate_bounds_are_clean():
    fails('let x := tensor([[1.0]]).\nlet y := t_dropout(x, 1.0, true).', "rate must be in [0, 1)")


# ------------------------------------------------------------------ init
def test_xavier_and_he_statistics():
    src = '''
let w := xavier(50, 200, 42).
var s := 0.0.
var n := 0.
repeat row in w:
    repeat v in row:
        s <- s + v * v.
        n <- n + 1.
    done.
done.
let var_x := s / to Real(n).
# U(-limit, limit) has variance limit^2/3 = 6/(3*(50+200)) ~= 0.008
emit var_x > 0.004 and var_x < 0.012.
let h := he_init(8, 32, 7).
let lim := 1.0.
emit len(h) == 8 and len(h[0]) == 32.
'''
    assert out(src) == "true\ntrue"


# ------------------------------------------------------------------ schedules
def test_lr_schedules():
    src = '''
emit round(lr_step_decay(0, 0.1, 0.5, 10), 6).
emit round(lr_step_decay(15, 0.1, 0.5, 10), 6).
emit round(lr_cosine(0, 100, 0.1, 0.0), 6).
emit round(lr_cosine(100, 100, 0.1, 0.0), 6).
emit round(lr_cosine(150, 100, 0.1, 0.02), 6).
'''
    assert out(src) == "0.1\n0.05\n0.1\n0.0\n0.02"


# ------------------------------------------------------------------ early stopping
def test_early_stopping_patience():
    src = '''
let es := early_stop(2).
emit get(early_stop_step(es, 0.5), "stop").
emit get(early_stop_step(es, 0.4), "stop").
emit get(early_stop_step(es, 0.45), "stop").
emit get(early_stop_step(es, 0.46), "stop").
emit get(early_stop_step(es, 0.9), "best").
emit get(early_stop_step(es, 0.91), "waiting").
'''
    assert out(src) == "false\nfalse\nfalse\ntrue\n0.4\n4"


def test_early_stop_step_rejects_garbage():
    # the type checker catches the wrong type before the runtime check
    fails('early_stop_step(5, 0.1).', "expects Map")


# ------------------------------------------------------------------ gradcheck
def test_gradcheck_is_zero_on_a_healthy_net():
    src = '''
let w1 := param(xavier(4, 6, 1)).
let b1 := param(zeros(6)).
let w2 := param(xavier(6, 3, 2)).
let b2 := param(zeros(3)).
let x := tensor([[1.0, 0.5, 0.25, 0.0], [0.1, 0.9, 0.3, 0.7], [0.4, 0.2, 0.8, 0.6], [0.0, 0.3, 0.7, 0.9]]).
let y := tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]).
fn loss() -> Any:
    let h := t_dropout(t_matmul_bias(x, w1, b1), 0.0, true).
    give ce_softmax_t(t_matmul_bias(t_relu(h), w2, b2), y).
done.
emit gradcheck([w1, b1, w2, b2], loss) < 1e-6.
'''
    assert out(src) == "true"


def test_gradcheck_flags_wrong_derivatives():
    """A node whose analytic derivative is wrong must be flagged."""
    from ailang import autodiff as ad
    from ailang.stdlib import _gradcheck

    x = ad.T([1.0, 2.0])
    x.requires_grad = True
    bad = ad.T([2.0, 4.0])  # forward is 2*x ...
    bad._parents = (x,)
    bad.requires_grad = True

    def backward(g):
        # ... but the stored derivative claims 1 (wrong)
        x._accum(list(g))

    bad._backward = backward
    err = _gradcheck([x], lambda: ad.t_sum(bad))
    assert err >= 0.4, f"expected a flagged error, got {err}"

# ------------------------------------------------------------------ f1
def test_f1_known_values():
    src = '''
emit f1([1.0, 0.0, 1.0, 1.0], [0.9, 0.1, 0.6, 0.4]).
emit f1([0.0, 0.0], [0.2, 0.8], 0.6).
emit f1([1.0], [0.2]).
emit f1([1.0, 1.0], [0.9, 0.9]).
'''
    assert out(src) == "0.8\n0.0\n0.0\n1.0"


def test_f1_length_mismatch_is_clean():
    fails("f1([1.0, 0.0], [0.5]).", "length mismatch")


# ------------------------------------------------------------------ numpy parity
def test_fused_ops_match_across_backends():
    """numpy engine vs reference engine: identical loss values (fresh
    processes, since each caches its tensor engine at first use)."""
    src = '''
let a := tensor([[1.0, 2.0, 0.5], [3.0, 4.0, 1.5], [0.7, 0.1, 0.9]]).
let w := param([[0.1, 0.9], [0.3, 0.7], [0.2, 0.8]]).
let b := param([0.0, 1.0]).
let h := t_matmul_bias(a, w, b).
let y := tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]).
emit round(value_of(ce_softmax_t(t_relu(h), y)), 6).
'''
    pure = out_subprocess(src, {"AILANG_NUMPY": "0", "AILANG_NATIVE": "0"})
    num = out_subprocess(src, {"AILANG_NUMPY": "1", "AILANG_NATIVE": "0"})
    # engines may differ in summation order; 6 decimals is exact in practice
    assert pure == num, f"{pure!r} vs {num!r}"
    assert 0.0 < float(pure) < 10.0
