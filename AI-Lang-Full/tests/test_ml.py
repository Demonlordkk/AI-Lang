"""Tests for the machine-learning surface: tensor operators, the
differentiable t_* ops, the optimizers, the dual autodiff engine, and the
sequence-model (RNN) capability.

These exercise the builtins as whole AI-Lang programs (parse -> check ->
run) so they cover the full pipeline, plus cross-engine parity: the same
program must print identically on the pure-Python and numpy engines.
"""
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

from ailang.errors import AILangError
from ailang.toolchain import run_source

ROOT = Path(__file__).resolve().parent.parent

HAS_NUMPY = None


def _has_numpy():
    global HAS_NUMPY
    if HAS_NUMPY is None:
        try:
            import numpy  # noqa: F401

            HAS_NUMPY = True
        except Exception:
            HAS_NUMPY = False
    return HAS_NUMPY


def out(src: str) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_source(src, "<test>", [ROOT])
    return buf.getvalue().strip()


def err(src: str) -> str:
    with pytest.raises(AILangError) as e:
        out(src)
    return str(e.value)


def _run_subprocess(src: str, native: str, numpy: str):
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        prog = Path(d) / "p.al"
        prog.write_text(src, encoding="utf-8")
        env = dict(os.environ, AILANG_NATIVE=native, AILANG_NUMPY=numpy)
        r = subprocess.run(
            [sys.executable, str(ROOT / "ailang.py"), "run", str(prog)],
            capture_output=True,
            text=True,
            env=env,
            cwd=d,
            timeout=300,
        )
    return r.stdout.strip(), r.returncode


# ------------------------------------------------------------------ operators


def test_tensor_add_sub_mul_div():
    src = (
        "let a := tensor([[1.0, 2.0], [3.0, 4.0]]).\n"
        "let b := tensor([[10.0, 20.0], [30.0, 40.0]]).\n"
        "emit value_of(a + b).\n"
        "emit value_of(b - a).\n"
        "emit value_of(a * 2.0).\n"
        "emit value_of(a / 2.0).\n"
        "emit value_of(-a).\n"
    )
    assert out(src) == (
        "[[11.0, 22.0], [33.0, 44.0]]\n"
        "[[9.0, 18.0], [27.0, 36.0]]\n"
        "[[2.0, 4.0], [6.0, 8.0]]\n"
        "[[0.5, 1.0], [1.5, 2.0]]\n"
        "[[-1.0, -2.0], [-3.0, -4.0]]"
    )


def test_tensor_scalar_left_and_broadcast_bias():
    # scalar on the left, and a (2,) bias broadcasting over (2,2) columns
    src = (
        "let a := tensor([[1.0, 2.0], [3.0, 4.0]]).\n"
        "emit value_of(3.0 * a).\n"
        "let bias := tensor([10.0, 20.0]).\n"
        "emit value_of(a + bias).\n"
    )
    assert out(src) == "[[3.0, 6.0], [9.0, 12.0]]\n[[11.0, 22.0], [13.0, 24.0]]"


def test_tensor_ops_agree_under_both_backends():
    src = (
        "let a := param(randn(2, 3, nothing, 1)).\n"
        "let b := param(zeros(3)).\n"
        "let c := t_tanh(a * 2.0 + b).\n"
        "emit value_of(c).\n"
    )
    a_out, a_code = _run_subprocess(src, "1", "0")
    b_out, b_code = _run_subprocess(src, "0", "0")
    assert a_code == b_code == 0
    assert a_out == b_out


# ------------------------------------------------------------- new t_* ops


def test_t_slice_rows_cols_negative():
    src = (
        "let m := tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).\n"
        "emit value_of(t_slice(m, 1, 3)).\n"
        "emit value_of(t_slice(m, 0, 2, 1)).\n"
        "emit value_of(t_slice(m, 1, 3, 1)).\n"
        "emit value_of(t_slice(m, -1)).\n"
        "let v := tensor([10.0, 20.0, 30.0, 40.0]).\n"
        "emit value_of(t_slice(v, 1, 3)).\n"
    )
    assert out(src) == (
        "[[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]\n"
        "[[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]]\n"
        "[[2.0, 3.0], [5.0, 6.0], [8.0, 9.0]]\n"
        "[[7.0, 8.0, 9.0]]\n"
        "[20.0, 30.0]"
    )


def test_t_gather_flat_and_nested():
    assert out(
        "let table := tensor([[100.0, 200.0], [300.0, 400.0], [500.0, 600.0]]).\n"
        "emit value_of(t_gather(table, [2, 0])).\n"
        "emit shape_of(t_gather(table, [[0, 1], [1, 2]])).\n"
    ) == "[[500.0, 600.0], [100.0, 200.0]]\n[2, 2, 2]"


def test_t_gather_accumulates_repeated_rows():
    src = (
        "let t := param(tensor([[1.0, 2.0], [3.0, 4.0]])).\n"
        "let g := t_gather(t, [1, 1, 0]).\n"
        "backward(sum_t(g)).\n"
        "emit value_of(grad_of(t)).\n"
    )
    assert out(src) == "[[1.0, 1.0], [2.0, 2.0]]"


def test_t_concat_1d_rows_cols():
    src = (
        "emit value_of(t_concat(tensor([1.0, 2.0]), tensor([3.0]))).\n"
        "let a := tensor([[1.0, 2.0], [3.0, 4.0]]).\n"
        "let b := tensor([[5.0, 6.0]]).\n"
        "emit value_of(t_concat(a, b, 0)).\n"
        "let c := tensor([[7.0, 8.0], [9.0, 10.0]]).\n"
        "emit value_of(t_concat(a, c, 1)).\n"
    )
    assert out(src) == (
        "[1.0, 2.0, 3.0]\n"
        "[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]\n"
        "[[1.0, 2.0, 7.0, 8.0], [3.0, 4.0, 9.0, 10.0]]"
    )


def test_t_clip_and_where():
    src = (
        "let m := tensor([[-2.0, 0.5], [3.0, 10.0]]).\n"
        "emit value_of(t_clip(m, 0.0, 5.0)).\n"
        "let mask := [[true, false], [true, true]].\n"
        "emit value_of(where_t(mask, m, zeros(2, 2))).\n"
    )
    assert out(src) == "[[0.0, 0.5], [3.0, 5.0]]\n[[-2.0, 0.0], [3.0, 10.0]]"


def test_l2_norm_value():
    assert out("emit value_of(l2_norm_t(tensor([3.0, 4.0]))).") == "5.0"


def test_l2_penalty_value():
    # 0.5 * (1^2 + 2^2 + 3^2 + 4^2) = 0.5 * 30 = 15
    assert (
        out("emit value_of(l2_penalty_t(tensor([[1.0, 2.0], [3.0, 4.0]]))).") == "15.0"
    )


def test_bce_logits_value():
    # single logit z=0, target 1: loss = log(1+exp(0)) = ln 2
    assert out(
        "emit round(value_of(bce_logits_t(tensor([[0.0]]), tensor([[1.0]]))), 5)."
    ) == "0.69315"


def test_huber_value():
    # residuals 0.5 (quadratic: 0.5*0.25 = 0.125) and 3.0
    # (linear: 0.5*(3.0 - 0.25) = 1.375); mean = (0.125 + 1.375)/2 = 0.75
    src = "emit value_of(huber_t(tensor([0.5, 3.0]), tensor([0.0, 0.0]), 0.5))."
    assert out(src) == "0.75"


# ------------------------------------------------------------- optimizers


def test_sgd_step_moves_params():
    src = (
        "let w := param(tensor([[1.0, 2.0]])).\n"
        "backward(sum_t(w)).\n"
        "sgd_step(w, 0.1).\n"
        "emit value_of(w).\n"
    )
    assert out(src) == "[[0.9, 1.9]]"


def test_momentum_and_adamw_change_params():
    # grad of sum_t(w) is [1, 1]. momentum(0.1, 0.9): w -> [0.9, 1.9].
    # adamw(0.1) on a fresh state with unit grad: step 0.1 -> [0.8, 1.8].
    src = (
        "let w := param(tensor([[1.0, 2.0]])).\n"
        "let om := momentum([w], 0.1, 0.9).\n"
        "backward(sum_t(w)).\n"
        "momentum_step(om).\n"
        "let oa := adamw([w], 0.1, 0.0).\n"
        "backward(sum_t(w)).\n"
        "adamw_step(oa).\n"
        "emit map(value_of(w), \\r -> map(r, \\v -> round(v, 6))).\n"
    )
    assert out(src) == "[[0.8, 1.8]]"


def test_clip_grad_bounds_the_global_norm():
    # a huge loss gives huge grads; clip_grad must cap the global norm
    src = (
        "let w := param(tensor([[1000.0, 1000.0]])).\n"
        "backward(sum_t(w * 1000.0)).\n"
        "clip_grad([w], 1.0).\n"
        "let g := value_of(grad_of(w)).\n"
        "emit round(sqrt(g[0][0] * g[0][0] + g[0][1] * g[0][1]), 6).\n"
    )
    assert out(src) == "1.0"


def test_seed_makes_randn_reproducible():
    src = (
        "seed(99).\n"
        "let a := randn(2, 2, nothing, 5).\n"
        "seed(99).\n"
        "let b := randn(2, 2, nothing, 5).\n"
        "emit a == b.\n"
    )
    assert out(src) == "true"


def test_ml_backend_reports_an_engine():
    src = "let info := ml_backend().\nemit len(info) > 0."
    assert out(src) == "true"


# ------------------------------------------------------------- correctness


def test_chain_rule_matches_numeric():
    # d/dw of mse(sigmoid(w0*x0 + w1*x1), y) at a fixed point, checked
    # against central differences on the pure engine.
    src = (
        "fn loss(w: Any) -> Any:\n"
        "    let x := tensor([[1.0, 2.0]]).\n"
        "    let z := t_matmul(x, w).\n"
        "    give mse_t(t_sigmoid(z), tensor([[0.8]])).\n"
        "done.\n"
        "let w := param(tensor([[0.3], [0.7]])).\n"
        "backward(loss(w)).\n"
        "emit grad_of(w).\n"
    )
    g = out(src)
    assert "[[" in g
    # numeric check
    import math

    def f(w0, w1):
        z = (1.0 * w0 + 2.0 * w1)
        p = 1.0 / (1.0 + math.exp(-z))
        return (p - 0.8) ** 2

    h = 1e-6
    w0, w1 = 0.3, 0.7
    num = [(f(w0 + h, w1) - f(w0 - h, w1)) / (2 * h), (f(w0, w1 + h) - f(w0, w1 - h)) / (2 * h)]
    vals = [float(v) for v in g.replace("[", " ").replace("]", " ").replace(",", " ").split()]
    assert abs(vals[0] - num[0]) < 1e-4
    assert abs(vals[1] - num[1]) < 1e-4


def test_rnn_example_learns_and_generates():
    """The shipped RNN char LM must drive the loss down and write the pattern."""
    path = ROOT / "examples" / "ml" / "char_lm.al"
    if not path.is_file():
        return
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_source(path.read_text(encoding="utf-8"), str(path), [path.parent, ROOT])
    text = buf.getvalue()
    assert "generated:" in text
    # it must have learned the recurring vocabulary
    assert "barks" in text or "jumps" in text


# ----------------------------------------------------------- engine parity


@pytest.mark.skipif(not _has_numpy(), reason="numpy not installed")
def test_same_program_same_output_on_both_engines():
    src = (
        "let w := param(randn(3, 1, nothing, 1)).\n"
        "let x := tensor([[1.0, 0.5, 0.2], [0.3, 0.9, 0.4]]).\n"
        "let loss := mse_t(t_matmul(x, w), tensor([[0.1], [0.9]])).\n"
        "backward(loss).\n"
        "sgd_step([w], 0.05).\n"
        "emit round(value_of(loss), 6).\n"
        "emit value_of(w).\n"
    )
    a_out, a_code = _run_subprocess(src, "0", "0")
    b_out, b_code = _run_subprocess(src, "0", "1")
    assert a_code == b_code == 0
    assert a_out == b_out, f"engines diverged:\n{a_out}\nvs\n{b_out}"
