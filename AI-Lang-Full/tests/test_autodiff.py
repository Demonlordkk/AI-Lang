"""Autodiff correctness.

Gradients are checked two ways: against closed-form derivatives, and against
central-difference numerical gradients for composite expressions.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ailang.autodiff import (  # noqa: E402
    Tensor, add, backward, bce_loss, ce_loss, div, matmul, mse_loss, mul,
    power, reshape, sub, t_exp, t_log, t_mean, t_relu, t_sigmoid, t_softmax,
    t_sqrt, t_sum, t_tanh, transpose,
)
from ailang.errors import AILangError  # noqa: E402
from ailang.toolchain import run_source  # noqa: E402


def out(source: str) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_source(source, "<test>", [ROOT])
    return buf.getvalue().strip()


def fails(source: str, fragment: str = ""):
    try:
        out(source)
    except AILangError as e:
        if fragment:
            assert fragment.lower() in str(e).lower(), f"expected {fragment!r} in {e!r}"
        return str(e)
    raise AssertionError(f"expected failure for:\n{source}")


def numeric_grad(f, xs, h=1e-6):
    g = []
    for i in range(len(xs)):
        a, b = list(xs), list(xs)
        a[i] += h
        b[i] -= h
        g.append((f(a) - f(b)) / (2 * h))
    return g


def check_against_numeric(name, xs, forward_plain, forward_grad, tol=1e-5):
    analytic = forward_grad(xs)
    numeric = numeric_grad(forward_plain, xs)
    err = max(abs(a - n) for a, n in zip(analytic, numeric))
    assert err < tol, f"{name}: gradient mismatch, max error {err:.3e}"


# --------------------------------------------------------- closed-form checks
def test_square_derivative():
    x = Tensor.of(3.0, True)
    backward(mul(x, x))
    assert abs(x.grad[0] - 6.0) < 1e-12


def test_power_derivative():
    x = Tensor.of(2.0, True)
    backward(power(x, 3))
    assert abs(x.grad[0] - 12.0) < 1e-12


def test_sigmoid_derivative_at_zero():
    x = Tensor.of(0.0, True)
    backward(t_sigmoid(x))
    assert abs(x.grad[0] - 0.25) < 1e-12


def test_tanh_derivative_at_zero():
    x = Tensor.of(0.0, True)
    backward(t_tanh(x))
    assert abs(x.grad[0] - 1.0) < 1e-12


def test_exp_and_log_derivatives():
    x = Tensor.of(0.0, True)
    backward(t_exp(x))
    assert abs(x.grad[0] - 1.0) < 1e-12

    y = Tensor.of(4.0, True)
    backward(t_log(y))
    assert abs(y.grad[0] - 0.25) < 1e-12


def test_sqrt_derivative():
    x = Tensor.of(4.0, True)
    backward(t_sqrt(x))
    assert abs(x.grad[0] - 0.25) < 1e-12


def test_relu_derivative_is_piecewise():
    a = Tensor.of([-1.0, 2.0], True)
    backward(t_sum(t_relu(a)))
    assert a.grad_value() == [0.0, 1.0]


def test_div_derivative():
    x = Tensor.of(6.0, True)
    y = Tensor.of(3.0, True)
    backward(div(x, y))
    assert abs(x.grad[0] - (1 / 3)) < 1e-12
    assert abs(y.grad[0] - (-6 / 9)) < 1e-12


def test_sub_and_neg_signs():
    a = Tensor.of(5.0, True)
    b = Tensor.of(2.0, True)
    backward(sub(a, b))
    assert a.grad[0] == 1.0 and b.grad[0] == -1.0


def test_mean_spreads_gradient_evenly():
    a = Tensor.of([1.0, 2.0, 3.0, 4.0], True)
    backward(t_mean(a))
    assert all(abs(g - 0.25) < 1e-12 for g in a.grad)


def test_gradient_accumulates_across_uses():
    """A tensor used twice must receive the sum of both paths."""
    x = Tensor.of(3.0, True)
    backward(add(x, x))          # d(2x)/dx = 2
    assert abs(x.grad[0] - 2.0) < 1e-12


def test_zero_grad_resets():
    x = Tensor.of(3.0, True)
    backward(mul(x, x))
    x.zero_grad()
    assert x.grad is None


# -------------------------------------------------------- numerical agreement
def test_matmul_tanh_sum_matches_numeric():
    def plain(xs):
        w = Tensor.of([[xs[0], xs[1]], [xs[2], xs[3]]])
        x = Tensor.of([[0.5], [1.5]])
        return t_sum(t_tanh(matmul(w, x))).data[0]

    def grad(xs):
        w = Tensor.of([[xs[0], xs[1]], [xs[2], xs[3]]], True)
        x = Tensor.of([[0.5], [1.5]])
        backward(t_sum(t_tanh(matmul(w, x))))
        return w.grad

    check_against_numeric("matmul+tanh", [0.3, -0.7, 1.1, 0.2], plain, grad)


def test_softmax_cross_entropy_matches_numeric():
    def plain(xs):
        return ce_loss(Tensor.of([xs]), [[0.0, 1.0, 0.0]]).data[0]

    def grad(xs):
        z = Tensor.of([xs], True)
        backward(ce_loss(z, [[0.0, 1.0, 0.0]]))
        return z.grad

    check_against_numeric("softmax+CE", [0.4, 1.2, -0.3], plain, grad)


def test_sigmoid_bce_matches_numeric():
    x = Tensor.of([[1.0, 2.0], [3.0, 1.0]])

    def plain(ws):
        w = Tensor.of([[ws[0]], [ws[1]]])
        return bce_loss(t_sigmoid(matmul(x, w)), [[1.0], [0.0]]).data[0]

    def grad(ws):
        w = Tensor.of([[ws[0]], [ws[1]]], True)
        backward(bce_loss(t_sigmoid(matmul(x, w)), [[1.0], [0.0]]))
        return w.grad

    check_against_numeric("sigmoid+BCE", [0.25, -0.4], plain, grad)


def test_broadcast_bias_matches_numeric():
    x = Tensor.of([[1.0, 2.0], [3.0, 4.0]])

    def plain(bs):
        d = add(x, Tensor.of([bs[0], bs[1]]))
        return t_mean(mul(d, d)).data[0]

    def grad(bs):
        b = Tensor.of([bs[0], bs[1]], True)
        d = add(x, b)
        backward(t_mean(mul(d, d)))
        return b.grad

    check_against_numeric("broadcast bias", [0.6, -0.2], plain, grad)


def test_two_layer_network_matches_numeric():
    one = Tensor.of([[1.0]])

    def plain(xs):
        w1 = Tensor.of([[xs[0], xs[1]]])
        w2 = Tensor.of([[xs[2]], [xs[3]]])
        h = t_relu(matmul(one, w1))
        return mse_loss(t_sigmoid(matmul(h, w2)), [[0.7]]).data[0]

    def grad(xs):
        w1 = Tensor.of([[xs[0], xs[1]]], True)
        w2 = Tensor.of([[xs[2]], [xs[3]]], True)
        h = t_relu(matmul(one, w1))
        backward(mse_loss(t_sigmoid(matmul(h, w2)), [[0.7]]))
        # flat in every engine: a plain list on the reference engine, an
        # ndarray under numpy, and list() flattens both identically
        return list(w1.grad) + list(w2.grad)

    check_against_numeric("2-layer net", [0.8, 0.3, 0.5, -0.6], plain, grad)


def test_mse_matches_numeric():
    def plain(xs):
        return mse_loss(Tensor.of(xs), [1.0, 2.0]).data[0]

    def grad(xs):
        p = Tensor.of(xs, True)
        backward(mse_loss(p, [1.0, 2.0]))
        return p.grad

    check_against_numeric("mse", [0.3, 2.4], plain, grad)


# ------------------------------------------------------------ shapes / safety
def test_shape_inference_and_reshape():
    t = Tensor.of([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert t.shape == (2, 3)
    assert reshape(t, [3, 2]).shape == (3, 2)
    assert transpose(t).shape == (3, 2)


def test_transpose_roundtrip_values():
    t = Tensor.of([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert transpose(transpose(t)).value() == t.value()


def test_ragged_input_is_rejected():
    try:
        Tensor.of([[1.0, 2.0], [3.0]])
    except AILangError as e:
        assert "ragged" in str(e).lower()
    else:
        raise AssertionError("expected a ragged-input error")


def test_matmul_shape_mismatch_message():
    try:
        matmul(Tensor.of([[1.0, 2.0]]), Tensor.of([[1.0, 2.0]]))
    except AILangError as e:
        assert "inner dimensions" in str(e)
    else:
        raise AssertionError("expected a shape error")


def test_incompatible_broadcast_message():
    try:
        add(Tensor.of([1.0, 2.0, 3.0]), Tensor.of([1.0, 2.0]))
    except AILangError as e:
        assert "cannot be combined" in str(e)
    else:
        raise AssertionError("expected a broadcast error")


def test_backward_on_non_scalar_is_rejected():
    try:
        backward(Tensor.of([1.0, 2.0], True))
    except AILangError as e:
        assert "single number" in str(e)
    else:
        raise AssertionError("expected a scalar-loss error")


def test_deep_graph_does_not_overflow_stack():
    """Backward must be iterative: a 3000-deep chain would blow recursion."""
    x = Tensor.of(0.01, True)
    acc = x
    for _ in range(3000):
        acc = add(acc, x)
    backward(acc)
    assert abs(x.grad[0] - 3001.0) < 1e-6


# ----------------------------------------------------------- language surface
def test_language_level_gradients():
    src = """
let x := param(3.0).
backward(t_mul(x, x)).
emit grad_of(x).
"""
    assert out(src) == "6.0"


def test_param_and_value_of_roundtrip():
    assert out("let w := param([[1.0, 2.0]]).\nemit value_of(w).") == "[[1.0, 2.0]]"


def test_shape_of():
    assert out("emit shape_of(param([[1.0, 2.0], [3.0, 4.0]])).") == "[2, 2]"


def test_grad_before_backward_is_a_clear_error():
    fails("let w := param(1.0).\nemit grad_of(w).", "call backward")


def test_sgd_step_moves_parameters_downhill():
    """Minimising w^2 from w=10 must drive w toward 0."""
    src = """
let w := param(10.0).
var i := 0.
while i < 50:
    zero_grad(w).
    backward(t_mul(w, w)).
    sgd_step([w], 0.1).
    i <- i + 1.
done.
emit abs(value_of(w)) < 0.001.
"""
    assert out(src) == "true"


def test_adam_optimiser_converges():
    src = """
let w := param([5.0]).
let opt := adam([w]).
var i := 0.
while i < 400:
    zero_grad(w).
    backward(mse_t(w, [1.0])).
    adam_step(opt, 0.1).
    i <- i + 1.
done.
emit round(first(value_of(w)), 2).
"""
    assert out(src) == "1.0"


def test_xor_network_example_learns():
    """The shipped neural net must reach 4/4 on XOR."""
    path = ROOT / "examples" / "ml" / "neural_net.al"
    if not path.is_file():
        return
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_source(path.read_text(encoding="utf-8"), str(path), [path.parent, ROOT])
    assert "accuracy: 4/4" in buf.getvalue()


def test_autodiff_logistic_example_matches_manual_version():
    path = ROOT / "examples" / "ml" / "logistic_autodiff.al"
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
