"""Reverse-mode automatic differentiation for AI-Lang.

This is what makes AI-Lang a language for *building* models rather than a
library of fixed ones. You write the forward computation with ordinary AI-Lang
operators and the gradients are derived for you:

    let w := param([0.0, 0.0]).
    let loss := mse_t(predict(w, x), y).
    backward(loss).
    emit grad_of(w).

A Tensor holds a flat buffer plus a shape. Every operation records how to push
gradient back to its inputs; `backward` walks that graph in reverse
topological order exactly once.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from .errors import VMError


def _prod(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _flatten(value, out: List[float], shape: Optional[List[int]], depth: int):
    """Flatten nested lists while verifying the structure is rectangular."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return
    if not isinstance(value, list):
        raise VMError(f"tensor: expected numbers or lists, found {type(value).__name__}")
    if len(shape) <= depth:
        shape.append(len(value))
    elif shape[depth] != len(value):
        raise VMError(
            f"tensor: ragged input — expected {shape[depth]} elements, found {len(value)}"
        )
    for item in value:
        _flatten(item, out, shape, depth + 1)


def _collect(value, out: List[float]):
    if isinstance(value, list):
        for item in value:
            _collect(item, out)
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VMError(f"tensor: values must be numbers, found {type(value).__name__}")
    out.append(float(value))


def _unflatten(data, shape, offset=0):
    if not shape:
        return data[offset]
    if len(shape) == 1:
        return [data[offset + i] for i in range(shape[0])]
    stride = _prod(shape[1:])
    return [_unflatten(data, shape[1:], offset + i * stride) for i in range(shape[0])]


class Tensor:
    """A differentiable n-dimensional array."""

    __slots__ = ("data", "shape", "grad", "_backward", "_parents", "requires_grad", "op")

    def __init__(self, data: List[float], shape: Tuple[int, ...], requires_grad=False, op=""):
        self.data = data
        self.shape = tuple(shape)
        self.requires_grad = requires_grad
        self.grad: Optional[List[float]] = None
        self._backward = None
        self._parents: Tuple[Tensor, ...] = ()
        self.op = op

    # ------------------------------------------------------------- factories
    @staticmethod
    def of(value, requires_grad=False):
        if isinstance(value, Tensor):
            return value
        shape: List[int] = []
        _flatten(value, [], shape, 0)
        data: List[float] = []
        _collect(value, data)
        if _prod(shape) != len(data):
            raise VMError("tensor: inconsistent shape")
        return Tensor(data, tuple(shape), requires_grad)

    @property
    def size(self):
        return len(self.data)

    def value(self):
        """Convert back to plain AI-Lang numbers / lists."""
        if not self.shape:
            return self.data[0]
        return _unflatten(self.data, list(self.shape))

    def grad_value(self):
        if self.grad is None:
            return None
        if not self.shape:
            return self.grad[0]
        return _unflatten(self.grad, list(self.shape))

    def zero_grad(self):
        self.grad = None

    # --------------------------------------------------------------- helpers
    def _child(self, data, shape, parents, backward, op):
        needs = any(p.requires_grad for p in parents)
        t = Tensor(data, shape, needs, op)
        if needs:
            t._parents = parents
            t._backward = backward
        return t

    def _accum(self, g: List[float]):
        if self.grad is None:
            self.grad = list(g)
        else:
            sg = self.grad
            for i, v in enumerate(g):
                sg[i] += v

    def __repr__(self):
        return f"Tensor(shape={list(self.shape)}, {self.value()})"


# ---------------------------------------------------------------- broadcasting
def _broadcast_shapes(a: Tuple[int, ...], b: Tuple[int, ...]):
    ra, rb = list(a)[::-1], list(b)[::-1]
    out = []
    for i in range(max(len(ra), len(rb))):
        da = ra[i] if i < len(ra) else 1
        db = rb[i] if i < len(rb) else 1
        if da == db or da == 1 or db == 1:
            out.append(max(da, db))
        else:
            raise VMError(
                f"shapes {list(a)} and {list(b)} cannot be combined "
                f"(dimension {da} vs {db})"
            )
    return tuple(out[::-1])


def _strides(shape):
    st = []
    acc = 1
    for d in reversed(shape):
        st.append(acc)
        acc *= d
    return st[::-1]


def _index_map(out_shape, in_shape):
    """Map each output flat index to the source flat index under broadcasting."""
    if out_shape == in_shape:
        return None  # identity, no mapping needed
    n = _prod(out_shape)
    out_strides = _strides(out_shape)
    in_strides = _strides(in_shape)
    pad = len(out_shape) - len(in_shape)
    mapping = [0] * n
    for flat in range(n):
        rem = flat
        src = 0
        for dim in range(len(out_shape)):
            idx = rem // out_strides[dim] if out_strides[dim] else 0
            rem = rem % out_strides[dim] if out_strides[dim] else rem
            j = dim - pad
            if j >= 0:
                if in_shape[j] != 1:
                    src += idx * in_strides[j]
        mapping[flat] = src
    return mapping


def _binary_op(a: Tensor, b: Tensor, fwd, back_a, back_b, name):
    shape = _broadcast_shapes(a.shape, b.shape)
    n = _prod(shape)
    ma = _index_map(shape, a.shape)
    mb = _index_map(shape, b.shape)
    ad, bd = a.data, b.data
    out = [0.0] * n
    for i in range(n):
        out[i] = fwd(ad[ma[i] if ma else i], bd[mb[i] if mb else i])

    def backward(g):
        if a.requires_grad:
            ga = [0.0] * a.size
            for i in range(n):
                si = ma[i] if ma else i
                ga[si] += back_a(ad[si], bd[mb[i] if mb else i], out[i]) * g[i]
            a._accum(ga)
        if b.requires_grad:
            gb = [0.0] * b.size
            for i in range(n):
                si = mb[i] if mb else i
                gb[si] += back_b(ad[ma[i] if ma else i], bd[si], out[i]) * g[i]
            b._accum(gb)

    return a._child(out, shape, (a, b), backward, name)


def _unary_op(a: Tensor, fwd, back, name):
    out = [fwd(x) for x in a.data]

    def backward(g):
        if a.requires_grad:
            a._accum([back(a.data[i], out[i]) * g[i] for i in range(len(out))])

    return a._child(out, a.shape, (a,), backward, name)


# ---------------------------------------------------------------------- ops
def add(a, b):
    return _binary_op(T(a), T(b), lambda x, y: x + y, lambda x, y, o: 1.0,
                      lambda x, y, o: 1.0, "add")


def sub(a, b):
    return _binary_op(T(a), T(b), lambda x, y: x - y, lambda x, y, o: 1.0,
                      lambda x, y, o: -1.0, "sub")


def mul(a, b):
    return _binary_op(T(a), T(b), lambda x, y: x * y, lambda x, y, o: y,
                      lambda x, y, o: x, "mul")


def div(a, b):
    def f(x, y):
        if y == 0:
            raise VMError("tensor division by zero")
        return x / y

    return _binary_op(T(a), T(b), f, lambda x, y, o: 1.0 / y,
                      lambda x, y, o: -x / (y * y), "div")


def power(a, p):
    p = float(p)
    return _unary_op(T(a), lambda x: x ** p,
                     lambda x, o: p * (x ** (p - 1.0)), "pow")


def neg(a):
    return _unary_op(T(a), lambda x: -x, lambda x, o: -1.0, "neg")


def t_exp(a):
    return _unary_op(T(a), lambda x: math.exp(min(x, 700.0)), lambda x, o: o, "exp")


def t_log(a):
    def f(x):
        if x <= 0:
            raise VMError("log of a non-positive number")
        return math.log(x)

    return _unary_op(T(a), f, lambda x, o: 1.0 / x, "log")


def t_sqrt(a):
    def f(x):
        if x < 0:
            raise VMError("sqrt of a negative number")
        return math.sqrt(x)

    return _unary_op(T(a), f, lambda x, o: 0.5 / o if o else 0.0, "sqrt")


def t_sigmoid(a):
    def f(x):
        if x < -500:
            return 0.0
        if x > 500:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    return _unary_op(T(a), f, lambda x, o: o * (1.0 - o), "sigmoid")


def t_relu(a):
    return _unary_op(T(a), lambda x: x if x > 0 else 0.0,
                     lambda x, o: 1.0 if x > 0 else 0.0, "relu")


def t_tanh(a):
    return _unary_op(T(a), math.tanh, lambda x, o: 1.0 - o * o, "tanh")


def t_sum(a):
    a = T(a)
    total = math.fsum(a.data)

    def backward(g):
        if a.requires_grad:
            a._accum([g[0]] * a.size)

    return a._child([total], (), (a,), backward, "sum")


def t_mean(a):
    a = T(a)
    n = a.size
    if n == 0:
        raise VMError("mean of an empty tensor")
    total = math.fsum(a.data) / n

    def backward(g):
        if a.requires_grad:
            a._accum([g[0] / n] * n)

    return a._child([total], (), (a,), backward, "mean")


def matmul(a, b):
    a, b = T(a), T(b)
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise VMError(
            f"matmul needs two 2-D tensors, got shapes {list(a.shape)} and {list(b.shape)}"
        )
    n, k = a.shape
    k2, m = b.shape
    if k != k2:
        raise VMError(f"matmul: inner dimensions differ ({k} vs {k2})")
    ad, bd = a.data, b.data
    out = [0.0] * (n * m)
    for i in range(n):
        ai = i * k
        oi = i * m
        for p in range(k):
            av = ad[ai + p]
            if av == 0.0:
                continue
            bp = p * m
            for j in range(m):
                out[oi + j] += av * bd[bp + j]

    def backward(g):
        if a.requires_grad:
            ga = [0.0] * (n * k)
            for i in range(n):
                for p in range(k):
                    s = 0.0
                    bp = p * m
                    gi = i * m
                    for j in range(m):
                        s += g[gi + j] * bd[bp + j]
                    ga[i * k + p] = s
            a._accum(ga)
        if b.requires_grad:
            gb = [0.0] * (k * m)
            for p in range(k):
                for j in range(m):
                    s = 0.0
                    for i in range(n):
                        s += ad[i * k + p] * g[i * m + j]
                    gb[p * m + j] = s
            b._accum(gb)

    return a._child(out, (n, m), (a, b), backward, "matmul")


def transpose(a):
    a = T(a)
    if len(a.shape) != 2:
        raise VMError("transpose needs a 2-D tensor")
    n, m = a.shape
    out = [0.0] * (n * m)
    for i in range(n):
        for j in range(m):
            out[j * n + i] = a.data[i * m + j]

    def backward(g):
        if a.requires_grad:
            ga = [0.0] * (n * m)
            for i in range(n):
                for j in range(m):
                    ga[i * m + j] = g[j * n + i]
            a._accum(ga)

    return a._child(out, (m, n), (a,), backward, "transpose")


def reshape(a, shape):
    a = T(a)
    shape = tuple(int(x) for x in shape)
    if _prod(shape) != a.size:
        raise VMError(
            f"reshape: cannot fit {a.size} elements into shape {list(shape)}"
        )

    def backward(g):
        if a.requires_grad:
            a._accum(list(g))

    return a._child(list(a.data), shape, (a,), backward, "reshape")


def t_softmax(a):
    """Row-wise softmax; numerically stabilised."""
    a = T(a)
    if len(a.shape) == 2:
        rows, cols = a.shape
    else:
        rows, cols = 1, a.size
    out = [0.0] * a.size
    for r in range(rows):
        base = r * cols
        row = a.data[base : base + cols]
        mx = max(row)
        exps = [math.exp(v - mx) for v in row]
        s = math.fsum(exps)
        for j in range(cols):
            out[base + j] = exps[j] / s

    def backward(g):
        if a.requires_grad:
            ga = [0.0] * a.size
            for r in range(rows):
                base = r * cols
                dot = math.fsum(g[base + j] * out[base + j] for j in range(cols))
                for j in range(cols):
                    ga[base + j] = out[base + j] * (g[base + j] - dot)
            a._accum(ga)

    return a._child(out, a.shape, (a,), backward, "softmax")


# --------------------------------------------------------------------- losses
def mse_loss(pred, target):
    pred, target = T(pred), T(target)
    diff = sub(pred, target)
    return t_mean(mul(diff, diff))


def mae_loss(pred, target):
    pred, target = T(pred), T(target)
    return t_mean(t_abs(sub(pred, target)))


def t_abs(a):
    return _unary_op(T(a), abs, lambda x, o: 1.0 if x >= 0 else -1.0, "abs")


def bce_loss(pred, target, eps=1e-12):
    """Binary cross-entropy. `pred` must already be in (0, 1)."""
    p, y = T(pred), T(target)
    if p.size != y.size:
        raise VMError(f"bce: size mismatch {p.size} vs {y.size}")
    n = p.size
    pd, yd = p.data, y.data
    total = 0.0
    for i in range(n):
        pi = min(max(pd[i], eps), 1.0 - eps)
        total += -(yd[i] * math.log(pi) + (1.0 - yd[i]) * math.log(1.0 - pi))
    total /= n

    def backward(g):
        if p.requires_grad:
            gp = [0.0] * n
            for i in range(n):
                pi = min(max(pd[i], eps), 1.0 - eps)
                gp[i] = g[0] * (pi - yd[i]) / (pi * (1.0 - pi) * n)
            p._accum(gp)

    return p._child([total], (), (p, y), backward, "bce")


def ce_loss(logits, target, eps=1e-12):
    """Softmax cross-entropy over logits; target is one-hot."""
    probs = t_softmax(logits)
    y = T(target)
    pd, yd = probs.data, y.data
    n = probs.shape[0] if len(probs.shape) == 2 else 1
    total = -math.fsum(
        yd[i] * math.log(max(pd[i], eps)) for i in range(len(pd))
    ) / n

    def backward(g):
        if probs.requires_grad:
            probs._accum([g[0] * (-yd[i] / max(pd[i], eps)) / n for i in range(len(pd))])

    return probs._child([total], (), (probs, y), backward, "cross_entropy")


# ------------------------------------------------------------------- backward
def backward(t: Tensor):
    """Populate .grad on every tensor that contributed to `t`."""
    t = T(t)
    if t.shape != ():
        raise VMError(
            f"backward expects a single number (a loss), but got shape {list(t.shape)}; "
            f"reduce it with sum_t(...) or mean_t(...) first"
        )
    order: List[Tensor] = []
    seen = set()

    # iterative post-order traversal: recursion would cap network depth
    stack = [(t, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.append((node, True))
        for p in node._parents:
            if id(p) not in seen:
                stack.append((p, False))

    t.grad = [1.0]
    for node in reversed(order):
        if node._backward is not None and node.grad is not None:
            node._backward(node.grad)
    return None


def T(v) -> Tensor:
    """Coerce an AI-Lang value to a Tensor."""
    if isinstance(v, Tensor):
        return v
    return Tensor.of(v)
