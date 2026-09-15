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

Two engines drive the same graph:

* the reference engine, pure Python, which runs on any host Python; and
* an optional numpy engine (see `accel`), used automatically when the host
  has numpy installed. It is a speed path for the implementation, not a
  different language: shapes, broadcasting, gradients and diagnostics are
  identical, and `AILANG_NUMPY=0` forces the reference engine.
"""

from __future__ import annotations

import math
import operator
from typing import List, Optional, Tuple

from . import accel
from .errors import VMError


def _prod(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _flat1d(t: "Tensor") -> List[float]:
    """A tensor's data as a flat 1-D python list (engine-agnostic)."""
    d = t.data
    if isinstance(d, list):
        return d
    return [float(v) for v in d.reshape(-1)]


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
        v = data[offset]
        return float(v) if hasattr(v, "item") else v
    if len(shape) == 1:
        return [_unflatten(data, (), offset + i) for i in range(shape[0])]
    stride = _prod(shape[1:])
    return [_unflatten(data, shape[1:], offset + i * stride) for i in range(shape[0])]


def _buf(flat) -> "List[float]":
    """Store a flat buffer in the active engine's native container."""
    if accel.have():
        return accel.asarr(flat)
    return list(flat)


def _asarr(x):
    """Gradient flow: to a 1-D array when the engine is numpy, else a list."""
    if accel.have():
        return accel.asarr(x)
    return list(x)


class Tensor:
    """A differentiable n-dimensional array."""

    __slots__ = ("data", "shape", "grad", "_backward", "_parents", "requires_grad", "op")

    def __init__(self, data, shape: Tuple[int, ...], requires_grad=False, op=""):
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
            # `param(t)` on a constant tensor turns it into a leaf that
            # accumulates gradients, sharing its buffer
            if requires_grad and not value.requires_grad:
                return Tensor(value.data, value.shape, True)
            return value
        shape: List[int] = []
        _flatten(value, [], shape, 0)
        data: List[float] = []
        _collect(value, data)
        if _prod(shape) != len(data):
            raise VMError("tensor: inconsistent shape")
        return Tensor(_buf(data), tuple(shape), requires_grad)

    @property
    def size(self):
        return len(self.data)

    def value(self):
        """Convert back to plain AI-Lang numbers / lists."""
        if not self.shape:
            v = self.data[0]
            return float(v) if hasattr(v, "item") else v
        return _unflatten(self.data, list(self.shape))

    def grad_value(self):
        if self.grad is None:
            return None
        if not self.shape:
            v = self.grad[0]
            return float(v) if hasattr(v, "item") else v
        return _unflatten(self.grad, list(self.shape))

    def zero_grad(self):
        self.grad = None

    # ------------------------------------------------ Python operators
    # The native backend compiles top-level loops into host code, where
    # `w + b` is a real Python operator. These dunders keep the operators
    # differentiable on both backends (the interpreter routes through
    # `vm._binary`, which handles Tensors identically).
    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return div(other, self)

    def __neg__(self):
        return neg(self)

    # --------------------------------------------------------------- helpers
    def _child(self, data, shape, parents, backward, op):
        needs = any(p.requires_grad for p in parents)
        t = Tensor(data, shape, needs, op)
        if needs:
            t._parents = parents
            t._backward = backward
        return t

    def _accum(self, g):
        if self.grad is None:
            # keep the buffer in the engine's native container: an ndarray
            # under numpy (fast in-place += and vectorised consumers), a
            # plain list on the reference engine
            self.grad = accel.asarr(g) if accel.have() else list(g)
        else:
            sg = self.grad
            if accel.have() and not isinstance(sg, list):
                sg += accel.asarr(g)
            else:
                self.grad = list(map(operator.add, sg, g))

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
    """Map each output flat index to the source flat index under broadcasting.

    Returns None for the identity case, else a list `mapping` with
    `mapping[flat]` = source flat index. The common broadcast patterns
    (scalar, trailing-suffix bias, per-dim 1s) use single-division
    fast paths instead of the general stride walk.
    """
    if out_shape == in_shape:
        return None  # identity, no mapping needed
    n = _prod(out_shape)
    if not in_shape:  # scalar broadcast: every output reads flat 0
        return [0] * n
    d = len(in_shape)
    if in_shape == out_shape[-d:]:
        # trailing-suffix broadcast, e.g. bias (m,) over (n, m)
        tail = _prod(in_shape)
        if tail == 1:
            return [0] * n
        return [flat % tail for flat in range(n)]
    if d == len(out_shape) and all(
        in_shape[i] in (1, out_shape[i]) for i in range(d)
    ):
        # in lines up with out dim-by-dim (each dim either equal or 1)
        out_strides = _strides(out_shape)
        in_strides = _strides(in_shape)
        keep = [
            (out_strides[i], in_strides[i], out_shape[i])
            for i in range(d)
            if in_shape[i] != 1
        ]
        if not keep:
            return [0] * n
        if len(keep) == 1:
            # one non-unit dim: flat -> (flat // so) % size, rescaled by si
            so, si, size = keep[0]
            return [((flat // so) % size) * si for flat in range(n)]
        mapping = [0] * n
        for flat in range(n):
            rem = flat
            src = 0
            for so, si, size in keep:
                src += (rem // so) % size * si
                rem %= so
            mapping[flat] = src
        return mapping
    # general case: full stride walk
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


def _reduce_grad(np_, contrib, out_shape, in_shape):
    """Sum a broadcast-shaped gradient contribution down to `in_shape`.

    `contrib` may be a scalar, or an array broadcast to `out_shape`. Every
    axis that broadcasting expanded must be summed: the leading axes
    broadcasting added, and any dimension the operand had size 1 in. That is
    exactly how a gradient must flow back through a broadcast op.
    """
    if isinstance(contrib, (int, float)):
        contrib = np_.full(out_shape, float(contrib), dtype=np_.float64)
    else:
        contrib = np_.asarray(contrib, dtype=np_.float64)
        if contrib.shape != out_shape:
            try:
                contrib = np_.broadcast_to(contrib, out_shape)
            except ValueError:
                raise VMError(
                    f"shapes {list(out_shape)} and {list(contrib.shape)} cannot be combined"
                ) from None
    if not in_shape:
        return np_.array([float(contrib.sum())])
    pad = len(contrib.shape) - len(in_shape)
    axes = list(range(pad))
    for d in range(len(in_shape)):
        if in_shape[d] == 1 and out_shape[d + pad] > 1:
            axes.append(d + pad)
    if axes:
        contrib = contrib.sum(axis=tuple(axes))
    return contrib.reshape(-1)


def _binary_op(a: Tensor, b: Tensor, fwd, back_a, back_b, name, np_fwd=None, np_back_a=None, np_back_b=None):
    shape = _broadcast_shapes(a.shape, b.shape)
    n = _prod(shape)
    ad, bd = a.data, b.data

    if accel.have():
        np_ = accel.np()
        va = ad.reshape(a.shape) if a.shape else ad.reshape(1)
        vb = bd.reshape(b.shape) if b.shape else bd.reshape(1)
        try:
            out = np_fwd(va, vb)
        except ValueError:
            raise VMError(f"{name}: incompatible operand shapes") from None
        out = np_.ascontiguousarray(out, dtype=np_.float64).reshape(-1)
        oa = out.reshape(shape) if shape else out.reshape(1)

        def backward(g):
            g = np_.asarray(g, dtype=np_.float64).reshape(-1)
            if shape:
                gv = g.reshape(shape)
            else:
                gv = g.reshape(1)
            if a.requires_grad:
                # local derivative times the upstream gradient, reduced from
                # the broadcast shape back to a's own shape
                contrib = np_back_a(va, vb, oa)
                if not isinstance(contrib, np_.ndarray):
                    contrib = gv * float(contrib)
                else:
                    contrib = contrib * gv
                a._accum(_reduce_grad(np_, contrib, shape if shape else (1,), a.shape if a.shape else ()))
            if b.requires_grad:
                contrib = np_back_b(va, vb, oa)
                if not isinstance(contrib, np_.ndarray):
                    contrib = gv * float(contrib)
                else:
                    contrib = contrib * gv
                b._accum(_reduce_grad(np_, contrib, shape if shape else (1,), b.shape if b.shape else ()))

        return a._child(out, shape, (a, b), backward, name)

    ma = _index_map(shape, a.shape)
    mb = _index_map(shape, b.shape)
    if ma is None and mb is None:
        out = [fwd(x, y) for x, y in zip(ad, bd)]

        def backward(g):
            if a.requires_grad:
                a._accum([back_a(x, y, o) * gi for x, y, o, gi in zip(ad, bd, out, g)])
            if b.requires_grad:
                b._accum([back_b(x, y, o) * gi for x, y, o, gi in zip(ad, bd, out, g)])
    else:
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


def _unary_op(a: Tensor, fwd, back, name, np_fwd=None, np_back=None):
    if accel.have():
        np_ = accel.np()
        ad = a.data
        out = np_.ascontiguousarray(np_fwd(ad), dtype=np_.float64).reshape(-1)

        def backward(g):
            if a.requires_grad:
                g = np_.asarray(g, dtype=np_.float64).reshape(-1)
                a._accum(np_back(ad, out, g))

        return a._child(out, a.shape, (a,), backward, name)

    out = [fwd(x) for x in a.data]

    def backward(g):
        if a.requires_grad:
            a._accum([back(a.data[i], out[i]) * g[i] for i in range(len(out))])

    return a._child(out, a.shape, (a,), backward, name)


# ---------------------------------------------------------------------- ops
def add(a, b):
    return _binary_op(
        T(a), T(b), lambda x, y: x + y, lambda x, y, o: 1.0, lambda x, y, o: 1.0,
        "add",
        np_fwd=lambda x, y: x + y,
        np_back_a=lambda x, y, o: 1.0,
        np_back_b=lambda x, y, o: 1.0,
    )


def sub(a, b):
    return _binary_op(
        T(a), T(b), lambda x, y: x - y, lambda x, y, o: 1.0, lambda x, y, o: -1.0,
        "sub",
        np_fwd=lambda x, y: x - y,
        np_back_a=lambda x, y, o: 1.0,
        np_back_b=lambda x, y, o: -1.0,
    )


def mul(a, b):
    return _binary_op(
        T(a), T(b), lambda x, y: x * y, lambda x, y, o: y, lambda x, y, o: x,
        "mul",
        np_fwd=lambda x, y: x * y,
        np_back_a=lambda x, y, o: y,
        np_back_b=lambda x, y, o: x,
    )


def div(a, b):
    def f(x, y):
        if y == 0:
            raise VMError("tensor division by zero")
        return x / y

    return _binary_op(
        T(a), T(b), f, lambda x, y, o: 1.0 / y, lambda x, y, o: -x / (y * y),
        "div",
        np_fwd=lambda x, y: np_div(x, y),
        np_back_a=lambda x, y, o: 1.0 / y,
        np_back_b=lambda x, y, o: -x / (y * y),
    )


def np_div(x, y):
    if (y == 0).any():
        raise VMError("tensor division by zero")
    return x / y


def power(a, p):
    p = float(p)
    return _unary_op(
        T(a), lambda x: x ** p, lambda x, o: p * (x ** (p - 1.0)), "pow",
        np_fwd=lambda x: _np_pow(x, p),
        np_back=lambda x, o, g: g * p * _np_pow(x, p - 1.0),
    )


def _np_pow(x, p):
    m = accel.np()
    if not float(p).is_integer() and (x < 0).any():
        raise VMError("pow: negative base with a fractional exponent")
    with m.errstate(over="ignore", invalid="ignore"):
        out = m.power(x, p)
    if m.any(m.isnan(out)) and (x == 0).any() and p < 0:
        raise VMError("tensor division by zero")
    return out


def neg(a):
    return _unary_op(T(a), lambda x: -x, lambda x, o: -1.0, "neg",
                     np_fwd=lambda x: -x, np_back=lambda x, o, g: -g)


def t_exp(a):
    return _unary_op(
        T(a), lambda x: math.exp(min(x, 700.0)), lambda x, o: o, "exp",
        np_fwd=lambda x: accel.np().exp(accel.np().clip(x, -745.0, 700.0)),
        np_back=lambda x, o, g: o * g,
    )


def t_log(a):
    def f(x):
        if x <= 0:
            raise VMError("log of a non-positive number")
        return math.log(x)

    return _unary_op(
        T(a), f, lambda x, o: 1.0 / x, "log",
        np_fwd=lambda x: _np_log(x),
        np_back=lambda x, o, g: g / x,
    )


def _np_log(x):
    if (x <= 0).any():
        raise VMError("log of a non-positive number")
    return accel.np().log(x)


def t_sqrt(a):
    def f(x):
        if x < 0:
            raise VMError("sqrt of a negative number")
        return math.sqrt(x)

    return _unary_op(
        T(a), f, lambda x, o: 0.5 / o if o else 0.0, "sqrt",
        np_fwd=lambda x: _np_sqrt(x),
        np_back=lambda x, o, g: accel.np().where(o > 0, 0.5 * g / o, 0.0),
    )


def _np_sqrt(x):
    if (x < 0).any():
        raise VMError("sqrt of a negative number")
    return accel.np().sqrt(x)


def t_sigmoid(a):
    def f(x):
        if x < -500:
            return 0.0
        if x > 500:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    return _unary_op(
        T(a), f, lambda x, o: o * (1.0 - o), "sigmoid",
        np_fwd=lambda x: _np_sigmoid(x),
        np_back=lambda x, o, g: o * (1.0 - o) * g,
    )


def _np_sigmoid(x):
    m = accel.np()
    out = m.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + m.exp(-x[pos]))
    e = m.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def t_relu(a):
    return _unary_op(
        T(a), lambda x: x if x > 0 else 0.0, lambda x, o: 1.0 if x > 0 else 0.0,
        "relu",
        np_fwd=lambda x: accel.np().maximum(x, 0.0),
        np_back=lambda x, o, g: accel.np().where(x > 0, g, 0.0),
    )


def t_tanh(a):
    return _unary_op(T(a), math.tanh, lambda x, o: 1.0 - o * o, "tanh",
                     np_fwd=lambda x: accel.np().tanh(x),
                     np_back=lambda x, o, g: (1.0 - o * o) * g)


def t_abs(a):
    return _unary_op(T(a), abs, lambda x, o: 1.0 if x >= 0 else -1.0, "abs",
                     np_fwd=lambda x: accel.np().abs(x),
                     np_back=lambda x, o, g: accel.np().where(x >= 0, g, -g))


def t_clip(a, lo, hi):
    """Element-wise clamp to [lo, hi]; gradient is zero outside the band."""
    lo, hi = float(lo), float(hi)
    if lo > hi:
        raise VMError(f"t_clip: low ({lo}) is above high ({hi})")

    def f(x):
        return max(lo, min(hi, x))

    return _unary_op(
        T(a), f, lambda x, o: 1.0 if lo <= x <= hi else 0.0, "clip",
        np_fwd=lambda x: accel.np().clip(x, lo, hi),
        np_back=lambda x, o, g: accel.np().where((x >= lo) & (x <= hi), g, 0.0),
    )


def t_sum(a):
    a = T(a)
    if accel.have():
        total = float(a.data.sum())

        def backward(g):
            if a.requires_grad:
                a._accum([float(g[0])] * a.size)

        return a._child(_buf([total]), (), (a,), backward, "sum")

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
    if accel.have():
        total = float(a.data.mean())

        def backward(g):
            if a.requires_grad:
                a._accum([float(g[0]) / n] * n)

        return a._child(_buf([total]), (), (a,), backward, "mean")

    total = math.fsum(a.data) / n

    def backward(g):
        if a.requires_grad:
            a._accum([g[0] / n] * n)

    return a._child([total], (), (a,), backward, "mean")


def l2_norm(a):
    """Differentiable 2-norm of a (possibly multi-dimensional) tensor."""
    a = T(a)
    if accel.have():
        total = float(accel.np().sqrt((a.data ** 2).sum()))

        def backward(g):
            if a.requires_grad:
                inv = float(g[0]) / total if total else 0.0
                a._accum([inv * v for v in a.data])

        return a._child(_buf([total]), (), (a,), backward, "l2_norm")

    total = math.sqrt(math.fsum(v * v for v in a.data))

    def backward(g):
        if a.requires_grad:
            inv = g[0] / total if total else 0.0
            a._accum([inv * v for v in a.data])

    return a._child([total], (), (a,), backward, "l2_norm")


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

    if accel.have():
        np_ = accel.np()
        A = ad.reshape(n, k)
        B = bd.reshape(k, m)
        out = np_.ascontiguousarray(A @ B, dtype=np_.float64).reshape(-1)

        def backward(g):
            g = np_.asarray(g, dtype=np_.float64).reshape(n, m)
            if a.requires_grad:
                a._accum((g @ B.T).reshape(-1))
            if b.requires_grad:
                b._accum((A.T @ g).reshape(-1))

        return a._child(out, (n, m), (a, b), backward, "matmul")

    # forward as dot products against B's columns: the inner loop is a
    # tight accumulator, which is ~1.5x faster in pure Python than
    # spreading each A element across a row of B
    bcols = [[bd[p * m + j] for p in range(k)] for j in range(m)]
    out = [0.0] * (n * m)
    for i in range(n):
        arow = ad[i * k:(i + 1) * k]
        oi = i * m
        for j, bc in enumerate(bcols):
            s = 0.0
            for p in range(k):
                s += arow[p] * bc[p]
            out[oi + j] = s

    def backward(g):
        brows = [bd[p * m:(p + 1) * m] for p in range(k)]
        if a.requires_grad:
            # ga = g @ B^T, as row dots against B's rows
            ga = [0.0] * (n * k)
            for i in range(n):
                grow = g[i * m:(i + 1) * m]
                base = i * k
                for p in range(k):
                    s = 0.0
                    for x, y in zip(grow, brows[p]):
                        s += x * y
                    ga[base + p] = s
            a._accum(ga)
        if b.requires_grad:
            # gb = A^T @ g, as column dots
            acols = [[ad[i * k + p] for i in range(n)] for p in range(k)]
            gcols = [[g[i * m + j] for i in range(n)] for j in range(m)]
            gb = [0.0] * (k * m)
            for p in range(k):
                ap = acols[p]
                for j in range(m):
                    s = 0.0
                    for x, y in zip(ap, gcols[j]):
                        s += x * y
                    gb[p * m + j] = s
            b._accum(gb)

    return a._child(out, (n, m), (a, b), backward, "matmul")


def matmul_bias(a, b, c):
    """`a @ b + c` in one node: the single most common MLP pattern.

    One graph node instead of two, one gradient accumulation instead of
    two — measurably faster in the reference engine, identical math.
    `c` is a length-m vector (or 1-D row) broadcast across the rows.
    """
    a, b, c = T(a), T(b), T(c)
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise VMError("matmul_bias needs two 2-D tensors")
    n, k = a.shape
    k2, m = b.shape
    if k != k2:
        raise VMError(f"matmul_bias: inner dimensions differ ({k} vs {k2})")
    cdata = _flat1d(c)
    if len(cdata) != m and len(cdata) != 1:
        raise VMError(f"matmul_bias: bias length {len(cdata)} does not fit width {m}")
    if len(cdata) == 1:
        cdata = cdata * m
    ad, bd = a.data, b.data

    if accel.have():
        np_ = accel.np()
        A = ad.reshape(n, k)
        B = bd.reshape(k, m)
        out = np_.ascontiguousarray(A @ B + np_.asarray(cdata, dtype=np_.float64),
                                   dtype=np_.float64).reshape(-1)

        def backward(g):
            g = np_.asarray(g, dtype=np_.float64).reshape(n, m)
            if a.requires_grad:
                a._accum((g @ B.T).reshape(-1))
            if b.requires_grad:
                b._accum((A.T @ g).reshape(-1))
            if c.requires_grad:
                c._accum((g.sum(axis=0)).reshape(-1))

        return a._child(out, (n, m), (a, b, c), backward, "matmul_bias")

    bcols = [[bd[p * m + j] for p in range(k)] for j in range(m)]
    out = [0.0] * (n * m)
    for i in range(n):
        arow = ad[i * k:(i + 1) * k]
        oi = i * m
        for j, bc in enumerate(bcols):
            s = cdata[j]
            for p in range(k):
                s += arow[p] * bc[p]
            out[oi + j] = s

    def backward(g):
        brows = [bd[p * m:(p + 1) * m] for p in range(k)]
        if a.requires_grad:
            ga = [0.0] * (n * k)
            for i in range(n):
                grow = g[i * m:(i + 1) * m]
                base = i * k
                for p in range(k):
                    s = 0.0
                    for x, y in zip(grow, brows[p]):
                        s += x * y
                    ga[base + p] = s
            a._accum(ga)
        if b.requires_grad:
            acols = [[ad[i * k + p] for i in range(n)] for p in range(k)]
            gcols = [[g[i * m + j] for i in range(n)] for j in range(m)]
            gb = [0.0] * (k * m)
            for p in range(k):
                ap = acols[p]
                for j in range(m):
                    s = 0.0
                    for x, y in zip(ap, gcols[j]):
                        s += x * y
                    gb[p * m + j] = s
            b._accum(gb)
        if c.requires_grad:
            gc = [0.0] * m
            for i in range(n):
                grow = g[i * m:(i + 1) * m]
                for j in range(m):
                    gc[j] += grow[j]
            c._accum(gc)

    return a._child(out, (n, m), (a, b, c), backward, "matmul_bias")


def dropout(a, rate, training=True):
    """Inverted dropout: scaled pass-through during training, identity at
    inference. `training=false` (or rate 0) returns `a` itself, so the same
    graph is correct for both training and evaluation."""
    a = T(a)
    rate = float(rate)
    if not training or rate <= 0.0:
        return a
    if rate >= 1.0:
        raise VMError("dropout: rate must be in [0, 1)")
    n = a.size
    keep = 1.0 - rate

    if accel.have():
        m = accel.np()
        mask = (m.random.rand(n) >= rate).astype(m.float64) / keep
        ad = a.data.reshape(-1)
        out = (ad * mask).reshape(a.shape) if a.shape else ad * mask

        def backward(g):
            if a.requires_grad:
                g = m.asarray(g, dtype=m.float64).reshape(-1)
                a._accum((g * mask).reshape(ad.shape))

        return a._child(out.reshape(-1), a.shape, (a,), backward, "dropout")

    import random as _random

    mask = [1.0 / keep if _random.random() >= rate else 0.0 for _ in range(n)]
    ad = list(a.data)
    out = [ad[i] * mask[i] for i in range(n)]

    def backward(g):
        if a.requires_grad:
            a._accum([g[i] * mask[i] for i in range(n)])

    return a._child(out, a.shape, (a,), backward, "dropout")


def ce_softmax(logits, target):
    """Softmax cross-entropy, fused: `ce_t(t_softmax(x), y)` in one node.

    Numerically stabilised (row-max shift), averaged over rows exactly like
    `ce_t`, and one gradient pass instead of two. `target` holds one-hot (or
    soft) label rows of the same shape as `logits`.
    """
    a, y = T(logits), T(target)
    if a.shape != y.shape or len(a.shape) != 2:
        raise VMError(
            f"ce_softmax needs matching 2-D shapes, got {list(a.shape)} and {list(y.shape)}"
        )
    rows, cols = a.shape
    ad = _flat1d(a)
    yd = _flat1d(y)

    if accel.have():
        m = accel.np()
        X = m.asarray(ad, dtype=m.float64).reshape(rows, cols)
        Y = m.asarray(yd, dtype=m.float64).reshape(rows, cols)
        M = X.max(axis=1, keepdims=True)
        E = m.exp(X - M)
        S = E.sum(axis=1, keepdims=True)
        P = (E / S).reshape(-1)
        total = float(-m.sum(Y * (X - M - m.log(S))) / rows)

        def backward(g):
            if a.requires_grad:
                a._accum((g[0] / rows) * (P - Y.reshape(-1)))

        return a._child(_buf([total]), (), (a, y), backward, "ce_softmax")

    total = 0.0
    P = [0.0] * (rows * cols)
    for r in range(rows):
        base = r * cols
        row = ad[base:base + cols]
        mx = max(row)
        exps = [math.exp(v - mx) for v in row]
        s = math.fsum(exps)
        for j in range(cols):
            P[base + j] = exps[j] / s
            total -= yd[base + j] * (row[j] - mx - math.log(s))
    total = total / rows

    def backward(g):
        if a.requires_grad:
            scale = g[0] / rows
            a._accum([scale * (P[i] - yd[i]) for i in range(rows * cols)])

    return a._child(_buf([total]), (), (a, y), backward, "ce_softmax")


def transpose(a):
    a = T(a)
    if len(a.shape) != 2:
        raise VMError("transpose needs a 2-D tensor")
    n, m = a.shape
    if accel.have():
        np_ = accel.np()
        out = np_.ascontiguousarray(
            a.data.reshape(n, m).T, dtype=np_.float64
        ).reshape(-1)

        def backward(g):
            if a.requires_grad:
                g = np_.asarray(g, dtype=np_.float64).reshape(m, n)
                a._accum(g.T.reshape(-1))

        return a._child(out, (m, n), (a,), backward, "transpose")

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

    data = a.data
    if accel.have():
        # copy, matching the pure path's `list(a.data)`: the child must not
        # alias the input buffer
        data = accel.asarr(data).reshape(shape).copy().reshape(-1)
    return a._child(data, shape, (a,), backward, "reshape")


def t_slice(t, start, stop=None, axis=0):
    """Rows of a 2-D tensor (axis 0), columns (axis 1), or elements of a 1-D.

    Negative indices count from the end, as in list slicing. The gradient is
    zero-padded back to the full input shape.
    """
    t = T(t)
    axis = int(axis)
    if len(t.shape) == 1:
        axis = 0
        axis_len = t.shape[0]
    elif len(t.shape) == 2:
        if axis not in (0, 1):
            raise VMError("t_slice: axis must be 0 or 1 for a 2-D tensor")
        axis_len = t.shape[axis]
    else:
        raise VMError(f"t_slice: needs a 1-D or 2-D tensor, got shape {list(t.shape)}")

    def norm(v):
        v = int(v)
        if v < 0:
            v += axis_len
        return max(0, min(axis_len, v))

    s = norm(start)
    e = norm(axis_len if stop is None else stop)
    e = max(e, s)
    count = e - s

    if len(t.shape) == 1 or axis == 0:
        row = 1 if len(t.shape) == 1 else t.shape[1]
        off = s * row
        out_shape = (count,) if len(t.shape) == 1 else (count, row)
        data = t.data[off : off + count * row]
    else:
        cols = t.shape[1]
        out_shape = (t.shape[0], count)
        data = []
        for r in range(t.shape[0]):
            data.extend(t.data[r * cols + s : r * cols + e])
    if accel.have():
        data = accel.np().array(data, dtype=accel.np().float64).reshape(-1)

    def backward(g):
        if not t.requires_grad:
            return
        if len(t.shape) == 1 or axis == 0:
            if accel.have():
                m = accel.np()
                full = m.zeros(t.size, dtype=m.float64)
                full[off : off + len(g)] = m.asarray(g, dtype=m.float64)
            else:
                full = [0.0] * t.size
                full[off : off + len(g)] = list(g)
        else:
            cols = t.shape[1]
            if accel.have():
                m = accel.np()
                full = m.zeros(t.size, dtype=m.float64)
                full.reshape(t.shape[0], cols)[:, s:e] = m.asarray(
                    g, dtype=m.float64
                ).reshape(t.shape[0], count)
            else:
                full = [0.0] * t.size
                for r in range(t.shape[0]):
                    full[r * cols + s : r * cols + e] = g[r * count : (r + 1) * count]
        t._accum(full)

    return t._child(data, out_shape, (t,), backward, "slice")


def t_gather(t, indices):
    """Row lookup with gradient — the embedding operation.

    `t` is a 2-D table (V, D). `indices` is a List of Ints, flat (B,) or
    nested (B, S). The result is (B, D) or (B, S, D). The gradient scatters
    back into the rows, accumulating when a row is used more than once.
    """
    t = T(t)
    if len(t.shape) != 2:
        raise VMError("t_gather: the table must be 2-D")
    V, D = t.shape
    if not isinstance(indices, list) or not indices:
        raise VMError("t_gather: indices must be a non-empty List of Ints")
    nested = isinstance(indices[0], list)
    flat = []
    if nested:
        rows, cols = len(indices), len(indices[0])
        _check_rect(indices, "t_gather")
        out_shape = (rows, cols, D)
    else:
        rows = len(indices)
        out_shape = (rows, D)
    for part in (indices if nested else [indices]):
        for i in part:
            if isinstance(i, bool) or not isinstance(i, int):
                raise VMError(f"t_gather: index must be an Int, got {i!r}")
            if not 0 <= i < V:
                raise VMError(f"t_gather: index {i} out of range for {V} rows")
            flat.append(i)

    if accel.have():
        m = accel.np()
        tab = t.data.reshape(V, D)
        flat_arr = m.asarray(flat, dtype=m.int64)
        out = m.ascontiguousarray(tab[flat_arr], dtype=m.float64).reshape(out_shape).reshape(-1)

        def backward(g):
            if t.requires_grad:
                g2 = m.asarray(g, dtype=m.float64).reshape(-1, D)
                ga = m.zeros((V, D), dtype=m.float64)
                m.add.at(ga, flat_arr, g2)
                t._accum(ga.reshape(-1))

    else:
        out = [0.0] * (len(flat) * D)
        for pos, i in enumerate(flat):
            base = i * D
            for j in range(D):
                out[pos * D + j] = t.data[base + j]

        def backward(g):
            if t.requires_grad:
                ga = [0.0] * t.size
                for pos, i in enumerate(flat):
                    gb = i * D
                    for j in range(D):
                        ga[gb + j] += g[pos * D + j]
                t._accum(ga)

    return t._child(out, out_shape, (t,), backward, "gather")


def _check_rect(rows, fname):
    width = len(rows[0])
    for i, r in enumerate(rows):
        if not isinstance(r, list) or len(r) != width:
            raise VMError(f"{fname}: ragged indices — row 0 has {width} entries, row {i} differs")


def t_concat(a, b, axis=0):
    """Join two tensors along an axis. Gradient splits it back."""
    a, b = T(a), T(b)
    axis = int(axis)
    if len(a.shape) == 1:
        if len(b.shape) != 1 or axis != 0:
            raise VMError("t_concat: 1-D tensors concatenate along axis 0 only")
        out_shape = (a.size + b.size,)
    elif len(a.shape) == 2:
        if len(b.shape) != 2:
            raise VMError("t_concat: both tensors must have the same rank")
        if axis == 0:
            if a.shape[1] != b.shape[1]:
                raise VMError(
                    f"t_concat: column counts differ ({a.shape[1]} vs {b.shape[1]})"
                )
            out_shape = (a.shape[0] + b.shape[0], a.shape[1])
        elif axis == 1:
            if a.shape[0] != b.shape[0]:
                raise VMError(
                    f"t_concat: row counts differ ({a.shape[0]} vs {b.shape[0]})"
                )
            out_shape = (a.shape[0], a.shape[1] + b.shape[1])
        else:
            raise VMError("t_concat: axis must be 0 or 1 for a 2-D tensor")
    else:
        raise VMError("t_concat: needs 1-D or 2-D tensors")

    if accel.have():
        m = accel.np()
        out = m.concatenate([a.data.reshape(a.shape), b.data.reshape(b.shape)], axis=axis)
        out = m.ascontiguousarray(out, dtype=m.float64).reshape(-1)
    else:
        out = list(a.data) + list(b.data) if axis == 0 or len(a.shape) == 1 else _concat_cols(a, b)

    na = a.size

    def backward(g):
        if a.requires_grad:
            if len(a.shape) == 2 and axis == 1:
                ga, gb = _split_cols(list(g), a.shape, b.shape)
            else:
                ga, gb = list(g[:na]), list(g[na:])
            a._accum(ga)
        if b.requires_grad:
            if len(a.shape) == 2 and axis == 1:
                _, gb = _split_cols(list(g), a.shape, b.shape)
            else:
                gb = list(g[na:])
            b._accum(gb)

    return a._child(out, out_shape, (a, b), backward, "concat")


def _concat_cols(a, b):
    _, ca = a.shape
    _, cb = b.shape
    out = []
    for r in range(a.shape[0]):
        out.extend(a.data[r * ca : (r + 1) * ca])
        out.extend(b.data[r * cb : (r + 1) * cb])
    return out


def _split_cols(g, sa, sb):
    ca, cb = sa[1], sb[1]
    rows = sa[0]
    ga, gb = [], []
    for r in range(rows):
        ga.extend(g[r * (ca + cb) : r * (ca + cb) + ca])
        gb.extend(g[r * (ca + cb) + ca : r * (ca + cb) + ca + cb])
    return ga, gb


def where(mask, a, b):
    """Element-wise select: mask of Bools, `a` and `b` the same shape."""
    a, b = T(a), T(b)
    if a.size != b.size:
        raise VMError(f"where: size mismatch {a.size} vs {b.size}")
    if not isinstance(mask, list):
        raise VMError("where: the mask must be a List of Bools")

    def check_mask(m, prefix):
        flat = []

        def walk(x):
            if isinstance(x, list):
                for y in x:
                    walk(y)
            elif isinstance(x, bool):
                flat.append(x)
            else:
                raise VMError(f"where: mask entries must be Bools, found {x!r}")

        walk(m)
        if len(flat) != len(prefix):
            raise VMError(f"where: mask has {len(flat)} entries, expected {len(prefix)}")
        return flat

    am = check_mask(mask, a.data)

    if accel.have():
        m = accel.np()
        av, bv = a.data, b.data
        out = m.ascontiguousarray(
            m.where(m.asarray(am, dtype=bool), av, bv), dtype=m.float64
        ).reshape(-1)

        def backward(g):
            g = m.asarray(g, dtype=m.float64)
            if a.requires_grad:
                a._accum(m.where(m.asarray(am, dtype=bool), g, 0.0))
            if b.requires_grad:
                b._accum(m.where(m.asarray(am, dtype=bool), 0.0, g))

        return a._child(out, a.shape, (a, b), backward, "where")

    out = [a.data[i] if am[i] else b.data[i] for i in range(a.size)]

    def backward(g):
        if a.requires_grad:
            a._accum([g[i] if am[i] else 0.0 for i in range(a.size)])
        if b.requires_grad:
            b._accum([g[i] if not am[i] else 0.0 for i in range(a.size)])

    return a._child(out, a.shape, (a, b), backward, "where")


def t_softmax(a):
    """Row-wise softmax; numerically stabilised."""
    a = T(a)
    if len(a.shape) == 2:
        rows, cols = a.shape
    else:
        rows, cols = 1, a.size

    if accel.have():
        m = accel.np()
        M = a.data.reshape(rows, cols)
        mx = M.max(axis=1, keepdims=True)
        E = m.exp(M - mx)
        S = E.sum(axis=1, keepdims=True)
        P = m.ascontiguousarray(E / S, dtype=m.float64).reshape(-1)

        def backward(g):
            if a.requires_grad:
                G = m.asarray(g, dtype=m.float64).reshape(rows, cols)
                Pv = P.reshape(rows, cols)
                dot = (G * Pv).sum(axis=1, keepdims=True)
                a._accum((Pv * (G - dot)).reshape(-1))

        return a._child(P, a.shape, (a,), backward, "softmax")

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


def bce_loss(pred, target, eps=1e-12):
    """Binary cross-entropy. `pred` must already be in (0, 1)."""
    p, y = T(pred), T(target)
    if p.size != y.size:
        raise VMError(f"bce: size mismatch {p.size} vs {y.size}")
    n = p.size
    pd, yd = p.data, y.data
    if accel.have():
        m = accel.np()
        pi = m.clip(pd, eps, 1.0 - eps)
        total = float(-(yd * m.log(pi) + (1.0 - yd) * m.log(1.0 - pi)).mean())

        def backward(g):
            g0 = float(g[0])
            if p.requires_grad:
                p._accum(g0 * (pi - yd) / (pi * (1.0 - pi) * n))
            if y.requires_grad:
                y._accum(g0 * (m.log(1.0 - pi) - m.log(pi)) / n)

        return p._child(_buf([total]), (), (p, y), backward, "bce")

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
        if y.requires_grad:
            gy = [0.0] * n
            for i in range(n):
                pi = min(max(pd[i], eps), 1.0 - eps)
                gy[i] = g[0] * (math.log(1.0 - pi) - math.log(pi)) / n
            y._accum(gy)

    return p._child([total], (), (p, y), backward, "bce")


def bce_logits_loss(logits, target, eps=1e-12):
    """Binary cross-entropy from raw logits — the numerically stable form.

    loss = mean( max(z,0) - z*y + log(1 + exp(-|z|)) ),
    grad = (sigmoid(z) - y) / n.
    """
    p, y = T(logits), T(target)
    if p.size != y.size:
        raise VMError(f"bce_logits: size mismatch {p.size} vs {y.size}")
    n = p.size
    pd, yd = p.data, y.data
    if accel.have():
        m = accel.np()
        z = pd
        total = float((m.maximum(z, 0) - z * yd + m.log1p(m.exp(-m.abs(z)))).mean())

        def backward(g):
            if p.requires_grad:
                p._accum(float(g[0]) * (_np_sigmoid(z) - yd) / n)
            if y.requires_grad:
                y._accum(-float(g[0]) * z / n)

        return p._child(_buf([total]), (), (p, y), backward, "bce_logits")

    total = 0.0
    for i in range(n):
        zi = pd[i]
        total += max(zi, 0.0) - zi * yd[i] + math.log1p(math.exp(-abs(zi)))
    total /= n

    def backward(g):
        if p.requires_grad:
            gp = [0.0] * n
            for i in range(n):
                s = _sigmoid_scalar(pd[i])
                gp[i] = g[0] * (s - yd[i]) / n
            p._accum(gp)
        if y.requires_grad:
            y._accum([-g[0] * pd[i] / n for i in range(n)])

    return p._child([total], (), (p, y), backward, "bce_logits")


def _sigmoid_scalar(x):
    if x < -500:
        return 0.0
    if x > 500:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def huber_loss(pred, target, delta=1.0):
    """Mean Huber loss: quadratic close to the target, linear far from it."""
    delta = float(delta)
    if delta <= 0:
        raise VMError("huber: delta must be positive")
    p, y = T(pred), T(target)
    if p.size != y.size:
        raise VMError(f"huber: size mismatch {p.size} vs {y.size}")
    n = p.size
    pd, yd = p.data, y.data
    if accel.have():
        m = accel.np()
        d = pd - yd
        ad = m.abs(d)
        quad = 0.5 * d * d
        lin = delta * (ad - 0.5 * delta)
        per = m.where(ad <= delta, quad, lin)
        total = float(per.mean())

        def backward(g):
            if p.requires_grad:
                p._accum(float(g[0]) * m.clip(d, -delta, delta) / n)

        return p._child(_buf([total]), (), (p, y), backward, "huber")

    total = 0.0
    for i in range(n):
        d = pd[i] - yd[i]
        ad = abs(d)
        total += 0.5 * d * d if ad <= delta else delta * (ad - 0.5 * delta)
    total /= n

    def backward(g):
        if p.requires_grad:
            gp = [0.0] * n
            for i in range(n):
                d = pd[i] - yd[i]
                gp[i] = g[0] * max(-delta, min(delta, d)) / n
            p._accum(gp)

    return p._child([total], (), (p, y), backward, "huber")


def ce_loss(logits, target, eps=1e-12):
    """Softmax cross-entropy over logits; target is one-hot."""
    probs = t_softmax(logits)
    y = T(target)
    pd, yd = probs.data, y.data
    n = probs.shape[0] if len(probs.shape) == 2 else 1
    if accel.have():
        m = accel.np()
        total = float(-m.sum(yd * m.log(m.clip(pd, eps, 1.0))) / n)

        def backward(g):
            if probs.requires_grad:
                probs._accum(float(g[0]) * (-yd / m.clip(pd, eps, 1.0)) / n)

        return probs._child(_buf([total]), (), (probs, y), backward, "cross_entropy")

    total = -math.fsum(
        yd[i] * math.log(max(pd[i], eps)) for i in range(len(pd))
    ) / n

    def backward(g):
        if probs.requires_grad:
            probs._accum([g[0] * (-yd[i] / max(pd[i], eps)) / n for i in range(len(pd))])

    return probs._child([total], (), (probs, y), backward, "cross_entropy")


def l2_penalty(value):
    """0.5 * sum(x^2) over a tensor, or a List of tensors.

    Add it to a loss for weight decay: `t_add(loss, t_mul(0.001, l2_penalty_t(w)))`.
    """
    if isinstance(value, list):
        ts = [T(x) for x in value]
    else:
        ts = [T(value)]
    for t in ts:
        if not isinstance(t, Tensor):
            raise VMError("l2_penalty: needs a tensor or a List of tensors")

    if accel.have():
        m = accel.np()
        total = 0.0
        for t in ts:
            total += float(0.5 * (t.data ** 2).sum())

        def backward(g):
            g0 = float(g[0])
            for t in ts:
                if t.requires_grad:
                    t._accum(g0 * t.data)

        return ts[0]._child(_buf([total]), (), tuple(ts), backward, "l2_penalty")

    total = 0.0
    for t in ts:
        total += 0.5 * math.fsum(v * v for v in t.data)

    def backward(g):
        for t in ts:
            if t.requires_grad:
                t._accum([g[0] * v for v in t.data])

    return ts[0]._child([total], (), tuple(ts), backward, "l2_penalty")


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
