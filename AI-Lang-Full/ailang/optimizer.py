"""AST optimiser: constant folding and dead-branch elimination.

Only performs transformations that preserve AI-Lang semantics, including the
short-circuit behaviour of `and`/`or`.
"""

from __future__ import annotations

from . import ast_nodes as A

_NUM = (int, float)


def _is_const(n):
    return isinstance(n, A.Literal)


def fold_expr(n):
    if isinstance(n, A.Binary):
        n.left = fold_expr(n.left)
        n.right = fold_expr(n.right)
        # Do not fold and/or: short-circuiting must survive to the compiler
        # unless both sides are already constant and side-effect free.
        if _is_const(n.left) and _is_const(n.right):
            a, b, op = n.left.value, n.right.value, n.op
            try:
                v = _apply(op, a, b)
            except Exception:
                return n
            if v is not _NOFOLD:
                return A.Literal(v, n.line, n.col)
        return n

    if isinstance(n, A.Unary):
        n.expr = fold_expr(n.expr)
        if _is_const(n.expr):
            v = n.expr.value
            if n.op == "-" and isinstance(v, _NUM) and not isinstance(v, bool):
                return A.Literal(-v, n.line, n.col)
            if n.op == "not":
                return A.Literal(not (v is not None and v is not False), n.line, n.col)
        return n

    if isinstance(n, A.ListExpr):
        n.items = [fold_expr(x) for x in n.items]
        return n
    if isinstance(n, A.MapExpr):
        n.items = [(fold_expr(k), fold_expr(v)) for k, v in n.items]
        return n
    if isinstance(n, A.Call):
        n.fn = fold_expr(n.fn)
        n.args = [(name, fold_expr(x)) for name, x in n.args]
        return n
    if isinstance(n, A.Index):
        n.obj = fold_expr(n.obj)
        n.index = fold_expr(n.index)
        return n
    if isinstance(n, A.Field):
        n.obj = fold_expr(n.obj)
        return n
    if isinstance(n, A.Convert):
        n.expr = fold_expr(n.expr)
        return n
    if isinstance(n, A.FnExpr):
        n.body = fold_body(n.body)
        return n
    return n


_NOFOLD = object()


def _apply(op, a, b):
    both_num = isinstance(a, _NUM) and isinstance(b, _NUM) and not isinstance(a, bool) and not isinstance(b, bool)
    if op == "+":
        if isinstance(a, str) and isinstance(b, str):
            return a + b
        if both_num:
            return a + b
        return _NOFOLD
    if op in ("-", "*"):
        if both_num:
            return a - b if op == "-" else a * b
        return _NOFOLD
    if op == "/":
        if both_num and b != 0:
            return a / b
        return _NOFOLD
    if op == "%":
        if both_num and b != 0:
            return a % b
        return _NOFOLD
    if op in ("<", "<=", ">", ">="):
        if both_num or (isinstance(a, str) and isinstance(b, str)):
            return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]
        return _NOFOLD
    if op in ("==", "!="):
        # must match vm._equal exactly, or folding would change semantics
        from .vm import _equal

        eq = _equal(a, b)
        return eq if op == "==" else not eq
    if op in ("and", "or"):
        ta = a is not None and a is not False
        tb = b is not None and b is not False
        return (ta and tb) if op == "and" else (ta or tb)
    if op == "??":
        return b if a is None else a
    return _NOFOLD


def fold_body(body):
    return [fold_stmt(s) for s in body]


def fold_stmt(s):
    for attr in ("expr", "cond", "iterable"):
        if hasattr(s, attr) and getattr(s, attr) is not None:
            setattr(s, attr, fold_expr(getattr(s, attr)))
    if isinstance(s, A.Assign):
        s.target = fold_expr(s.target)
    if isinstance(s, A.Fn):
        s.body = fold_body(s.body)
    if isinstance(s, A.When):
        kept = []
        for br in s.branches:
            br.cond = fold_expr(br.cond)
            br.body = fold_body(br.body)
            # a literally-false branch can never run
            if isinstance(br.cond, A.Literal) and br.cond.value in (False, None):
                continue
            kept.append(br)
        s.branches = kept or s.branches
        if s.else_body is not None:
            s.else_body = fold_body(s.else_body)
    if isinstance(s, (A.While, A.Repeat)):
        s.body = fold_body(s.body)
    if isinstance(s, A.Attempt):
        s.body = fold_body(s.body)
        s.rescue_body = fold_body(s.rescue_body)
    return s


def optimize(program: A.Program) -> A.Program:
    program.statements = fold_body(program.statements)
    return program
