"""Function contracts: `needs` (preconditions) and `ensures` (postconditions).

Contracts are desugared into ordinary AST nodes before compilation, so every
later stage -- the type checker, the compiler, the peephole pass and the
native backend -- sees plain code and needs no special case.

    fn withdraw(balance: Real, amount: Real) -> Real:
        needs amount > 0.
        needs amount <= balance.
        ensures result >= 0.
        give balance - amount.
    done.

becomes, in effect:

    fn withdraw(balance, amount):
        when not (amount > 0):
            raise "withdraw: precondition failed: amount > 0".
        done.
        ...
        let result := balance - amount.
        when not (result >= 0):
            raise "withdraw: postcondition failed: result >= 0".
        done.
        give result.

A postcondition applies to *every* path out of the function, including
implicit fall-off-the-end, and `result` is bound to the value being returned.
"""
from __future__ import annotations

import os

from . import ast_nodes as A

RESULT = "result"


def enabled():
    """Contracts are checked unless AILANG_CONTRACTS=0.

    Turning them off strips the checks entirely rather than making them
    cheap, so a released build pays nothing for them.
    """
    return os.environ.get("AILANG_CONTRACTS", "1") != "0"


def _check(cond, message, line, col):
    """`when not (cond): raise message. done.`"""
    guard = A.Unary("not", cond, line, col)
    raise_stmt = A.Raise(A.Literal(message, line, col), line, col)
    return A.When([A.Branch(guard, [raise_stmt])], None, line, col)


def _describe(kind, fn_name, text):
    label = "precondition" if kind == "needs" else "postcondition"
    if not text:
        return f"{fn_name}: {label} failed"
    return f"{fn_name}: {label} failed: {text}"


def split(body):
    """Separate leading contracts from the rest of a function body."""
    pre, post, rest = [], [], []
    for i, st in enumerate(body):
        if isinstance(st, A.Needs) and not rest:
            pre.append(st)
        elif isinstance(st, A.Ensures) and not rest:
            post.append(st)
        else:
            rest = body[i:]
            break
    return pre, post, rest


def _has_contracts(body):
    return any(isinstance(s, (A.Needs, A.Ensures)) for s in body)


class _PostRewriter:
    """Rewrites every `give` so the postconditions run first."""

    def __init__(self, checks):
        self.checks = checks

    def rewrite(self, stmts):
        out = []
        for st in stmts:
            out.extend(self.stmt(st))
        return out

    def stmt(self, st):
        if isinstance(st, A.Give):
            # bind the value, check it, then return it
            bind = A.Let(RESULT, st.expr, None, st.line, st.col)
            checks = [c(st.line, st.col) for c in self.checks]
            give = A.Give(A.Name(RESULT, st.line, st.col), st.line, st.col)
            return [bind, *checks, give]
        if isinstance(st, A.When):
            branches = [A.Branch(b.cond, self.rewrite(b.body)) for b in st.branches]
            els = self.rewrite(st.else_body) if st.else_body is not None else None
            return [A.When(branches, els, st.line, st.col)]
        if isinstance(st, A.Attempt):
            st.body = self.rewrite(st.body)
            if getattr(st, "rescue_body", None) is not None:
                st.rescue_body = self.rewrite(st.rescue_body)
            return [st]
        for attr in ("body",):
            inner = getattr(st, attr, None)
            # loops and other block statements
            if isinstance(inner, list) and not isinstance(st, A.Fn):
                setattr(st, attr, self.rewrite(inner))
        return [st]


def desugar_body(fn_name, body, check=True):
    """Return `body` with contracts turned into ordinary checks."""
    if not _has_contracts(body):
        return body
    if not check:
        # strip them: keep only the real statements
        return split(body)[2]
    pre, post, rest = split(body)

    out = []
    for p in pre:
        out.append(_check(p.expr, _describe("needs", fn_name, p.text), p.line, p.col))

    if post:
        def make(p):
            return lambda line, col: _check(
                p.expr, _describe("ensures", fn_name, p.text), p.line, p.col
            )

        checks = [make(p) for p in post]
        rest = _PostRewriter(checks).rewrite(rest)
        # a function that falls off the end returns nothing; check that too
        if not (rest and isinstance(rest[-1], A.Give)):
            line = post[-1].line
            rest = rest + [
                A.Let(RESULT, A.Literal(None, line, 0), None, line, 0),
                *[c(line, 0) for c in checks],
            ]
    out.extend(rest)
    return out


def desugar(node, check=None):
    """Walk a program (or any node list) applying contract desugaring."""
    if check is None:
        check = enabled()
    stmts = node.statements if isinstance(node, A.Program) else node
    for st in stmts:
        _walk(st, check)
    return node


def _walk(st, check=True):
    if isinstance(st, (A.Fn, A.FnExpr)):
        st.body = desugar_body(st.name, st.body, check)
        for inner in st.body:
            _walk(inner, check)
        return
    for attr in ("body", "else_body", "rescue_body"):
        inner = getattr(st, attr, None)
        if isinstance(inner, list):
            for x in inner:
                _walk(x, check)
    for b in getattr(st, "branches", []) or []:
        for x in getattr(b, "body", []):
            _walk(x, check)
    # functions can also appear inside expressions (let f := fn() ...)
    ex = getattr(st, "expr", None)
    if isinstance(ex, A.FnExpr):
        _walk(ex, check)
