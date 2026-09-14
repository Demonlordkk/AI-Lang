"""Recursive-descent parser producing the AI-Lang AST."""

from __future__ import annotations

from typing import List, Optional

from . import ast_nodes as A
from .errors import ParseError
from .lexer import lex

# binary operator precedence (higher binds tighter)
_COMPARE = {"LT": "<", "LE": "<=", "GT": ">", "GE": ">="}
_EQUALITY = {"EQEQ": "==", "NE": "!="}
_TERM = {"PLUS": "+", "MINUS": "-"}
_FACTOR = {"STAR": "*", "SLASH": "/", "PERCENT": "%"}

_BLOCK_ENDERS = {"DONE", "ELSE", "ELIF", "RESCUE", "EOF"}

# `x +<- 1` desugars to `x <- x + 1`
_COMPOUND = {
    "PLUS_ASSIGN": "+",
    "MINUS_ASSIGN": "-",
    "STAR_ASSIGN": "*",
    "SLASH_ASSIGN": "/",
}


class Parser:
    def __init__(self, source: str):
        self.source = source
        self.tokens = lex(source)
        self.i = 0

    # -------------------------------------------------------------- utilities
    def cur(self):
        return self.tokens[self.i]

    def peek(self, offset: int = 1):
        idx = min(self.i + offset, len(self.tokens) - 1)
        return self.tokens[idx]

    def at(self, kind: str) -> bool:
        return self.cur().kind == kind

    def match(self, kind: str) -> bool:
        if self.at(kind):
            self.i += 1
            return True
        return False

    def take(self, kind: str, what: str = ""):
        t = self.cur()
        if t.kind != kind:
            hint = f" while parsing {what}" if what else ""
            raise ParseError(
                f"expected {_friendly(kind)} but found {_friendly(t.kind)}{hint}", t.line, t.col
            )
        self.i += 1
        return t

    def dot(self):
        self.take("DOT", "end of statement")

    # ----------------------------------------------------------------- program
    def program(self) -> A.Program:
        stmts = []
        while not self.at("EOF"):
            st = self.statement()
            # a statement may expand to several nodes (destructuring)
            stmts.extend(st) if isinstance(st, list) else stmts.append(st)
        return A.Program(stmts)

    def block(self, *, enders=("DONE",), consume_done=True, consume_dot=True) -> List:
        """Parse statements until one of `enders`. Optionally consume `done.`.

        `consume_dot` is False for anonymous functions, where the trailing '.'
        belongs to the enclosing statement (`let f := fn(): ... done.`).
        """
        body = []
        while True:
            k = self.cur().kind
            if k == "EOF":
                t = self.cur()
                raise ParseError("unterminated block: missing 'done.'", t.line, t.col)
            if k in enders:
                break
            st = self.statement()
            body.extend(st) if isinstance(st, list) else body.append(st)
        if consume_done:
            self.take("DONE", "end of block")
            if consume_dot:
                self.dot()
        return body

    # -------------------------------------------------------------- statements
    def statement(self):
        t = self.cur()
        k = t.kind

        if k == "LET" or k == "VAR":
            self.i += 1
            node = A.Let if k == "LET" else A.Var

            # destructuring: `let [a, b] := pair.` / `let {name, age} := person.`
            if self.at("LBRACKET") or self.at("LBRACE"):
                return self._destructure(node, t)

            name = self.take("IDENT", "binding name").value
            declared = None
            if self.match("COLON"):
                declared = self.take("TYPE", "type annotation").value
            self.take("DEFINE", "binding (use ':=')")
            expr = self.expr()
            self.dot()
            return node(name, expr, declared, t.line, t.col)

        if k == "GIVEN":
            return self._given(t)

        if k == "EMIT":
            self.i += 1
            expr = self.expr()
            self.dot()
            return A.Emit(expr, t.line, t.col)

        if k == "GIVE":
            self.i += 1
            if self.at("DOT"):
                self.dot()
                return A.Give(A.Literal(None, t.line, t.col), t.line, t.col)
            expr = self.expr()
            self.dot()
            return A.Give(expr, t.line, t.col)

        if k == "USE":
            self.i += 1
            parts = [self.take("IDENT", "module path").value]
            while self.match("SLASH"):
                parts.append(self.take("IDENT", "module path").value)
            path = "/".join(parts)
            alias = parts[-1]
            if self.match("AS"):
                alias = self.take("IDENT", "module alias").value
            self.dot()
            return A.Use(path, alias, t.line, t.col)

        if k == "RAISE":
            self.i += 1
            expr = self.expr()
            self.dot()
            return A.Raise(expr, t.line, t.col)

        if k == "STOP":
            self.i += 1
            self.dot()
            return A.Stop(t.line, t.col)

        if k == "NEXT":
            self.i += 1
            self.dot()
            return A.Next(t.line, t.col)

        if k == "ATTEMPT":
            self.i += 1
            self.take("COLON", "attempt block")
            body = self.block(enders=("RESCUE",), consume_done=False)
            self.take("RESCUE", "rescue clause")
            err_name = "error"
            if self.at("IDENT"):
                err_name = self.take("IDENT").value
            self.take("COLON", "rescue block")
            rescue = self.block()
            return A.Attempt(body, err_name, rescue, t.line, t.col)

        if k == "FN" and self.peek().kind == "IDENT":
            self.i += 1
            name = self.take("IDENT", "function name").value
            params = self.param_list()
            ret = None
            if self.match("ARROW"):
                ret = self.take("TYPE", "return type").value
            self.take("COLON", "function body")
            body = self.block()
            return A.Fn(name, params, ret, body, t.line, t.col)

        if k == "WHEN":
            self.i += 1
            cond = self.expr()
            self.take("COLON", "when block")
            first = self.block(enders=("DONE", "ELSE", "ELIF"), consume_done=False)
            branches = [A.Branch(cond, first)]
            else_body = None
            while True:
                if self.match("ELIF"):
                    c2 = self.expr()
                    self.take("COLON", "elif block")
                    b2 = self.block(enders=("DONE", "ELSE", "ELIF"), consume_done=False)
                    branches.append(A.Branch(c2, b2))
                    continue
                if self.match("ELSE"):
                    self.take("COLON", "else block")
                    else_body = self.block(enders=("DONE",), consume_done=False)
                break
            self.take("DONE", "end of when")
            self.dot()
            return A.When(branches, else_body, t.line, t.col)

        if k == "WHILE":
            self.i += 1
            cond = self.expr()
            self.take("COLON", "while block")
            body = self.block()
            return A.While(cond, body, t.line, t.col)

        if k == "REPEAT":
            self.i += 1
            name = self.take("IDENT", "loop variable").value
            index_name = None
            if self.match("AT"):
                index_name = self.take("IDENT", "index variable").value
            self.take("IN", "repeat loop")
            iterable = self.expr()
            self.take("COLON", "repeat block")
            body = self.block()
            return A.Repeat(name, iterable, body, index_name, t.line, t.col)

        if k == "RECORD":
            self.i += 1
            name = self.take("IDENT", "record name").value
            self.take("COLON", "record body")
            fields = []
            # `done: Bool.` is a field; a bare `done.` closes the record.
            while not (self.at("DONE") and self.peek().kind != "COLON"):
                if self.at("EOF"):
                    raise ParseError("unterminated record", t.line, t.col)
                if not _is_field_name(self.cur()):
                    tk = self.cur()
                    raise ParseError(
                        f"expected a field name but found {_friendly(tk.kind)}", tk.line, tk.col
                    )
                fname = str(self.cur().value)
                self.i += 1
                ftype = None
                if self.match("COLON"):
                    ftype = self.take("TYPE", "field type").value
                self.dot()
                fields.append((fname, ftype))
            self.take("DONE")
            self.dot()
            return A.Record(name, fields, t.line, t.col)

        # assignment or bare expression
        expr = self.expr()
        compound = _COMPOUND.get(self.cur().kind)
        if compound:
            self.i += 1
            if not isinstance(expr, (A.Name, A.Index, A.Field)):
                raise ParseError("invalid assignment target", t.line, t.col)
            value = self.expr()
            self.dot()
            return A.Assign(
                expr, A.Binary(expr, compound, value, t.line, t.col), t.line, t.col
            )
        if self.at("ASSIGN"):
            self.i += 1
            if not isinstance(expr, (A.Name, A.Index, A.Field)):
                raise ParseError("invalid assignment target", t.line, t.col)
            value = self.expr()
            self.dot()
            return A.Assign(expr, value, t.line, t.col)
        self.dot()
        return A.ExprStmt(expr, t.line, t.col)

    def _interpolate(self, tok):
        """Turn "a={x}b" into ("a" + to Text(x)) + "b" at parse time.

        Interpolation is pure syntax: it lowers to ordinary Text conversion
        and concatenation, so it costs nothing extra at runtime and reports
        errors inside {...} with the enclosing line/column.
        """
        node = None
        for kind, value in tok.value:
            if kind == "lit":
                piece = A.Literal(value, tok.line, tok.col)
            else:
                try:
                    sub = Parser(value)
                    piece = sub.expr()
                    if not sub.at("EOF"):
                        bad = sub.cur()
                        raise ParseError(
                            f"unexpected {_friendly(bad.kind)} inside text interpolation",
                            tok.line,
                            tok.col,
                        )
                except ParseError as e:
                    raise ParseError(
                        f"in text interpolation {{{value}}}: {e.message}", tok.line, tok.col
                    ) from e
                piece = A.Convert("Text", piece, tok.line, tok.col)
            node = piece if node is None else A.Binary(node, "+", piece, tok.line, tok.col)
        return node if node is not None else A.Literal("", tok.line, tok.col)

    def _destructure(self, node, t):
        """Lower `let [a, b] := xs.` and `let {x, y} := m.` into plain bindings.

        The subject is evaluated once into a hidden temporary, then each name
        is bound to an index (for lists) or a field (for maps). Because this
        expands to ordinary Let/Var nodes, the type checker, compiler and VM
        need no special cases.
        """
        is_list = self.at("LBRACKET")
        close = "RBRACKET" if is_list else "RBRACE"
        self.i += 1
        names = []
        while not self.at(close):
            n = self.take("IDENT", "name in destructuring pattern")
            names.append(n)
            if not self.match("COMMA"):
                break
        self.take(close, "']'" if is_list else "'}'")
        if not names:
            raise ParseError("destructuring needs at least one name", t.line, t.col)
        self.take("DEFINE", "binding (use ':=')")
        subject = self.expr()
        self.dot()

        # a hidden temporary keeps the subject from being evaluated repeatedly
        tmp = f"__d{t.line}_{t.col}"
        out = [A.Let(tmp, subject, None, t.line, t.col)]
        for i, n in enumerate(names):
            src = A.Name(tmp, n.line, n.col)
            if is_list:
                access = A.Index(src, A.Literal(i, n.line, n.col), n.line, n.col)
            else:
                access = A.Field(src, n.value, n.line, n.col)
            out.append(node(n.value, access, None, n.line, n.col))
        return out

    def _given(self, t):
        """`given x: is 1: ... is 2, 3: ... else: ... done.`

        A multi-way branch on one subject. It lowers to the same When node a
        chain of `elif`s would produce, with the subject evaluated once into a
        hidden temporary, so there is no new runtime machinery and no
        fall-through surprises.
        """
        self.i += 1
        subject = self.expr()
        self.take("COLON", "':' after the given subject")
        tmp = f"__g{t.line}_{t.col}"

        branches = []
        else_body = None
        while True:
            k = self.cur().kind
            if k == "IS":
                it = self.cur()
                self.i += 1
                values = [self.expr()]
                while self.match("COMMA"):
                    values.append(self.expr())
                self.take("COLON", "':' after the given value")
                body = self.block(enders=("IS", "ELSE", "DONE"), consume_done=False)
                cond = None
                for v in values:
                    test = A.Binary(
                        A.Name(tmp, it.line, it.col), "==", v, it.line, it.col
                    )
                    cond = test if cond is None else A.Binary(
                        cond, "or", test, it.line, it.col
                    )
                branches.append(A.Branch(cond, body))
                continue
            if k == "ELSE":
                self.i += 1
                self.take("COLON", "':' after else")
                else_body = self.block(enders=("DONE",), consume_done=False)
                continue
            break
        self.take("DONE", "'done' to close the given block")
        self.dot()
        if not branches and else_body is None:
            raise ParseError("given needs at least one 'is' branch", t.line, t.col)
        return [
            A.Let(tmp, subject, None, t.line, t.col),
            A.When(branches, else_body, t.line, t.col),
        ]

    def param_list(self):
        self.take("LPAREN", "parameter list")
        params = []
        if not self.at("RPAREN"):
            while True:
                pname = self.take("IDENT", "parameter name").value
                ptype = None
                if self.match("COLON"):
                    ptype = self.take("TYPE", "parameter type").value
                params.append((pname, ptype))
                if not self.match("COMMA"):
                    break
        self.take("RPAREN", "parameter list")
        return params

    # ------------------------------------------------------------- expressions
    def expr(self):
        return self.pipeline()

    def pipeline(self):
        """`x |> f |> g` desugars to `g(f(x))` — a core brevity feature."""
        left = self.coalesce()
        while self.at("PIPE"):
            t = self.cur()
            self.i += 1
            right = self.coalesce()
            if isinstance(right, A.Call):
                right.args.insert(0, (None, left))
                left = right
            else:
                left = A.Call(right, [(None, left)], t.line, t.col)
        return left

    def coalesce(self):
        left = self.or_expr()
        while self.at("COALESCE"):
            t = self.cur()
            self.i += 1
            left = A.Binary(left, "??", self.or_expr(), t.line, t.col)
        return left

    def or_expr(self):
        left = self.and_expr()
        while self.at("OR"):
            t = self.cur()
            self.i += 1
            left = A.Binary(left, "or", self.and_expr(), t.line, t.col)
        return left

    def and_expr(self):
        left = self.equality()
        while self.at("AND"):
            t = self.cur()
            self.i += 1
            left = A.Binary(left, "and", self.equality(), t.line, t.col)
        return left

    def equality(self):
        left = self.comparison()
        while self.cur().kind in _EQUALITY:
            t = self.cur()
            op = _EQUALITY[t.kind]
            self.i += 1
            left = A.Binary(left, op, self.comparison(), t.line, t.col)
        return left

    def comparison(self):
        left = self.term()
        while True:
            k = self.cur().kind
            if k in _COMPARE:
                t = self.cur()
                op = _COMPARE[t.kind]
                self.i += 1
                left = A.Binary(left, op, self.term(), t.line, t.col)
                continue
            # membership reads as prose: `when name in names:` and
            # `when key not in seen:`. Lowered to the `contains` builtin.
            if k == "IN":
                t = self.cur()
                self.i += 1
                right = self.term()
                left = A.Call(
                    A.Name("contains", t.line, t.col),
                    [(None, right), (None, left)],
                    t.line,
                    t.col,
                )
                continue
            if k == "NOT" and self.peek(1).kind == "IN":
                t = self.cur()
                self.i += 2
                right = self.term()
                inner = A.Call(
                    A.Name("contains", t.line, t.col),
                    [(None, right), (None, left)],
                    t.line,
                    t.col,
                )
                left = A.Unary("not", inner, t.line, t.col)
                continue
            break
        return left

    def term(self):
        left = self.factor()
        while self.cur().kind in _TERM:
            t = self.cur()
            op = _TERM[t.kind]
            self.i += 1
            left = A.Binary(left, op, self.factor(), t.line, t.col)
        return left

    def factor(self):
        left = self.unary()
        while self.cur().kind in _FACTOR:
            t = self.cur()
            op = _FACTOR[t.kind]
            self.i += 1
            left = A.Binary(left, op, self.unary(), t.line, t.col)
        return left

    def unary(self):
        t = self.cur()
        if self.match("NOT"):
            return A.Unary("not", self.unary(), t.line, t.col)
        if self.match("MINUS"):
            return A.Unary("-", self.unary(), t.line, t.col)
        return self.postfix()

    def postfix(self):
        node = self.primary()
        while True:
            t = self.cur()
            if self.at("LPAREN"):
                self.i += 1
                args = []
                if not self.at("RPAREN"):
                    while True:
                        # named argument: NAME ':' expr (keywords allowed, e.g. done:)
                        if _is_field_name(self.cur()) and self.peek().kind == "COLON":
                            argname = str(self.cur().value)
                            self.i += 1
                            self.take("COLON")
                            args.append((argname, self.expr()))
                        else:
                            args.append((None, self.expr()))
                        if not self.match("COMMA"):
                            break
                self.take("RPAREN", "argument list")
                node = A.Call(node, args, t.line, t.col)
            elif self.at("LBRACKET"):
                self.i += 1
                idx = self.expr()
                self.take("RBRACKET", "index expression")
                node = A.Index(node, idx, t.line, t.col)
            elif self.at("DOT") and t.tight and _is_field_name(self.peek()):
                self.i += 1
                fname = str(self.cur().value)
                self.i += 1
                node = A.Field(node, fname, t.line, t.col)
            else:
                break
        return node

    def primary(self):
        t = self.cur()
        k = t.kind

        if k in ("INT", "REAL", "TEXT", "BOOL"):
            self.i += 1
            return A.Literal(t.value, t.line, t.col)

        if k == "TEXT_PARTS":
            self.i += 1
            return self._interpolate(t)
        if k == "NOTHING":
            self.i += 1
            return A.Literal(None, t.line, t.col)
        if k == "IDENT":
            self.i += 1
            return A.Name(t.value, t.line, t.col)
        if k == "TYPE":
            # Types are first-class enough to be used as constructors: Int(x)
            self.i += 1
            return A.Name(t.value, t.line, t.col)

        if k == "LPAREN":
            self.i += 1
            inner = self.expr()
            self.take("RPAREN", "parenthesised expression")
            return inner

        if k == "LBRACKET":
            self.i += 1
            items = []
            if not self.at("RBRACKET"):
                while True:
                    if self.at("RBRACKET"):
                        break
                    items.append(self.expr())
                    if not self.match("COMMA"):
                        break
            self.take("RBRACKET", "list literal")
            return A.ListExpr(items, t.line, t.col)

        if k == "LBRACE":
            self.i += 1
            pairs = []
            if not self.at("RBRACE"):
                while True:
                    if self.at("RBRACE"):
                        break
                    key = self.expr()
                    self.take("COLON", "map entry")
                    pairs.append((key, self.expr()))
                    if not self.match("COMMA"):
                        break
            self.take("RBRACE", "map literal")
            return A.MapExpr(pairs, t.line, t.col)

        if k == "TO":
            self.i += 1
            typ = self.take("TYPE", "conversion target").value
            return A.Convert(typ, self.unary(), t.line, t.col)

        # anonymous function: fn(a, b) -> T: ... done.
        if k == "FN":
            self.i += 1
            params = self.param_list()
            ret = None
            if self.match("ARROW"):
                ret = self.take("TYPE", "return type").value
            self.take("COLON", "lambda body")
            body = self.block(consume_dot=False)
            return A.FnExpr(params, ret, body, "<lambda>", t.line, t.col)

        # short lambda:  \x -> x * 2      \a, b -> a + b
        if k == "BACKSLASH":
            self.i += 1
            params = []
            if not self.at("ARROW"):
                while True:
                    pname = self.take("IDENT", "lambda parameter").value
                    ptype = None
                    if self.match("COLON"):
                        ptype = self.take("TYPE").value
                    params.append((pname, ptype))
                    if not self.match("COMMA"):
                        break
            self.take("ARROW", "lambda arrow")
            body_expr = self.expr()
            return A.FnExpr(
                params, None, [A.Give(body_expr, t.line, t.col)], "<lambda>", t.line, t.col
            )

        raise ParseError(f"expected an expression but found {_friendly(k)}", t.line, t.col)


_FRIENDLY = {
    "DOT": "'.'",
    "COLON": "':'",
    "DEFINE": "':='",
    "ASSIGN": "'<-'",
    "ARROW": "'->'",
    "LPAREN": "'('",
    "RPAREN": "')'",
    "LBRACKET": "'['",
    "RBRACKET": "']'",
    "LBRACE": "'{'",
    "RBRACE": "'}'",
    "COMMA": "','",
    "IDENT": "an identifier",
    "TYPE": "a type name",
    "EOF": "end of file",
    "DONE": "'done'",
    "INT": "an integer",
    "REAL": "a number",
    "TEXT": "a text literal",
}


def _is_field_name(tok) -> bool:
    """After a tight '.', keywords are valid field names (`task.done`)."""
    if tok.kind in ("IDENT", "TYPE"):
        return True
    return tok.kind in _KEYWORD_FIELDS


# keyword tokens whose spelling is a plausible record/map field
_KEYWORD_FIELDS = {
    "DONE", "IN", "AT", "TO", "AS", "USE", "EMIT", "GIVE", "LET", "VAR",
    "WHEN", "ELSE", "ELIF", "REPEAT", "WHILE", "RECORD", "FN", "AND", "OR",
    "NOT", "STOP", "NEXT", "RAISE", "ATTEMPT", "RESCUE", "BOOL", "NOTHING",
}


def _friendly(kind: str) -> str:
    return _FRIENDLY.get(kind, f"'{kind.lower()}'")


def parse(source: str) -> A.Program:
    return Parser(source).program()
