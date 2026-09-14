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
            stmts.append(self.statement())
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
            body.append(self.statement())
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
            name = self.take("IDENT", "binding name").value
            declared = None
            if self.match("COLON"):
                declared = self.take("TYPE", "type annotation").value
            self.take("DEFINE", "binding (use ':=')")
            expr = self.expr()
            self.dot()
            node = A.Let if k == "LET" else A.Var
            return node(name, expr, declared, t.line, t.col)

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
        if self.at("ASSIGN"):
            self.i += 1
            if not isinstance(expr, (A.Name, A.Index, A.Field)):
                raise ParseError("invalid assignment target", t.line, t.col)
            value = self.expr()
            self.dot()
            return A.Assign(expr, value, t.line, t.col)
        self.dot()
        return A.ExprStmt(expr, t.line, t.col)

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
        while self.cur().kind in _COMPARE:
            t = self.cur()
            op = _COMPARE[t.kind]
            self.i += 1
            left = A.Binary(left, op, self.term(), t.line, t.col)
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
