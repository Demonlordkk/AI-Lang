"""Static checker for AI-Lang.

Performs scope resolution, mutability enforcement, arity checking and a
gradual type discipline: `Any` unifies with everything, so untyped code keeps
working while annotated code gets real guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import ast_nodes as A
from .errors import CheckError


@dataclass(frozen=True)
class Ty:
    name: str
    elem: Optional["Ty"] = None

    def __str__(self):
        if self.name in ("List", "Map") and self.elem:
            return f"{self.name}[{self.elem}]"
        return self.name


ANY = Ty("Any")
INT = Ty("Int")
REAL = Ty("Real")
BOOL = Ty("Bool")
TEXT = Ty("Text")
BYTE = Ty("Byte")
VOID = Ty("Void")
NOTHING = Ty("Nothing")
LIST = Ty("List")
MAP = Ty("Map")
FUNCTION = Ty("Function")

NUMERIC = {"Int", "Real", "Byte"}


def ty(name: Optional[str]) -> Ty:
    if not name:
        return ANY
    return Ty(name)


@dataclass
class FnSig:
    params: list
    ret: Ty
    name: str = ""
    optional: int = 0  # how many trailing params may be omitted

    @property
    def required(self):
        return len(self.params) - self.optional


# name -> (param specs, return type)
BUILTIN_SIGS = {
    "len": ([("value", ANY)], INT),
    "type_of": ([("value", ANY)], TEXT),
    "abs": ([("value", ANY)], ANY),
    "floor": ([("value", ANY)], INT),
    "ceil": ([("value", ANY)], INT),
    "round": ([("value", ANY), ("digits", INT)], ANY, 1),
    "sqrt": ([("value", ANY)], REAL),
    "pow": ([("base", ANY), ("exp", ANY)], REAL),
    "min": ([("a", ANY), ("b", ANY)], ANY, 1),
    "max": ([("a", ANY), ("b", ANY)], ANY, 1),
    "sum": ([("items", LIST)], ANY),
    "clock": ([], REAL),
    "str": ([("value", ANY)], TEXT),
    "int": ([("value", ANY)], INT),
    "real": ([("value", ANY)], REAL),
    "bool": ([("value", ANY)], BOOL),
    "print": ([("value", ANY)], VOID, 1),
    "join": ([("items", LIST), ("sep", TEXT)], TEXT, 1),
    "split": ([("text", TEXT), ("sep", TEXT)], LIST, 1),
    "upper": ([("text", TEXT)], TEXT),
    "lower": ([("text", TEXT)], TEXT),
    "trim": ([("text", TEXT)], TEXT),
    "replace": ([("text", TEXT), ("old", TEXT), ("new", TEXT)], TEXT),
    "contains": ([("haystack", ANY), ("needle", ANY)], BOOL),
    "starts_with": ([("text", TEXT), ("prefix", TEXT)], BOOL),
    "ends_with": ([("text", TEXT), ("suffix", TEXT)], BOOL),
    "range": ([("n", INT)], LIST),
    "range_from": ([("start", INT), ("stop", INT), ("step", INT)], LIST, 1),
    "first": ([("items", LIST)], ANY),
    "last": ([("items", LIST)], ANY),
    "push": ([("items", LIST), ("value", ANY)], LIST),
    "concat": ([("a", LIST), ("b", LIST)], LIST),
    "slice": ([("items", ANY), ("start", INT), ("stop", INT)], ANY),
    "reverse": ([("items", ANY)], ANY),
    "sort": ([("items", LIST)], LIST),
    "sort_by": ([("items", LIST), ("key", FUNCTION)], LIST),
    "map": ([("items", LIST), ("fn", FUNCTION)], LIST),
    "filter": ([("items", LIST), ("fn", FUNCTION)], LIST),
    "reduce": ([("items", LIST), ("fn", FUNCTION), ("initial", ANY)], ANY),
    "any": ([("items", LIST), ("fn", FUNCTION)], BOOL),
    "all": ([("items", LIST), ("fn", FUNCTION)], BOOL),
    "find": ([("items", LIST), ("fn", FUNCTION)], ANY),
    "count": ([("items", LIST), ("fn", FUNCTION)], INT),
    "unique": ([("items", LIST)], LIST),
    "zip": ([("a", LIST), ("b", LIST)], LIST),
    "enumerate": ([("items", LIST)], LIST),
    "flatten": ([("items", LIST)], LIST),
    "keys": ([("m", MAP)], LIST),
    "values": ([("m", MAP)], LIST),
    "entries": ([("m", MAP)], LIST),
    "has": ([("m", ANY), ("key", ANY)], BOOL),
    "get": ([("m", ANY), ("key", ANY), ("default", ANY)], ANY, 1),
    "set": ([("m", MAP), ("key", ANY), ("value", ANY)], MAP),
    "remove": ([("m", MAP), ("key", ANY)], MAP),
    "merge": ([("a", MAP), ("b", MAP)], MAP),
    "json_encode": ([("value", ANY), ("indent", INT)], TEXT, 1),
    "json_decode": ([("text", TEXT)], ANY),
    "assert": ([("cond", ANY), ("message", ANY)], VOID, 1),
    "now": ([], REAL),
    "read_file": ([("path", TEXT)], TEXT),
    "write_file": ([("path", TEXT), ("content", TEXT)], VOID),
    "append_file": ([("path", TEXT), ("content", TEXT)], VOID),
    "pad_left": ([("text", TEXT), ("width", INT)], TEXT),
    "env": ([("name", TEXT)], ANY),
    "args": ([], LIST),
    "input": ([("prompt", ANY)], TEXT, 1),
    "format": ([("template", TEXT), ("values", LIST)], TEXT),
    "pad": ([("text", TEXT), ("width", INT)], TEXT),
    "repeat_text": ([("text", TEXT), ("times", INT)], TEXT),
    "index_of": ([("items", ANY), ("value", ANY)], INT),
    "chars": ([("text", TEXT)], LIST),
    "code_of": ([("text", TEXT)], INT),
    "text_of": ([("code", INT)], TEXT),
    "is_nothing": ([("value", ANY)], BOOL),
    "random": ([], REAL),
    "random_int": ([("low", INT), ("high", INT)], INT),
    "sleep": ([("seconds", ANY)], VOID),
    "hash_text": ([("text", TEXT)], TEXT),
    "uuid": ([], TEXT),
    "spawn": ([("fn", FUNCTION), ("arg", ANY)], ANY, 1),
    "await_all": ([("tasks", LIST)], LIST),
    "http_get": ([("url", TEXT), ("headers", MAP)], MAP, 1),
    "http_post": ([("url", TEXT), ("body", ANY), ("headers", MAP)], MAP, 2),
}


class Scope:
    __slots__ = ("vars", "parent", "kind")

    def __init__(self, parent=None, kind="block"):
        self.vars = {}
        self.parent = parent
        self.kind = kind

    def declare(self, name, t, mutable):
        self.vars[name] = (t, mutable)

    def lookup(self, name):
        s = self
        while s:
            if name in s.vars:
                return s.vars[name]
            s = s.parent
        return None

    def local(self, name):
        return name in self.vars


class TypeChecker:
    def __init__(self, module_resolver=None):
        self.diagnostics = []
        self.global_scope = Scope(None, "global")
        self.scope = self.global_scope
        self.functions = {}
        self.records = {}
        self.modules = {}
        self.module_resolver = module_resolver
        self.return_stack = []
        self.loop_depth = 0
        for name, spec in BUILTIN_SIGS.items():
            params, ret = spec[0], spec[1]
            optional = spec[2] if len(spec) > 2 else 0
            self.functions[name] = FnSig(params, ret, name, optional)
            self.global_scope.declare(name, FUNCTION, False)

    # ------------------------------------------------------------ diagnostics
    def error(self, msg, node=None):
        line = getattr(node, "line", 0) or 0
        col = getattr(node, "col", 0) or 0
        self.diagnostics.append((line, col, msg))

    def compatible(self, expected: Ty, actual: Ty) -> bool:
        if expected.name in ("Any", "Void") or actual.name in ("Any", "Void"):
            return True
        if actual.name == "Nothing":
            return True  # nothing inhabits every type
        if expected == actual:
            return True
        if expected.name == "Real" and actual.name in ("Int", "Byte"):
            return True
        if expected.name == "Int" and actual.name == "Byte":
            return True
        if expected.name == actual.name:
            return True
        return False

    # ----------------------------------------------------------------- driver
    def check(self, program: A.Program):
        self.hoist(program.statements, self.global_scope)
        for s in program.statements:
            self.stmt(s)
        if self.diagnostics:
            self.diagnostics.sort(key=lambda d: (d[0], d[1]))
            line, col, msg = self.diagnostics[0]
            detail = "\n".join(f"  line {l}:{c}: {m}" for l, c, m in self.diagnostics)
            raise CheckError(
                f"{msg}" if len(self.diagnostics) == 1 else f"{len(self.diagnostics)} type errors:\n{detail}",
                line,
                col,
                self.diagnostics,
            )
        return True

    def hoist(self, statements, scope):
        """Pre-declare functions and records so order of definition is free."""
        for s in statements:
            if isinstance(s, A.Fn):
                if s.name in self.functions and s.name not in BUILTIN_SIGS:
                    self.error(f"duplicate function '{s.name}'", s)
                params = [(n, ty(t)) for n, t in s.params]
                self.functions[s.name] = FnSig(params, ty(s.return_type or "Void"), s.name)
                scope.declare(s.name, FUNCTION, False)
            elif isinstance(s, A.Record):
                if s.name in self.records:
                    self.error(f"duplicate record '{s.name}'", s)
                self.records[s.name] = dict(s.fields)
                scope.declare(s.name, Ty("RecordType"), False)
                self.functions[s.name] = FnSig(
                    [(f, ty(t)) for f, t in s.fields], Ty(s.name), s.name
                )
            elif isinstance(s, A.Use):
                scope.declare(s.alias, Ty("Module"), False)
                if self.module_resolver:
                    try:
                        exports = self.module_resolver(s.path)
                        self.modules[s.alias] = exports
                    except Exception as e:
                        self.error(f"cannot resolve module '{s.path}': {e}", s)

    def push(self, kind="block"):
        self.scope = Scope(self.scope, kind)

    def pop(self):
        self.scope = self.scope.parent

    # ------------------------------------------------------------- statements
    def stmt(self, s):
        if isinstance(s, (A.Let, A.Var)):
            t = self.expr(s.expr)
            if s.declared_type:
                declared = ty(s.declared_type)
                if not self.compatible(declared, t):
                    self.error(
                        f"cannot bind {t} to '{s.name}' declared as {declared}", s
                    )
                t = declared
            # user bindings may shadow builtins; only real redeclaration is an error
            if self.scope.local(s.name) and not (
                self.scope is self.global_scope and s.name in BUILTIN_SIGS
            ):
                self.error(f"'{s.name}' is already defined in this scope", s)
            if s.name in BUILTIN_SIGS:
                self.functions.pop(s.name, None)
            self.scope.declare(s.name, t, isinstance(s, A.Var))
            return

        if isinstance(s, A.Assign):
            target = s.target
            value_t = self.expr(s.expr)
            if isinstance(target, A.Name):
                found = self.scope.lookup(target.value)
                if not found:
                    self.error(f"undefined name '{target.value}'", s)
                    return
                t, mutable = found
                if not mutable:
                    self.error(
                        f"cannot assign to '{target.value}': it is a let binding "
                        f"(declare it with 'var' to allow mutation)",
                        s,
                    )
                if not self.compatible(t, value_t):
                    self.error(f"cannot assign {value_t} to '{target.value}' of type {t}", s)
            else:
                self.expr(target)
            return

        if isinstance(s, (A.Emit, A.ExprStmt)):
            self.expr(s.expr)
            return

        if isinstance(s, A.Raise):
            self.expr(s.expr)
            return

        if isinstance(s, A.Give):
            if not self.return_stack:
                self.error("'give' is only valid inside a function", s)
                return
            t = self.expr(s.expr)
            expected = self.return_stack[-1]
            if not self.compatible(expected, t):
                self.error(f"function returns {expected} but this gives {t}", s)
            return

        if isinstance(s, A.Use):
            return

        if isinstance(s, A.Record):
            return

        if isinstance(s, A.Fn):
            self.check_function(s.name, s.params, s.return_type, s.body, s)
            return

        if isinstance(s, A.When):
            for br in s.branches:
                self.expr(br.cond)
                self.push()
                for x in br.body:
                    self.stmt(x)
                self.pop()
            if s.else_body is not None:
                self.push()
                for x in s.else_body:
                    self.stmt(x)
                self.pop()
            return

        if isinstance(s, A.While):
            self.expr(s.cond)
            self.loop_depth += 1
            self.push("loop")
            for x in s.body:
                self.stmt(x)
            self.pop()
            self.loop_depth -= 1
            return

        if isinstance(s, A.Repeat):
            it = self.expr(s.iterable)
            if it.name not in ("List", "Map", "Text", "Any", "Void"):
                self.error(f"'repeat' needs a List, Map or Text but got {it}", s)
            elem = it.elem if (it.name == "List" and it.elem) else ANY
            self.loop_depth += 1
            self.push("loop")
            self.scope.declare(s.name, elem, False)
            if s.index_name:
                self.scope.declare(s.index_name, INT, False)
            for x in s.body:
                self.stmt(x)
            self.pop()
            self.loop_depth -= 1
            return

        if isinstance(s, (A.Stop, A.Next)):
            if self.loop_depth == 0:
                word = "stop" if isinstance(s, A.Stop) else "next"
                self.error(f"'{word}' is only valid inside a loop", s)
            return

        if isinstance(s, A.Attempt):
            self.push()
            for x in s.body:
                self.stmt(x)
            self.pop()
            self.push()
            self.scope.declare(s.error_name, ANY, False)
            for x in s.rescue_body:
                self.stmt(x)
            self.pop()
            return

        self.error(f"unsupported statement {type(s).__name__}", s)

    def check_function(self, name, params, return_type, body, node):
        ret = ty(return_type or "Void")
        self.return_stack.append(ret)
        saved_loop = self.loop_depth
        self.loop_depth = 0
        self.push("function")
        for pname, ptype in params:
            self.scope.declare(pname, ty(ptype), False)
        # nested declarations are visible within the function body
        self.hoist(body, self.scope)
        for x in body:
            self.stmt(x)
        self.pop()
        self.loop_depth = saved_loop
        self.return_stack.pop()
        if return_type and return_type != "Void" and not self.always_returns(body):
            self.error(
                f"function '{name}' declares -> {return_type} but can finish without 'give'",
                node,
            )

    def always_returns(self, body) -> bool:
        for s in body:
            if isinstance(s, A.Give):
                return True
            if isinstance(s, A.Raise):
                return True
            if isinstance(s, A.When) and s.else_body is not None:
                if all(self.always_returns(b.body) for b in s.branches) and self.always_returns(
                    s.else_body
                ):
                    return True
            if isinstance(s, A.Attempt):
                if self.always_returns(s.body) and self.always_returns(s.rescue_body):
                    return True
            if isinstance(s, A.While):
                cond = s.cond
                if isinstance(cond, A.Literal) and cond.value is True and self.always_returns(s.body):
                    return True
        return False

    # ------------------------------------------------------------ expressions
    def expr(self, n) -> Ty:
        if isinstance(n, A.Literal):
            v = n.value
            if v is None:
                return NOTHING
            if isinstance(v, bool):
                return BOOL
            if isinstance(v, int):
                return INT
            if isinstance(v, float):
                return REAL
            if isinstance(v, str):
                return TEXT
            return ANY

        if isinstance(n, A.Name):
            found = self.scope.lookup(n.value)
            if found:
                return found[0]
            if n.value in self.functions:
                return FUNCTION
            self.error(f"undefined name '{n.value}'", n)
            return ANY

        if isinstance(n, A.ListExpr):
            ts = [self.expr(x) for x in n.items]
            if not ts:
                return Ty("List", ANY)
            head = ts[0]
            if all(self.compatible(head, t) and self.compatible(t, head) for t in ts[1:]):
                return Ty("List", head)
            return Ty("List", ANY)

        if isinstance(n, A.MapExpr):
            vts = []
            for k, v in n.items:
                self.expr(k)
                vts.append(self.expr(v))
            if vts and all(vts[0] == t for t in vts[1:]):
                return Ty("Map", vts[0])
            return Ty("Map", ANY)

        if isinstance(n, A.Convert):
            self.expr(n.expr)
            return ty(n.type_name)

        if isinstance(n, A.Unary):
            t = self.expr(n.expr)
            if n.op == "not":
                return BOOL
            if t.name not in NUMERIC and t.name != "Any":
                self.error(f"unary '-' needs a number but got {t}", n)
            return t

        if isinstance(n, A.Binary):
            return self.binary(n)

        if isinstance(n, A.Index):
            obj = self.expr(n.obj)
            idx = self.expr(n.index)
            if obj.name == "List":
                if idx.name not in ("Int", "Byte", "Any"):
                    self.error(f"list index must be Int but got {idx}", n)
                return obj.elem or ANY
            if obj.name == "Map":
                return obj.elem or ANY
            if obj.name == "Text":
                return TEXT
            if obj.name in ("Any", "Void"):
                return ANY
            self.error(f"cannot index a value of type {obj}", n)
            return ANY

        if isinstance(n, A.Field):
            obj = self.expr(n.obj)
            if obj.name in self.records:
                fields = self.records[obj.name]
                if n.name not in fields:
                    self.error(f"record {obj.name} has no field '{n.name}'", n)
                    return ANY
                return ty(fields[n.name])
            return ANY

        if isinstance(n, A.FnExpr):
            self.check_function("<lambda>", n.params, n.return_type, n.body, n)
            return FUNCTION

        if isinstance(n, A.Call):
            return self.call(n)

        self.error(f"unsupported expression {type(n).__name__}", n)
        return ANY

    def binary(self, n) -> Ty:
        op = n.op
        a = self.expr(n.left)
        b = self.expr(n.right)

        if op in ("and", "or"):
            return BOOL
        if op == "??":
            return b if a.name == "Nothing" else a
        if op in ("==", "!="):
            return BOOL
        if op in ("<", "<=", ">", ">="):
            ok = {a.name, b.name} <= NUMERIC | {"Any", "Void"} or (
                a.name == "Text" and b.name == "Text"
            )
            if not ok:
                self.error(f"'{op}' needs comparable operands but got {a} and {b}", n)
            return BOOL
        if op == "+":
            if a.name == "Text" or b.name == "Text":
                if a.name in ("Text", "Any", "Void") and b.name in ("Text", "Any", "Void"):
                    return TEXT
                self.error(
                    f"cannot add {a} and {b}; convert with to Text(...) first", n
                )
                return TEXT
            if a.name == "List" and b.name == "List":
                return Ty("List", a.elem if a.elem == b.elem else ANY)
            if a.name == "Map" and b.name == "Map":
                return MAP
        if op in ("+", "-", "*", "/", "%"):
            if a.name in ("Any", "Void") or b.name in ("Any", "Void"):
                return ANY
            if a.name not in NUMERIC or b.name not in NUMERIC:
                self.error(f"'{op}' needs numbers but got {a} and {b}", n)
                return ANY
            if op == "/":
                return REAL
            return REAL if "Real" in (a.name, b.name) else INT
        self.error(f"unknown operator '{op}'", n)
        return ANY

    def call(self, n) -> Ty:
        # module member call: mod.fn(...)
        fn = n.fn
        arg_types = []
        for name, expr in n.args:
            arg_types.append((name, self.expr(expr)))

        target_name = None
        if isinstance(fn, A.Name):
            target_name = fn.value
        elif isinstance(fn, A.Field) and isinstance(fn.obj, A.Name):
            mod = self.modules.get(fn.obj.value)
            if mod is not None:
                if fn.name not in mod:
                    self.error(f"module '{fn.obj.value}' has no member '{fn.name}'", n)
                    return ANY
                sig = mod[fn.name]
                if isinstance(sig, FnSig):
                    self.check_arity(sig, arg_types, n)
                    return sig.ret
                return ANY
            self.expr(fn)
            return ANY

        if target_name and target_name in self.records:
            sig = self.functions[target_name]
            self.check_arity(sig, arg_types, n, is_record=True)
            return Ty(target_name)

        if target_name and target_name in self.functions:
            local = self.scope.lookup(target_name)
            # a local binding shadows a global function of the same name
            if not (local and local[0].name != "Function"):
                sig = self.functions[target_name]
                self.check_arity(sig, arg_types, n)
                return sig.ret

        ft = self.expr(fn)
        if ft.name not in ("Function", "Any", "Void", "RecordType"):
            self.error(f"value of type {ft} is not callable", n)
        return ANY

    def check_arity(self, sig: FnSig, arg_types, node, is_record=False):
        positional = [t for name, t in arg_types if name is None]
        named = {name: t for name, t in arg_types if name is not None}
        total = len(positional) + len(named)
        low, high = sig.required, len(sig.params)
        if not (low <= total <= high):
            what = "record" if is_record else "function"
            expect = str(high) if low == high else f"{low} to {high}"
            self.error(
                f"{what} '{sig.name}' expects {expect} argument(s) but got {total}",
                node,
            )
            return
        param_names = [p[0] for p in sig.params]
        for key in named:
            if key not in param_names:
                self.error(f"'{sig.name}' has no parameter named '{key}'", node)
                return
        for (pname, ptype), at in zip(sig.params, positional):
            if not self.compatible(ptype, at):
                self.error(
                    f"argument '{pname}' of '{sig.name}' expects {ptype} but got {at}", node
                )
        types_by_name = dict(sig.params)
        for key, at in named.items():
            ptype = types_by_name.get(key, ANY)
            if not self.compatible(ptype, at):
                self.error(f"argument '{key}' of '{sig.name}' expects {ptype} but got {at}", node)


def check(program: A.Program, module_resolver=None):
    return TypeChecker(module_resolver).check(program)
