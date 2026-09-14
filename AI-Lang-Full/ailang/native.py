"""Native backend: compile AI-Lang functions to host-machine bytecode.

The bytecode VM walks an instruction list and resolves every name through a
scope chain. That indirection is the interpreter's speed ceiling. This backend
removes it for functions it can prove are safe: their parameters and locals
become real machine-level slots, and control flow becomes native branches.

Design rules, in order of importance:

1.  **Semantics never change.** Every operation whose AI-Lang meaning differs
    from the host's is emitted as a guarded expression: a fast path that is
    provably identical, and the interpreter's own operator as the fallback.
    `type(x) is int` is used rather than `isinstance`, because AI-Lang's `Bool`
    is not an `Int` and `isinstance(True, int)` would wrongly say it is.
2.  **Refuse rather than guess.** `_Supported` walks a function first and
    rejects anything this backend does not model exactly -- closures, modules,
    records, keyword arguments. A rejected function keeps running on the VM,
    so the two backends are always interchangeable.
3.  **Observable behaviour is identical**, including error messages: the
    fallback path raises the same `VMError` text the VM would.

`AILANG_NATIVE=0` disables the backend entirely, which makes differential
testing against the VM a one-line change.
"""

from __future__ import annotations

import os
import sys

from . import ast_nodes as A
from .errors import AILangError, VMError
from .values import display, is_truthy, type_name

# Statements and expressions this backend models exactly. Anything else keeps
# running on the bytecode VM.
_OK_STMT = (
    A.Let, A.Var, A.Assign, A.Emit, A.Give, A.When, A.Repeat, A.While,
    A.Stop, A.Next, A.ExprStmt, A.Raise, A.Attempt,
)
_OK_EXPR = (
    A.Literal, A.Name, A.Binary, A.Unary, A.Call, A.Index, A.Field,
    A.ListExpr, A.MapExpr, A.Convert,
)


class _Unsupported(Exception):
    """Raised while screening a function that this backend will not compile."""


def _screen_expr(e):
    if not isinstance(e, _OK_EXPR):
        raise _Unsupported(type(e).__name__)
    if isinstance(e, A.Call):
        # named arguments change binding order; leave those to the VM
        for name, arg in e.args:
            if name is not None:
                raise _Unsupported("named argument")
            _screen_expr(arg)
        _screen_expr(e.fn)
    elif isinstance(e, A.Binary):
        _screen_expr(e.left)
        _screen_expr(e.right)
    elif isinstance(e, A.Unary):
        _screen_expr(e.expr)
    elif isinstance(e, A.Index):
        _screen_expr(e.obj)
        _screen_expr(e.index)
    elif isinstance(e, A.Field):
        _screen_expr(e.obj)
    elif isinstance(e, A.ListExpr):
        for x in e.items:
            _screen_expr(x)
    elif isinstance(e, A.MapExpr):
        for k, v in e.items:
            _screen_expr(k)
            _screen_expr(v)
    elif isinstance(e, A.Convert):
        _screen_expr(e.expr)


def _screen_body(body):
    for s in body:
        if not isinstance(s, _OK_STMT):
            raise _Unsupported(type(s).__name__)
        if isinstance(s, (A.Let, A.Var)):
            _screen_expr(s.expr)
        elif isinstance(s, A.Assign):
            if not isinstance(s.target, (A.Name, A.Index, A.Field)):
                raise _Unsupported("assignment target")
            _screen_expr(s.target)
            _screen_expr(s.expr)
        elif isinstance(s, (A.Emit, A.Give, A.ExprStmt, A.Raise)):
            _screen_expr(s.expr)
        elif isinstance(s, A.When):
            for br in s.branches:
                _screen_expr(br.cond)
                _screen_body(br.body)
            if s.else_body:
                _screen_body(s.else_body)
        elif isinstance(s, A.Repeat):
            _screen_expr(s.iterable)
            _screen_body(s.body)
        elif isinstance(s, A.While):
            _screen_expr(s.cond)
            _screen_body(s.body)
        elif isinstance(s, A.Attempt):
            _screen_body(s.body)
            _screen_body(s.rescue_body)


def _declared_and_assigned(body, declared, assigned):
    """Split the names a body binds into declarations and bare assignments."""
    for s in body:
        k = s.__class__.__name__
        if k in ("Let", "Var"):
            declared.add(s.name)
        elif k == "Assign":
            t = s.target
            if isinstance(t, A.Name):
                assigned.add(t.value)
        elif k == "When":
            for br in s.branches:
                _declared_and_assigned(br.body, declared, assigned)
            if s.else_body:
                _declared_and_assigned(s.else_body, declared, assigned)
        elif k in ("Repeat", "While"):
            if k == "Repeat":
                declared.add(s.name)
                if s.index_name:
                    declared.add(s.index_name)
            _declared_and_assigned(s.body, declared, assigned)
        elif k == "Attempt":
            declared.add(s.error_name)
            _declared_and_assigned(s.body, declared, assigned)
            _declared_and_assigned(s.rescue_body, declared, assigned)


class _Gen:
    """Emit host source for one AI-Lang function."""

    def __init__(self, fn):
        self.fn = fn
        self.lines = []
        self.depth = 1
        self.tmp = 0
        # every name bound in this function becomes a real local slot
        self.locals = set(p[0] if isinstance(p, tuple) else p for p in fn.params)
        # names resolved dynamically, i.e. not locals -- must all be globals
        self.free = set()

    def w(self, text, node=None):
        if getattr(self, "track", False):
            src_line = getattr(node, "line", 0) or getattr(self, "_cur_line", 0)
            if src_line:
                self.linemap[len(self.lines) + 1] = src_line
        self.lines.append("    " * self.depth + text)

    def fresh(self, prefix="_t"):
        self.tmp += 1
        return f"{prefix}{self.tmp}"

    # ------------------------------------------------------------ expressions
    def expr(self, e):
        if isinstance(e, A.Literal):
            return repr(e.value)

        if isinstance(e, A.Name):
            if e.value in self.locals:
                return e.value
            if e.value == "true":
                return "True"
            if e.value == "false":
                return "False"
            if e.value == "nothing":
                return "None"
            # a free name resolves through the global scope
            self.free.add(e.value)
            return f"_lk({e.value!r})"

        if isinstance(e, A.Unary):
            v = self.expr(e.expr)
            if e.op == "not":
                return f"(({v}) is None or ({v}) is False)"
            return f"_un({e.op!r}, {v})"

        if isinstance(e, A.Binary):
            return self.binary(e)

        if isinstance(e, A.Call):
            fn = self.expr(e.fn)
            args = ", ".join(self.expr(a) for _, a in e.args)
            return f"{fn}({args})"

        if isinstance(e, A.Index):
            return f"_ix({self.expr(e.obj)}, {self.expr(e.index)})"

        if isinstance(e, A.Field):
            return f"_fd({self.expr(e.obj)}, {e.name!r})"

        if isinstance(e, A.ListExpr):
            return "[" + ", ".join(self.expr(x) for x in e.items) + "]"

        if isinstance(e, A.MapExpr):
            pairs = ", ".join(
                f"_hk({self.expr(k)}): {self.expr(v)}" for k, v in e.items
            )
            return "{" + pairs + "}"

        if isinstance(e, A.Convert):
            return f"_cv({e.target!r}, {self.expr(e.expr)})"

        raise _Unsupported(type(e).__name__)

    def binary(self, e):
        op = e.op
        # `and`/`or` must short-circuit and must yield AI-Lang truthiness
        if op == "and":
            return f"(_tr({self.expr(e.left)}) and _tr({self.expr(e.right)}))"
        if op == "or":
            return f"(_tr({self.expr(e.left)}) or _tr({self.expr(e.right)}))"
        if op == "??":
            a = self.expr(e.left)
            return f"(_nz if (_nz := {a}) is not None else {self.expr(e.right)})"

        a, b = self.expr(e.left), self.expr(e.right)
        ta, tb = self.fresh("_a"), self.fresh("_b")

        # Both operands are bound BEFORE the type test. Writing the walrus
        # assignments inside an `and` would short-circuit and leave the second
        # temporary unbound whenever the first operand is not an Int -- the
        # fallback call would then raise UnboundLocalError instead of the
        # interpreter's real error message.
        def guarded(test, fast, slow):
            return f"(({fast}) if ({test}) else ({slow}))"

        bind = f"(({ta} := {a}), ({tb} := {b}))"
        both_int = f"{bind} and type({ta}) is int and type({tb}) is int"

        # `type(x) is int` deliberately excludes Bool: AI-Lang does not treat
        # true as 1, and isinstance would.
        if op in ("+", "-", "*", "<", "<=", ">", ">="):
            return guarded(both_int, f"{ta} {op} {tb}", f"_bin({op!r}, {ta}, {tb})")
        if op == "==":
            return guarded(both_int, f"{ta} == {tb}", f"_eq({ta}, {tb})")
        if op == "!=":
            return guarded(both_int, f"{ta} != {tb}", f"not _eq({ta}, {tb})")
        # `/` always yields Real and must reject a zero divisor with the
        # interpreter's message, so only a non-zero Int pair takes the fast path
        if op == "/":
            return guarded(
                f"{both_int} and {tb} != 0", f"{ta} / {tb}", f"_bin('/', {ta}, {tb})"
            )
        return f"_bin({op!r}, {a}, {b})"

    # ------------------------------------------------------------- statements
    def body(self, stmts):
        if not stmts:
            self.w("pass")
            return
        for s in stmts:
            self.stmt(s)

    def stmt(self, s):
        line = getattr(s, "line", 0)
        if line:
            self._cur_line = line
        if isinstance(s, (A.Let, A.Var)):
            self.locals.add(s.name)
            self.w(f"{s.name} = {self.expr(s.expr)}")
        elif isinstance(s, A.While):
            # a loop compiled to host bytecode would otherwise run forever
            # without touching the interpreter's fuel meter; the per-iteration
            # check gives it exactly the same "execution limit exceeded"
            # behaviour a bytecode loop has
            self.w(f"while _tr({self.expr(s.cond)}):")
            self.depth += 1
            self.w("_fc()")
            self.body(s.body)
            self.depth -= 1

        elif isinstance(s, A.Assign):
            t = s.target
            if isinstance(t, A.Name):
                self.locals.add(t.value)
                self.w(f"{t.value} = {self.expr(s.expr)}")
            elif isinstance(t, A.Index):
                self.w(
                    f"_setix({self.expr(t.obj)}, {self.expr(t.index)}, "
                    f"{self.expr(s.expr)})"
                )
            else:
                self.w(
                    f"_setfd({self.expr(t.obj)}, {t.name!r}, {self.expr(s.expr)})"
                )

        elif isinstance(s, A.Emit):
            self.w(f"print(_disp({self.expr(s.expr)}))")

        elif isinstance(s, A.Give):
            self.w(f"return {self.expr(s.expr)}")

        elif isinstance(s, A.ExprStmt):
            self.w(self.expr(s.expr))

        elif isinstance(s, A.Raise):
            self.w(f"_raise({self.expr(s.expr)})")

        elif isinstance(s, A.When):
            kw = "if"
            for br in s.branches:
                self.w(f"{kw} _tr({self.expr(br.cond)}):")
                self.depth += 1
                self.body(br.body)
                self.depth -= 1
                kw = "elif"
            if s.else_body:
                self.w("else:")
                self.depth += 1
                self.body(s.else_body)
                self.depth -= 1

        elif isinstance(s, A.Repeat):
            self.locals.add(s.name)
            it = s.iterable
            # `_fcit` wraps the iterable so each iteration charges the shared
            # fuel meter, matching the interpreter's per-instruction cost.
            # `repeat i in range(n)` maps onto a native counted loop
            if (
                not s.index_name
                and isinstance(it, A.Call)
                and isinstance(it.fn, A.Name)
                and it.fn.value == "range"
                and len(it.args) == 1
                and it.args[0][0] is None
            ):
                self.w(
                    f"for {s.name} in _fcit(range(_rng({self.expr(it.args[0][1])}))):"
                )
            elif s.index_name:
                self.locals.add(s.index_name)
                self.w(
                    f"for {s.index_name}, {s.name} in "
                    f"_fcit(enumerate(_iter({self.expr(it)}))):"
                )
            else:
                self.w(f"for {s.name} in _fcit(_iter({self.expr(it)})):")
            self.depth += 1
            self.body(s.body)
            self.depth -= 1

        elif isinstance(s, A.Stop):
            self.w("break")

        elif isinstance(s, A.Next):
            self.w("continue")

        elif isinstance(s, A.Attempt):
            self.w("try:")
            self.depth += 1
            self.body(s.body)
            self.depth -= 1
            self.w("except (KeyboardInterrupt, SystemExit):")
            self.depth += 1
            self.w("raise")
            self.depth -= 1
            self.w("except BaseException as _exc:")
            self.depth += 1
            self.locals.add(s.error_name)
            self.w(f"{s.error_name} = _errval(_exc)")
            self.body(s.rescue_body)
            self.depth -= 1

        else:
            raise _Unsupported(type(s).__name__)

    def generate(self):
        params = [p[0] if isinstance(p, tuple) else p for p in self.fn.params]
        header = f"def _fn({', '.join(params)}):"
        self.body(self.fn.body)
        self.w("return None")
        return header + "\n" + "\n".join(self.lines)

    def generate_located(self):
        """Source plus a map from host line number to AI-Lang line number.

        Errors raised inside compiled code are reported at the AI-Lang
        statement that caused them, matching the interpreter exactly.
        """
        params = [p[0] if isinstance(p, tuple) else p for p in self.fn.params]
        self.track = True
        self.lines = []
        self.linemap = {}
        self.body(self.fn.body)
        self.w("return None")
        # +1 because the header occupies host line 1
        return (
            f"def _fn({', '.join(params)}):\n" + "\n".join(self.lines),
            {h + 1: a for h, a in self.linemap.items()},
        )


# ------------------------------------------------------------------- runtime
def _make_runtime(lookup, fuel_box=None):
    """Helpers the generated code calls. Each mirrors the VM exactly."""
    from .errors import AILangRaise, VMError
    from .values import RecordValue
    from .vm import _binary, _equal, _field, _hashable, _index
    from .vm import _convert, _error_value

    def _fuel_check(_box=fuel_box, _count=[256]):
        # Charge one unit of the shared budget per iteration. The box is
        # decremented every time and only tested every 256 charges, so a
        # simple loop pays a few list operations, not a comparison storm.
        if _box is None:
            return
        _box[0] -= 1
        _count[0] -= 1
        if not _count[0]:
            _count[0] = 256
            if _box[0] <= 0:
                raise VMError("execution limit exceeded")

    def _fuel_iter(it):
        for item in it:
            _fuel_check()
            yield item

    def _unary(op, v):
        if op == "not":
            return not is_truthy(v)
        if v.__class__ is bool or not isinstance(v, (int, float)):
            raise VMError(f"cannot negate {type_name(v)}")
        return -v

    def _set_index(obj, key, value):
        if obj.__class__ is list:
            if key.__class__ is not int:
                raise VMError("list index must be an Int")
            if not -len(obj) <= key < len(obj):
                raise VMError(f"index {key} out of range for list of {len(obj)}")
            obj[key] = value
        elif obj.__class__ is dict:
            obj[_hashable(key)] = value
        else:
            raise VMError(f"cannot index-assign into {type_name(obj)}")
        return None

    def _set_field(obj, name, value):
        if obj.__class__ is RecordValue:
            obj.set(name, value)
        elif obj.__class__ is dict:
            obj[name] = value
        else:
            raise VMError(f"cannot set field on {type_name(obj)}")
        return None

    def _raise(value):
        raise AILangRaise(value)

    def _rng(n):
        if type(n) is not int:
            raise VMError("range loop needs an Int bound")
        return n

    def _iterate(v):
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            return list(v)
        if isinstance(v, dict):
            return list(v.keys())
        raise VMError(f"cannot iterate over {type_name(v)}")

    return {
        "_lk": lookup,
        "_bin": _binary,
        "_eq": _equal,
        "_un": _unary,
        "_ix": _index,
        "_fd": _field,
        "_hk": _hashable,
        "_cv": _convert,
        "_setix": _set_index,
        "_setfd": _set_field,
        "_tr": is_truthy,
        "_disp": display,
        "_iter": _iterate,
        "_rng": _rng,
        "_raise": _raise,
        "_errval": _error_value,
        "_fc": _fuel_check,
        "_fcit": _fuel_iter,
    }


def enabled():
    return os.environ.get("AILANG_NATIVE", "1") != "0"


def try_compile(fn_node, lookup, name="<fn>", is_global=None, fuel_box=None):
    """Compile one AI-Lang function to a host function, or return None.

    `lookup` resolves a free name against the global scope. `is_global` says
    whether a name exists there; any free name that does not is a capture from
    an enclosing scope, which a cached host function cannot model, so the
    function is left to the interpreter. `fuel_box` is the VM's shared fuel
    budget; the generated loops charge it so a runaway loop is bounded exactly
    as it would be on the interpreter.
    """
    if not enabled():
        return None
    try:
        _screen_body(fn_node.body)
        # A function that assigns to a name it never declares is mutating an
        # enclosing scope (a closure counter, say). Host locals cannot express
        # that, so leave those functions to the interpreter.
        declared, assigned = set(), set()
        _declared_and_assigned(fn_node.body, declared, assigned)
        params = set(
            p[0] if isinstance(p, tuple) else p for p in fn_node.params
        )
        if assigned - declared - params:
            return None
        gen = _Gen(fn_node)
        src, linemap = gen.generate_located()
        # every name the generated code resolves dynamically must be a global
        if is_global is not None:
            for free in gen.free:
                if not is_global(free):
                    return None
    except _Unsupported:
        return None
    except Exception:
        return None

    env = _make_runtime(lookup, fuel_box)
    try:
        code = compile(src, f"<ailang:{name}>", "exec")
        exec(code, env)
    except Exception:
        return None
    raw = env["_fn"]
    # No wrapper: an extra Python frame on every call is measurable on
    # call-heavy code. The line map travels on the function object and the VM
    # applies it once, when an error actually escapes.
    raw._ailang_lines = linemap
    return raw


# ------------------------------------------------------- top-level hot loops
def try_compile_loop(node, read_names, name="<loop>"):
    """Compile a top-level `repeat`/`while` into a host function.

    Top-level code cannot become host locals wholesale: the names it declares
    are globals that functions elsewhere must see. A loop is different -- the
    variables it *reads and writes* can be lifted into slots for the duration
    of the loop and written back when it finishes, provided the loop does not
    declare anything that outlives it.

    Returns `(fn, names)` where calling `fn(*values)` runs the loop and
    returns the final values of `names`, or None if the loop is not eligible.
    """
    if not enabled():
        return None
    try:
        _screen_body([node])
    except _Unsupported:
        return None

    declared, assigned = set(), set()
    _declared_and_assigned([node], declared, assigned)
    # names the loop mutates that live outside it
    outer = sorted(assigned - declared)
    if not outer:
        return None

    gen = _Gen(_FakeFn(outer))
    try:
        gen.stmt(node)
    except (_Unsupported, Exception):
        return None
    # a loop that touches anything non-global besides its own carried
    # variables cannot be lifted
    for free in gen.free:
        if not read_names(free):
            return None

    body = "\n".join(gen.lines)
    src = (
        f"def _fn({', '.join(outer)}):\n"
        + body
        + "\n    return (" + ", ".join(outer) + ",)\n"
    )
    return src, outer, gen.free


class _FakeFn:
    """Minimal stand-in so _Gen can treat carried variables as parameters."""

    def __init__(self, params):
        self.params = list(params)
        self.body = []
