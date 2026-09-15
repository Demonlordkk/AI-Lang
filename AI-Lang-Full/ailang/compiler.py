"""Bytecode compiler for AI-Lang.

Key correctness rules encoded here:
  * `CALL n` / `MAKE_LIST n` always carry an explicit count; the VM never uses
    negative-index slicing (`stack[-0:]` is the whole stack in Python, which
    silently corrupted the operand stack in the previous implementation).
  * Nested functions are hoisted into the enclosing program's function table
    and captured as closures, so inner functions resolve correctly.
  * `and` / `or` compile to real short-circuit jumps rather than eager BINARY.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import ast_nodes as A
from .errors import CompileError
from .opcodes import OPS

globals().update(OPS)


@dataclass
class FunctionCode:
    name: str
    params: List[str]
    code: List[tuple]
    constants: List[Any]
    captures: List[str] = field(default_factory=list)
    defaults: Dict[str, Any] = field(default_factory=dict)
    line: int = 0
    # the source structure, kept so the native backend can compile from it
    body: Any = None
    native: Any = None
    native_tried: bool = False
    lines: Dict[int, int] = field(default_factory=dict)
    # the file this function was written in, so an error raised inside an
    # imported module is reported against the module, not the importer
    origin: str = ""

    def __post_init__(self):
        # shared by every invocation; parameters are always immutable bindings
        self.param_set = frozenset(self.params)


@dataclass
class ProgramCode:
    main: FunctionCode
    functions: Dict[str, FunctionCode]
    records: Dict[str, List[tuple]] = field(default_factory=dict)
    imports: List[tuple] = field(default_factory=list)
    native_loops: List[tuple] = field(default_factory=list)


class _FnScope:
    """Tracks names bound inside a function so we can detect free variables."""

    def __init__(self, parent=None, params=()):
        self.parent = parent
        self.names = set(params)
        self.captures = []

    def declare(self, name):
        self.names.add(name)

    def knows(self, name):
        return name in self.names


class Compiler:
    def __init__(self, lift_loops=True):
        self.functions: Dict[str, FunctionCode] = {}
        self.native_loops: List[tuple] = []
        self.lift_loops = lift_loops
        self.records: Dict[str, List[tuple]] = {}
        self.imports: List[tuple] = []
        self._anon = 0
        self.filename = ""

    # ------------------------------------------------------------------ entry
    def compile(self, program: A.Program, filename: str = "") -> ProgramCode:
        self.filename = filename
        ctx = _Ctx(self, None)
        ctx.hoist(program.statements)
        for s in program.statements:
            ctx.stmt(s)
        ctx.emit("HALT")
        main = FunctionCode(
            "<main>", [], ctx.code, ctx.consts, lines=ctx.lines, origin=filename
        )
        program = ProgramCode(
            main, self.functions, self.records, self.imports, self.native_loops
        )
        from .peephole import optimise_program

        return optimise_program(program)

    def anon_name(self, hint="lambda"):
        self._anon += 1
        return f"<{hint}#{self._anon}>"



def _declares(body):
    """True if a block body introduces bindings needing their own scope.

    Loop bodies get a fresh Environment per iteration so that `let` inside the
    loop is legal on every pass. When a body declares nothing that allocation
    is pure overhead, so we skip emitting the scope entirely. Nested `when`
    branches are inspected too, since they share the enclosing scope; an
    `attempt` always needs a frame because `rescue` binds the error name.
    """
    for st in body:
        k = st.__class__.__name__
        if k in ("Let", "Var", "Record", "Fn", "Use"):
            return True
        if k == "When":
            for br in st.branches:
                if _declares(br.body):
                    return True
            if st.else_body and _declares(st.else_body):
                return True
        elif k == "Attempt":
            return True
        elif k in ("Repeat", "While"):
            if _declares(st.body):
                return True
    return False


class _Ctx:
    """Per-function compilation context."""

    def __init__(self, owner: Compiler, scope: Optional[_FnScope], params=()):
        self.owner = owner
        self.code: List[tuple] = []
        self.consts: List[Any] = []
        self.lines: Dict[int, int] = {}
        self.scope = scope if scope is not None else _FnScope(None, params)
        self.loops: List[dict] = []
        # Number of runtime Environment frames currently active on the normal
        # path.  Control-flow exits use this to unwind branch/loop frames
        # before jumping, without changing the compile-time state used for the
        # fall-through path.
        self.scope_depth = 0
        self._hoisted: Dict[int, bool] = {}
        # 0 for the top-level context; sub-contexts (function bodies) set 1
        self._fn_depth = 0

    # ------------------------------------------------------------- emit utils
    def emit(self, op, *args, line=0):
        if op.__class__ is str:
            op = OPS[op]
        self.code.append((op, *args))
        idx = len(self.code) - 1
        # Keep line numbers beside the code rather than inside instructions:
        # the hot dispatch loop stays untouched, and a runtime error can still
        # be reported at the statement that caused it.
        if line:
            self.lines[idx] = line
        return idx

    def patch(self, pos, target):
        ins = self.code[pos]
        self.code[pos] = (ins[0], target, *ins[2:])

    def here(self):
        return len(self.code)

    def hoist(self, body):
        """Pre-bind every function declared in `body` so order is irrelevant."""
        for s in body:
            if isinstance(s, A.Fn):
                code = self.compile_function(s.name, s.params, s.body, s.line)
                self.owner.functions[code.name] = code
                self.scope.declare(s.name)
                self.emit("CLOSURE", code.name, s.name, line=s.line)
                self._hoisted[id(s)] = True

    # -------------------------------------------------------------- statements
    def stmt(self, s):
        m = getattr(self, f"_s_{type(s).__name__}", None)
        if m is None:
            raise CompileError(
                f"unsupported statement {type(s).__name__}",
                getattr(s, "line", 0),
                getattr(s, "col", 0),
            )
        m(s)

    def _s_Let(self, s):
        self.expr(s.expr)
        self.scope.declare(s.name)
        self.emit("STORE", s.name, False, line=s.line)

    def _s_Var(self, s):
        self.expr(s.expr)
        self.scope.declare(s.name)
        self.emit("STORE", s.name, True, line=s.line)

    def _s_Assign(self, s):
        t = s.target
        if isinstance(t, A.Name):
            # `i <- i + 1` becomes a single INC_FAST opcode
            e = s.expr
            if (
                isinstance(e, A.Binary)
                and e.op in ("+", "-")
                and isinstance(e.left, A.Name)
                and e.left.value == t.value
                and isinstance(e.right, A.Literal)
                and e.right.value.__class__ is int
            ):
                delta = e.right.value if e.op == "+" else -e.right.value
                self.emit("INC_FAST", t.value, delta, line=s.line)
                return
            self.expr(s.expr)
            self.emit("SET", t.value, line=s.line)
        elif isinstance(t, A.Index):
            self.expr(t.obj)
            self.expr(t.index)
            self.expr(s.expr)
            self.emit("SET_INDEX", line=s.line)
        elif isinstance(t, A.Field):
            self.expr(t.obj)
            self.expr(s.expr)
            self.emit("SET_FIELD", t.name, line=s.line)
        else:
            raise CompileError("invalid assignment target", s.line, s.col)

    def _s_Emit(self, s):
        self.expr(s.expr)
        self.emit("PRINT", line=s.line)

    def _s_ExprStmt(self, s):
        self.expr(s.expr)
        self.emit("POP")

    def _s_Give(self, s):
        self.expr(s.expr)
        self.emit("RETURN", line=s.line)

    def _s_Raise(self, s):
        self.expr(s.expr)
        self.emit("RAISE", line=s.line)

    def _s_Use(self, s):
        self.owner.imports.append((s.path, s.alias))
        self.scope.declare(s.alias)
        self.emit("IMPORT", s.path, s.alias, line=s.line)

    def _s_Record(self, s):
        self.owner.records[s.name] = list(s.fields)
        self.scope.declare(s.name)
        self.emit("RECORD", s.name, line=s.line)

    def _s_Fn(self, s):
        if self._hoisted.pop(id(s), False):
            return
        code = self.compile_function(s.name, s.params, s.body, s.line)
        self.owner.functions[code.name] = code
        self.scope.declare(s.name)
        self.emit("CLOSURE", code.name, s.name, line=s.line)


    def _s_When(self, s):
        end_jumps = []
        for br in s.branches:
            self.expr(br.cond)
            skip = self.emit("JUMP_IF_FALSE", None, line=s.line)
            # Branch declarations are block-local in the checker.  Give each
            # selected branch its own runtime frame so a binding cannot leak
            # into a later branch or the enclosing block, and closures keep the
            # correct defining environment.
            self.emit("SCOPE_PUSH")
            self.scope_depth += 1
            for x in br.body:
                self.stmt(x)
            self.emit("SCOPE_POP")
            self.scope_depth -= 1
            end_jumps.append(self.emit("JUMP", None))
            # A false condition skips both the branch body and its frame
            # teardown; no frame was pushed on that path.
            self.patch(skip, self.here())
        if s.else_body is not None:
            self.emit("SCOPE_PUSH")
            self.scope_depth += 1
            for x in s.else_body:
                self.stmt(x)
            self.emit("SCOPE_POP")
            self.scope_depth -= 1
        for j in end_jumps:
            self.patch(j, self.here())

    def _s_While(self, s):
        if self._try_native_loop(s):
            return
        start = self.here()
        self.expr(s.cond)
        exit_jump = self.emit("JUMP_IF_FALSE", None, line=s.line)
        scoped = _declares(s.body)
        loop_scope = self.scope_depth
        self.loops.append({"continue": start, "breaks": [], "scoped": scoped,
                           "scope_depth": loop_scope, "iterated": False})
        if scoped:
            self.emit("SCOPE_PUSH")
            self.scope_depth += 1
        for x in s.body:
            self.stmt(x)
        if scoped:
            self.emit("SCOPE_POP")
            self.scope_depth -= 1
        self.emit("JUMP", start)
        self.patch(exit_jump, self.here())
        frame = self.loops.pop()
        for b in frame["breaks"]:
            self.patch(b, self.here())

    def _s_Repeat(self, s):
        if self._try_native_loop(s):
            return
        it = s.iterable
        # `repeat i in range(n):` iterates lazily instead of materialising a list
        if (
            not s.index_name
            and isinstance(it, A.Call)
            and isinstance(it.fn, A.Name)
            and it.fn.value == "range"
            and len(it.args) == 1
            and it.args[0][0] is None
        ):
            self.expr(it.args[0][1])
            self.emit("RANGE_INIT", line=s.line)
            start = self.here()
            nxt = self.emit("RANGE_NEXT", None, s.name)
            self.scope.declare(s.name)
            scoped = _declares(s.body)
            loop_scope = self.scope_depth
            self.loops.append({"continue": start, "breaks": [], "scoped": scoped,
                               "scope_depth": loop_scope, "iterated": True})
            if scoped:
                self.emit("SCOPE_PUSH")
                self.scope_depth += 1
            for x in s.body:
                self.stmt(x)
            if scoped:
                self.emit("SCOPE_POP")
                self.scope_depth -= 1
            self.emit("JUMP", start)
            self.patch(nxt, self.here())
            frame = self.loops.pop()
            for b in frame["breaks"]:
                self.patch(b, self.here())
            self.emit("ITER_END")
            return
        self.expr(s.iterable)
        self.emit("ITER_INIT", line=s.line)
        start = self.here()
        nxt = self.emit("ITER_NEXT", None, s.name, s.index_name)
        self.scope.declare(s.name)
        if s.index_name:
            self.scope.declare(s.index_name)
        scoped = _declares(s.body)
        loop_scope = self.scope_depth
        self.loops.append({"continue": start, "breaks": [], "scoped": scoped,
                           "scope_depth": loop_scope, "iterated": True})
        if scoped:
            self.emit("SCOPE_PUSH")
            self.scope_depth += 1
        for x in s.body:
            self.stmt(x)
        if scoped:
            self.emit("SCOPE_POP")
            self.scope_depth -= 1
        self.emit("JUMP", start)
        self.patch(nxt, self.here())
        frame = self.loops.pop()
        for b in frame["breaks"]:
            self.patch(b, self.here())
        self.emit("ITER_END")

    @property
    def fn_depth(self):
        """0 while compiling top-level code, >0 inside a function body."""
        return getattr(self, "_fn_depth", 0)

    def _try_native_loop(self, s):
        """Emit NATIVE_LOOP when this loop can run as host bytecode.

        Skipped when `lift_loops` is off (line tracing needs the
        interpreter to see every instruction).

        Only top-level loops qualify: inside a function the whole function is
        already a native-backend candidate. The loop must not contain `stop`
        or `next` targeting an outer construct, and the VM re-checks
        eligibility before using the compiled form.
        """
        if not self.owner.lift_loops:
            return False
        # Safe VM execution is the default.  Native loop source is only
        # embedded in an in-memory ProgramCode when explicitly opted in.
        from .native import enabled as native_enabled
        if not native_enabled():
            return False
        if self.fn_depth > 0:
            return False
        from .native import try_compile_loop
        result = try_compile_loop(s, lambda n: True)
        if result is None:
            return False
        src, carried, free = result
        idx = len(self.owner.native_loops)
        self.owner.native_loops.append((src, carried, sorted(free)))
        self.emit("NATIVE_LOOP", idx, line=getattr(s, "line", 0))
        return True

    def _s_Stop(self, s):
        if not self.loops:
            raise CompileError("'stop' outside of a loop", s.line, s.col)
        # A stop/next may be nested in one or more block frames (for example a
        # `when` inside a loop).  Unwind every frame created since the loop
        # entry; the fall-through compile state is intentionally unchanged.
        unwind = self.scope_depth - self.loops[-1]["scope_depth"]
        for _ in range(max(0, unwind)):
            self.emit("SCOPE_POP")
        opcode = "ITER_BREAK" if self.loops[-1].get("iterated") else "BREAK"
        j = self.emit(opcode, None, line=s.line)
        self.loops[-1]["breaks"].append(j)

    def _s_Next(self, s):
        if not self.loops:
            raise CompileError("'next' outside of a loop", s.line, s.col)
        unwind = self.scope_depth - self.loops[-1]["scope_depth"]
        for _ in range(max(0, unwind)):
            self.emit("SCOPE_POP")
        self.emit("JUMP", self.loops[-1]["continue"], line=s.line)

    def _s_Attempt(self, s):
        # `attempt` is a recoverable control-flow construct, not a lexical
        # block: bindings made in its body remain available to the following
        # code and rescue shares that same scope.  This is useful for parsing
        # into a value and handling a failure immediately afterwards, and it is
        # also the behavior of the language's existing examples.
        setup = self.emit("TRY_PUSH", None, s.error_name, line=s.line)
        for x in s.body:
            self.stmt(x)
        self.emit("TRY_POP")
        done = self.emit("JUMP", None)
        self.patch(setup, self.here())
        self.scope.declare(s.error_name)
        for x in s.rescue_body:
            self.stmt(x)
        self.patch(done, self.here())

    # ------------------------------------------------------------- functions
    def compile_function(self, name, params, body, line=0) -> FunctionCode:
        pnames = [p for p, _ in params]
        inner_scope = _FnScope(self.scope, pnames)
        sub = _Ctx(self.owner, inner_scope)
        sub._fn_depth = 1
        sub.hoist(body)
        for x in body:
            sub.stmt(x)
        sub.emit("PUSH", None)
        sub.emit("RETURN")
        unique = name
        if unique in self.owner.functions:
            unique = self.owner.anon_name(name.strip("<>#") or "fn")
        return FunctionCode(
            unique, pnames, sub.code, sub.consts,
            captures=inner_scope.captures, line=line, body=body, lines=sub.lines,
            origin=self.owner.filename,
        )

    # ------------------------------------------------------------ expressions
    def expr(self, n):
        m = getattr(self, f"_e_{type(n).__name__}", None)
        if m is None:
            raise CompileError(
                f"unsupported expression {type(n).__name__}",
                getattr(n, "line", 0),
                getattr(n, "col", 0),
            )
        m(n)

    def _e_Literal(self, n):
        self.emit("PUSH", n.value)

    def _e_Name(self, n):
        self.emit("LOAD", n.value, line=n.line)

    def _e_ListExpr(self, n):
        for x in n.items:
            self.expr(x)
        self.emit("MAKE_LIST", len(n.items))

    def _e_MapExpr(self, n):
        for k, v in n.items:
            self.expr(k)
            self.expr(v)
        self.emit("MAKE_MAP", len(n.items))

    def _e_Convert(self, n):
        self.expr(n.expr)
        self.emit("CONVERT", n.type_name, line=n.line)

    def _e_Unary(self, n):
        self.expr(n.expr)
        self.emit("UNARY", n.op, line=n.line)

    def _e_Binary(self, n):
        op = n.op
        # real short-circuit evaluation
        if op == "and":
            self.expr(n.left)
            j = self.emit("JUMP_IF_FALSE_KEEP", None)
            self.emit("POP")
            self.expr(n.right)
            self.emit("TO_BOOL")
            end = self.here()
            self.patch(j, end)
            return
        if op == "or":
            self.expr(n.left)
            j = self.emit("JUMP_IF_TRUE_KEEP", None)
            self.emit("POP")
            self.expr(n.right)
            self.emit("TO_BOOL")
            self.patch(j, self.here())
            return
        if op == "??":
            self.expr(n.left)
            j = self.emit("JUMP_IF_SOME", None)
            self.emit("POP")
            self.expr(n.right)
            self.patch(j, self.here())
            return
        self.expr(n.left)
        self.expr(n.right)
        fast = _FAST_BINARY.get(op)
        if fast is not None:
            self.emit(fast, line=n.line)
        else:
            self.emit("BINARY", op, line=n.line)

    def _e_Index(self, n):
        self.expr(n.obj)
        self.expr(n.index)
        self.emit("INDEX", line=n.line)

    def _e_Field(self, n):
        self.expr(n.obj)
        self.emit("FIELD", n.name, line=n.line)

    def _e_FnExpr(self, n):
        code = self.compile_function(self.owner.anon_name("lambda"), n.params, n.body, n.line)
        self.owner.functions[code.name] = code
        self.emit("CLOSURE", code.name, None, line=n.line)

    def _e_Call(self, n):
        self.expr(n.fn)
        names = []
        for argname, expr in n.args:
            self.expr(expr)
            names.append(argname)
        if any(names):
            self.emit("CALL_KW", len(names), tuple(names), line=n.line)
        else:
            self.emit("CALL", len(names), line=n.line)


def compile_program(program: A.Program, filename: str = "") -> ProgramCode:
    return Compiler().compile(program, filename)


# operator -> specialised opcode name
_FAST_BINARY = {
    "+": "ADD_NN",
    "-": "SUB_NN",
    "*": "MUL_NN",
    "/": "DIV_NN",
    "%": "MOD_NN",
    "<": "LT_NN",
    "<=": "LE_NN",
    ">": "GT_NN",
    ">=": "GE_NN",
    "==": "EQ",
    "!=": "NE",
}
