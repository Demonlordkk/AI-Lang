"""Stack virtual machine for AI-Lang bytecode."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .errors import AILangRaise, VMError
from .opcodes import NAMES, OPS
from .values import Module, RecordType, RecordValue, display, is_truthy, type_name

# bind opcode ints as module-level constants for fast local lookup
globals().update(OPS)


def _builtin_names():
    from .stdlib import build_globals

    return frozenset(build_globals().keys()) | {"true", "false", "nothing"}


_BUILTIN_NAMES = frozenset()


def _init_builtin_names():
    global _BUILTIN_NAMES
    if not _BUILTIN_NAMES:
        _BUILTIN_NAMES = _builtin_names()
    return _BUILTIN_NAMES


class Environment:
    """Lexical scope chain with let/var mutability tracking."""

    __slots__ = ("vars", "immutable", "parent")

    def __init__(self, parent: Optional["Environment"] = None, initial: Dict[str, Any] = None):
        self.vars: Dict[str, Any] = initial if initial is not None else {}
        self.immutable = set()
        self.parent = parent

    def lookup(self, name):
        env = self
        while env is not None:
            if name in env.vars:
                return env.vars[name]
            env = env.parent
        raise VMError(f"undefined name '{name}'")

    def has(self, name):
        env = self
        while env is not None:
            if name in env.vars:
                return True
            env = env.parent
        return False

    def declare(self, name, value, mutable):
        # Builtins live in the root scope and are shadowable by user code.
        if name in self.vars and not (self.parent is None and name in _BUILTIN_NAMES):
            raise VMError(
                f"'{name}' is already defined in this scope; use '<-' to reassign a var"
            )
        self.vars[name] = value
        if self.immutable.__class__ is frozenset:
            self.immutable = set(self.immutable)   # copy-on-write
        if mutable:
            self.immutable.discard(name)
        else:
            self.immutable.add(name)

    def assign(self, name, value):
        env = self
        while env is not None:
            if name in env.vars:
                if name in env.immutable:
                    raise VMError(
                        f"cannot assign to '{name}': it is a let binding "
                        f"(declare it with 'var' to allow mutation)"
                    )
                env.vars[name] = value
                return
            env = env.parent
        raise VMError(f"undefined name '{name}'")

    def child(self):
        return Environment(self)


class Closure:
    """A function value bound to its defining environment and program."""

    __slots__ = ("code", "env", "vm", "ailang_name", "program")

    def __init__(self, code, env, vm, program=None):
        self.code = code
        self.env = env
        self.vm = vm
        self.program = program if program is not None else vm.program
        self.ailang_name = code.name

    @property
    def arity(self):
        return len(self.code.params)

    def __call__(self, *args, **kwargs):
        return self.vm.invoke(self, list(args), kwargs)

    def __repr__(self):
        return f"<function {self.code.name}>"


class _Break(Exception):
    pass


class VM:
    MAX_DEPTH = 2500

    def __init__(self, globals_dict: Dict[str, Any] = None, fuel: int = 50_000_000,
                 module_loader=None):
        self.globals = Environment(None, globals_dict or {})
        self.globals.vars.setdefault("true", True)
        self.globals.vars.setdefault("false", False)
        self.globals.vars.setdefault("nothing", None)
        self.fuel = fuel
        self.depth = 0
        self.program = None
        self.module_loader = module_loader
        _init_builtin_names()

    # ----------------------------------------------------------------- driver
    def run(self, program):
        self.program = program
        for name, fields in program.records.items():
            self.globals.vars[name] = RecordType(name, fields)
        return self.execute(program.main, self.globals)

    def invoke(self, closure, args, kwargs=None):
        code = closure.code
        params = code.params
        nparams = len(params)

        if not kwargs:
            if len(args) != nparams:
                raise VMError(
                    f"'{code.name}' expects {nparams} argument(s) but got {len(args)}"
                )
            # zip into a dict in one step; parameters share a precomputed
            # immutable set instead of rebuilding it on every call
            env = Environment(closure.env, dict(zip(params, args)))
            env.immutable = code.param_set
        else:
            if len(args) + len(kwargs) != nparams:
                raise VMError(
                    f"'{code.name}' expects {nparams} argument(s) "
                    f"but got {len(args) + len(kwargs)}"
                )
            env = Environment(closure.env, dict(zip(params, args)))
            for key, value in kwargs.items():
                if key not in params:
                    raise VMError(f"'{code.name}' has no parameter named '{key}'")
                if key in env.vars:
                    raise VMError(f"duplicate value for parameter '{key}'")
                env.vars[key] = value
            env.immutable = set(params)
        self.depth += 1
        if self.depth > self.MAX_DEPTH:
            self.depth -= 1
            raise VMError("recursion limit exceeded (possible infinite recursion)")
        try:
            return self.execute(code, env, closure.program)
        finally:
            self.depth -= 1

    # -------------------------------------------------------------- execution
    def execute(self, fn, env, program=None):
        """Interpreter loop.

        Dispatch is on small integers through a chain ordered by measured
        frequency, and the hottest arithmetic/compare/loop opcodes are
        specialised so they never touch the generic type-dispatch helpers.
        """
        program = program if program is not None else self.program
        code = fn.code
        stack = []
        push = stack.append
        pop = stack.pop
        iters = []
        traps = []
        scopes = []
        ip = 0
        n = len(code)
        fuel = self.fuel

        # local aliases: attribute lookups in a hot loop are expensive
        _PUSH = PUSH; _LOAD = LOAD; _STORE = STORE; _SET = SET
        _ADD_NN = ADD_NN; _SUB_NN = SUB_NN; _MUL_NN = MUL_NN
        _LT_NN = LT_NN; _LE_NN = LE_NN; _GT_NN = GT_NN; _GE_NN = GE_NN
        _JUMP = JUMP; _JUMP_IF_FALSE = JUMP_IF_FALSE
        _ITER_NEXT = ITER_NEXT; _CALL = CALL; _RETURN = RETURN
        _BINARY = BINARY; _INC_FAST = INC_FAST; _ADD_CONST = ADD_CONST
        _LOAD_LOAD = LOAD_LOAD; _LOAD_PUSH = LOAD_PUSH
        _LOAD_ADD_NN = LOAD_ADD_NN; _LOAD_LT_NN = LOAD_LT_NN
        _LOAD_FIELD = LOAD_FIELD
        _LOAD_INDEX = LOAD_INDEX

        while ip < n:
            fuel -= 1
            if fuel < 0:
                self.fuel = fuel
                raise VMError("execution limit exceeded")
            ins = code[ip]
            op = ins[0]
            ip += 1

            try:
                # ---- hottest opcodes first -------------------------------
                if op == _LOAD:
                    name = ins[1]
                    e = env
                    while e is not None:
                        v = e.vars
                        if name in v:
                            push(v[name])
                            break
                        e = e.parent
                    else:
                        raise VMError(f"undefined name '{name}'")

                # ---- fused pairs: one dispatch instead of two -------------
                elif op == _LOAD_PUSH:
                    name = ins[1]
                    e = env
                    while e is not None:
                        v = e.vars
                        if name in v:
                            push(v[name])
                            break
                        e = e.parent
                    else:
                        raise VMError(f"undefined name '{name}'")
                    push(ins[2])

                elif op == _LOAD_LOAD:
                    name = ins[1]
                    e = env
                    while e is not None:
                        v = e.vars
                        if name in v:
                            push(v[name])
                            break
                        e = e.parent
                    else:
                        raise VMError(f"undefined name '{name}'")
                    name = ins[2]
                    e = env
                    while e is not None:
                        v = e.vars
                        if name in v:
                            push(v[name])
                            break
                        e = e.parent
                    else:
                        raise VMError(f"undefined name '{name}'")

                elif op == _LOAD_ADD_NN:
                    # stack already holds the left operand; the local is right
                    name = ins[1]
                    e = env
                    while e is not None:
                        v = e.vars
                        if name in v:
                            b = v[name]
                            break
                        e = e.parent
                    else:
                        raise VMError(f"undefined name '{name}'")
                    a = pop()
                    if a.__class__ is int and b.__class__ is int:
                        push(a + b)
                    else:
                        push(_binary("+", a, b))

                elif op == _LOAD_LT_NN:
                    # stack already holds the left operand; the local is right
                    name = ins[1]
                    e = env
                    while e is not None:
                        v = e.vars
                        if name in v:
                            b = v[name]
                            break
                        e = e.parent
                    else:
                        raise VMError(f"undefined name '{name}'")
                    a = pop()
                    if a.__class__ is int and b.__class__ is int:
                        push(a < b)
                    else:
                        push(_binary("<", a, b))

                elif op == _LOAD_FIELD:
                    name = ins[1]
                    e = env
                    while e is not None:
                        v = e.vars
                        if name in v:
                            push(_field(v[name], ins[2]))
                            break
                        e = e.parent
                    else:
                        raise VMError(f"undefined name '{name}'")

                elif op == _LOAD_INDEX:
                    # LOAD pushed the key; the object is already on the stack
                    name = ins[1]
                    e = env
                    while e is not None:
                        v = e.vars
                        if name in v:
                            key = v[name]
                            break
                        e = e.parent
                    else:
                        raise VMError(f"undefined name '{name}'")
                    obj = pop()
                    if obj.__class__ is list and key.__class__ is int:
                        if -len(obj) <= key < len(obj):
                            push(obj[key])
                        else:
                            raise VMError(
                                f"index {key} is out of range for a list of {len(obj)}"
                            )
                    else:
                        push(_index(obj, key))


                elif op == _PUSH:
                    push(ins[1])

                elif op == _ADD_NN:
                    b = pop(); a = pop()
                    if a.__class__ is int and b.__class__ is int:
                        push(a + b)
                    else:
                        push(_binary("+", a, b))

                elif op == _SUB_NN:
                    b = pop(); a = pop()
                    if a.__class__ is int and b.__class__ is int:
                        push(a - b)
                    else:
                        push(_binary("-", a, b))

                elif op == _MUL_NN:
                    b = pop(); a = pop()
                    if a.__class__ is int and b.__class__ is int:
                        push(a * b)
                    else:
                        push(_binary("*", a, b))

                elif op == _LT_NN:
                    b = pop(); a = pop()
                    if a.__class__ is int and b.__class__ is int:
                        push(a < b)
                    else:
                        push(_binary("<", a, b))

                elif op == _GT_NN:
                    b = pop(); a = pop()
                    if a.__class__ is int and b.__class__ is int:
                        push(a > b)
                    else:
                        push(_binary(">", a, b))

                elif op == _LE_NN:
                    b = pop(); a = pop()
                    if a.__class__ is int and b.__class__ is int:
                        push(a <= b)
                    else:
                        push(_binary("<=", a, b))

                elif op == _GE_NN:
                    b = pop(); a = pop()
                    if a.__class__ is int and b.__class__ is int:
                        push(a >= b)
                    else:
                        push(_binary(">=", a, b))

                elif op == _JUMP_IF_FALSE:
                    v = pop()
                    if v is False or v is None:
                        ip = ins[1]
                    elif v is not True and not is_truthy(v):
                        ip = ins[1]

                elif op == _JUMP:
                    ip = ins[1]

                elif op == _ITER_NEXT:
                    it = iters[-1]
                    nxt = next(it, _SENTINEL)
                    if nxt is _SENTINEL:
                        iters.pop()
                        ip = ins[1]
                    else:
                        index, value = nxt
                        ev = env.vars
                        ev[ins[2]] = value
                        if env.immutable:
                            if env.immutable.__class__ is frozenset:
                                env.immutable = set(env.immutable)
                            env.immutable.discard(ins[2])
                        if ins[3]:
                            ev[ins[3]] = index

                elif op == _CALL:
                    count = ins[1]
                    if count:
                        args = stack[-count:]
                        del stack[-count:]
                    else:
                        args = []
                    callee = pop()
                    if callee.__class__ is Closure:
                        self.fuel = fuel
                        push(self.invoke(callee, args))
                        fuel = self.fuel
                    else:
                        self.fuel = fuel
                        push(self.call_value(callee, args))
                        fuel = self.fuel

                elif op == _SET:
                    name = ins[1]
                    value = pop()
                    e = env
                    while e is not None:
                        if name in e.vars:
                            if name in e.immutable:
                                raise VMError(
                                    f"cannot assign to '{name}': it is a let binding "
                                    f"(declare it with 'var' to allow mutation)"
                                )
                            e.vars[name] = value
                            break
                        e = e.parent
                    else:
                        raise VMError(f"undefined name '{name}'")

                elif op == _INC_FAST:
                    # var <- var + <const>   (single opcode, no stack traffic)
                    name = ins[1]
                    e = env
                    while e is not None:
                        if name in e.vars:
                            if name in e.immutable:
                                raise VMError(
                                    f"cannot assign to '{name}': it is a let binding "
                                    f"(declare it with 'var' to allow mutation)"
                                )
                            cur = e.vars[name]
                            if cur.__class__ is int:
                                e.vars[name] = cur + ins[2]
                            else:
                                e.vars[name] = _binary("+", cur, ins[2])
                            break
                        e = e.parent
                    else:
                        raise VMError(f"undefined name '{name}'")

                elif op == _ADD_CONST:
                    a = pop()
                    if a.__class__ is int:
                        push(a + ins[1])
                    else:
                        push(_binary("+", a, ins[1]))

                elif op == _STORE:
                    env.declare(ins[1], pop(), ins[2])

                elif op == _RETURN:
                    self.fuel = fuel
                    return pop() if stack else None

                elif op == _BINARY:
                    b = pop(); a = pop()
                    push(_binary(ins[1], a, b))

                # ---- everything else -------------------------------------
                elif op == EQ:
                    b = pop(); a = pop()
                    push(_equal(a, b))

                elif op == NE:
                    b = pop(); a = pop()
                    push(not _equal(a, b))

                elif op == DIV_NN:
                    b = pop(); a = pop()
                    push(_binary("/", a, b))

                elif op == MOD_NN:
                    b = pop(); a = pop()
                    if a.__class__ is int and b.__class__ is int and b:
                        push(a % b)
                    else:
                        push(_binary("%", a, b))

                elif op == POP:
                    pop()

                elif op == INDEX:
                    key = pop(); obj = pop()
                    if obj.__class__ is list and key.__class__ is int:
                        if -len(obj) <= key < len(obj):
                            push(obj[key])
                        else:
                            raise VMError(
                                f"index {key} is out of range for a list of {len(obj)}"
                            )
                    elif obj.__class__ is dict:
                        k = _hashable(key)
                        if k not in obj:
                            raise VMError(f"map has no key {display(key)}")
                        push(obj[k])
                    else:
                        push(_index(obj, key))

                elif op == FIELD:
                    push(_field(pop(), ins[1]))

                elif op == PRINT:
                    print(display(pop()))

                elif op == TO_BOOL:
                    push(is_truthy(pop()))

                elif op == UNARY:
                    v = pop()
                    if ins[1] == "not":
                        push(not is_truthy(v))
                    else:
                        if v.__class__ is bool or not isinstance(v, (int, float)):
                            raise VMError(f"cannot negate {type_name(v)}")
                        push(-v)

                elif op == MAKE_LIST:
                    count = ins[1]
                    if count:
                        items = stack[-count:]
                        del stack[-count:]
                        push(items)
                    else:
                        push([])

                elif op == MAKE_MAP:
                    count = ins[1]
                    out = {}
                    if count:
                        flat = stack[-count * 2:]
                        del stack[-count * 2:]
                        for i in range(0, len(flat), 2):
                            out[_hashable(flat[i])] = flat[i + 1]
                    push(out)

                elif op == RANGE_INIT:
                    src = pop()
                    if src.__class__ is not int:
                        raise VMError("range loop needs an Int bound")
                    iters.append(iter(range(src)))

                elif op == RANGE_NEXT:
                    it = iters[-1]
                    nxt = next(it, _SENTINEL)
                    if nxt is _SENTINEL:
                        iters.pop()
                        ip = ins[1]
                    else:
                        env.vars[ins[2]] = nxt

                elif op == ITER_INIT:
                    iters.append(_make_iter(pop()))

                elif op == ITER_BREAK:
                    if iters:
                        iters.pop()
                    ip = ins[1]

                elif op == ITER_END:
                    pass

                elif op == SCOPE_PUSH:
                    scopes.append(env)
                    env = Environment(env)

                elif op == SCOPE_POP:
                    if scopes:
                        env = scopes.pop()

                elif op == JUMP_IF_FALSE_KEEP:
                    if not is_truthy(stack[-1]):
                        stack[-1] = False
                        ip = ins[1]

                elif op == JUMP_IF_TRUE_KEEP:
                    if is_truthy(stack[-1]):
                        stack[-1] = True
                        ip = ins[1]

                elif op == JUMP_IF_TRUE:
                    if is_truthy(pop()):
                        ip = ins[1]

                elif op == JUMP_IF_SOME:
                    if stack[-1] is not None:
                        ip = ins[1]

                elif op == CALL_KW:
                    count = ins[1]
                    if count:
                        args = stack[-count:]
                        del stack[-count:]
                    else:
                        args = []
                    callee = pop()
                    names = ins[2]
                    kwargs = {}
                    positional = []
                    for nm, value in zip(names, args):
                        if nm is None:
                            positional.append(value)
                        else:
                            kwargs[nm] = value
                    self.fuel = fuel
                    push(self.call_value(callee, positional, kwargs))
                    fuel = self.fuel

                elif op == CLOSURE:
                    fcode = program.functions[ins[1]]
                    clo = Closure(fcode, env, self, program)
                    if ins[2]:
                        if ins[2] in env.vars:
                            env.vars[ins[2]] = clo
                        else:
                            env.declare(ins[2], clo, False)
                    else:
                        push(clo)

                elif op == SET_INDEX:
                    value = pop(); key = pop(); obj = pop()
                    if obj.__class__ is list:
                        if key.__class__ is not int:
                            raise VMError("list index must be an Int")
                        if not -len(obj) <= key < len(obj):
                            raise VMError(
                                f"index {key} out of range for list of {len(obj)}"
                            )
                        obj[key] = value
                    elif obj.__class__ is dict:
                        obj[_hashable(key)] = value
                    else:
                        raise VMError(f"cannot index-assign into {type_name(obj)}")

                elif op == SET_FIELD:
                    value = pop(); obj = pop()
                    if obj.__class__ is RecordValue:
                        obj.set(ins[1], value)
                    elif obj.__class__ is dict:
                        obj[ins[1]] = value
                    else:
                        raise VMError(f"cannot set field on {type_name(obj)}")

                elif op == CONVERT:
                    push(_convert(ins[1], pop()))

                elif op == TRY_PUSH:
                    traps.append((ins[1], ins[2], len(stack), len(iters), len(scopes)))

                elif op == TRY_POP:
                    if traps:
                        traps.pop()

                elif op == RAISE:
                    raise AILangRaise(pop())

                elif op == RECORD:
                    name = ins[1]
                    if not env.has(name):
                        env.declare(name, RecordType(name, program.records[name]), False)

                elif op == IMPORT:
                    path, alias = ins[1], ins[2]
                    if self.module_loader is None:
                        raise VMError(f"cannot import '{path}': no module loader configured")
                    module = self.module_loader(path)
                    if env.has(alias):
                        env.assign(alias, module)
                    else:
                        env.declare(alias, module, False)

                elif op == RETURN_NONE:
                    self.fuel = fuel
                    return None

                elif op == DUP:
                    push(stack[-1])

                elif op == HALT:
                    self.fuel = fuel
                    return None

                else:
                    raise VMError(f"unknown opcode {NAMES.get(op, op)}")

            except _Break:
                raise
            except Exception as exc:  # noqa: BLE001 - translated below
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                if not traps:
                    self.fuel = fuel
                    raise _promote(exc)
                target, err_name, sdepth, idepth, scdepth = traps.pop()
                del stack[sdepth:]
                del iters[idepth:]
                while len(scopes) > scdepth:
                    env = scopes.pop()
                env.vars[err_name] = _error_value(exc)
                if env.immutable:
                    if env.immutable.__class__ is frozenset:
                        env.immutable = set(env.immutable)
                    env.immutable.discard(err_name)
                ip = target

        self.fuel = fuel
        return None

    def call_value(self, callee, args, kwargs=None):
        kwargs = kwargs or {}
        if isinstance(callee, Closure):
            return self.invoke(callee, args, kwargs)
        if isinstance(callee, RecordType):
            return callee(*args, **kwargs)
        if callable(callee):
            try:
                return callee(*args, **kwargs)
            except TypeError as e:
                name = getattr(callee, "ailang_name", getattr(callee, "__name__", "function"))
                raise VMError(f"{name}: {e}") from e
        raise VMError(f"value of type {type_name(callee)} is not callable")


_SENTINEL = object()


def _make_iter(src):
    if isinstance(src, list):
        return iter(enumerate(src))
    if isinstance(src, str):
        return iter(enumerate(src))
    if isinstance(src, dict):
        return iter(enumerate(list(src.keys())))
    if isinstance(src, range):
        return iter(enumerate(src))
    raise VMError(f"cannot repeat over {type_name(src)}; expected List, Map or Text")


def _hashable(key):
    if isinstance(key, list):
        return tuple(key)
    if isinstance(key, dict):
        raise VMError("a Map cannot be used as a Map key")
    return key


def _index(obj, key):
    if isinstance(obj, list):
        if isinstance(key, bool) or not isinstance(key, int):
            raise VMError(f"list index must be an Int, got {type_name(key)}")
        if not -len(obj) <= key < len(obj):
            raise VMError(f"index {key} is out of range for a list of {len(obj)}")
        return obj[key]
    if isinstance(obj, str):
        if isinstance(key, bool) or not isinstance(key, int):
            raise VMError(f"text index must be an Int, got {type_name(key)}")
        if not -len(obj) <= key < len(obj):
            raise VMError(f"index {key} is out of range for text of length {len(obj)}")
        return obj[key]
    if isinstance(obj, dict):
        k = _hashable(key)
        if k not in obj:
            raise VMError(f"map has no key {display(key)}")
        return obj[k]
    if isinstance(obj, RecordValue):
        return obj.get(str(key))
    if isinstance(obj, Module):
        return obj.get(str(key))
    raise VMError(f"cannot index a value of type {type_name(obj)}")


def _field(obj, name):
    if isinstance(obj, RecordValue):
        return obj.get(name)
    if isinstance(obj, Module):
        return obj.get(name)
    if isinstance(obj, dict):
        if name not in obj:
            raise VMError(f"map has no key \"{name}\"")
        return obj[name]
    raise VMError(f"cannot read field '{name}' from {type_name(obj)}")


def _numeric(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _binary(op, a, b):
    if op == "==":
        return _equal(a, b)
    if op == "!=":
        return not _equal(a, b)

    if op in ("<", "<=", ">", ">="):
        if _numeric(a) and _numeric(b):
            pass
        elif isinstance(a, str) and isinstance(b, str):
            pass
        else:
            raise VMError(f"cannot compare {type_name(a)} with {type_name(b)} using '{op}'")
        return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]

    if op == "+":
        if isinstance(a, str) or isinstance(b, str):
            if isinstance(a, str) and isinstance(b, str):
                return a + b
            raise VMError(
                f"cannot add {type_name(a)} and {type_name(b)}; "
                f"wrap the non-text side in to Text(...)"
            )
        if isinstance(a, list) and isinstance(b, list):
            return a + b
        if isinstance(a, dict) and isinstance(b, dict):
            out = dict(a)
            out.update(b)
            return out
        if _numeric(a) and _numeric(b):
            return a + b
        raise VMError(f"cannot add {type_name(a)} and {type_name(b)}")

    if op == "*":
        if isinstance(a, str) and isinstance(b, int) and not isinstance(b, bool):
            return a * max(b, 0)
        if isinstance(a, list) and isinstance(b, int) and not isinstance(b, bool):
            return a * max(b, 0)

    if not (_numeric(a) and _numeric(b)):
        raise VMError(f"cannot apply '{op}' to {type_name(a)} and {type_name(b)}")

    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if b == 0:
            raise VMError("division by zero")
        return a / b
    if op == "%":
        if b == 0:
            raise VMError("modulo by zero")
        return a % b
    raise VMError(f"unknown operator '{op}'")


def _equal(a, b):
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if _numeric(a) and _numeric(b):
        return a == b
    if type(a) is not type(b):
        if isinstance(a, RecordValue) and isinstance(b, RecordValue):
            return a == b
        return False
    return a == b


def _convert(target, v):
    try:
        if target == "Int":
            if isinstance(v, bool):
                return 1 if v else 0
            if isinstance(v, str):
                return int(float(v)) if ("." in v or "e" in v.lower()) else int(v.strip())
            if v is None:
                raise VMError("cannot convert nothing to Int")
            return int(v)
        if target == "Real":
            if isinstance(v, bool):
                return 1.0 if v else 0.0
            if v is None:
                raise VMError("cannot convert nothing to Real")
            return float(v)
        if target == "Text":
            return display(v)
        if target == "Bool":
            return is_truthy(v)
        if target == "Byte":
            return int(v) % 256
        if target == "List":
            if isinstance(v, list):
                return v
            if isinstance(v, str):
                return list(v)
            if isinstance(v, dict):
                return list(v.keys())
            raise VMError(f"cannot convert {type_name(v)} to List")
        if target == "Map":
            if isinstance(v, dict):
                return v
            raise VMError(f"cannot convert {type_name(v)} to Map")
    except VMError:
        raise
    except Exception as e:
        raise VMError(f"cannot convert {display(v)} to {target}: {e}") from e
    raise VMError(f"unknown conversion target '{target}'")


def _error_value(exc):
    """Expose a caught error to `rescue` as an inspectable Map."""
    from .errors import AILangError

    if isinstance(exc, AILangRaise):
        value = exc.value
        if isinstance(value, dict):
            return value
        return {"kind": "raised", "message": display(value), "value": value}
    if isinstance(exc, AILangError):
        return {"kind": exc.stage, "message": exc.message, "value": exc.message}
    return {"kind": "runtime error", "message": str(exc), "value": str(exc)}


def _promote(exc):
    from .errors import AILangError

    if isinstance(exc, AILangError):
        return exc
    if isinstance(exc, RecursionError):
        return VMError("recursion limit exceeded (possible infinite recursion)")
    return VMError(str(exc))
