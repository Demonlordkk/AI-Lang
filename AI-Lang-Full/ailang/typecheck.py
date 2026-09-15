"""Static checker for AI-Lang.

Performs scope resolution, mutability enforcement, arity checking and a
gradual type discipline: `Any` unifies with everything, so untyped code keeps
working while annotated code gets real guarantees.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
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
    variadic: bool = False  # implementation takes *args; named args rejected

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
    "capability_check": ([("name", TEXT), ("resource", TEXT)], BOOL, 1),
    "capability_require": ([("name", TEXT), ("resource", TEXT)], BOOL, 1),
    "capability_scope": ([("requested", MAP), ("fn", FUNCTION)], FUNCTION),
    "capability_info": ([], MAP),
    "device_profile": ([], MAP),
    "batch_budget": ([("item_bytes", INT), ("requested", INT)], INT, 2),
    "sleep_save": ([("path", TEXT), ("state", ANY), ("cursor", INT), ("metrics", MAP), ("epoch", INT), ("metadata", MAP), ("signing_key", TEXT), ("key_id", TEXT)], MAP, 6),
    "sleep_load": ([("path", TEXT), ("signing_key", TEXT), ("require_signature", BOOL)], MAP, 2),
    "active_train": ([("data", LIST), ("step_fn", FUNCTION), ("checkpoint_path", TEXT), ("options", MAP)], MAP, 1),
    "model_save": ([("path", TEXT), ("model", ANY), ("metadata", MAP), ("signing_key", TEXT), ("key_id", TEXT)], MAP, 3),
    "model_load": ([("path", TEXT), ("template", ANY), ("signing_key", TEXT), ("require_signature", BOOL)], ANY, 3),
    "optimizer_save": ([("path", TEXT), ("optimizer", MAP), ("metadata", MAP), ("signing_key", TEXT), ("key_id", TEXT)], MAP, 3),
    "optimizer_load": ([("path", TEXT), ("params", ANY), ("signing_key", TEXT), ("require_signature", BOOL)], MAP, 3),
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
    "append": ([("items", LIST), ("value", ANY)], LIST),
    "extend": ([("items", LIST), ("more", LIST)], LIST),
    "insert": ([("items", LIST), ("index", INT), ("value", ANY)], LIST),
    "pop": ([("items", LIST), ("index", INT)], ANY, 1),
    "clear": ([("items", LIST)], LIST),
    "concat": ([("a", LIST), ("b", LIST)], LIST),
    "slice": ([("items", ANY), ("start", INT), ("stop", INT)], ANY),
    "reverse": ([("items", ANY)], ANY),
    "sort": ([("items", LIST)], LIST),
    "sort_by": ([("items", LIST), ("key", FUNCTION)], LIST),
    "sort_desc": ([("items", LIST)], LIST),
    "group_by": ([("items", LIST), ("key", FUNCTION)], MAP),
    "count_by": ([("items", LIST), ("key", FUNCTION)], MAP),
    "sum_by": ([("items", LIST), ("key", FUNCTION)], ANY),
    "max_by": ([("items", LIST), ("key", FUNCTION)], ANY),
    "min_by": ([("items", LIST), ("key", FUNCTION)], ANY),
    "chunk": ([("items", LIST), ("size", INT)], LIST),
    "windows": ([("items", LIST), ("size", INT)], LIST),
    "partition": ([("items", LIST), ("fn", FUNCTION)], LIST),
    "take": ([("items", LIST), ("n", INT)], LIST),
    "drop": ([("items", LIST), ("n", INT)], LIST),
    "take_while": ([("items", LIST), ("fn", FUNCTION)], LIST),
    "drop_while": ([("items", LIST), ("fn", FUNCTION)], LIST),
    "zip_with": ([("a", LIST), ("b", LIST), ("fn", FUNCTION)], LIST),
    "flat_map": ([("items", LIST), ("fn", FUNCTION)], LIST),
    "pluck": ([("items", LIST), ("field", TEXT)], LIST),
    "index_where": ([("items", LIST), ("fn", FUNCTION)], INT),
    "counts": ([("items", LIST)], MAP),
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
    # numeric / statistics
    # autodiff / tensors
    "tensor": ([("value", ANY), ("requires_grad", BOOL)], ANY, 1),
    "param": ([("value", ANY)], ANY),
    "is_tensor": ([("value", ANY)], BOOL),
    "value_of": ([("t", ANY)], ANY),
    "grad_of": ([("t", ANY)], ANY),
    "zero_grad": ([("t", ANY)], VOID),
    "backward": ([("loss", ANY)], VOID),
    "sgd_step": ([("params", ANY), ("rate", ANY)], VOID),
    "adam": ([("params", ANY)], MAP),
    "adam_step": ([("state", MAP), ("rate", ANY), ("b1", ANY), ("b2", ANY), ("eps", ANY)], VOID, 4),
    "momentum": ([("params", ANY), ("rate", ANY), ("mu", ANY)], MAP, 2),
    "momentum_step": ([("state", MAP), ("rate", ANY), ("mu", ANY)], VOID, 2),
    "adamw": ([("params", ANY), ("rate", ANY), ("wd", ANY)], MAP, 2),
    "adamw_step": ([("state", MAP), ("rate", ANY), ("wd", ANY), ("b1", ANY), ("b2", ANY), ("eps", ANY)], VOID, 5),
    "clip_grad": ([("params", ANY), ("max_norm", ANY)], VOID),
    "seed": ([("n", ANY)], VOID),
    "randn": ([("rows", INT), ("cols", INT), ("scale", ANY), ("seed", ANY)], LIST, 3),
    "shape_of": ([("t", ANY)], LIST),
    "t_add": ([("a", ANY), ("b", ANY)], ANY),
    "t_sub": ([("a", ANY), ("b", ANY)], ANY),
    "t_mul": ([("a", ANY), ("b", ANY)], ANY),
    "t_div": ([("a", ANY), ("b", ANY)], ANY),
    "t_pow": ([("a", ANY), ("p", ANY)], ANY),
    "t_neg": ([("a", ANY)], ANY),
    "t_exp": ([("a", ANY)], ANY),
    "t_log": ([("a", ANY)], ANY),
    "t_sqrt": ([("a", ANY)], ANY),
    "t_abs": ([("a", ANY)], ANY),
    "t_sigmoid": ([("a", ANY)], ANY),
    "t_relu": ([("a", ANY)], ANY),
    "t_tanh": ([("a", ANY)], ANY),
    "t_softmax": ([("a", ANY)], ANY),
    "t_matmul": ([("a", ANY), ("b", ANY)], ANY),
    "t_transpose": ([("a", ANY)], ANY),
    "t_reshape": ([("a", ANY), ("shape", LIST)], ANY),
    "t_slice": ([("t", ANY), ("start", ANY), ("stop", ANY), ("axis", ANY)], ANY, 2),
    "t_gather": ([("t", ANY), ("indices", LIST)], ANY),
    "t_concat": ([("a", ANY), ("b", ANY), ("axis", ANY)], ANY, 1),
    "t_clip": ([("t", ANY), ("low", ANY), ("high", ANY)], ANY),
    "where_t": ([("cond", LIST), ("a", ANY), ("b", ANY)], ANY),
    "l2_norm_t": ([("t", ANY)], ANY),
    "bce_logits_t": ([("logits", ANY), ("target", ANY)], ANY),
    "huber_t": ([("pred", ANY), ("target", ANY), ("delta", ANY)], ANY, 1),
    "l2_penalty_t": ([("value", ANY)], ANY),
    "ml_backend": ([], TEXT),
    "sum_t": ([("a", ANY)], ANY),
    "mean_t": ([("a", ANY)], ANY),
    "mse_t": ([("pred", ANY), ("target", ANY)], ANY),
    "mae_t": ([("pred", ANY), ("target", ANY)], ANY),
    "bce_t": ([("pred", ANY), ("target", ANY)], ANY),
    "ce_t": ([("logits", ANY), ("target", ANY)], ANY),
    "dot": ([("a", LIST), ("b", LIST)], REAL),
    "vadd": ([("a", ANY), ("b", ANY)], ANY),
    "vsub": ([("a", ANY), ("b", ANY)], ANY),
    "vmul": ([("a", ANY), ("b", ANY)], ANY),
    "vdiv": ([("a", ANY), ("b", ANY)], ANY),
    "mean": ([("xs", LIST)], REAL),
    "median": ([("xs", LIST)], REAL),
    "variance": ([("xs", LIST), ("sample", BOOL)], REAL, 1),
    "stddev": ([("xs", LIST), ("sample", BOOL)], REAL, 1),
    "percentile": ([("xs", LIST), ("q", ANY)], REAL),
    "normalize": ([("xs", LIST)], LIST),
    "standardize": ([("xs", LIST)], LIST),
    "correlation": ([("a", LIST), ("b", LIST)], REAL),
    # matrices
    "shape": ([("m", LIST)], LIST),
    "matmul": ([("a", LIST), ("b", LIST)], LIST),
    "transpose": ([("m", LIST)], LIST),
    "identity": ([("n", INT)], LIST),
    "zeros": ([("rows", INT), ("cols", INT)], LIST, 1),
    # machine learning
    "sigmoid": ([("x", ANY)], ANY),
    "relu": ([("x", ANY)], ANY),
    "tanh": ([("x", ANY)], ANY),
    "softmax": ([("xs", LIST)], LIST),
    "argmax": ([("xs", LIST)], INT),
    "argmin": ([("xs", LIST)], INT),
    "mse": ([("pred", LIST), ("actual", LIST)], REAL),
    "mae": ([("pred", LIST), ("actual", LIST)], REAL),
    "cross_entropy": ([("pred", LIST), ("actual", LIST), ("eps", REAL)], REAL, 1),
    "accuracy": ([("pred", LIST), ("actual", LIST)], REAL),
    "linear_fit": ([("xs", LIST), ("ys", LIST)], MAP),
    "train_test_split": ([("items", LIST), ("ratio", ANY), ("seed", ANY)], MAP, 2),
    "shuffle": ([("items", LIST), ("seed", ANY)], LIST, 1),
    "sample": ([("items", LIST), ("k", INT), ("seed", ANY)], LIST, 1),
    "one_hot": ([("index", INT), ("size", INT)], LIST),
    "bincount": ([("items", LIST)], MAP),
    "exp": ([("x", ANY)], REAL),
    "log": ([("x", ANY), ("base", ANY)], REAL, 1),
    "json_encode": ([("value", ANY), ("indent", INT)], TEXT, 1),
    "json_decode": ([("text", TEXT)], ANY),
    "assert": ([("cond", ANY), ("message", ANY)], VOID, 1),
    "now": ([], REAL),
    "read_file": ([("path", TEXT)], TEXT),
    "write_file": ([("path", TEXT), ("content", TEXT)], VOID),
    "append_file": ([("path", TEXT), ("content", TEXT)], VOID),
    "pad_left": ([("text", TEXT), ("width", INT)], TEXT),
    "env": ([("name", TEXT)], ANY),
    # automation / filesystem
    "list_dir": ([("path", TEXT)], LIST, 1),
    "path_exists": ([("path", TEXT)], BOOL),
    "is_dir": ([("path", TEXT)], BOOL),
    "make_dir": ([("path", TEXT)], VOID),
    "delete_file": ([("path", TEXT)], VOID),
    "find_files": ([("root", TEXT), ("suffix", TEXT)], LIST, 1),
    "read_lines": ([("path", TEXT)], LIST),
    "read_line": ([], TEXT),
    "exit": ([("code", INT)], VOID, 1),
    "memo": ([("fn", FUNCTION)], FUNCTION),
    "memo_rec": ([("fn", FUNCTION)], FUNCTION),
    "script_dir": ([], TEXT),
    "write_lines": ([("path", TEXT), ("lines", LIST)], VOID),
    "read_csv": ([("path", TEXT), ("sep", TEXT)], LIST, 1),
    "write_csv": ([("path", TEXT), ("rows", LIST), ("sep", TEXT)], VOID, 1),
    "run": ([("cmd", TEXT), ("timeout", ANY)], MAP, 1),
    "pi": ([], REAL),
    "e": ([], REAL),
    "sin": ([("x", ANY)], REAL),
    "cos": ([("x", ANY)], REAL),
    "tan": ([("x", ANY)], REAL),
    "asin": ([("x", ANY)], REAL),
    "acos": ([("x", ANY)], REAL),
    "atan": ([("x", ANY)], REAL),
    "atan2": ([("y", ANY), ("x", ANY)], REAL),
    "sinh": ([("x", ANY)], REAL),
    "cosh": ([("x", ANY)], REAL),
    "log10": ([("x", ANY)], REAL),
    "log2": ([("x", ANY)], REAL),
    "hypot": ([("a", ANY), ("b", ANY)], REAL),
    "degrees": ([("x", ANY)], REAL),
    "radians": ([("x", ANY)], REAL),
    "sign": ([("x", ANY)], INT),
    "clamp": ([("value", ANY), ("low", ANY), ("high", ANY)], ANY),
    "gcd": ([("a", INT), ("b", INT)], INT),
    "lcm": ([("a", INT), ("b", INT)], INT),
    "trunc": ([("x", ANY)], INT),
    "is_finite": ([("x", ANY)], BOOL),
    "factorial": ([("n", INT)], INT),
    "canvas": ([("width", INT), ("height", INT)], ANY),
    "canvas_size": ([("canvas", ANY)], MAP),
    "canvas_fill": ([("canvas", ANY), ("colour", ANY)], VOID),
    "canvas_save": ([("canvas", ANY), ("path", TEXT)], INT),
    "pixel": ([("canvas", ANY), ("x", INT), ("y", INT), ("colour", ANY)], VOID),
    "rect": ([("canvas", ANY), ("x", INT), ("y", INT), ("w", INT), ("h", INT), ("colour", ANY), ("filled", BOOL)], VOID, 1),
    "line": ([("canvas", ANY), ("x0", INT), ("y0", INT), ("x1", INT), ("y1", INT), ("colour", ANY)], VOID),
    "circle": ([("canvas", ANY), ("cx", INT), ("cy", INT), ("r", INT), ("colour", ANY), ("filled", BOOL)], VOID, 1),
    "text": ([("canvas", ANY), ("x", INT), ("y", INT), ("message", ANY), ("colour", ANY), ("scale", INT)], VOID, 1),
    "plot": ([("path", TEXT), ("series", LIST), ("width", INT), ("height", INT), ("colour", TEXT), ("background", TEXT)], INT, 4),
    "db_open": ([("path", TEXT)], ANY, 1),
    "db_exec": ([("db", ANY), ("sql", TEXT), ("args", ANY)], INT, 1),
    "db_query": ([("db", ANY), ("sql", TEXT), ("args", ANY)], LIST, 1),
    "db_one": ([("db", ANY), ("sql", TEXT), ("args", ANY)], ANY, 1),
    "db_many": ([("db", ANY), ("sql", TEXT), ("rows", LIST)], INT),
    "db_transaction": ([("db", ANY), ("fn", FUNCTION)], ANY),
    "db_tables": ([("db", ANY)], LIST),
    "db_close": ([("db", ANY)], VOID),
    "store_open": ([("path", TEXT)], ANY),
    "store_put": ([("db", ANY), ("key", TEXT), ("value", ANY)], VOID),
    "store_get": ([("db", ANY), ("key", TEXT), ("fallback", ANY)], ANY, 1),
    "store_delete": ([("db", ANY), ("key", TEXT)], BOOL),
    "store_keys": ([("db", ANY), ("prefix", TEXT)], LIST, 1),
    "serve": ([("port", INT), ("handler", FUNCTION), ("host", TEXT), ("background", BOOL), ("ssl", ANY)], ANY, 3),
    "serve_stop": ([("server", ANY)], VOID),
    "tcp_listen": ([("port", INT), ("host", TEXT), ("backlog", INT)], ANY, 2),
    "tcp_accept": ([("listener", ANY), ("timeout", ANY)], ANY, 1),
    "tcp_connect": ([("host", TEXT), ("port", INT), ("timeout", ANY)], ANY, 1),
    "tcp_send": ([("conn", ANY), ("data", TEXT)], INT),
    "tcp_receive": ([("conn", ANY), ("limit", INT), ("timeout", ANY)], TEXT, 2),
    "tcp_close": ([("sock", ANY)], VOID),
    "ffi_open": ([("name", TEXT)], ANY),
    "ffi_fn": ([("lib", ANY), ("symbol", TEXT), ("argtypes", LIST), ("restype", TEXT)], ANY, 2),
    "ffi_call": ([("fn", ANY), ("args", LIST)], ANY, 1),
    "ffi_symbol": ([("lib", ANY), ("symbol", TEXT)], BOOL),
    "ffi_info": ([("lib", ANY)], MAP),
    "timestamp": ([("fmt", TEXT)], TEXT, 1),
    "parallel_map": ([("items", LIST), ("fn", FUNCTION), ("workers", INT)], LIST, 1),
    "retry": ([("fn", FUNCTION), ("attempts", INT), ("delay", ANY)], ANY, 2),
    "timed": ([("fn", FUNCTION)], MAP),
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
    "panic": ([("message", ANY)], VOID),
    "hash_text": ([("text", TEXT)], TEXT),
    "uuid": ([], TEXT),
    "spawn": ([("fn", FUNCTION), ("arg", ANY)], ANY, 1),
    "await_all": ([("tasks", LIST)], LIST),
    "await": ([("task", ANY), ("timeout", ANY)], ANY, 1),
    "t_matmul_bias": ([("a", ANY), ("b", ANY), ("c", ANY)], ANY),
    "ce_softmax_t": ([("logits", ANY), ("y", ANY)], ANY),
    "t_dropout": ([("x", ANY), ("rate", REAL), ("training", BOOL)], ANY, 1),
    "xavier": ([("n_in", INT), ("n_out", INT), ("seed", ANY)], LIST, 1),
    "he_init": ([("n_in", INT), ("n_out", INT), ("seed", ANY)], LIST, 1),
    "lr_step_decay": ([("step", INT), ("initial", REAL), ("factor", REAL), ("every", INT)], REAL),
    "lr_cosine": ([("step", INT), ("total", INT), ("initial", REAL), ("floor", REAL)], REAL, 1),
    "early_stop": ([("patience", INT)], MAP),
    "early_stop_step": ([("state", MAP), ("val_loss", REAL)], MAP),
    "gradcheck": ([("params", LIST), ("loss_fn", FUNCTION), ("eps", REAL)], REAL, 1),
    "f1": ([("y_true", LIST), ("y_pred", LIST), ("threshold", REAL)], REAL, 1),
    "mutex": ([], MAP),
    "lock": ([("m", MAP), ("timeout", ANY)], MAP, 1),
    "unlock": ([("m", MAP)], VOID),
    "channel": ([], MAP),
    "channel_send": ([("c", MAP), ("value", ANY)], VOID),
    "channel_recv": ([("c", MAP)], ANY),
    "channel_try_recv": ([("c", MAP)], ANY),
    "channel_close": ([("c", MAP)], VOID),
    "http_get": ([("url", TEXT), ("headers", MAP)], MAP, 1),
    "http_post": ([("url", TEXT), ("body", ANY), ("headers", MAP)], MAP, 2),
    "http_request": ([("url", TEXT), ("method", TEXT), ("body", ANY), ("headers", MAP), ("verify", BOOL), ("timeout", ANY)], MAP, 4),
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

    def all_names(self):
        """Every name visible from here, innermost first."""
        out = []
        s = self
        while s:
            out.extend(s.vars)
            s = s.parent
        return out


def _closest(name, candidates, limit=3):
    """Names within a small edit distance of `name`, best first.

    Uses a bounded Levenshtein distance rather than a similarity ratio so the
    threshold scales sensibly with word length: one typo in a short name is a
    strong signal, one typo in a long name even more so.
    """
    name_l = name.lower()
    scored = []
    # candidates are passed innermost-scope-first; earlier entries win ties so
    # a local name is suggested ahead of a builtin with the same distance
    rank = {}
    for i, c in enumerate(candidates):
        rank.setdefault(c, i)
    for cand in rank:
        if cand == name:
            continue
        cand_l = cand.lower()
        d = _edit_distance(name_l, cand_l)
        threshold = 1 if len(name) <= 4 else 2 if len(name) <= 8 else 3
        # a transposition or a shared prefix is a strong signal even when the
        # raw distance is larger, e.g. `lenght` for `length`
        if d > threshold and (
            sorted(name_l) == sorted(cand_l)
            or (len(name_l) > 4 and cand_l.startswith(name_l[:3]) and d <= 3)
        ):
            d = threshold
        if d <= threshold:
            scored.append((d, rank[cand], cand))
    scored.sort()
    return [c for _, _, c in scored[:limit]]


def _edit_distance(a, b):
    """Levenshtein distance, iterative and allocation-light."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if len(a) - len(b) > 3:
        return 99
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]



# -------------------------------------------------- builtin signature alignment

_GLOBALS = None
_BUILTIN_SIGS = None


def _builtin_globals():
    global _GLOBALS
    if _GLOBALS is None:
        from .stdlib import build_globals
        _GLOBALS = build_globals([])
    return _GLOBALS


def _builtin_sigs():
    """Reconcile public signatures with the real implementations.

    The implementation supplies reliable arity and default information, while
    ``BUILTIN_SIGS`` remains the source of public AI-Lang parameter names.  A
    Python helper may use short/private names such as ``v`` without changing
    the language API.  Implementations that take ``*args`` are marked
    variadic (named arguments are rejected for them at check time).

    This keeps the checker and runtime from drifting: a named argument that
    passes the checker is guaranteed to exist in the documented AI-Lang
    signature, and the VM's runtime guard still protects unchecked calls.
    """
    global _BUILTIN_SIGS
    if _BUILTIN_SIGS is None:
        g = _builtin_globals()
        out = {}
        for name, spec in BUILTIN_SIGS.items():
            params, ret = spec[0], spec[1]
            optional = spec[2] if len(spec) > 2 else 0
            variadic = False
            fn = g.get(name)
            if fn is not None:
                try:
                    sig = inspect.signature(fn)
                except (TypeError, ValueError):
                    sig = None
                if sig is not None:
                    psig = list(sig.parameters.values())
                    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in psig):
                        variadic = True
                    else:
                        # Derive both names and optionality from the callable
                        # that the VM will actually invoke.  Keeping the
                        # hand-written arity while a builtin gains a default
                        # argument was the source of several checker/runtime
                        # mismatches (notably http_request and await).
                        real = [
                            p for p in psig
                            if p.kind in (
                                p.POSITIONAL_ONLY,
                                p.POSITIONAL_OR_KEYWORD,
                                p.KEYWORD_ONLY,
                            )
                        ]
                        # The language-level names are the public API and
                        # may intentionally differ from private implementation
                        # names (for example `value` versus Python's `v`).
                        # Reconcile arity/optionality by position, but do not
                        # silently rename a documented AI-Lang parameter.
                        public = list(params)
                        params = [
                            (
                                public[i][0] if i < len(public) else p.name,
                                public[i][1] if i < len(public) else ANY,
                            )
                            for i, p in enumerate(real)
                        ]
                        optional = sum(
                            p.default is not inspect.Parameter.empty for p in real
                        )
            out[name] = FnSig(params, ret, name, optional, variadic)
        _BUILTIN_SIGS = out
    return _BUILTIN_SIGS


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
        # unknown type names are reported once, at their first occurrence
        self._unknown_types = set()
        for name, sig in _builtin_sigs().items():
            self.functions[name] = replace(sig)
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
    def _did_you_mean(self, name):
        """Suffix suggesting near-miss names, or an empty string."""
        # names the user wrote come first so they outrank a builtin at the
        # same edit distance -- a mistyped local is the likelier mistake
        builtins = set(BUILTIN_SIGS)
        visible = list(self.scope.all_names())
        pool = [n for n in visible if n not in builtins]
        pool.extend(n for n in self.functions if n not in builtins)
        pool.extend(n for n in visible if n in builtins)
        pool.extend(BUILTIN_SIGS)
        near = _closest(name, pool)
        if not near:
            return ""
        if len(near) == 1:
            return f"; did you mean '{near[0]}'?"
        quoted = ", ".join(f"'{n}'" for n in near)
        return f"; did you mean one of {quoted}?"

    def check(self, program: A.Program):
        self.hoist(program.statements, self.global_scope)
        self._check_type_names(program.statements)
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

    def _check_type_names(self, statements):
        """Reject annotations naming a type that was never declared.

        Records are hoisted first, so by now every legitimate record name is
        known; anything else capitalised is a typo or a missing declaration.
        """
        from .lexer import TYPE_NAMES

        def ok(name):
            return not name or name in TYPE_NAMES or name in self.records

        def check_fn_like(fn_node):
            for pname, ptype in fn_node.params:
                if not ok(ptype):
                    self._unknown_type(ptype, fn_node)
            if not ok(fn_node.return_type):
                self._unknown_type(fn_node.return_type, fn_node)
            visit(fn_node.body)

        def visit(stmts):
            for s in stmts:
                if isinstance(s, (A.Fn, A.FnExpr)):
                    check_fn_like(s)
                elif isinstance(s, A.Record):
                    for _f, ftype in s.fields:
                        if not ok(ftype):
                            self._unknown_type(ftype, s)
                elif isinstance(s, (A.Let, A.Var)):
                    if not ok(getattr(s, "declared_type", None)):
                        self._unknown_type(s.declared_type, s)
                    # lambdas bound in a binding carry their own annotations
                    if isinstance(s.expr, A.FnExpr):
                        check_fn_like(s.expr)
                    elif isinstance(s.expr, A.Call):
                        for _name, arg in s.expr.args:
                            if isinstance(arg, A.FnExpr):
                                check_fn_like(arg)
                else:
                    for attr in ("body", "else_body", "rescue_body"):
                        inner = getattr(s, attr, None)
                        if isinstance(inner, list):
                            visit(inner)
                    for b in getattr(s, "branches", []) or []:
                        visit(getattr(b, "body", []))

        visit(statements)

    def _unknown_type(self, name, node):
        if name in self._unknown_types:
            return
        self._unknown_types.add(name)
        hint = _closest(name, sorted(set(self.records) | set(__import__(
            "ailang.lexer", fromlist=["TYPE_NAMES"]).TYPE_NAMES))) if name else []
        extra = ""
        if hint:
            extra = f"; did you mean '{hint[0]}'?" if len(hint) == 1 else \
                "; did you mean one of " + ", ".join(f"'{h}'" for h in hint) + "?"
        self.error(f"unknown type '{name}'{extra}", node)

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
                if s.declared_type in self._unknown_types:
                    # the declared type itself is the error; a bind check
                    # against it would only repeat the same unknown name
                    pass
                else:
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
                    self.error(
                        f"undefined name '{target.value}'"
                        + self._did_you_mean(target.value),
                        s,
                    )
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
            # `attempt` does not introduce a lexical block.  Its body, rescue
            # clause, and following statements share the enclosing scope; this
            # lets a parsed value survive a recoverable failure and matches the
            # VM's recovery environment.
            for x in s.body:
                self.stmt(x)
            if self.scope.local(s.error_name) and not (
                self.scope is self.global_scope and s.error_name in BUILTIN_SIGS
            ):
                self.error(
                    f"'{s.error_name}' is already defined in this scope; "
                    "choose a different rescue name",
                    s,
                )
            self.scope.declare(s.error_name, ANY, False)
            for x in s.rescue_body:
                self.stmt(x)
            return

        self.error(f"unsupported statement {type(s).__name__}", s)

    def check_function(self, name, params, return_type, body, node):
        ret = ty(return_type or "Void")
        self.return_stack.append(ret)
        saved_loop = self.loop_depth
        self.loop_depth = 0

        # Function declarations are lexical.  The old checker kept nested
        # declarations in the shared ``self.functions`` dictionary, so a name
        # declared inside one function could be called from an unrelated outer
        # scope.  Keep a private registry while checking this body and restore
        # the enclosing registry even when diagnostics are raised later.
        saved_functions = self.functions
        saved_records = self.records
        saved_modules = self.modules
        self.functions = dict(saved_functions)
        self.records = dict(saved_records)
        self.modules = dict(saved_modules)
        self.push("function")
        try:
            seen = set()
            for pname, ptype in params:
                if pname in seen:
                    self.error(f"function '{name}' has a duplicate parameter '{pname}'", node)
                seen.add(pname)
                self.scope.declare(pname, ty(ptype), False)
            # nested declarations are visible within the function body, but
            # only while that function is being checked.
            self.hoist(body, self.scope)
            for x in body:
                self.stmt(x)
            if return_type and return_type != "Void" and not self.always_returns(body):
                self.error(
                    f"function '{name}' declares -> {return_type} but can finish without 'give'",
                    node,
                )
        finally:
            self.pop()
            self.functions = saved_functions
            self.records = saved_records
            self.modules = saved_modules
            self.loop_depth = saved_loop
            self.return_stack.pop()

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
            self.error(
                f"undefined name '{n.value}'" + self._did_you_mean(n.value), n
            )
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
        # Preserve source order here instead of collapsing named arguments into
        # a dict immediately.  This catches duplicate names and the ambiguous
        # ``f(a: 1, 2)`` form before the VM would have to guess how to bind it.
        positional = []
        named = []
        saw_named = False
        for name, t in arg_types:
            if name is None:
                if saw_named:
                    self.error(
                        f"function '{sig.name}' does not allow a positional argument after a named argument",
                        node,
                    )
                    return
                positional.append(t)
            else:
                saw_named = True
                named.append((name, t))

        if sig.variadic:
            if named:
                self.error(f"function '{sig.name}' takes positional arguments only", node)
                return
            if len(positional) < 1:
                self.error(
                    f"function '{sig.name}' expects at least 1 argument but got {len(positional)}",
                    node,
                )
                return
            return

        param_names = [p[0] for p in sig.params]
        types_by_name = dict(sig.params)
        seen_named = set()
        for key, _at in named:
            if key not in types_by_name:
                self.error(f"'{sig.name}' has no parameter named '{key}'", node)
                return
            if key in seen_named:
                self.error(f"duplicate value for parameter '{key}'", node)
                return
            seen_named.add(key)

        # A named argument cannot fill a slot already occupied by a
        # positional argument.  Positional arguments always bind from the
        # left, as they do in Closure.invoke and Python builtins.
        for key, _at in named:
            if param_names.index(key) < len(positional):
                self.error(f"duplicate value for parameter '{key}'", node)
                return

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

        # Count alone is insufficient for optional signatures: ``round(digits:
        # 2)`` has one argument but is still missing its required value.
        bound = set(param_names[: len(positional)]) | seen_named
        missing = [p for p in param_names[:low] if p not in bound]
        if missing:
            self.error(
                f"function '{sig.name}' is missing required parameter(s): {', '.join(missing)}",
                node,
            )
            return

        for (pname, ptype), at in zip(sig.params, positional):
            if not self.compatible(ptype, at):
                self.error(
                    f"argument '{pname}' of '{sig.name}' expects {ptype} but got {at}", node
                )
        for key, at in named:
            ptype = types_by_name[key]
            if not self.compatible(ptype, at):
                self.error(f"argument '{key}' of '{sig.name}' expects {ptype} but got {at}", node)


def check(program: A.Program, module_resolver=None):
    return TypeChecker(module_resolver).check(program)
