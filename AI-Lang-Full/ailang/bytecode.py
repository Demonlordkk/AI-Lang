"""Deterministic and defensive AI-Lang bytecode artifacts.

Artifacts are canonical JSON.  They are deliberately a *data* format: native
host-code loop lowering is useful for source execution in the current process,
but is never embedded in a file that can later be loaded.  ``read`` validates
the complete schema and instruction stream before ``load`` creates executable
objects.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import tempfile
from pathlib import Path

from .compiler import FunctionCode, ProgramCode
from .lexer import TYPE_NAMES
from .opcodes import NAMES, OPS
from .version import BYTECODE_FORMAT, LANGUAGE, VERSION


_NATIVE_LOOP_OPCODE = OPS["NATIVE_LOOP"]
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_FUNCTIONS = 100_000
_MAX_INSTRUCTIONS = 10_000_000
_MAX_SEQUENCE_COUNT = 10_000_000
_MAX_JSON_DEPTH = 1_000
_MAX_INTEGER_DIGITS = 10_000

# The VM currently supports every opcode listed here.  Keeping the arity table
# next to the artifact validator makes a malformed file fail before dispatch,
# rather than producing an IndexError or an unknown-opcode traceback halfway
# through execution.
_ARITY = {
    "PUSH": 1,
    "POP": 0,
    "DUP": 0,
    "LOAD": 1,
    "LOAD_FAST": 1,
    "STORE": 2,
    "SET": 1,
    "SET_FAST": 1,
    "BINARY": 1,
    "UNARY": 1,
    "TO_BOOL": 0,
    "ADD_NN": 0,
    "SUB_NN": 0,
    "MUL_NN": 0,
    "DIV_NN": 0,
    "MOD_NN": 0,
    "LT_NN": 0,
    "LE_NN": 0,
    "GT_NN": 0,
    "GE_NN": 0,
    "EQ": 0,
    "NE": 0,
    "ADD_CONST": 1,
    "INC_FAST": 2,
    "MAKE_LIST": 1,
    "MAKE_MAP": 1,
    "INDEX": 0,
    "FIELD": 1,
    "SET_INDEX": 0,
    "SET_FIELD": 1,
    "CONVERT": 1,
    "JUMP": 1,
    "JUMP_IF_FALSE": 1,
    "JUMP_IF_TRUE": 1,
    "JUMP_IF_FALSE_KEEP": 1,
    "JUMP_IF_TRUE_KEEP": 1,
    "JUMP_IF_SOME": 1,
    "CALL": 1,
    "CALL_KW": 2,
    "CLOSURE": 2,
    "RETURN": 0,
    "RETURN_NONE": 0,
    "ITER_INIT": 0,
    "ITER_NEXT": 3,
    "ITER_BREAK": 1,
    "ITER_END": 0,
    "RANGE_INIT": 0,
    "RANGE_NEXT": 2,
    "BREAK": 1,
    "SCOPE_PUSH": 0,
    "SCOPE_POP": 0,
    "TRY_PUSH": 2,
    "TRY_POP": 0,
    "RAISE": 0,
    "PRINT": 0,
    "RECORD": 1,
    "IMPORT": 2,
    "HALT": 0,
    "LOAD_LOAD": 2,
    "LOAD_PUSH": 2,
    "LOAD_ADD_NN": 1,
    "LOAD_LT_NN": 1,
    "LOAD_FIELD": 2,
    "LOAD_INDEX": 1,
    # NATIVE_LOOP is intentionally not an accepted artifact instruction; it
    # remains in the opcode table for explicitly in-memory execution.
}

_JUMP_OPS = {
    "JUMP",
    "JUMP_IF_FALSE",
    "JUMP_IF_TRUE",
    "JUMP_IF_FALSE_KEEP",
    "JUMP_IF_TRUE_KEEP",
    "JUMP_IF_SOME",
    "ITER_NEXT",
    "ITER_BREAK",
    "RANGE_NEXT",
    "BREAK",
    "TRY_PUSH",
}
_COUNT_OPS = {"MAKE_LIST", "MAKE_MAP", "CALL", "CALL_KW"}
_NAME_SLOTS = {
    "LOAD": (1,),
    "LOAD_FAST": (1,),
    "SET": (1,),
    "SET_FAST": (1,),
    "FIELD": (1,),
    "SET_FIELD": (1,),
    "CONVERT": (1,),
    "RECORD": (1,),
    "LOAD_LOAD": (1, 2),
    "LOAD_ADD_NN": (1,),
    "LOAD_LT_NN": (1,),
    "LOAD_FIELD": (1, 2),
    "LOAD_INDEX": (1,),
    "INC_FAST": (1,),
}
_BINARY_OPS = {"+", "-", "*", "/", "%", "<", "<=", ">", ">=", "==", "!="}
_UNARY_OPS = {"not", "-"}


def _corrupt(name: str, detail: str):
    raise ValueError(f"corrupt bytecode artifact {name}: {detail}")


def _reject_json_constant(value):
    raise ValueError(f"non-finite JSON constant {value}")


def _reject_duplicate_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON object key {key!r}")
        out[key] = value
    return out


def _limited_int(text: str):
    if len(text.lstrip("-")) > _MAX_INTEGER_DIGITS:
        raise ValueError("integer literal in artifact is too long")
    return int(text)


def _safe(v, _depth=0, _active=None):
    """Convert a runtime value to JSON without lossy ``repr`` fallbacks.

    Artifact metadata must round-trip as data.  In particular, an arbitrary
    Python object, a cyclic container, a non-finite float, or a non-string JSON
    object key is rejected instead of becoming text that the VM cannot restore.
    """
    if _depth > _MAX_JSON_DEPTH:
        raise ValueError("bytecode value is nested too deeply to serialize")
    if v is None or isinstance(v, (str, int, bool)):
        if isinstance(v, str) and len(v) > _MAX_ARTIFACT_BYTES:
            raise ValueError("bytecode string is too large to serialize")
        return v
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError("cannot serialize a non-finite number in bytecode")
        return v
    if _active is None:
        _active = set()
    if isinstance(v, (list, tuple)):
        ident = id(v)
        if ident in _active:
            raise ValueError("cannot serialize a cyclic container in bytecode")
        _active.add(ident)
        try:
            if len(v) > _MAX_SEQUENCE_COUNT:
                raise ValueError("bytecode sequence is too large to serialize")
            return [_safe(x, _depth + 1, _active) for x in v]
        finally:
            _active.remove(ident)
    if isinstance(v, dict):
        ident = id(v)
        if ident in _active:
            raise ValueError("cannot serialize a cyclic container in bytecode")
        _active.add(ident)
        try:
            if len(v) > _MAX_SEQUENCE_COUNT:
                raise ValueError("bytecode map is too large to serialize")
            out = {}
            for key in sorted(v, key=lambda x: x if isinstance(x, str) else str(x)):
                if not isinstance(key, str):
                    raise ValueError(
                        "cannot serialize a map with a non-text key in bytecode"
                    )
                if key in out:
                    raise ValueError(f"duplicate bytecode map key {key!r}")
                out[key] = _safe(v[key], _depth + 1, _active)
            return out
        finally:
            _active.remove(ident)
    raise ValueError(f"cannot serialize runtime value {type(v).__name__} in bytecode")


def _fn(f: FunctionCode):
    if not isinstance(f, FunctionCode):
        raise ValueError("cannot serialize a non-FunctionCode function")
    return {
        "name": f.name,
        "params": list(f.params),
        "code": [[_safe(part) for part in ins] for ins in f.code],
        "constants": _safe(f.constants),
        # Extra fidelity for rehydrated code: error line numbers and fields
        # consulted by the runtime/native eligibility code.
        "captures": list(f.captures),
        "defaults": _safe(dict(f.defaults)),
        "line": f.line,
        "lines": {str(k): v for k, v in sorted(f.lines.items())},
    }


def _has_native_loop(code) -> bool:
    """Return whether an instruction stream contains a host-code loop."""
    if not isinstance(code, list):
        return False
    return any(
        isinstance(ins, (list, tuple))
        and ins
        and type(ins[0]) is int
        and ins[0] == _NATIVE_LOOP_OPCODE
        for ins in code
    )


def _validate_json_value(value, name, depth=0):
    """Validate values produced by json.loads and cap recursive structures."""
    if depth > _MAX_JSON_DEPTH:
        _corrupt(name, "JSON value is nested too deeply")
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and len(value) > _MAX_ARTIFACT_BYTES:
            _corrupt(name, "JSON string is too large")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _corrupt(name, "JSON contains a non-finite number")
        return
    if isinstance(value, list):
        if len(value) > _MAX_SEQUENCE_COUNT:
            _corrupt(name, "JSON list is too large")
        for item in value:
            _validate_json_value(item, name, depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > _MAX_SEQUENCE_COUNT:
            _corrupt(name, "JSON object is too large")
        for key, item in value.items():
            if not isinstance(key, str):
                _corrupt(name, "JSON object key is not text")
            _validate_json_value(item, name, depth + 1)
        return
    _corrupt(name, f"unsupported JSON value {type(value).__name__}")


def _int_arg(value, name, index, detail="integer"):
    if type(value) is not int:
        _corrupt(name, f"instruction {index} has a non-{detail} argument")
    return value


def _text_arg(value, name, index, what="text"):
    if not isinstance(value, str) or not value:
        _corrupt(name, f"instruction {index} has a malformed {what} argument")
    return value


def _validate_instruction(ins, name, index, code_len, function_names, record_names):
    if not isinstance(ins, list) or not ins:
        _corrupt(name, f"instruction {index} is not a non-empty list")
    op = ins[0]
    if type(op) is not int or op not in NAMES:
        _corrupt(name, f"instruction {index} has an unknown opcode")
    opname = NAMES[op]
    if opname == "NATIVE_LOOP":
        raise ValueError(
            f"unsafe bytecode artifact {name}: executable native-loop source is not allowed"
        )
    arity = _ARITY.get(opname)
    if arity is None:
        _corrupt(name, f"instruction {index} uses an unsupported opcode {opname}")
    if len(ins) - 1 != arity:
        _corrupt(
            name,
            f"instruction {index} ({opname}) expects {arity} argument(s), "
            f"got {len(ins) - 1}",
        )

    if opname in _JUMP_OPS:
        target = _int_arg(ins[1], name, index, "jump target")
        if not 0 <= target <= code_len:
            _corrupt(name, f"instruction {index} jumps outside its function")
    if opname in _COUNT_OPS:
        count = _int_arg(ins[1], name, index, "count")
        if count < 0 or count > _MAX_SEQUENCE_COUNT:
            _corrupt(name, f"instruction {index} has an invalid count")
    if opname == "CALL_KW":
        names = ins[2]
        if not isinstance(names, list) or len(names) != ins[1] or any(
            n is not None and (not isinstance(n, str) or not n)
            for n in names
        ):
            _corrupt(name, f"instruction {index} has malformed named-argument metadata")
    if opname in _NAME_SLOTS:
        if any(not isinstance(ins[slot], str) or not ins[slot] for slot in _NAME_SLOTS[opname]):
            _corrupt(name, f"instruction {index} has a malformed name argument")
    if opname == "INC_FAST" and type(ins[2]) is not int:
        _corrupt(name, f"instruction {index} has a malformed increment")
    if opname == "ADD_CONST":
        if type(ins[1]) not in (int, float) or isinstance(ins[1], bool) or (
            isinstance(ins[1], float) and not math.isfinite(ins[1])
        ):
            _corrupt(name, f"instruction {index} has a non-numeric constant")
    if opname == "BINARY" and ins[1] not in _BINARY_OPS:
        _corrupt(name, f"instruction {index} has an unknown binary operator")
    if opname == "UNARY" and ins[1] not in _UNARY_OPS:
        _corrupt(name, f"instruction {index} has an unknown unary operator")
    if opname == "CONVERT" and ins[1] not in TYPE_NAMES | set(record_names):
        _corrupt(name, f"instruction {index} has an unknown conversion target")
    if opname == "RECORD" and ins[1] not in record_names:
        _corrupt(name, f"instruction {index} references an unknown record")
    if opname == "IMPORT":
        _text_arg(ins[1], name, index, "module path")
        _text_arg(ins[2], name, index, "module alias")
    if opname == "ITER_NEXT" and (
        not isinstance(ins[2], str) or not ins[2]
        or ins[3] is not None and (not isinstance(ins[3], str) or not ins[3])
    ):
        _corrupt(name, f"instruction {index} has malformed iteration metadata")
    if opname == "RANGE_NEXT" and (not isinstance(ins[2], str) or not ins[2]):
        _corrupt(name, f"instruction {index} has malformed range metadata")
    if opname == "CLOSURE":
        if not isinstance(ins[1], str) or ins[1] not in function_names:
            _corrupt(name, f"instruction {index} references an unknown function")
        if ins[2] is not None and (not isinstance(ins[2], str) or not ins[2]):
            _corrupt(name, f"instruction {index} has malformed closure binding")
    if opname == "TRY_PUSH" and (not isinstance(ins[2], str) or not ins[2]):
        _corrupt(name, f"instruction {index} has malformed rescue binding")


def _validate_function(value, name, function_names, record_names, is_main=False):
    if not isinstance(value, dict):
        _corrupt(name, "function entry is not an object")
    required = {"name", "params", "code", "constants"}
    if not required <= value.keys():
        _corrupt(name, "function entry is missing required fields")
    allowed = required | {"captures", "defaults", "line", "lines"}
    unknown = set(value) - allowed
    if unknown:
        _corrupt(name, f"function entry has unknown field(s): {', '.join(sorted(unknown))}")
    if not isinstance(value["name"], str) or not value["name"]:
        _corrupt(name, "function name is not a non-empty string")
    if is_main and value["name"] != "<main>":
        _corrupt(name, "main function has the wrong name")
    params = value["params"]
    if not isinstance(params, list) or len(params) > _MAX_SEQUENCE_COUNT or any(
        not isinstance(p, str) or not p for p in params
    ):
        _corrupt(name, "function parameters are malformed")
    if len(params) != len(set(params)):
        _corrupt(name, "function parameters contain duplicates")
    code = value["code"]
    if not isinstance(code, list) or len(code) > _MAX_INSTRUCTIONS:
        _corrupt(name, "function code is malformed or too large")
    if not isinstance(value["constants"], list) or len(value["constants"]) > _MAX_SEQUENCE_COUNT:
        _corrupt(name, "function constants are malformed or too large")
    captures = value.get("captures", [])
    if not isinstance(captures, list) or any(not isinstance(c, str) or not c for c in captures):
        _corrupt(name, "function captures are malformed")
    if len(captures) != len(set(captures)):
        _corrupt(name, "function captures contain duplicates")
    defaults = value.get("defaults", {})
    if not isinstance(defaults, dict) or any(
        not isinstance(k, str) or not k or k not in params for k in defaults
    ):
        _corrupt(name, "function defaults are malformed")
    line = value.get("line", 0)
    if type(line) is not int or line < 0:
        _corrupt(name, "function line is malformed")
    lines = value.get("lines", {})
    if not isinstance(lines, dict):
        _corrupt(name, "function line table is not a map")
    for key, line_no in lines.items():
        if (
            not isinstance(key, str)
            or not key.isdigit()
            or int(key) >= len(code)
            or type(line_no) is not int
            or line_no < 0
        ):
            _corrupt(name, "function line table is malformed")
    for index, ins in enumerate(code):
        _validate_instruction(ins, name, index, len(code), function_names, record_names)
    return value


def artifact(program: ProgramCode, source: str | None = None) -> dict:
    """Return a canonical, safe artifact object for ``program``."""
    if not isinstance(program, ProgramCode):
        raise ValueError("cannot serialize a non-ProgramCode program")
    if source is not None and not isinstance(source, str):
        raise ValueError("bytecode source must be Text or None")
    if program.native_loops or _has_native_loop(program.main.code) or any(
        _has_native_loop(fn.code) for fn in program.functions.values()
    ):
        raise ValueError(
            "cannot serialize native loops; compile artifacts with lift_loops=False"
        )

    obj = {
        "format": BYTECODE_FORMAT,
        "language": LANGUAGE,
        "version": VERSION,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest()
        if source is not None
        else None,
        "main": _fn(program.main),
        "functions": {name: _fn(f) for name, f in sorted(program.functions.items())},
        "records": {k: [list(x) for x in v] for k, v in sorted(program.records.items())},
        "imports": [list(i) for i in program.imports],
        # Native loops are deliberately absent from safe artifacts.  Native
        # compilation remains available for source execution in this process;
        # artifacts use ordinary validated VM instructions.
        "native_loops": [],
    }
    raw = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    obj["artifact_sha256"] = hashlib.sha256(raw).hexdigest()
    return obj


def write(program: ProgramCode, path, source: str | None = None) -> dict:
    """Atomically write a deterministic artifact and return its object."""
    obj = artifact(program, source)
    target = Path(path)
    if target.exists() and target.is_dir():
        raise ValueError(f"cannot write bytecode artifact to directory {target}")
    parent = target.parent
    if not parent.is_dir():
        raise ValueError(f"artifact output directory does not exist: {parent}")
    text = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=parent, prefix=f".{target.name}.",
            suffix=".tmp", delete=False
        ) as fh:
            temporary = Path(fh.name)
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, target)
        temporary = None
    except OSError as e:
        raise ValueError(f"cannot write bytecode artifact {target}: {e}") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass
    return obj


def read(path) -> dict:
    """Read and strictly validate an executable artifact.

    Validation happens before rehydration.  The integrity field is a
    tamper/truncation detector, not an authenticity mechanism; applications
    that execute artifacts from an untrusted source should authenticate the
    file at a higher layer as well.
    """
    target = Path(path)
    name = target.name or str(target)
    try:
        raw_bytes = target.read_bytes()
    except OSError as e:
        raise ValueError(f"cannot read bytecode artifact {name}: {e}") from None
    if len(raw_bytes) > _MAX_ARTIFACT_BYTES:
        raise ValueError(
            f"bytecode artifact {name} is too large; maximum is {_MAX_ARTIFACT_BYTES} bytes"
        )
    try:
        text = raw_bytes.decode("utf-8")
        obj = json.loads(
            text,
            parse_int=_limited_int,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as e:
        raise ValueError(f"corrupt bytecode artifact {name}: {e}") from None
    if not isinstance(obj, dict):
        _corrupt(name, "top level is not an object")
    if obj.get("format") != BYTECODE_FORMAT:
        raise ValueError(
            f"unsupported bytecode format {obj.get('format')!r}; expected {BYTECODE_FORMAT}"
        )
    if obj.get("language") != LANGUAGE:
        raise ValueError(
            f"foreign bytecode artifact {name}: language {obj.get('language')!r}, "
            f"expected {LANGUAGE!r}"
        )
    if obj.get("version") != VERSION:
        raise ValueError(
            f"bytecode artifact {name} was built by {LANGUAGE} {obj.get('version')!r}; "
            f"this toolchain is {VERSION}"
        )

    required_top = {
        "format",
        "language",
        "version",
        "source_sha256",
        "main",
        "functions",
        "records",
        "imports",
        "native_loops",
        "artifact_sha256",
    }
    missing = required_top - obj.keys()
    if missing:
        if "main" in missing:
            _corrupt(name, "missing or malformed 'main'")
        _corrupt(name, f"missing required field(s): {', '.join(sorted(missing))}")
    unknown = set(obj) - required_top
    if unknown:
        _corrupt(name, f"unknown top-level field(s): {', '.join(sorted(unknown))}")

    _validate_json_value(obj, name)
    source_hash = obj["source_sha256"]
    if source_hash is not None and (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(c not in "0123456789abcdef" for c in source_hash)
    ):
        _corrupt(name, "source_sha256 is malformed")
    digest = obj["artifact_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(
        c not in "0123456789abcdef" for c in digest
    ):
        _corrupt(name, "artifact_sha256 is malformed")

    # Verify the canonical digest before interpreting any executable metadata.
    # A changed instruction, function name, or operand is reported as tampering
    # first; an attacker cannot make malformed code look like a normal load by
    # relying on an incidental validation-order detail.
    check = {key: value for key, value in obj.items() if key != "artifact_sha256"}
    canonical = json.dumps(
        check,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if not hmac.compare_digest(hashlib.sha256(canonical).hexdigest(), digest):
        raise ValueError(
            f"bytecode artifact {name} failed its integrity check "
            "(artifact_sha256 mismatch - the file was modified or truncated)"
        )

    records = obj["records"]
    if not isinstance(records, dict):
        _corrupt(name, "records is not a map")
    if len(records) > _MAX_FUNCTIONS:
        _corrupt(name, "too many record declarations")
    record_names = set()
    for record_name, fields in records.items():
        if not isinstance(record_name, str) or not record_name:
            _corrupt(name, "record name is malformed")
        record_names.add(record_name)
        if not isinstance(fields, list) or len(fields) > _MAX_SEQUENCE_COUNT:
            _corrupt(name, "record metadata is malformed")
        seen_fields = set()
        for field in fields:
            if (
                not isinstance(field, list)
                or len(field) != 2
                or not all(isinstance(part, str) and part for part in field)
            ):
                _corrupt(name, "record field metadata is malformed")
            if field[0] in seen_fields:
                _corrupt(name, f"record {record_name!r} has duplicate fields")
            seen_fields.add(field[0])

    functions = obj["functions"]
    if not isinstance(functions, dict) or any(not isinstance(key, str) for key in functions):
        _corrupt(name, "functions is not a map of names")
    if len(functions) > _MAX_FUNCTIONS:
        _corrupt(name, "too many functions")
    function_names = set(functions)
    _validate_function(obj["main"], f"{name}:main", function_names, record_names, True)
    for key, fn in functions.items():
        if not isinstance(fn, dict) or key != fn.get("name"):
            _corrupt(name, f"function key {key!r} does not match its name")
        _validate_function(fn, f"{name}:function:{key}", function_names, record_names)

    imports = obj["imports"]
    if not isinstance(imports, list) or len(imports) > _MAX_SEQUENCE_COUNT or any(
        not isinstance(item, list)
        or len(item) != 2
        or not all(isinstance(part, str) and part for part in item)
        for item in imports
    ):
        _corrupt(name, "imports metadata is malformed")
    native_loops = obj["native_loops"]
    if not isinstance(native_loops, list) or native_loops:
        raise ValueError(
            f"unsafe bytecode artifact {name}: executable native-loop source is not allowed"
        )

    return obj


def load(path) -> ProgramCode:
    """Rehydrate a validated artifact into a ``ProgramCode``."""
    obj = read(path)

    def mk(d):
        return FunctionCode(
            d["name"],
            list(d["params"]),
            [tuple(x) for x in d["code"]],
            list(d["constants"]),
            captures=list(d.get("captures", [])),
            defaults=dict(d.get("defaults", {})),
            line=d.get("line", 0),
            lines={int(k): v for k, v in d.get("lines", {}).items()},
        )

    return ProgramCode(
        mk(obj["main"]),
        {key: mk(value) for key, value in obj["functions"].items()},
        {key: [tuple(x) for x in value] for key, value in obj["records"].items()},
        [tuple(x) for x in obj["imports"]],
        [],
    )
