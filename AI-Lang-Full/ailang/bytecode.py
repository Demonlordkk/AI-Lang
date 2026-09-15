"""Deterministic bytecode serialisation for AI-Lang.

The artifact is canonical JSON: the same source always produces byte-identical
output, which makes builds reproducible and cacheable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .compiler import FunctionCode, ProgramCode
from .version import BYTECODE_FORMAT, LANGUAGE, VERSION


def _safe(v):
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _safe(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    return repr(v)


def _fn(f: FunctionCode):
    return {
        "name": f.name,
        "params": list(f.params),
        "code": [[_safe(part) for part in ins] for ins in f.code],
        "constants": _safe(f.constants),
    }


def artifact(program: ProgramCode, source: str = None) -> dict:
    obj = {
        "format": BYTECODE_FORMAT,
        "language": LANGUAGE,
        "version": VERSION,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest() if source is not None else None,
        "main": _fn(program.main),
        "functions": {name: _fn(f) for name, f in sorted(program.functions.items())},
        "records": {k: [list(x) for x in v] for k, v in sorted(program.records.items())},
        "imports": [list(i) for i in program.imports],
    }
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    obj["artifact_sha256"] = hashlib.sha256(raw).hexdigest()
    return obj


def write(program: ProgramCode, path, source: str = None) -> dict:
    obj = artifact(program, source)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return obj


def read(path) -> dict:
    """Load and strictly validate an artifact.

    Rejects (with a clear ValueError, not a KeyError/TypeError/JSONDecodeError):
    truncated JSON, non-object top levels, foreign/unsupported formats,
    foreign languages, mismatching toolchain versions, malformed 'main'
    entries, and tampered files (artifact_sha256 mismatch).
    """
    name = Path(path).name
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"corrupt bytecode artifact {name}: {e}") from None
    if not isinstance(obj, dict):
        raise ValueError(f"corrupt bytecode artifact {name}: top level is not an object")
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
    main = obj.get("main")
    if not isinstance(main, dict) or not {"name", "params", "code", "constants"} <= main.keys():
        raise ValueError(f"corrupt bytecode artifact {name}: missing or malformed 'main'")
    if "artifact_sha256" in obj:
        check = {k: v for k, v in obj.items() if k != "artifact_sha256"}
        raw = json.dumps(check, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode()
        if hashlib.sha256(raw).hexdigest() != obj["artifact_sha256"]:
            raise ValueError(
                f"bytecode artifact {name} failed its integrity check "
                "(artifact_sha256 mismatch - the file was modified or truncated)"
            )
    return obj


def load(path) -> ProgramCode:
    """Rehydrate a ProgramCode so a built artifact can be executed directly."""
    obj = read(path)

    def mk(d):
        return FunctionCode(
            d["name"], list(d["params"]), [tuple(x) for x in d["code"]], list(d["constants"])
        )

    return ProgramCode(
        mk(obj["main"]),
        {k: mk(v) for k, v in obj["functions"].items()},
        {k: [tuple(x) for x in v] for k, v in obj.get("records", {}).items()},
        [tuple(x) for x in obj.get("imports", [])],
    )
