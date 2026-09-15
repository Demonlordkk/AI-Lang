"""AI-Lang — a concise, general-purpose programming language.

Public API:
    import ailang
    ailang.run('emit "hello".')
    ailang.check(source)      # static analysis only
    ailang.compile(source)    # -> ProgramCode
"""

from __future__ import annotations

from .capabilities import CapabilitySet
from .errors import (
    AILangError,
    AILangRaise,
    CheckError,
    CapabilityError,
    CompileError,
    LexError,
    ParseError,
    VMError,
)
from .version import BYTECODE_FORMAT, LANGUAGE, VERSION

__all__ = [
    "run",
    "check",
    "compile",
    "parse",
    "evaluate",
    "LANGUAGE",
    "VERSION",
    "BYTECODE_FORMAT",
    "AILangError",
    "CapabilitySet",
    "LexError",
    "ParseError",
    "CheckError",
    "CapabilityError",
    "CompileError",
    "VMError",
    "AILangRaise",
]


def parse(source: str):
    from .parser import parse as _parse

    return _parse(source)


def check(source: str, search_paths=None):
    from pathlib import Path

    from .toolchain import _resolver_for
    from .typecheck import TypeChecker

    program = parse(source)
    TypeChecker(module_resolver=_resolver_for(search_paths or [Path.cwd()])).check(program)
    return True


def compile(source: str, **kw):  # noqa: A001 - deliberate public name
    from .toolchain import compile_source

    return compile_source(source, **kw)


def run(source: str, **kw):
    from .toolchain import run_source

    return run_source(source, **kw)


def evaluate(source: str, **kw):
    """Run source and return the value of the final expression statement."""
    from .toolchain import run_source

    vm = run_source(source, **kw)
    return vm
