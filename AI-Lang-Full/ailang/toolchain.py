"""High-level AI-Lang toolchain: parse -> check -> optimize -> compile -> run."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .compiler import Compiler, ProgramCode
from .errors import AILangError
from .optimizer import optimize
from .parser import parse
from .stdlib import build_globals
from .typecheck import TypeChecker
from .vm import VM


def _resolver_for(search_paths):
    from .modules import ModuleLoader

    loader = ModuleLoader(search_paths, build_globals)

    def resolve(path):
        return loader.signatures(path)

    return resolve


def compile_source(
    source: str,
    filename: str = "<source>",
    search_paths: Optional[List] = None,
    check: bool = True,
    opt: bool = True,
) -> ProgramCode:
    program = parse(source)
    if check:
        resolver = _resolver_for(search_paths or [Path.cwd()])
        TypeChecker(module_resolver=resolver).check(program)
    if opt:
        optimize(program)
    return Compiler().compile(program)


def run_source(
    source: str,
    filename: str = "<source>",
    search_paths: Optional[List] = None,
    argv: Optional[List[str]] = None,
    check: bool = True,
    fuel: int = 50_000_000,
):
    from .modules import ModuleLoader

    paths = search_paths or [Path(filename).parent if filename != "<source>" else Path.cwd()]
    program = compile_source(source, filename, paths, check=check)
    loader = ModuleLoader(paths, lambda: build_globals(argv), fuel)
    vm = VM(build_globals(argv), fuel=fuel, module_loader=loader.load)
    vm.run(program)
    return vm


def run_file(path, argv=None, check=True, fuel=50_000_000):
    p = Path(path)
    source = p.read_text(encoding="utf-8")
    # relative paths inside the program resolve against the program's own
    # directory, so a script behaves the same no matter where it is run from
    from .stdlib import set_script_dir

    set_script_dir(p.parent.resolve())
    return run_source(source, str(p), [p.parent.resolve(), Path.cwd()], argv, check, fuel)


def eval_source(source: str, search_paths=None):
    """Compile and run, returning the VM for inspection (used by tests)."""
    return run_source(source, "<eval>", search_paths or [Path.cwd()])
