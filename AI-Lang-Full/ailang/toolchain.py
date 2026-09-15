"""High-level AI-Lang toolchain: parse -> check -> optimize -> compile -> run."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .capabilities import normalize as normalize_capabilities, use as use_capabilities
from .compiler import Compiler, ProgramCode
from .contracts import desugar
from .errors import AILangError
from .optimizer import optimize
from .parser import parse
from .resources import scope as resource_scope
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
    lift_loops: bool = True,
) -> ProgramCode:
    program = parse(source)
    # contracts become ordinary checks before anything else looks at the tree
    desugar(program)
    if check:
        resolver = _resolver_for(search_paths or [Path.cwd()])
        TypeChecker(module_resolver=resolver).check(program)
    if opt:
        optimize(program)
    return Compiler(lift_loops=lift_loops).compile(program, filename)


def run_source(
    source: str,
    filename: str = "<source>",
    search_paths: Optional[List] = None,
    argv: Optional[List[str]] = None,
    check: bool = True,
    fuel: Optional[int] = None,
    trace: bool = False,
    capabilities=None,
):
    from .modules import ModuleLoader
    from .vm import fuel_default

    if fuel is None:
        fuel = fuel_default()
    policy = normalize_capabilities(capabilities)
    paths = search_paths or [Path(filename).parent if filename != "<source>" else Path.cwd()]
    # Install the policy while compiling and executing.  This is important for
    # imported modules and ContextVar-aware spawned tasks, not just the root
    # VM's globals.
    with resource_scope():
        with use_capabilities(policy):
            program = compile_source(source, filename, paths, check=check,
                                     lift_loops=not trace)
            loader = ModuleLoader(paths, lambda: build_globals(argv, policy), fuel)
            vm = VM(build_globals(argv, policy), fuel=fuel, module_loader=loader.load,
                    trace=trace, trace_lines=source.splitlines())
            vm.run(program)
            return vm


def _project_root(start: Path):
    """Nearest ancestor directory holding the project marker, if any.

    A project is rooted where `ailang.project.json` lives; packages in
    `packages/` at that root are importable from anywhere inside it.
    """
    for d in (start, *start.parents):
        if (d / "ailang.project.json").is_file():
            return d
    return None


def run_file(path, argv=None, check=True, fuel=None, trace=False, capabilities=None):
    p = Path(path)
    source = p.read_text(encoding="utf-8")
    # relative paths inside the program resolve against the program's own
    # directory, so a script behaves the same no matter where it is run from
    from .stdlib import script_dir, set_script_dir

    # script dir is process-global; save/restore so one run never leaks its
    # base directory into the next (tests run many programs in one process)
    prev = script_dir()
    set_script_dir(p.parent.resolve())
    try:
        paths = [p.parent.resolve(), Path.cwd()]
        root = _project_root(p.parent.resolve())
        if root is not None and root not in paths:
            paths.append(root)
        return run_source(source, str(p), paths, argv, check, fuel, trace=trace,
                          capabilities=capabilities)
    finally:
        set_script_dir(prev)


def run_artifact(path, argv=None, fuel=None, capabilities=None,
                 signing_key=None, require_signature=False):
    """Execute a compiled .albc.json artifact directly.

    The artifact is rehydrated into a ProgramCode and run through the same
    VM/module setup as source, so a built file and its source behave
    identically (including `use` resolution relative to the artifact's
    directory and the enclosing project root).
    """
    from .bytecode import load
    from .modules import ModuleLoader
    from .stdlib import build_globals, script_dir, set_script_dir
    from .vm import VM

    p = Path(path)
    program = load(p, signing_key=signing_key, require_signature=require_signature)
    policy = normalize_capabilities(capabilities)
    prev = script_dir()
    set_script_dir(p.parent.resolve())
    try:
        with resource_scope():
            with use_capabilities(policy):
                paths = [p.parent.resolve(), Path.cwd()]
                root = _project_root(p.parent.resolve())
                if root is not None and root not in paths:
                    paths.append(root)
                loader = ModuleLoader(paths, lambda: build_globals(argv, policy), fuel)
                vm = VM(build_globals(argv, policy), fuel=fuel, module_loader=loader.load)
                vm.run(program)
                return vm
    finally:
        set_script_dir(prev)


def eval_source(source: str, search_paths=None):
    """Compile and run, returning the VM for inspection (used by tests)."""
    return run_source(source, "<eval>", search_paths or [Path.cwd()])
