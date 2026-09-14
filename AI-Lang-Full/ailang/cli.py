"""`ailang` command line interface."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .errors import AILangError
from .version import LANGUAGE, VERSION

BANNER = f"{LANGUAGE} {VERSION}"


def _fail(exc: AILangError, source=None, filename="<source>"):
    print(exc.render(source, filename), file=sys.stderr)
    return 1


def cmd_run(args):
    from .toolchain import run_file

    path = Path(args.file)
    if not path.is_file():
        print(f"ailang: no such file: {path}", file=sys.stderr)
        return 1
    source = path.read_text(encoding="utf-8")
    try:
        run_file(path, argv=args.args, check=not args.no_check)
        return 0
    except AILangError as e:
        return _fail(e, source, str(path))
    except RecursionError:
        print("ailang: recursion limit exceeded", file=sys.stderr)
        return 1


def cmd_check(args):
    from .toolchain import compile_source

    path = Path(args.file)
    source = path.read_text(encoding="utf-8")
    try:
        compile_source(source, str(path), [path.parent.resolve(), Path.cwd()])
    except AILangError as e:
        return _fail(e, source, str(path))
    print(f"{path}: no problems found")
    return 0


def cmd_build(args):
    from .bytecode import write
    from .toolchain import compile_source

    path = Path(args.file)
    source = path.read_text(encoding="utf-8")
    try:
        program = compile_source(source, str(path), [path.parent.resolve(), Path.cwd()])
    except AILangError as e:
        return _fail(e, source, str(path))
    out = Path(args.output) if args.output else path.with_suffix(".albc.json")
    obj = write(program, out, source)
    print(f"{out}  ({obj['artifact_sha256'][:16]})")
    return 0



def _project_root(start=None):
    """Nearest ancestor directory containing the project manifest."""
    from .packages import PROJECT

    d = Path(start or Path.cwd()).resolve()
    for cand in [d, *d.parents]:
        if (cand / PROJECT).is_file():
            return cand
    return None


def _registry(args):
    from .packages import Registry

    if getattr(args, "registry", None):
        return Registry(Path(args.registry))
    env = os.environ.get("AILANG_REGISTRY")
    if env:
        return Registry(Path(env))
    return Registry(Path.home() / ".ailang" / "registry")


def cmd_install(args):
    from .packages import PackageError, install

    root = _project_root()
    if root is None:
        print("ailang: no ailang.project.json found", file=sys.stderr)
        return 1
    try:
        resolved = install(root, _registry(args))
    except PackageError as e:
        print(f"ailang: {e}", file=sys.stderr)
        return 1
    if not resolved:
        print("no dependencies to install")
        return 0
    for name, info in sorted(resolved.items()):
        print(f"  installed {name} {info['version']}")
    print(f"{len(resolved)} package(s) installed into ai_modules/")
    return 0


def cmd_verify(args):
    from .packages import verify

    root = _project_root()
    if root is None:
        print("ailang: no ailang.project.json found", file=sys.stderr)
        return 1
    problems = verify(root)
    if problems:
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print(f"{len(problems)} problem(s)", file=sys.stderr)
        return 1
    print("all packages verified")
    return 0


def cmd_publish(args):
    from .packages import PackageError

    try:
        name, version = _registry(args).publish(Path(args.dir))
    except PackageError as e:
        print(f"ailang: {e}", file=sys.stderr)
        return 1
    print(f"published {name} {version}")
    return 0


def cmd_add(args):
    """Record a dependency in the project manifest, then install."""
    from .packages import PROJECT, PackageError, read_json, write_json

    root = _project_root()
    if root is None:
        print("ailang: no ailang.project.json found", file=sys.stderr)
        return 1
    manifest = read_json(root / PROJECT)
    deps = manifest.setdefault("dependencies", {})
    deps[args.name] = args.constraint
    write_json(root / PROJECT, manifest)
    print(f"added {args.name} {args.constraint}")
    return cmd_install(args)


def cmd_fmt(args):
    from .format import format_source

    path = Path(args.file)
    source = path.read_text(encoding="utf-8")
    formatted = format_source(source)
    if args.check:
        if formatted != source:
            print(f"{path}: needs formatting")
            return 1
        print(f"{path}: already formatted")
        return 0
    if formatted != source:
        path.write_text(formatted, encoding="utf-8")
        print(f"{path}: formatted")
    else:
        print(f"{path}: unchanged")
    return 0


def cmd_lint(args):
    from .format import lint

    path = Path(args.file)
    source = path.read_text(encoding="utf-8")
    issues = lint(source)
    for line, message in issues:
        print(f"{path}:{line}: {message}")
    if issues:
        print(f"{len(issues)} issue(s)", file=sys.stderr)
        return 1
    print(f"{path}: clean")
    return 0


def cmd_test(args):
    """Run every `fn test_*` in a file and report pass/fail."""
    from .stdlib import build_globals
    from .toolchain import compile_source
    from .vm import VM, Closure
    from .modules import ModuleLoader

    path = Path(args.file)
    source = path.read_text(encoding="utf-8")
    try:
        program = compile_source(source, str(path), [path.parent.resolve(), Path.cwd()])
    except AILangError as e:
        return _fail(e, source, str(path))
    loader = ModuleLoader([path.parent.resolve(), Path.cwd()], build_globals)
    vm = VM(build_globals(), module_loader=loader.load)
    vm.run(program)

    tests = [
        (name, value)
        for name, value in vm.globals.vars.items()
        if name.startswith("test_") and isinstance(value, Closure) and value.arity == 0
    ]
    if not tests:
        print("no tests found (define functions named test_*)")
        return 0
    passed = failed = 0
    start = time.perf_counter()
    for name, fn in sorted(tests):
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AILangError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
    elapsed = time.perf_counter() - start
    print(f"\n{passed} passed, {failed} failed in {elapsed:.3f}s")
    return 1 if failed else 0


def cmd_repl(args):
    """Interactive session that keeps bindings alive across lines."""
    from .compiler import Compiler
    from .modules import ModuleLoader
    from .optimizer import optimize
    from .parser import parse
    from .stdlib import build_globals
    from .typecheck import TypeChecker, ANY
    from .vm import VM

    print(f"{BANNER} — type 'exit.' to leave, ':help' for help")
    base_globals = build_globals()
    loader = ModuleLoader([Path.cwd()], build_globals)
    vm = VM(base_globals, module_loader=loader.load)

    # one persistent checker so `let x := 1.` is visible on the next line
    checker = TypeChecker()
    buffer = []

    while True:
        try:
            line = input("... " if buffer else ">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        stripped = line.strip()
        if not buffer:
            if stripped in ("exit.", "exit", "quit.", "quit", ":q"):
                return 0
            if stripped == ":help":
                print("  Enter statements ending in '.'  Multi-line blocks end with 'done.'")
                print("  :vars   list bindings      :clear  reset session      exit.   quit")
                continue
            if stripped == ":vars":
                hidden = set(base_globals) | {"true", "false", "nothing"}
                for k in sorted(vm.globals.vars):
                    if k not in hidden:
                        from .values import display

                        print(f"  {k} = {display(vm.globals.vars[k])}")
                continue
            if stripped == ":clear":
                vm = VM(build_globals(), module_loader=loader.load)
                checker = TypeChecker()
                buffer = []
                print("session cleared")
                continue

        buffer.append(line)
        text = "\n".join(buffer)

        if _needs_more(text):
            continue
        buffer = []
        if not text.strip():
            continue

        try:
            program = parse(text)
            checker.diagnostics = []
            checker.hoist(program.statements, checker.global_scope)
            for st in program.statements:
                checker.stmt(st)
            if checker.diagnostics:
                checker.diagnostics.sort(key=lambda d: (d[0], d[1]))
                ln, col, msg = checker.diagnostics[0]
                checker.diagnostics = []
                print(f"<repl>:{ln}:{col}: type error: {msg}", file=sys.stderr)
                continue
            optimize(program)
            compiled = Compiler().compile(program)
            # merge newly compiled functions/records into the live program
            if vm.program is None:
                vm.program = compiled
            else:
                vm.program.functions.update(compiled.functions)
                vm.program.records.update(compiled.records)
                compiled.functions = vm.program.functions
                compiled.records = vm.program.records
            for rname, rfields in compiled.records.items():
                if rname not in vm.globals.vars:
                    from .values import RecordType

                    vm.globals.vars[rname] = RecordType(rname, rfields)
            vm.execute(compiled.main, vm.globals, compiled)
        except AILangError as e:
            print(e.render(text, "<repl>"), file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"<repl>: {e}", file=sys.stderr)


def _needs_more(text: str) -> bool:
    """True when the buffered source has unclosed blocks or brackets."""
    from .errors import AILangError
    from .lexer import lex

    try:
        tokens = lex(text)
    except AILangError:
        return not text.rstrip().endswith(".")
    depth = 0
    openers = 0
    dones = 0
    for t in tokens:
        if t.kind in ("LPAREN", "LBRACKET", "LBRACE"):
            depth += 1
        elif t.kind in ("RPAREN", "RBRACKET", "RBRACE"):
            depth -= 1
        elif t.kind in ("FN", "WHEN", "REPEAT", "WHILE", "RECORD", "ATTEMPT"):
            openers += 1
        elif t.kind == "DONE":
            dones += 1
    if depth > 0:
        return True
    if openers > dones:
        return True
    return not text.rstrip().endswith(".")


def cmd_version(args):
    print(BANNER)
    return 0


def build_parser():
    ap = argparse.ArgumentParser(prog="ailang", description=f"{LANGUAGE} toolchain")
    ap.add_argument("--version", action="version", version=BANNER)
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("run", help="run a .al program")
    p.add_argument("file")
    p.add_argument("args", nargs="*", help="arguments passed to the program")
    p.add_argument("--no-check", action="store_true", help="skip static checking")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("check", help="type-check without running")
    p.add_argument("file")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("build", help="compile to a bytecode artifact")
    p.add_argument("file")
    p.add_argument("-o", "--output")
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("fmt", help="format source in place")
    p.add_argument("file")
    p.add_argument("--check", action="store_true", help="report instead of writing")
    p.set_defaults(fn=cmd_fmt)

    p = sub.add_parser("lint", help="report style and correctness issues")
    p.add_argument("file")
    p.set_defaults(fn=cmd_lint)

    p = sub.add_parser("test", help="run test_* functions")
    p.add_argument("file")
    p.set_defaults(fn=cmd_test)

    p = sub.add_parser("install", help="install dependencies into ai_modules/")
    p.add_argument("--registry")
    p.set_defaults(fn=cmd_install)

    p = sub.add_parser("add", help="add a dependency and install it")
    p.add_argument("name")
    p.add_argument("constraint", nargs="?", default="*")
    p.add_argument("--registry")
    p.set_defaults(fn=cmd_add)

    p = sub.add_parser("verify", help="check installed packages against the lockfile")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("publish", help="publish a package directory to the registry")
    p.add_argument("dir")
    p.add_argument("--registry")
    p.set_defaults(fn=cmd_publish)

    p = sub.add_parser("repl", help="start an interactive session")
    p.set_defaults(fn=cmd_repl)

    p = sub.add_parser("version", help="print the version")
    p.set_defaults(fn=cmd_version)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "cmd", None):
        ap.print_help()
        return 0
    try:
        return args.fn(args)
    except AILangError as e:
        print(f"ailang: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"ailang: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
