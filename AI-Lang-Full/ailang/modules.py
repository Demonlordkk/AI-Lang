"""Module loading for AI-Lang.

`use math/stats as stats.` resolves to `math/stats.al` relative to the
importing file, then the project root. Modules are cached and cycle-safe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .errors import AILangError, ImportError_
from .values import Module


class ModuleLoader:
    def __init__(self, search_paths: List[Path], globals_factory, fuel=50_000_000):
        self.search_paths = [Path(p).resolve() for p in search_paths]
        self.globals_factory = globals_factory
        self.cache: Dict[str, Module] = {}
        self.loading: List[str] = []
        self.fuel = fuel

    def resolve_path(self, path: str) -> Path:
        rel = Path(*path.split("/"))
        candidates = []
        for base in self.search_paths:
            candidates.append(base / rel.with_suffix(".al"))
            candidates.append(base / rel / "main.al")
        for c in candidates:
            if c.is_file():
                return c.resolve()
        tried = "\n  ".join(str(c) for c in candidates)
        raise ImportError_(f"module '{path}' not found; looked in:\n  {tried}")

    def load(self, path: str) -> Module:
        if path in self.cache:
            return self.cache[path]
        if path in self.loading:
            chain = " -> ".join(self.loading + [path])
            raise ImportError_(f"circular import detected: {chain}")

        file = self.resolve_path(path)
        source = file.read_text(encoding="utf-8")

        from .toolchain import compile_source
        from .vm import VM

        self.loading.append(path)
        try:
            program = compile_source(source, filename=str(file), search_paths=[file.parent] + self.search_paths)
            child = ModuleLoader([file.parent] + self.search_paths, self.globals_factory, self.fuel)
            child.cache = self.cache
            child.loading = self.loading
            vm = VM(self.globals_factory(), fuel=self.fuel, module_loader=child.load)
            vm.run(program)
            exported = {
                k: v
                for k, v in vm.globals.vars.items()
                if not k.startswith("_") and k not in self.globals_factory()
            }
            module = Module(path, exported)
            self.cache[path] = module
            return module
        except AILangError as e:
            raise ImportError_(f"while importing '{path}' ({file}): {e}") from e
        finally:
            self.loading.pop()

    def signatures(self, path: str):
        """Static export signatures used by the type checker."""
        try:
            file = self.resolve_path(path)
        except ImportError_:
            return {}
        from .parser import parse
        from . import ast_nodes as A
        from .typecheck import FnSig, ty

        try:
            program = parse(file.read_text(encoding="utf-8"))
        except AILangError as e:
            # Never swallow a broken module: surface it at the import site.
            raise ImportError_(
                f"module '{path}' ({file}) failed to parse: {e}", e.line, e.col
            ) from e
        out = {}
        for s in program.statements:
            if isinstance(s, A.Fn):
                out[s.name] = FnSig(
                    [(n, ty(t)) for n, t in s.params], ty(s.return_type or "Void"), s.name
                )
            elif isinstance(s, A.Record):
                out[s.name] = FnSig([(f, ty(t)) for f, t in s.fields], ty(s.name), s.name)
            elif isinstance(s, (A.Let, A.Var)):
                out[s.name] = None
        return out
