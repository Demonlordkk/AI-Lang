"""Safe module loading for AI-Lang.

``use math/stats as stats.`` resolves to ``math/stats.al`` relative to the
importing project/search roots.  Modules are cached by their canonical file
path and cycle-safe.  Import names are language identifiers, never arbitrary
filesystem paths, so ``..``, absolute paths, backslashes, and symlink escapes
cannot turn an import into a file-read primitive.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Dict, List

from .errors import AILangError, ImportError_
from .values import Module


_MODULE_PART = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _module_parts(path: str) -> List[str]:
    """Validate and split a source-level module name."""
    if not isinstance(path, str) or not path:
        raise ImportError_("module name must be non-empty Text")
    if path.startswith(("/", "\\")) or "\\" in path:
        raise ImportError_(f"invalid module name '{path}': use identifier segments separated by '/'")
    parts = path.split("/")
    if any(not part or part in {".", ".."} or not _MODULE_PART.fullmatch(part) for part in parts):
        raise ImportError_(
            f"invalid module name '{path}': use identifier segments separated by '/'"
        )
    return parts


def _inside(path: Path, root: Path) -> bool:
    """Whether canonical ``path`` stays below canonical ``root``."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


class ModuleLoader:
    def __init__(self, search_paths: List[Path], globals_factory, fuel=None):
        self.search_paths = [Path(p).resolve() for p in search_paths]
        self.globals_factory = globals_factory
        self.cache: Dict[str, Module] = {}
        self.loading: List[str] = []
        # Module execution mutates the shared cache/loading stack.  A
        # re-entrant lock makes concurrent handlers/tasks see one complete
        # module load and still permits a same-thread import chain.
        self._lock = threading.RLock()
        from .vm import fuel_default

        self.fuel = fuel if fuel is not None else fuel_default()

    def resolve_path(self, path: str) -> Path:
        parts = _module_parts(path)
        rel = Path(*parts)
        candidates = []
        for base in self.search_paths:
            root = base.resolve()
            candidates.extend(
                (
                    base / rel.with_suffix(".al"),
                    base / rel / "main.al",
                    # installed packages live in ai_modules/<name>/...
                    base / "ai_modules" / rel.with_suffix(".al"),
                    base / "ai_modules" / rel / "main.al",
                )
            )
            for candidate in candidates[-4:]:
                try:
                    resolved = candidate.resolve()
                except OSError:
                    continue
                if resolved.is_file() and _inside(resolved, root):
                    return resolved
        tried = "\n  ".join(str(c) for c in candidates)
        raise ImportError_(f"module '{path}' not found; looked in:\n  {tried}")

    def _bundled_source(self, path: str):
        """Read a source package embedded in the standalone zipapp, if any."""
        try:
            from ._bundled_sources import SOURCES
        except ImportError:
            return None
        return SOURCES.get(path)

    def load(self, path: str) -> Module:
        with self._lock:
            return self._load_locked(path)

    def _load_locked(self, path: str) -> Module:
        # Resolve before consulting the cache. Different spellings of a
        # module name must not create duplicate module instances or evade cycle
        # detection. A bundle can provide source through importlib resources
        # when its package files live inside the zipapp rather than a directory.
        file = None
        bundled = None
        try:
            file = self.resolve_path(path)
            key = str(file)
            filename = str(file)
            source = file.read_text(encoding="utf-8")
        except ImportError_ as resolution_error:
            bundled = self._bundled_source(path)
            if bundled is None:
                raise resolution_error
            key = f"<bundle:{path}>"
            filename = key
            source = bundled
        except UnicodeDecodeError as e:
            raise ImportError_(f"module '{path}' ({file}) is not valid UTF-8: {e.reason}") from None
        except OSError as e:
            raise ImportError_(f"cannot read module '{path}' ({file}): {e}") from None
        if key in self.cache:
            return self.cache[key]
        if key in self.loading:
            chain = " -> ".join(self.loading + [key])
            raise ImportError_(f"circular import detected: {chain}")

        from .toolchain import compile_source
        from .vm import VM

        self.loading.append(key)
        try:
            search = [file.parent] + self.search_paths if file is not None else self.search_paths
            program = compile_source(source, filename=filename, search_paths=search)
            child = ModuleLoader(search, self.globals_factory, self.fuel)
            child.cache = self.cache
            child.loading = self.loading
            child._lock = self._lock
            base = self.globals_factory()
            # VM adopts this dict by reference and the module's definitions
            # land in it, so snapshot the builtins beforehand.
            baseline = dict(base)
            vm = VM(base, fuel=self.fuel, module_loader=child.load)
            vm.run(program)
            # Export what the module actually defines. Comparing against the
            # builtin *names* would hide a function that intentionally shadows
            # a builtin; compare identity instead.
            exported = {}
            for name, value in vm.globals.vars.items():
                if name.startswith("_"):
                    continue
                if name in baseline and baseline[name] is value:
                    continue
                exported[name] = value
            module = Module(path, exported)
            self.cache[key] = module
            return module
        except AILangError as e:
            raise ImportError_(f"while importing '{path}' ({filename}): {e}") from e
        finally:
            self.loading.pop()

    def signatures(self, path: str):
        """Return static export signatures used by the type checker."""
        file = None
        try:
            file = self.resolve_path(path)
            source = file.read_text(encoding="utf-8")
            filename = str(file)
        except ImportError_:
            source = self._bundled_source(path)
            if source is None:
                return {}
            filename = f"<bundle:{path}>"
        except UnicodeDecodeError as e:
            raise ImportError_(f"module '{path}' ({file}) is not valid UTF-8: {e.reason}") from None
        except OSError as e:
            raise ImportError_(f"cannot read module '{path}' ({file}): {e}") from None
        from .parser import parse
        from . import ast_nodes as A
        from .typecheck import FnSig, ty

        try:
            program = parse(source)
        except AILangError as e:
            # Never swallow a broken module: surface it at the import site.
            raise ImportError_(
                f"module '{path}' ({filename}) failed to parse: {e}", e.line, e.col
            ) from e
        out = {}
        for statement in program.statements:
            if isinstance(statement, A.Fn):
                out[statement.name] = FnSig(
                    [(name, ty(type_name)) for name, type_name in statement.params],
                    ty(statement.return_type or "Void"),
                    statement.name,
                )
            elif isinstance(statement, A.Record):
                out[statement.name] = FnSig(
                    [(field, ty(type_name)) for field, type_name in statement.fields],
                    ty(statement.name),
                    statement.name,
                )
            elif isinstance(statement, (A.Let, A.Var)):
                out[statement.name] = None
        return out
