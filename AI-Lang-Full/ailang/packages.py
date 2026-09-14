"""Package management for AI-Lang.

A package is a directory containing `ailang.package.json` and `.al` sources.
Packages are installed into `ai_modules/` beside the project manifest, where
the module loader already looks, so `use http/client as http.` works with no
extra configuration.

Design decisions:

*   **Content addressed.** Every installed package records the SHA-256 of its
    sources in the lockfile. `verify` recomputes them, so a tampered or
    half-written install is detected rather than silently used.
*   **Deterministic.** `ailang.lock.json` pins an exact version and digest for
    every package, direct or transitive, and the resolver walks dependencies
    in sorted order, so the same manifest always produces the same lockfile.
*   **Offline first.** Sources are local directories or a registry directory.
    Nothing here requires a network, which keeps the "runs on any device"
    promise intact.

Version constraints supported: exact (`1.2.3`), caret (`^1.2.3`, same major),
tilde (`~1.2.3`, same minor) and `*` (any).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MANIFEST = "ailang.package.json"
PROJECT = "ailang.project.json"
LOCKFILE = "ailang.lock.json"
MODULES_DIR = "ai_modules"


class PackageError(Exception):
    """A packaging problem stated in terms the user can act on."""


# --------------------------------------------------------------------- version
def parse_version(text: str) -> Tuple[int, int, int]:
    parts = str(text).strip().split(".")
    if len(parts) != 3:
        raise PackageError(f"version '{text}' must look like MAJOR.MINOR.PATCH")
    try:
        return tuple(int(p) for p in parts)  # type: ignore[return-value]
    except ValueError:
        raise PackageError(f"version '{text}' must be three numbers") from None


def satisfies(version: str, constraint: str) -> bool:
    """Does `version` meet `constraint`?"""
    constraint = str(constraint).strip()
    if constraint in ("*", ""):
        return True
    v = parse_version(version)
    if constraint.startswith("^"):
        c = parse_version(constraint[1:])
        return v[0] == c[0] and v >= c
    if constraint.startswith("~"):
        c = parse_version(constraint[1:])
        return v[0] == c[0] and v[1] == c[1] and v >= c
    if constraint.startswith(">="):
        return v >= parse_version(constraint[2:])
    return v == parse_version(constraint)


def best_match(versions: List[str], constraint: str) -> Optional[str]:
    """Highest version satisfying the constraint."""
    ok = [v for v in versions if satisfies(v, constraint)]
    if not ok:
        return None
    return max(ok, key=parse_version)


# ---------------------------------------------------------------- integrity
def digest_dir(path: Path) -> str:
    """SHA-256 over every `.al` file and the manifest, in sorted order.

    Sorting makes the digest independent of filesystem ordering, so the same
    sources always hash the same on every machine.
    """
    h = hashlib.sha256()
    files = sorted(
        [p for p in path.rglob("*.al") if p.is_file()]
        + [p for p in path.glob(MANIFEST)],
        key=lambda p: str(p.relative_to(path)).replace("\\", "/"),
    )
    for f in files:
        h.update(str(f.relative_to(path)).replace("\\", "/").encode("utf-8"))
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return "sha256:" + h.hexdigest()


# ------------------------------------------------------------------ manifests
def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PackageError(f"{path.name} not found at {path}") from None
    except ValueError as e:
        raise PackageError(f"{path} is not valid JSON: {e}") from None


def write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_manifest(pkg_dir: Path) -> dict:
    m = read_json(pkg_dir / MANIFEST)
    for key in ("name", "version"):
        if key not in m:
            raise PackageError(f"{pkg_dir / MANIFEST} is missing '{key}'")
    parse_version(m["version"])
    return m


class Registry:
    """A directory of packages, laid out as `<name>/<version>/`."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def versions(self, name: str) -> List[str]:
        d = self.root / name
        if not d.is_dir():
            return []
        return sorted(
            (p.name for p in d.iterdir() if p.is_dir() and (p / MANIFEST).is_file()),
            key=parse_version,
        )

    def path(self, name: str, version: str) -> Path:
        p = self.root / name / version
        if not (p / MANIFEST).is_file():
            raise PackageError(f"{name} {version} is not in the registry at {self.root}")
        return p

    def publish(self, src: Path) -> Tuple[str, str]:
        m = read_manifest(src)
        name, version = m["name"], m["version"]
        dest = self.root / name / version
        if dest.exists():
            raise PackageError(
                f"{name} {version} is already published; bump the version first"
            )
        dest.mkdir(parents=True)
        for f in sorted(src.rglob("*.al")):
            rel = f.relative_to(src)
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest / rel)
        shutil.copy2(src / MANIFEST, dest / MANIFEST)
        return name, version


# ------------------------------------------------------------------- resolver
def resolve(manifest: dict, registry: Registry) -> Dict[str, dict]:
    """Resolve direct and transitive dependencies to exact versions.

    Walks breadth-first in sorted order so the result is deterministic. A
    conflicting requirement is reported with both constraints named, rather
    than silently picking one.
    """
    resolved: Dict[str, dict] = {}
    constraints: Dict[str, List[Tuple[str, str]]] = {}

    queue: List[Tuple[str, str, str]] = [
        (n, c, manifest.get("name", "<project>"))
        for n, c in sorted(manifest.get("dependencies", {}).items())
    ]

    while queue:
        name, constraint, requester = queue.pop(0)
        constraints.setdefault(name, []).append((constraint, requester))

        available = registry.versions(name)
        if not available:
            raise PackageError(
                f"package '{name}' (required by {requester}) is not in the registry"
            )

        # a version satisfying EVERY constraint gathered so far
        candidates = [
            v
            for v in available
            if all(satisfies(v, c) for c, _ in constraints[name])
        ]
        if not candidates:
            detail = ", ".join(f"{c} (from {r})" for c, r in constraints[name])
            raise PackageError(
                f"cannot satisfy all constraints on '{name}': {detail}; "
                f"available: {', '.join(available)}"
            )
        chosen = max(candidates, key=parse_version)

        if name in resolved and resolved[name]["version"] == chosen:
            continue

        pkg_dir = registry.path(name, chosen)
        sub = read_manifest(pkg_dir)
        resolved[name] = {
            "version": chosen,
            "digest": digest_dir(pkg_dir),
            "dependencies": sub.get("dependencies", {}),
        }
        for dn, dc in sorted(sub.get("dependencies", {}).items()):
            queue.append((dn, dc, f"{name} {chosen}"))

    return resolved


# ------------------------------------------------------------------ installer
def install(project_dir: Path, registry: Registry) -> Dict[str, dict]:
    """Resolve, copy into `ai_modules/`, and write the lockfile."""
    project_dir = Path(project_dir)
    manifest = read_json(project_dir / PROJECT)
    resolved = resolve(manifest, registry)

    modules = project_dir / MODULES_DIR
    modules.mkdir(exist_ok=True)

    for name, info in sorted(resolved.items()):
        src = registry.path(name, info["version"])
        dest = modules / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        actual = digest_dir(dest)
        if actual != info["digest"]:
            raise PackageError(
                f"integrity check failed for {name} {info['version']}: "
                f"expected {info['digest']}, got {actual}"
            )

    write_json(
        project_dir / LOCKFILE,
        {
            "lockfile_version": 1,
            "packages": {
                n: {"version": i["version"], "digest": i["digest"]}
                for n, i in sorted(resolved.items())
            },
        },
    )
    return resolved


def verify(project_dir: Path) -> List[str]:
    """Recompute digests of installed packages. Returns a list of problems."""
    project_dir = Path(project_dir)
    lock_path = project_dir / LOCKFILE
    if not lock_path.is_file():
        return [f"no {LOCKFILE}; run 'ailang install' first"]
    lock = read_json(lock_path)
    problems = []
    for name, info in sorted(lock.get("packages", {}).items()):
        d = project_dir / MODULES_DIR / name
        if not d.is_dir():
            problems.append(f"{name}: not installed")
            continue
        actual = digest_dir(d)
        if actual != info["digest"]:
            problems.append(
                f"{name}: digest mismatch (expected {info['digest'][:19]}..., "
                f"got {actual[:19]}...)"
            )
    return problems
