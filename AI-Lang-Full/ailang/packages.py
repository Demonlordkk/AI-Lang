"""Offline-first package management for AI-Lang.

Packages are directories containing ``ailang.package.json`` and AI-Lang source
files.  The resolver writes an exact lockfile with content digests; optional
HMAC signatures authenticate registry contents.  All filesystem names are
validated before being joined to a registry/project path, and installs use a
staging directory so an interrupted copy is never mistaken for a complete
package.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MANIFEST = "ailang.package.json"
PROJECT = "ailang.project.json"
LOCKFILE = "ailang.lock.json"
MODULES_DIR = "ai_modules"
SIGNATURE = "signature.json"
SIGNING_KEY_ENV = "AILANG_REGISTRY_KEY"

_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_VERSION_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_PACKAGE_FILE_BYTES = 64 * 1024 * 1024
_MAX_PACKAGE_FILES = 100_000


class PackageError(Exception):
    """A packaging problem stated in terms the user can act on."""


# --------------------------------------------------------------------- checks
def _package_name(value, what="package name") -> str:
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        raise PackageError(
            f"{what} {value!r} must contain only letters, numbers, '_' or '-' "
            "and start with a letter"
        )
    return value


def _version_text(value) -> str:
    if not isinstance(value, str) or not _VERSION_RE.fullmatch(value):
        raise PackageError(f"version {value!r} must look like MAJOR.MINOR.PATCH")
    return value


def _dependency_map(value, where) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PackageError(f"{where} dependencies must be a map")
    out = {}
    for name, constraint in value.items():
        _package_name(name, "dependency name")
        if not isinstance(constraint, str) or not constraint.strip():
            raise PackageError(f"dependency '{name}' has an invalid version constraint")
        # Parse now so a malformed constraint cannot sit in a lock resolution
        # queue and fail much later with an unhelpful message.
        _constraint_parts(constraint)
        out[name] = constraint.strip()
    return out


# --------------------------------------------------------------------- version
def parse_version(text: str) -> Tuple[int, int, int]:
    raw = _version_text(text)
    return tuple(int(part) for part in raw.split("."))  # type: ignore[return-value]


def _constraint_parts(constraint: str):
    raw = str(constraint).strip()
    if raw in ("", "*"):
        return "*", None
    if raw.startswith(">="):
        return ">=", parse_version(raw[2:])
    if raw[:1] in {"^", "~"}:
        return raw[0], parse_version(raw[1:])
    return "=", parse_version(raw)


def satisfies(version: str, constraint: str) -> bool:
    """Return whether a concrete version meets a supported constraint."""
    v = parse_version(version)
    kind, c = _constraint_parts(constraint)
    if kind == "*":
        return True
    if kind == "^":
        return v[0] == c[0] and v >= c
    if kind == "~":
        return v[0] == c[0] and v[1] == c[1] and v >= c
    if kind == ">=":
        return v >= c
    return v == c


def best_match(versions: List[str], constraint: str) -> Optional[str]:
    """Return the highest valid version satisfying ``constraint``."""
    ok = [version for version in versions if satisfies(version, constraint)]
    return max(ok, key=parse_version) if ok else None


# ---------------------------------------------------------------- integrity
def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _signable_files(pkg_dir: Path):
    """Return manifest and source files in deterministic relative order.

    Symlinks are rejected rather than followed.  A symlink in a package could
    otherwise make a digest/signature depend on files outside the package and
    could copy an unexpected file during installation.
    """
    pkg_dir = Path(pkg_dir)
    if not pkg_dir.is_dir():
        raise PackageError(f"package directory does not exist: {pkg_dir}")
    files = []
    try:
        for path in pkg_dir.rglob("*"):
            if path.is_symlink():
                raise PackageError(f"package contains unsupported symlink: {_relative(path, pkg_dir)}")
            if path.is_file() and path.suffix == ".al":
                if path.stat().st_size > _MAX_PACKAGE_FILE_BYTES:
                    raise PackageError(
                        f"package file is too large: {_relative(path, pkg_dir)}"
                    )
                files.append(path)
        if len(files) > _MAX_PACKAGE_FILES:
            raise PackageError("package contains too many source files")
        manifest = pkg_dir / MANIFEST
        if manifest.is_symlink():
            raise PackageError(f"package manifest is a symlink: {manifest}")
        if not manifest.is_file():
            raise PackageError(f"{manifest} not found")
        if manifest.stat().st_size > _MAX_JSON_BYTES:
            raise PackageError(f"package manifest is too large: {manifest}")
        files.append(manifest)
        if len(files) > _MAX_PACKAGE_FILES:
            raise PackageError("package contains too many source files")
    except OSError as e:
        raise PackageError(f"cannot inspect package {pkg_dir}: {e}") from None
    return sorted(files, key=lambda path: _relative(path, pkg_dir))


def _read_package_file(path: Path) -> bytes:
    try:
        data = Path(path).read_bytes()
    except OSError as e:
        raise PackageError(f"cannot read package file {path}: {e}") from None
    if len(data) > _MAX_PACKAGE_FILE_BYTES:
        raise PackageError(f"package file is too large: {path}")
    return data


def digest_dir(path: Path) -> str:
    """SHA-256 over every source and the manifest, in relative-path order."""
    root = Path(path)
    h = hashlib.sha256()
    for file in _signable_files(root):
        h.update(_relative(file, root).encode("utf-8"))
        h.update(b"\0")
        h.update(_read_package_file(file))
        h.update(b"\0")
    return "sha256:" + h.hexdigest()


# ------------------------------------------------------------------ manifests
def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _json_load(path: Path):
    path = Path(path)
    try:
        raw = path.read_bytes()
        if len(raw) > _MAX_JSON_BYTES:
            raise ValueError(f"JSON file exceeds {_MAX_JSON_BYTES} bytes")
        text = raw.decode("utf-8")
        return json.loads(
            text,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except FileNotFoundError:
        raise PackageError(f"{path.name} not found at {path}") from None
    except (OSError, UnicodeDecodeError) as e:
        raise PackageError(f"cannot read {path}: {e}") from None
    except (ValueError, json.JSONDecodeError, RecursionError) as e:
        raise PackageError(f"{path} is not valid JSON: {e}") from None


def read_json(path: Path) -> dict:
    value = _json_load(Path(path))
    if not isinstance(value, dict):
        raise PackageError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, data: dict) -> None:
    if not isinstance(data, dict):
        raise PackageError(f"{path} must contain a JSON object")
    target = Path(path)
    if target.exists() and target.is_dir():
        raise PackageError(f"cannot write JSON to directory {target}")
    parent = target.parent
    if not parent.is_dir():
        raise PackageError(f"directory does not exist: {parent}")
    text = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=parent, prefix=f".{target.name}.",
            suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    except OSError as e:
        raise PackageError(f"cannot write {target}: {e}") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


def read_manifest(pkg_dir: Path) -> dict:
    root = Path(pkg_dir)
    manifest = root / MANIFEST
    if manifest.is_symlink():
        raise PackageError(f"package manifest is a symlink: {manifest}")
    m = read_json(manifest)
    if "name" not in m:
        raise PackageError(f"{root / MANIFEST} is missing 'name'")
    if "version" not in m:
        raise PackageError(f"{root / MANIFEST} is missing 'version'")
    name = _package_name(m["name"])
    version = _version_text(m["version"])
    deps = _dependency_map(m.get("dependencies", {}), str(root / MANIFEST))
    # Return a normalized copy so callers never operate on an arbitrary
    # mutable object from a manifest parser.
    out = dict(m)
    out["name"] = name
    out["version"] = version
    out["dependencies"] = deps
    return out


def _signed_payload(name: str, version: str, pkg_dir: Path) -> bytes:
    name = _package_name(name)
    version = _version_text(version)
    manifest = read_manifest(pkg_dir)
    if manifest["name"] != name or manifest["version"] != version:
        raise PackageError("package manifest does not match its registry path")
    files = {
        _relative(file, Path(pkg_dir)): hashlib.sha256(_read_package_file(file)).hexdigest()
        for file in _signable_files(Path(pkg_dir))
    }
    doc = {"name": name, "version": version, "manifest": manifest, "files": files}
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sign_payload(payload: bytes, key: str) -> str:
    if not isinstance(payload, bytes) or not isinstance(key, str) or not key:
        raise PackageError("signing needs bytes payload and a non-empty text key")
    return hmac.new(key.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def verify_signature(payload: bytes, sig: str, key: str) -> bool:
    if not isinstance(sig, str) or not re.fullmatch(r"[0-9a-f]{64}", sig):
        return False
    try:
        expected = sign_payload(payload, key)
    except PackageError:
        return False
    return hmac.compare_digest(expected, sig)


# ------------------------------------------------------------------ registry
def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _copy_signable(src: Path, dest: Path, include_signature=True):
    """Copy only package-owned files, never arbitrary registry extras."""
    src = Path(src)
    dest = Path(dest)
    for file in _signable_files(src):
        rel = file.relative_to(src)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(file, target)
        except OSError as e:
            raise PackageError(f"cannot copy package file {file}: {e}") from None
    sig = src / SIGNATURE
    if include_signature and sig.is_file():
        if sig.is_symlink():
            raise PackageError("package signature is a symlink")
        shutil.copy2(sig, dest / SIGNATURE)


class Registry:
    """A local registry laid out as ``<name>/<version>/``."""

    def __init__(self, root: Path, key: Optional[str] = None):
        self.root = Path(root).resolve()
        self.key = key

    def _component_path(self, name: str, version: str) -> Path:
        name = _package_name(name)
        version = _version_text(version)
        path = self.root / name / version
        if not _inside(path.resolve(strict=False), self.root):
            raise PackageError("registry path escapes the registry root")
        return path

    def signature_path(self, name: str, version: str) -> Path:
        return self._component_path(name, version) / SIGNATURE

    def check_signature(self, name: str, version: str) -> Tuple[str, str]:
        path = self.signature_path(name, version)
        if path.is_symlink():
            return "invalid", "publisher signature is a symlink"
        if not path.is_file():
            return "unsigned", "no publisher signature"
        sig = read_json(path)
        if sig.get("alg") != "hmac-sha256":
            return "invalid", "unsupported signature algorithm"
        publisher = sig.get("publisher") or "unknown"
        if self.key is None:
            return "unverifiable", f"signed by {publisher}; no key set"
        payload = _signed_payload(name, version, self.path(name, version))
        if verify_signature(payload, sig.get("sig", ""), self.key):
            return "ok", f"signature by {publisher} verified"
        return "invalid", f"signature by {publisher} does not match (wrong key or tampered package)"

    def publish(self, src: Path, publisher: Optional[str] = None) -> Tuple[str, str]:
        src = Path(src)
        m = read_manifest(src)
        name, version = m["name"], m["version"]
        dest = self._component_path(name, version)
        if dest.exists():
            raise PackageError(f"{name} {version} is already published; bump the version first")
        self.root.mkdir(parents=True, exist_ok=True)
        package_parent = self.root / name
        package_parent.mkdir(parents=True, exist_ok=True)
        stage = package_parent / f".{version}.{os.getpid()}.{time.time_ns()}.tmp"
        try:
            stage.mkdir(parents=True)
            _copy_signable(src, stage, include_signature=False)
            if self.key is not None:
                payload = _signed_payload(name, version, stage)
                write_json(
                    stage / SIGNATURE,
                    {
                        "alg": "hmac-sha256",
                        "publisher": publisher or m.get("publisher", ""),
                        "created": int(time.time()),
                        "sig": sign_payload(payload, self.key),
                    },
                )
            os.replace(stage, dest)
            stage = None
        except (OSError, PackageError) as e:
            raise PackageError(f"could not publish {name} {version}: {e}") from None
        finally:
            if stage is not None:
                shutil.rmtree(stage, ignore_errors=True)
        return name, version

    def versions(self, name: str) -> List[str]:
        try:
            name = _package_name(name)
        except PackageError:
            return []
        directory = self.root / name
        if not directory.is_dir() or directory.is_symlink():
            return []
        out = []
        try:
            candidates = list(directory.iterdir())
        except OSError:
            return []
        for candidate in candidates:
            if not candidate.is_dir() or candidate.is_symlink():
                continue
            try:
                manifest = read_manifest(candidate)
                if manifest["name"] == name:
                    out.append(manifest["version"])
            except PackageError:
                continue
        return sorted(set(out), key=parse_version)

    def path(self, name: str, version: str) -> Path:
        path = self._component_path(name, version)
        if path.is_symlink() or not (path / MANIFEST).is_file():
            raise PackageError(f"{name} {version} is not in the registry at {self.root}")
        resolved = path.resolve()
        if not _inside(resolved, self.root):
            raise PackageError("registry package escapes the registry root")
        return resolved


# ------------------------------------------------------------------- resolver
def _validate_project_manifest(manifest: dict) -> dict:
    if not isinstance(manifest, dict):
        raise PackageError("project manifest must be a JSON object")
    name = manifest.get("name", "<project>")
    if name != "<project>":
        _package_name(name, "project name")
    out = dict(manifest)
    out["dependencies"] = _dependency_map(manifest.get("dependencies", {}), PROJECT)
    return out


def resolve(manifest: dict, registry: Registry) -> Dict[str, dict]:
    """Resolve direct and transitive dependencies deterministically."""
    manifest = _validate_project_manifest(manifest)
    resolved: Dict[str, dict] = {}
    constraints: Dict[str, List[Tuple[str, str]]] = {}
    queue: List[Tuple[str, str, str]] = [
        (name, constraint, manifest.get("name", "<project>"))
        for name, constraint in sorted(manifest["dependencies"].items())
    ]

    while queue:
        name, constraint, requester = queue.pop(0)
        _package_name(name, "dependency name")
        # Constraints can be supplied directly to this public API, not only
        # through read_manifest, so validate them here too.
        _constraint_parts(constraint)
        constraints.setdefault(name, []).append((constraint, requester))
        available = registry.versions(name)
        if not available:
            raise PackageError(f"package '{name}' (required by {requester}) is not in the registry")
        candidates = [
            version for version in available
            if all(satisfies(version, current) for current, _ in constraints[name])
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
        package_dir = registry.path(name, chosen)
        sub = read_manifest(package_dir)
        resolved[name] = {
            "version": chosen,
            "digest": digest_dir(package_dir),
            "dependencies": sub.get("dependencies", {}),
        }
        for dep, dep_constraint in sorted(sub["dependencies"].items()):
            queue.append((dep, dep_constraint, f"{name} {chosen}"))
    return resolved


# ------------------------------------------------------------------ installer
def install(project_dir: Path, registry: Registry) -> Dict[str, dict]:
    """Resolve, atomically copy packages into ``ai_modules/``, and lock."""
    project_dir = Path(project_dir)
    manifest_path = project_dir / PROJECT
    if manifest_path.is_symlink():
        raise PackageError(f"project manifest is a symlink: {manifest_path}")
    manifest = _validate_project_manifest(read_json(manifest_path))
    resolved = resolve(manifest, registry)
    modules = project_dir / MODULES_DIR
    if modules.is_symlink() or (modules.exists() and not modules.is_dir()):
        raise PackageError(f"unsafe modules directory: {modules}")
    modules.mkdir(parents=True, exist_ok=True)

    for name, info in sorted(resolved.items()):
        src = registry.path(name, info["version"])
        dest = modules / name
        stage = modules / f".{name}.{os.getpid()}.{time.time_ns()}.tmp"
        try:
            stage.mkdir()
            _copy_signable(src, stage)
            actual = digest_dir(stage)
            if actual != info["digest"]:
                raise PackageError(
                    f"integrity check failed for {name} {info['version']}: "
                    f"expected {info['digest']}, got {actual}"
                )
            status, detail = registry.check_signature(name, info["version"])
            if status == "invalid":
                raise PackageError(
                    f"signature verification failed for {name} {info['version']}: {detail}"
                )
            if dest.exists() or dest.is_symlink():
                if dest.is_symlink() or not dest.is_dir():
                    raise PackageError(f"cannot replace unsafe installed package path {dest}")
                shutil.rmtree(dest)
            os.replace(stage, dest)
            stage = None
            info["signature"] = status
        except (OSError, PackageError) as e:
            raise PackageError(str(e)) from None
        finally:
            if stage is not None:
                shutil.rmtree(stage, ignore_errors=True)

    write_json(
        project_dir / LOCKFILE,
        {
            "lockfile_version": 1,
            "packages": {
                name: {
                    "version": info["version"],
                    "digest": info["digest"],
                    **(
                        {"signature": info["signature"]}
                        if info.get("signature") != "unsigned"
                        else {}
                    ),
                }
                for name, info in sorted(resolved.items())
            },
        },
    )
    return resolved


# ---------------------------------------------------------------- verification
def verify(project_dir: Path, key: Optional[str] = None) -> List[str]:
    """Recompute installed digests/signatures and return human-readable issues."""
    project_dir = Path(project_dir)
    lock_path = project_dir / LOCKFILE
    if lock_path.is_symlink():
        return [f"{lock_path} is a symlink; refusing to verify it"]
    if not lock_path.is_file():
        return [f"no {LOCKFILE}; run 'ailang install' first"]
    try:
        lock = read_json(lock_path)
    except PackageError as e:
        return [str(e)]
    if lock.get("lockfile_version") != 1 or not isinstance(lock.get("packages"), dict):
        return [f"{lock_path} has an unsupported or malformed lockfile"]

    problems = []
    modules = project_dir / MODULES_DIR
    if modules.is_symlink() or (modules.exists() and not modules.is_dir()):
        return [f"unsafe modules directory: {modules}"]
    for name, info in sorted(lock["packages"].items()):
        try:
            _package_name(name, "package name")
            if not isinstance(info, dict):
                raise PackageError("entry is not an object")
            version = _version_text(info.get("version"))
            digest = info.get("digest")
            if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
                raise PackageError("lockfile digest is malformed")
        except PackageError as e:
            problems.append(f"{name}: {e}")
            continue
        directory = project_dir / MODULES_DIR / name
        if directory.is_symlink() or not directory.is_dir():
            problems.append(f"{name}: not installed")
            continue
        try:
            actual = digest_dir(directory)
        except PackageError as e:
            problems.append(f"{name}: {e}")
            continue
        if actual != digest:
            problems.append(
                f"{name}: digest mismatch (expected {digest[:19]}..., got {actual[:19]}...)"
            )
        sig_path = directory / SIGNATURE
        if sig_path.is_file() and key is not None:
            try:
                sig = read_json(sig_path)
                payload = _signed_payload(name, version, directory)
                if not verify_signature(payload, sig.get("sig", ""), key):
                    problems.append(f"{name}: signature does not match")
            except PackageError as e:
                problems.append(f"{name}: invalid signature: {e}")
    return problems
