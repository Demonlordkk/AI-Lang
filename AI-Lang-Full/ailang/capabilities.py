"""Capability-based resource policy for AI-Lang.

AI-Lang keeps computation pleasant by making effects explicit at the runtime
boundary.  A :class:`CapabilitySet` is an attenuatable allow-list: a program
can use only the capabilities its host granted, and a nested operation may
further restrict (never amplify) that set.

The policy is deliberately dependency-free and works with the reference VM,
the native source backend, spawned tasks, HTTP handlers, and imported modules
because it is carried by a ``ContextVar``.  It is not a process sandbox: FFI,
subprocesses, and the host itself remain trust boundaries.  For hostile code,
run in a separate operating-system sandbox as well.
"""

from __future__ import annotations

import fnmatch
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Iterable, Mapping, Optional, Tuple

from .errors import CapabilityError


# Public names.  Aliases are accepted by the CLI and ``capability_scope`` but
# the runtime always records the canonical names below.
CAPABILITY_ALIASES = {
    "fs": ("fs.read", "fs.write"),
    "filesystem": ("fs.read", "fs.write"),
    "net": ("net.connect", "net.listen"),
    "network": ("net.connect", "net.listen"),
    "db": ("db.open",),
    "database": ("db.open",),
    "ffi": ("ffi.load",),
    "process": ("process.exec",),
    "terminal": ("terminal.read", "terminal.write"),
    "env": ("env.read",),
    "graphics": ("graphics.write",),
    "clock": ("clock.read",),
    "random": ("random.use",),
}

CAPABILITIES = frozenset({
    "fs.read",
    "fs.write",
    "net.connect",
    "net.listen",
    "db.open",
    "ffi.load",
    "process.exec",
    "terminal.read",
    "terminal.write",
    "env.read",
    "graphics.write",
    "clock.read",
    "random.use",
})

_SAFE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


def _canonical_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("capability name must be Text")
    name = name.strip().lower()
    if name == "all":
        return "*"
    if name in CAPABILITY_ALIASES:
        return name
    if not _SAFE_NAME.fullmatch(name):
        raise ValueError(f"invalid capability name {name!r}")
    if name != "*" and name not in CAPABILITIES and not name.endswith(".*"):
        raise ValueError(f"unknown capability '{name}'")
    return name


def resource_path(value) -> str:
    """Canonicalize a path for policy matching without requiring it to exist."""
    raw = os.fspath(value)
    return os.path.abspath(os.path.expanduser(raw))


def resource_host(value) -> str:
    return str(value).strip().lower()


def resource_command(value) -> str:
    return str(value).strip()


def _resource_pattern(capability: str, pattern: str) -> str:
    """Normalize filesystem globs the same way effect paths are normalized."""
    if capability in {"fs.read", "fs.write"} and pattern != "*":
        return resource_path(pattern)
    return pattern


@dataclass(frozen=True)
class CapabilitySet:
    """An immutable allow-list, optionally attenuated from a parent policy.

    ``rules`` is a tuple of ``(name, patterns)``.  An empty pattern tuple
    means the capability is granted for every resource.  A parent policy is
    always checked too, which makes attenuation safe even when glob patterns
    overlap in surprising ways.
    """

    rules: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    parent: Optional["CapabilitySet"] = None
    unrestricted: bool = False

    @classmethod
    def none(cls) -> "CapabilitySet":
        return cls()

    @classmethod
    def all(cls) -> "CapabilitySet":
        return cls(unrestricted=True)

    @classmethod
    def from_specs(cls, specs: Iterable[str], parent=None) -> "CapabilitySet":
        """Parse ``name`` or ``name=resource-glob`` specifications.

        ``all`` grants the unrestricted policy.  It is intentionally useful
        only to hosts; a nested language map cannot manufacture a parent grant.
        """
        entries = {}
        for item in specs or ():
            if not isinstance(item, str) or not item.strip():
                raise ValueError("capability specification must be non-empty Text")
            raw_name, sep, raw_pattern = item.partition("=")
            name = _canonical_name(raw_name)
            pattern = raw_pattern if sep else "*"
            if not pattern:
                raise ValueError(f"capability '{raw_name}' has an empty resource pattern")
            names = CAPABILITIES if name == "*" else CAPABILITY_ALIASES.get(name, (name,))
            for canonical in names:
                entries.setdefault(canonical, []).append(
                    _resource_pattern(canonical, pattern)
                )
        return cls(
            tuple(sorted((name, tuple(patterns)) for name, patterns in entries.items())),
            parent=parent,
            unrestricted=False,
        )

    @classmethod
    def from_map(cls, value: Mapping, parent=None) -> "CapabilitySet":
        """Build an attenuated set from an AI-Lang Map.

        Values may be ``true``/``false``, one Text resource, or a List of Text
        resources.  ``true`` means any resource for that capability.
        """
        if not isinstance(value, Mapping):
            raise ValueError("capability scope needs a Map")
        specs = []
        for raw_name, raw_patterns in value.items():
            name = str(raw_name)
            if raw_patterns is True:
                specs.append(name)
            elif raw_patterns is False or raw_patterns is None:
                continue
            elif isinstance(raw_patterns, str):
                specs.append(f"{name}={raw_patterns}")
            elif isinstance(raw_patterns, (list, tuple)):
                for pattern in raw_patterns:
                    if not isinstance(pattern, str):
                        raise ValueError(f"capability '{name}' resource must be Text")
                    specs.append(f"{name}={pattern}")
            else:
                raise ValueError(
                    f"capability '{name}' must be Bool, Text, or a List of Text"
                )
        return cls.from_specs(specs, parent=parent)

    def _own_allows(self, name: str, resource: Optional[str]) -> bool:
        if self.unrestricted:
            return True
        matching = []
        for rule, patterns in self.rules:
            if rule == name or rule == "*" or (rule.endswith(".*") and name.startswith(rule[:-1])):
                matching.extend(patterns)
        if not matching:
            return False
        if resource is None:
            return True
        resource = str(resource)
        return any(pattern == "*" or fnmatch.fnmatchcase(resource, pattern)
                   for pattern in matching)

    def allows(self, name: str, resource: Optional[str] = None) -> bool:
        try:
            canonical = _canonical_name(name)
        except ValueError:
            return False
        if canonical in CAPABILITY_ALIASES:
            return all(self.allows(child, resource) for child in CAPABILITY_ALIASES[canonical])
        if not self._own_allows(canonical, resource):
            return False
        return self.parent is None or self.parent.allows(canonical, resource)

    def require(self, name: str, resource: Optional[str] = None, where: str = "operation") -> None:
        if self.allows(name, resource):
            return
        detail = f" for {resource!r}" if resource is not None else ""
        raise CapabilityError(
            f"{where} requires capability '{name}'{detail}; "
            "grant it explicitly or run with --allow"
        )

    def attenuate(self, requested: Mapping) -> "CapabilitySet":
        return CapabilitySet.from_map(requested, parent=self)

    def describe(self):
        if self.unrestricted:
            return {"all": True}
        return {
            name: list(patterns) if patterns else True
            for name, patterns in self.rules
        }


_CURRENT: ContextVar[Optional[CapabilitySet]] = ContextVar(
    "ailang_capabilities", default=None
)


def current() -> CapabilitySet:
    return _CURRENT.get() or CapabilitySet.all()


def _active(fallback=None) -> CapabilitySet:
    """Use an attenuated context policy, falling back to a host root policy."""
    return _CURRENT.get() or normalize(fallback)


def normalize(value=None) -> CapabilitySet:
    if value is None:
        return current()
    if isinstance(value, CapabilitySet):
        return value
    if isinstance(value, Mapping):
        return CapabilitySet.from_map(value)
    if isinstance(value, (list, tuple, set)):
        return CapabilitySet.from_specs(value)
    raise ValueError("capabilities must be a CapabilitySet, Map, or list of specifications")


@contextmanager
def use(policy: CapabilitySet):
    policy = normalize(policy)
    token = _CURRENT.set(policy)
    try:
        yield policy
    finally:
        _CURRENT.reset(token)


def require(name: str, resource=None, where="operation", policy=None) -> None:
    _active(policy).require(name, resource, where)


def guarded(fn: Callable, name: str, capability: str,
            resource: Optional[Callable] = None, policy: Optional[CapabilitySet] = None):
    """Return a callable that checks a capability before each effect."""
    @wraps(fn)
    def checked(*args, **kwargs):
        target = resource(args, kwargs) if resource is not None else None
        require(capability, target, name, policy=policy)
        return fn(*args, **kwargs)
    checked._ailang_capability = capability
    checked._ailang_original = fn
    return checked


def capability_check(name, resource=None, policy=None) -> bool:
    if resource is not None and not isinstance(resource, str):
        resource = str(resource)
    return _active(policy).allows(str(name), resource)


def capability_require(name, resource=None, policy=None):
    if resource is not None and not isinstance(resource, str):
        resource = str(resource)
    require(str(name), resource, "capability_require", policy=policy)
    return True


def capability_scope(requested, fn, policy=None):
    """Run a callable under an attenuated policy, preserving its return value."""
    if not isinstance(requested, Mapping):
        raise CapabilityError("capability_scope: first argument must be a Map")
    if not callable(fn):
        raise CapabilityError("capability_scope: second argument must be a function")
    child = _active(policy).attenuate(requested)

    @wraps(fn)
    def scoped(*args, **kwargs):
        with use(child):
            return fn(*args, **kwargs)
    scoped.ailang_name = getattr(fn, "ailang_name", "scoped")
    return scoped


__all__ = [
    "CapabilityError",
    "CapabilitySet",
    "CAPABILITIES",
    "current",
    "normalize",
    "use",
    "guarded",
    "capability_check",
    "capability_require",
    "capability_scope",
    "resource_path",
    "resource_host",
    "resource_command",
]
