"""Bounded lifetime management for AI-Lang host resources.

The VM can create threads, sockets, servers, databases, and other host-backed
objects.  A completed execution must not leave those resources attached to a
long-lived interpreter or test process.  ``ResourceScope`` is carried through
``ContextVar`` so nested modules and spawned tasks register with the same
execution lifetime.

Cleanup is best effort and idempotent: an explicit ``close``/``serve_stop``
may run before scope teardown.  A cleanup failure is recorded rather than
masking the program's original exception.
"""

from __future__ import annotations

import atexit
import contextlib
import threading
from contextvars import ContextVar


class ResourceScope:
    """Thread-safe collection of bounded, idempotent cleanup callbacks."""

    def __init__(self):
        self._lock = threading.RLock()
        self._cleanups = []
        self._closed = False
        self.errors = []

    def register(self, cleanup):
        if not callable(cleanup):
            raise TypeError("resource cleanup must be callable")
        with self._lock:
            if self._closed:
                # A late registration should not leak a resource.  Cleanup is
                # run synchronously because no live scope remains to own it.
                run_now = True
            else:
                self._cleanups.append(cleanup)
                run_now = False
        if run_now:
            self._run_one(cleanup)
        return cleanup

    def _run_one(self, cleanup):
        try:
            cleanup()
        except BaseException as exc:  # noqa: BLE001 - teardown must continue
            with self._lock:
                self.errors.append(exc)

    def close(self):
        with self._lock:
            if self._closed:
                return list(self.errors)
            self._closed = True
            cleanups = list(reversed(self._cleanups))
            self._cleanups.clear()
        for cleanup in cleanups:
            self._run_one(cleanup)
        return list(self.errors)


_CURRENT: ContextVar[ResourceScope | None] = ContextVar(
    "ailang_resource_scope", default=None
)

# Resources created by direct host API calls outside a run_source scope are
# retained here only until interpreter shutdown.  Language executions always
# get a bounded per-run scope.
_PROCESS_SCOPE = ResourceScope()


def current() -> ResourceScope | None:
    return _CURRENT.get()


def register(cleanup):
    """Register cleanup in the active execution, or at process shutdown."""
    return (_CURRENT.get() or _PROCESS_SCOPE).register(cleanup)


def cleanup_process_resources():
    """Close resources created through direct host APIs and start a fresh scope."""
    global _PROCESS_SCOPE
    old = _PROCESS_SCOPE
    old.close()
    _PROCESS_SCOPE = ResourceScope()
    return list(old.errors)


@contextlib.contextmanager
def scope():
    """Install and close an execution scope, restoring any outer scope."""
    owned = ResourceScope()
    token = _CURRENT.set(owned)
    try:
        yield owned
    finally:
        try:
            owned.close()
        finally:
            _CURRENT.reset(token)


@atexit.register
def _close_process_scope():
    _PROCESS_SCOPE.close()


__all__ = ["ResourceScope", "current", "register", "cleanup_process_resources", "scope"]
