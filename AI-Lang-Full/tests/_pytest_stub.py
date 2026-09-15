"""A minimal stand-in for the pytest pieces the AI-Lang test suite uses.

The suite only relies on two pytest features: `pytest.raises` and
`pytest.mark.parametrize`. This shim implements exactly those, so the test
files run with nothing but the host Python installed -- no pip, no network,
no third-party package. `tools/run_tests.py` installs this module as
`pytest` before importing the suite; individual test files fall back to it
automatically when real pytest is absent.

Deliberately small: anything beyond `raises` and `parametrize` should be
made plain test code, not another dependency on the real framework.
"""

from __future__ import annotations

import re
import sys


def raises(expected, match=None):
    """`with pytest.raises(AILangError):` -- asserts the exception is raised.

    Supports the optional `match` argument (a regex searched in str(exc)),
    which is all the suite uses.
    """
    class _Context:
        def __enter__(self):
            return self

        def __exit__(self, etype, value, tb):
            if etype is None:
                raise AssertionError(
                    f"DID NOT RAISE {getattr(expected, '__name__', expected)}"
                )
            if not issubclass(etype, expected):
                return False  # some other exception: let it propagate
            if match is not None and not re.search(match, str(value)):
                raise AssertionError(
                    f"{getattr(expected, '__name__', expected)} raised, "
                    f"but {match!r} not found in {str(value)!r}"
                )
            self.value = value
            self.exception = value
            return True

    return _Context()


class _Parametrize:
    """`@pytest.mark.parametrize("a, b", cases, ids=...)`.

    Expands the decorated function into one `test_...[id]` function per
    case, registered on the module it was defined in so plain name-scanning
    discovery (pytest and the standalone runner) finds each case.
    """

    def __init__(self, argnames, argvalues, ids=None):
        self.names = [n.strip() for n in argnames.split(",")]
        self.values = list(argvalues)
        self.ids = ids

    def __call__(self, fn):
        if len(self.names) == 1:
            cases = [(v,) for v in self.values]
        else:
            cases = [tuple(v) for v in self.values]

        mod = sys.modules.get(fn.__module__)
        used = set()

        for i, vals in enumerate(cases):
            try:
                label = str(self.ids(vals[0] if len(self.names) == 1 else vals))
            except Exception:  # noqa: BLE001 - ids is a convenience hook
                label = ""
            if not label:
                label = str(i)
            label = re.sub(r"\W+", "_", label)[:40] or str(i)
            case = dict(zip(self.names, vals))
            base = f"{fn.__name__}[{label}]"
            # distinct cases can sanitise to the same label (empty strings,
            # long similar blobs); a second one must not clobber the first
            name = base
            k = 1
            while name in used:
                name = f"{base}__{k}"
                k += 1
            used.add(name)

            def wrapper(*args, _case=case, **kwargs):
                fn(*args, **{**_case, **kwargs})

            import inspect as _inspect

            wrapper.__name__ = name
            wrapper.__qualname__ = name
            wrapper.__doc__ = fn.__doc__
            # expose the original signature so standalone runners can still
            # see fixture needs (tmp_path, monkeypatch) behind the parameters
            try:
                wrapper.__signature__ = _inspect.signature(fn)
            except (TypeError, ValueError):
                pass
            if mod is not None:
                setattr(mod, name, wrapper)

        # replace the original with a placeholder that is *not* discovered
        # (its name no longer starts with `test_`), so each case runs once
        def _placeholder(*_args, **_kwargs):
            raise AssertionError(f"{fn.__name__} was parametrized; run its [id] cases")

        _placeholder.__name__ = f"_parametrized_{fn.__name__}"
        _placeholder.__qualname__ = _placeholder.__name__
        if mod is not None:
            # the decorator's return value also rebinds fn.__name__ in the
            # module, which must not shadow anything -- use the placeholder
            setattr(mod, fn.__name__, _placeholder)
        return _placeholder


class Skipped(Exception):
    """Raised by a skipped test wrapper; the runner counts it, not fails it."""


class _Mark:
    parametrize = staticmethod(_Parametrize)

    @staticmethod
    def skipif(condition, reason=""):
        def deco(fn):
            if not condition:
                return fn

            def skipped(*_args, **_kwargs):
                raise Skipped(reason or "skipped")

            skipped.__name__ = fn.__name__
            skipped.__qualname__ = fn.__qualname__
            try:
                import inspect as _inspect

                skipped.__signature__ = _inspect.signature(fn)
            except (TypeError, ValueError):
                pass
            return skipped

        return deco


class _MonkeyPatch:
    """Enough of pytest's `monkeypatch` for setattr-based tests."""

    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value):
        if isinstance(target, str):
            import importlib

            target = importlib.import_module(target)
        old = getattr(target, name, object())
        had = old is not object()
        setattr(target, name, value)
        self._undo.append((target, name, old, had))

    def undo(self):
        while self._undo:
            target, name, old, had = self._undo.pop()
            if had:
                setattr(target, name, old)
            else:
                try:
                    delattr(target, name)
                except AttributeError:
                    pass


mark = _Mark()
