#!/usr/bin/env python3
"""Run the whole AI-Lang test suite with no third-party dependencies.

The suite is written for pytest, but it only uses two pytest features
(`raises`, `mark.parametrize`). This runner installs the tiny shim in
`tests/_pytest_stub.py` as `pytest`, imports every `tests/test_*.py`, and
executes each test -- including every parametrized case -- with:

  * fixture support for `tmp_path` and `monkeypatch`;
  * a per-test wall-clock timeout (default 120 s, `RUN_TESTS_TEST_TIMEOUT`),
    so a non-terminating test (fuzz input, subprocess, whatever) is reported
    and the run moves on instead of hanging;
  * a hard cap on total suite time (default 900 s, `RUN_TESTS_TOTAL_TIMEOUT`;
    0 disables).

Usage:
    python3 tools/run_tests.py            # full suite
    python3 tools/run_tests.py fuzz cli   # only test_fuzz.py, test_cli.py
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import signal
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"

PER_TEST_TIMEOUT = int(os.environ.get("RUN_TESTS_TEST_TIMEOUT", "120"))
TOTAL_TIMEOUT = int(os.environ.get("RUN_TESTS_TOTAL_TIMEOUT", "900"))


def _install_pytest_stub():
    """Make `import pytest` resolve to the zero-dependency shim."""
    spec = importlib.util.spec_from_file_location(
        "_pytest_stub", TESTS / "_pytest_stub.py")
    stub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stub)
    sys.modules["pytest"] = stub
    sys.modules["_pytest_stub"] = stub
    return stub


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise _Timeout("test exceeded its time limit")


def _load_module(path: Path):
    name = f"ailang_tests_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _collect(mod):
    """Yield (attr_name, fn) for every runnable test, in definition order."""
    for attr, fn in vars(mod).items():
        if not attr.startswith("test_") or not callable(fn):
            continue
        if not getattr(fn, "__name__", "").startswith("test_"):
            continue  # a parametrization placeholder
        yield attr, fn


def _fixtures(fn):
    params = inspect.signature(fn).parameters
    fixtures = {}
    tmp = None
    monkey = None
    if "tmp_path" in params:
        tmp = tempfile.mkdtemp(prefix="ailang_test_")
        fixtures["tmp_path"] = Path(tmp)
    if "monkeypatch" in params:
        from _pytest_stub import _MonkeyPatch

        monkey = _MonkeyPatch()
        fixtures["monkeypatch"] = monkey
    return fixtures, tmp, monkey


def main(argv):
    _install_pytest_stub()
    from _pytest_stub import _MonkeyPatch, Skipped  # noqa: F401

    wanted = set(argv)
    files = sorted(TESTS.glob("test_*.py"))
    if wanted:
        files = [f for f in files if f.stem.replace("test_", "") in wanted
                 or f.name in wanted]
    if not files:
        print("no test files matched", file=sys.stderr)
        return 2

    signal.signal(signal.SIGALRM, _alarm_handler)
    started = time.monotonic()
    passed = failed = skipped = 0
    failures = []

    for path in files:
        if TOTAL_TIMEOUT and time.monotonic() - started > TOTAL_TIMEOUT:
            print(f"\nTOTAL TIMEOUT after {TOTAL_TIMEOUT}s; "
                  f"skipping {len(files)} remaining file(s)")
            break
        try:
            mod = _load_module(path)
        except Exception:
            failed += 1
            failures.append((path.name, "import", traceback.format_exc()))
            continue
        for attr, fn in _collect(mod):
            if TOTAL_TIMEOUT and time.monotonic() - started > TOTAL_TIMEOUT:
                print(f"\nTOTAL TIMEOUT after {TOTAL_TIMEOUT}s; stopping")
                break
            fixtures, tmp, monkey = _fixtures(fn)
            signal.alarm(PER_TEST_TIMEOUT)
            t0 = time.monotonic()
            try:
                fn(**fixtures)
                signal.alarm(0)
                passed += 1
                print(f"ok {path.name}::{attr} ({time.monotonic() - t0:.1f}s)")
            except _Timeout:
                signal.alarm(0)
                failed += 1
                failures.append((f"{path.name}::{attr}", "timeout",
                                 f"exceeded {PER_TEST_TIMEOUT}s"))
                print(f"TIMEOUT {path.name}::{attr}")
            except Skipped as e:
                signal.alarm(0)
                skipped += 1
                print(f"skip {path.name}::{attr} ({e})")
            except BaseException:  # noqa: BLE001 - report, keep going
                signal.alarm(0)
                failed += 1
                failures.append((f"{path.name}::{attr}", "error",
                                 traceback.format_exc()))
                print(f"FAIL {path.name}::{attr}")
            finally:
                if monkey is not None:
                    monkey.undo()
                if tmp is not None:
                    import shutil

                    shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped, "
          f"{passed + failed + skipped} total in {time.monotonic() - started:.1f}s")
    for name, kind, detail in failures:
        print(f"\n--- {name} ({kind}) ---")
        print(detail)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
