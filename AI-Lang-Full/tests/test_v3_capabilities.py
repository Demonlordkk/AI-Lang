"""V3 capability, authentication, lifetime, and native-isolation checks."""
from __future__ import annotations

import io
import os
import subprocess
import sys
import threading
import time
from contextlib import redirect_stdout
from pathlib import Path

try:
    import pytest
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _pytest_stub as pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ailang.bytecode import artifact, load, read, write  # noqa: E402
from ailang.capabilities import CapabilitySet, use  # noqa: E402
from ailang.errors import CapabilityError, VMError  # noqa: E402
from ailang.resources import cleanup_process_resources  # noqa: E402
from ailang.toolchain import compile_source, run_source  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _run(source, policy=None, native=False):
    old = os.environ.get("AILANG_NATIVE")
    os.environ["AILANG_NATIVE"] = "1" if native else "0"
    out = io.StringIO()
    try:
        with redirect_stdout(out):
            run_source(source, "<v3>", [ROOT], capabilities=policy)
        return out.getvalue().strip()
    finally:
        if old is None:
            os.environ.pop("AILANG_NATIVE", None)
        else:
            os.environ["AILANG_NATIVE"] = old


def test_capability_aliases_and_resource_patterns():
    policy = CapabilitySet.from_specs([
        "fs.read=/tmp/ailang-v3/**",
        "net.connect=localhost:443",
    ])
    assert policy.allows("fs.read", "/tmp/ailang-v3/a.txt")
    assert not policy.allows("fs.write", "/tmp/ailang-v3/a.txt")
    assert policy.allows("fs", "/tmp/ailang-v3/a.txt") is False
    assert policy.allows("net.connect", "localhost:443")
    assert not policy.allows("net.connect", "example.com:443")


def test_capability_scope_can_only_attenuate(tmp_path):
    allowed = tmp_path / "allowed.txt"
    denied = tmp_path / "denied.txt"
    policy = CapabilitySet.from_specs([f"fs.write={tmp_path}/*"])
    source = f'''\
fn writer(p: Text) -> Void:
    write_file(p, "ok").
done.
let scoped := capability_scope({{"fs.write": "{allowed}"}}, writer).
scoped("{allowed}").
'''
    _run(source, policy)
    assert allowed.read_text(encoding="utf-8") == "ok"
    source = f'''\
fn writer(p: Text) -> Void:
    write_file(p, "no").
done.
let scoped := capability_scope({{"fs.write": "{denied}"}}, writer).
scoped("{denied}").
'''
    # The parent policy does not contain denied, even though the child asks
    # for it; attenuation cannot mint authority.
    try:
        _run(source, CapabilitySet.from_specs([f"fs.write={allowed}"]))
    except CapabilityError:
        pass
    else:
        raise AssertionError("attenuation amplified a parent capability")


def test_effects_are_denied_by_default_when_host_policy_is_empty(tmp_path):
    path = tmp_path / "x"
    for source in (
        f'write_file("{path}", "x").',
        f'emit read_file("{path}").',
        "emit 1.",
    ):
        try:
            _run(source, CapabilitySet.none())
        except CapabilityError:
            pass
        else:
            raise AssertionError(f"effect was not denied: {source}")


def test_spawn_inherits_capability_context(tmp_path):
    path = tmp_path / "worker.txt"
    source = f'''\
fn worker() -> Void:
    write_file("{path}", "from worker").
done.
let task := spawn(worker).
await(task).
'''
    _run(source, CapabilitySet.from_specs([f"fs.write={tmp_path}/*"]))
    assert path.read_text(encoding="utf-8") == "from worker"
    try:
        _run(source, CapabilitySet.none())
    except (CapabilityError, VMError) as exc:
        assert "capability" in str(exc).lower()
    else:
        raise AssertionError("worker escaped its capability context")


def test_signed_artifact_requires_authentication(tmp_path):
    source = "emit 7."
    program = compile_source(source, check=False)
    path = tmp_path / "signed.albc.json"
    write(program, path, source, signing_key="v3-secret", key_id="release")
    assert read(path, signing_key="v3-secret", require_signature=True)["signature"]["key_id"] == "release"
    for key in ("wrong", None):
        try:
            read(path, signing_key=key, require_signature=True)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid or absent artifact key was accepted")
    unsigned = tmp_path / "unsigned.albc.json"
    write(program, unsigned, source)
    try:
        load(unsigned, require_signature=True)
    except ValueError:
        pass
    else:
        raise AssertionError("unsigned artifact was accepted in required mode")


def test_signature_tampering_fails_even_if_integrity_is_recomputed(tmp_path):
    source = "emit 7."
    path = tmp_path / "signed.albc.json"
    write(compile_source(source, check=False), path, source, signing_key="secret")
    obj = read(path)
    obj["signature"]["key_id"] = "attacker"
    # Recompute only the public corruption hash; authentication must still
    # reject the changed signed payload.
    import hashlib
    import json
    base = {k: v for k, v in obj.items() if k != "artifact_sha256"}
    obj["artifact_sha256"] = hashlib.sha256(json.dumps(
        base, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    path.write_text(json.dumps(obj), encoding="utf-8")
    try:
        read(path, signing_key="secret", require_signature=True)
    except ValueError as exc:
        assert "signature" in str(exc)
    else:
        raise AssertionError("tampered signed artifact was accepted")


def test_native_generated_namespace_has_no_import_or_file_builtins():
    # The validator is an explicit fail-closed gate in addition to the empty
    # builtin table.  A generated source fragment containing an import is not
    # accepted for host execution.
    from ailang.native import _Unsupported, _validate_generated_source

    try:
        _validate_generated_source("def _fn():\n    import os\n")
    except _Unsupported:
        pass
    else:
        raise AssertionError("native AST gate accepted an import")


def test_run_source_cleans_unfinished_spawn_task_quickly():
    source = """\
fn loop() -> Void:
    while true:
        let x := 1.
    done.
done.
let task := spawn(loop).
"""
    started = time.monotonic()
    run_source(source, "<v3-cleanup>", [ROOT], check=True)
    assert time.monotonic() - started < 2.0
    # The executor may retain an idle reusable worker, but no unfinished task
    # should remain after scope teardown.
    time.sleep(0.05)
    cleanup_process_resources()


def test_vm_and_native_enforce_emit_capability():
    for native in (False, True):
        try:
            _run("emit 1.", CapabilitySet.none(), native=native)
        except CapabilityError:
            pass
        else:
            raise AssertionError(f"native={native} bypassed emit policy")
