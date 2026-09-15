"""Focused tests for portable active training and sleep-memory persistence."""

from __future__ import annotations

import io
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

try:
    import pytest
except ImportError:  # the repository runner supplies a small compatible stub
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _pytest_stub as pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ailang.capabilities import CapabilitySet, use  # noqa: E402
from ailang.errors import VMError  # noqa: E402
from ailang.toolchain import run_source  # noqa: E402
from ailang.training import (  # noqa: E402
    active_train,
    batch_budget,
    device_profile,
    load_checkpoint,
    save_checkpoint,
)

ROOT = Path(__file__).resolve().parent.parent


def _policy(path):
    return CapabilitySet.from_specs([
        f"fs.read={path.parent}/*",
        f"fs.write={path.parent}/*",
        "terminal.write",
        "random.use",
    ])


def _run(source, policy):
    output = io.StringIO()
    with redirect_stdout(output):
        run_source(source, "<training-test>", [ROOT], capabilities=policy)
    return output.getvalue().strip().splitlines()


def test_device_budget_is_conservative_and_caps_requested_work(monkeypatch):
    import ailang.training as training

    monkeypatch.setattr(training, "_memory_available", lambda: 64 * 1024 * 1024)
    profile = device_profile()
    assert profile["tier"] == "small"
    assert profile["memory_budget"] == int(64 * 1024 * 1024 * 0.60)
    assert batch_budget(1024, 3) == 3
    assert batch_budget(1024, 100000000) <= 100000000
    with pytest.raises(VMError, match="non-negative"):
        batch_budget(-1, 1)


def test_checkpoint_roundtrip_authentication_and_tamper_detection(tmp_path):
    path = tmp_path / "signed.alstate"
    policy = _policy(path)
    with use(policy):
        saved = save_checkpoint(
            path,
            {"weights": [1.0, 2.0], "cursor_name": "mobile"},
            cursor=2,
            metrics={"loss": 0.25},
            epoch=3,
            signing_key="test-secret",
            key_id="unit-test",
        )
        loaded = load_checkpoint(path, "test-secret", True)
    assert saved["signed"] is True
    assert loaded["cursor"] == 2
    assert loaded["epoch"] == 3
    assert loaded["state"]["weights"] == [1.0, 2.0]

    tampered = path.read_text(encoding="utf-8").replace('"cursor": 2', '"cursor": 99')
    path.write_text(tampered, encoding="utf-8")
    with use(policy):
        with pytest.raises(VMError, match="integrity|envelope|authentication"):
            load_checkpoint(path, "test-secret", True)


def test_active_train_caps_explicit_batch_on_a_constrained_budget(tmp_path, monkeypatch):
    import ailang.training as training

    monkeypatch.setattr(training, "_memory_available", lambda: 1024 * 1024)
    path = tmp_path / "constrained.alstate"
    batches = []

    def step(batch, state):
        batches.append(list(batch))
        return {"state": state or 0, "loss": 0.0}

    with use(_policy(path)):
        result = active_train(
            list(range(4)), step, path,
            {"batch_size": 100, "reserved_bytes": 1024 * 1024, "max_items": 2},
        )
    assert [len(batch) for batch in batches] == [1, 1]
    assert result["items"] == 2


def test_active_train_honors_a_time_window_between_batches(tmp_path):
    path = tmp_path / "timed.alstate"
    calls = 0

    def step(batch, state):
        nonlocal calls
        calls += 1
        time.sleep(0.01)
        return {"state": (state or 0) + len(batch), "loss": 0.0}

    with use(_policy(path)):
        result = active_train(
            list(range(10)), step, path,
            {"batch_size": 1, "max_seconds": 0.001},
        )
    assert calls == 1
    assert result["items"] == 1
    assert result["complete"] is False
    assert result["sleeping"] is True


def test_active_train_accepts_a_batch_only_callback(tmp_path):
    path = tmp_path / "batch-only.alstate"
    seen = []

    def step(batch):
        seen.append(list(batch))
        return {"loss": float(len(batch))}

    with use(_policy(path)):
        result = active_train([1, 2], step, path, {"batch_size": 2})
    assert seen == [[1, 2]]
    assert result["complete"] is True


def test_active_train_respects_final_batch_and_resumes_next_chunk(tmp_path):
    path = tmp_path / "resume.alstate"
    policy = _policy(path)
    batches = []

    def step(batch, state):
        batches.append(list(batch))
        total = 0 if state is None else state
        return {"state": total + len(batch), "loss": float(len(batch))}

    with use(policy):
        first = active_train(
            list(range(7)), step, path,
            {"batch_size": 3, "max_items": 5, "epochs": 1},
        )
        second = active_train(
            list(range(7)), step, path,
            {"batch_size": 3, "max_items": 5, "epochs": 1},
        )
    assert [len(batch) for batch in batches] == [3, 2, 2]
    assert first["items"] == 5
    assert first["cursor"] == 5
    assert first["complete"] is False
    assert first["sleeping"] is True
    assert second["items"] == 2
    assert second["state"] == 7
    assert second["complete"] is True
    assert second["epoch"] == 1
    assert json.loads(path.read_text(encoding="utf-8"))["cursor"] == 0


def test_active_train_empty_dataset_completes_without_a_callback(tmp_path):
    path = tmp_path / "empty.alstate"
    calls = []

    def step(batch, state):
        calls.append(batch)
        return {"state": state, "loss": 0.0}

    with use(_policy(path)):
        result = active_train([], step, path, {"epochs": 3})
    assert calls == []
    assert result["complete"] is True
    assert result["sleeping"] is False
    assert result["epoch"] == 3
    assert result["cursor"] == 0


def test_active_train_restores_tensor_state_for_a_later_window(tmp_path):
    path = tmp_path / "tensor-resume.alstate"
    policy = _policy(path)
    source = f'''\
let w := param([1.0]).
fn step(batch: List, state: Any) -> Map:
    give {{"state": {{"w": state.w + len(batch), "seen": state.seen + len(batch)}}, "loss": 0.0}}.
done.
let first := active_train([1, 2], step, "{path}", {{"state": {{"w": w, "seen": 0}}, "batch_size": 1, "max_items": 1}}).
let second := active_train([1, 2], step, "{path}", {{"batch_size": 1, "max_items": 1}}).
emit first.complete.
emit second.state.seen.
emit is_tensor(second.state.w).
'''
    assert _run(source, policy) == ["false", "2", "true"]


def test_model_checkpoint_can_require_a_signature(tmp_path):
    path = tmp_path / "signed-model.almodel"
    policy = _policy(path)
    source = f'''\
let model := {{"weights": param([1.0, 2.0])}}.
model_save("{path}", model, {{}}, "model-secret", "test").
let restored := model_load("{path}", nothing, "model-secret", true).
emit is_tensor(restored.weights).
'''
    assert _run(source, policy) == ["true"]


def test_model_and_optimizer_bundle_roundtrip(tmp_path):
    model_path = tmp_path / "model.almodel"
    optimizer_path = tmp_path / "optimizer.alstate"
    policy = CapabilitySet.from_specs([
        f"fs.read={tmp_path}/*",
        f"fs.write={tmp_path}/*",
        "terminal.write",
        "random.use",
    ])
    source = f'''\
use packages/neural as nn.
let net := nn.mlp([1, 2, 1], 4).
let optimizer := adamw(nn.params(net), 0.01, 0.0).
let x := tensor([[1.0], [2.0]]).
backward(sum_t(nn.forward(net, x))).
adamw_step(optimizer).
let saved := nn.save_bundle(net, optimizer, "{model_path}", "{optimizer_path}").
let restored := nn.load_bundle("{model_path}", "{optimizer_path}", net).
emit is_tensor(restored.model.layers[0].w).
emit value_of(restored.model.layers[0].w) == value_of(net.layers[0].w).
emit restored.optimizer.t.
'''
    assert _run(source, policy) == ["true", "true", "1"]


def test_neural_active_training_sleeps_and_resumes_with_model_state(tmp_path):
    path = tmp_path / "neural-active.alstate"
    policy = _policy(path)
    source = f'''\
use packages/neural as nn.
let net := nn.mlp([1, 2, 1], 8).
let rows := [[[0.0], [0.0]], [[1.0], [1.0]], [[2.0], [2.0]], [[3.0], [3.0]]].
fn loss_fn(output: Any, target: Any) -> Any:
    give mse_t(output, target).
done.
let first := nn.active(net, rows, loss_fn, "{path}", {{"batch_size": 2, "max_items": 2, "rate": 0.01, "wd": 0.0}}).
let second := nn.active(net, rows, loss_fn, "{path}", {{"batch_size": 2, "max_items": 2, "rate": 0.01, "wd": 0.0}}).
emit first.complete.
emit second.complete.
emit is_tensor(second.state.net.layers[0].w).
emit second.state.optimizer.t.
'''
    assert _run(source, policy) == ["false", "true", "true", "2"]


def test_data_and_metrics_packages_work_together():
    source = '''\
use packages/data as d.
use packages/metrics as m.
let batches := d.chunked([1, 2, 3, 4, 5], 2).
emit batches.
emit d.windows([1, 2, 3, 4], 3).
emit m.accuracy([0, 1, 1], [0, 0, 1]).
emit m.f1([true, false, true], [true, true, true], true).
'''
    assert _run(source, CapabilitySet.from_specs(["terminal.write"])) == [
        "[[1, 2], [3, 4], [5]]",
        "[[1, 2, 3], [2, 3, 4]]",
        "0.6666666666666666",
        "0.8",
    ]
