"""End-to-end test of the professional CLI reference app (examples/apps/tasks.al).

Runs the full add/list/done/rm/stats flow as a real subprocess against both
backends and asserts the output is byte-identical (plus the JSON state is
sane). This is the "normal app building" acceptance test: a terminal product
built from the cli package, ordinary AI-Lang and JSON persistence.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run_flow(env_native: str) -> str:
    state = Path(tempfile.mkdtemp()) / "tasks.json"
    env = dict(os.environ)
    env["AILANG_NATIVE"] = env_native
    cmd = [PY, str(ROOT / "ailang.py"), "run", str(ROOT / "examples" / "apps" / "tasks.al"),
           "--", "--file", str(state)]

    def step(*args: str) -> str:
        p = subprocess.run(cmd + list(args), capture_output=True, text=True, env=env,
                           cwd=ROOT, timeout=120)
        return p.stdout + p.stderr

    out = []
    out.append(step("add", "ship the release"))
    out.append(step("add", "fix crash in parser"))
    out.append(step("add", "write docs"))
    out.append(step("list"))
    out.append(step("done", "2"))
    out.append(step("list", "--all", "--limit", "2"))
    out.append(step("rm", "1"))
    out.append(step("stats"))
    out.append(step("bogus"))
    out.append(step("done", "99"))
    out.append(step("add"))

    doc = json.loads(state.read_text(encoding="utf-8"))
    assert doc["next_id"] == 4
    ids = sorted(t["id"] for t in doc["tasks"])
    assert ids == [2, 3]
    by_id = {t["id"]: t for t in doc["tasks"]}
    assert by_id[2]["done"] is True
    assert by_id[3]["done"] is False
    return "".join(out)


def test_tasks_app_flow_and_persistence():
    text = run_flow("1")
    assert "added #1: ship the release" in text
    assert "ship the release" in text and "fix crash in parser" in text
    assert "done #2: fix crash in parser" in text
    assert "removed #1" in text
    assert "50" in text  # percent complete
    assert "unknown command: bogus" in text
    assert "no task #99" in text
    assert "add needs a title" in text


def test_tasks_app_agrees_under_both_backends():
    assert run_flow("1") == run_flow("0")


def test_tasks_app_usage_with_no_args():
    p = subprocess.run(
        [PY, str(ROOT / "ailang.py"), "run", str(ROOT / "examples" / "apps" / "tasks.al")],
        capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert p.returncode == 0
    assert "usage: tasks" in p.stdout
    assert "add | list | done | rm | stats | clear" in p.stdout
