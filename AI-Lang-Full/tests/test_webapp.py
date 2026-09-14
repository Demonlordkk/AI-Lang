"""Tests for the webapp package (packages/webapp) and app-building support.

The package is ordinary AI-Lang, so these tests drive it the way a user
would: whole AI-Lang programs (parse -> check -> run) that build an app,
dispatch synthetic request maps, and print observable results. One test
runs the finished app over real HTTP; one proves that `use packages/...`
resolves from anywhere inside the project (the project-root search path).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ailang.toolchain import run_source  # noqa: E402
from contextlib import redirect_stdout  # noqa: E402
import io  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def out(src: str) -> str:
    """Run an AI-Lang program in-process (searching the project root) and
    return its stdout."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_source(src, filename="<webapp-test>", search_paths=[ROOT])
    return buf.getvalue()


# ------------------------------------------------------------------- routing
def test_route_exact_and_params():
    src = (
        "use packages/webapp as w.\n"
        "let app := w.new().\n"
        "w.route(app, \"GET\", \"/hello\", \\req -> w.text(200, \"hi\")).\n"
        "w.route(app, \"GET\", \"/notes/:id/comments/:cid\", \\req -> w.json(200, req.params)).\n"
        "w.route(app, \"POST\", \"/notes\", \\req -> w.json(201, {\"ok\": true})).\n"
        "emit w.dispatch(app, {\"method\": \"GET\", \"path\": \"/hello\", \"body\": \"\"}).body.\n"
        "emit w.dispatch(app, {\"method\": \"GET\", \"path\": \"/notes/7/comments/9\", \"body\": \"\"}).json.\n"
        "emit w.dispatch(app, {\"method\": \"POST\", \"path\": \"/notes\", \"body\": \"\"}).status.\n"
        "emit w.dispatch(app, {\"method\": \"DELETE\", \"path\": \"/notes\", \"body\": \"\"}).status.\n"
        "emit w.dispatch(app, {\"method\": \"GET\", \"path\": \"/notes/7\", \"body\": \"\"}).status.\n"
    )
    assert out(src) == (
        "hi\n"
        "{\"id\": \"7\", \"cid\": \"9\"}\n"
        "201\n"
        "404\n"
        "404\n"
    )


def test_route_trailing_slash_and_root():
    src = (
        "use packages/webapp as w.\n"
        "let app := w.new().\n"
        "w.route(app, \"GET\", \"/a/b\", \\req -> w.text(200, \"found\")).\n"
        "w.route(app, \"GET\", \"/\", \\req -> w.text(200, \"root\")).\n"
        "emit w.dispatch(app, {\"method\": \"GET\", \"path\": \"/a/b/\", \"body\": \"\"}).body.\n"
        "emit w.dispatch(app, {\"method\": \"GET\", \"path\": \"/\", \"body\": \"\"}).body.\n"
    )
    assert out(src) == "found\nroot\n"


def test_404_json_shape():
    src = (
        "use packages/webapp as w.\n"
        "let app := w.new().\n"
        "let r := w.dispatch(app, {\"method\": \"GET\", \"path\": \"/missing\", \"body\": \"\"}).\n"
        "emit r.status.\n"
        "emit r.json.error.\n"
    )
    assert out(src) == "404\nno route for /missing\n"


# ------------------------------------------------------------------- static
def test_static_files_and_index(tmp_path):
    d = tmp_path / "site"
    d.mkdir()
    (d / "index.html").write_text("<html>home</html>", encoding="utf-8")
    (d / "app.js").write_text("console.log(1)", encoding="utf-8")
    (d / "data.txt").write_text("plain", encoding="utf-8")
    src = (
        "use packages/webapp as w.\n"
        f"let app := w.new().\n"
        f"w.static(app, \"/\", \"{d}\").\n"
        "emit w.dispatch(app, {\"method\": \"GET\", \"path\": \"/\", \"body\": \"\"}).body.\n"
        "emit w.dispatch(app, {\"method\": \"GET\", \"path\": \"/app.js\", \"body\": \"\"}).headers[\"content-type\"].\n"
        "emit w.dispatch(app, {\"method\": \"GET\", \"path\": \"/data.txt\", \"body\": \"\"}).body.\n"
        "emit w.dispatch(app, {\"method\": \"GET\", \"path\": \"/nope.txt\", \"body\": \"\"}).status.\n"
    )
    assert out(src) == (
        "<html>home</html>\n"
        "application/javascript\n"
        "plain\n"
        "404\n"
    )


def test_static_traversal_rejected(tmp_path):
    (tmp_path / "secret.txt").write_text("top secret", encoding="utf-8")
    d = tmp_path / "site"
    d.mkdir()
    src = (
        "use packages/webapp as w.\n"
        f"let app := w.new().\n"
        f"w.static(app, \"/files\", \"{d}\").\n"
        "emit w.dispatch(app, {\"method\": \"GET\", \"path\": \"/files/../secret.txt\", \"body\": \"\"}).status.\n"
        "emit w.dispatch(app, {\"method\": \"GET\", \"path\": \"/files/..%2Fsecret.txt\", \"body\": \"\"}).status.\n"
    )
    assert out(src) == "404\n404\n"


# ---------------------------------------------------------------- middleware
def test_middleware_wraps_routes_static_and_404():
    src = (
        "use packages/webapp as w.\n"
        "let app := w.new().\n"
        "w.route(app, \"GET\", \"/ok\", \\req -> w.text(200, \"routed\")).\n"
        "var seen := 0.\n"
        "w.middleware(app, fn(req: Map, cont: Function) -> Map:\n"
        "    seen <- seen + 1.\n"
        "    give cont(req).\n"
        "done).\n"
        "w.dispatch(app, {\"method\": \"GET\", \"path\": \"/ok\", \"body\": \"\"}).\n"
        "w.dispatch(app, {\"method\": \"GET\", \"path\": \"/missing\", \"body\": \"\"}).\n"
        "emit seen.\n"
    )
    assert out(src) == "2\n"


def test_middleware_order():
    src = (
        "use packages/webapp as w.\n"
        "let app := w.new().\n"
        "w.route(app, \"GET\", \"/x\", \\req -> w.text(200, \"\" + get(req, \"tag\", \"none\"))).\n"
        "w.middleware(app, fn(req: Map, cont: Function) -> Map:\n"
        "    let r := set(req, \"tag\", \"a\").\n"
        "    give cont(r).\n"
        "done).\n"
        "w.middleware(app, fn(req: Map, cont: Function) -> Map:\n"
        "    let r := set(req, \"tag\", get(req, \"tag\", \"\") + \"b\").\n"
        "    give cont(r).\n"
        "done).\n"
        "emit w.dispatch(app, {\"method\": \"GET\", \"path\": \"/x\", \"body\": \"\"}).body.\n"
    )
    # first middleware sees the plain request, second sees tag=a
    assert out(src) == "ab\n"


# ------------------------------------------------------------------ helpers
def test_response_helpers_and_body_json():
    src = (
        "use packages/webapp as w.\n"
        "let app := w.new().\n"
        "emit w.json(202, {\"n\": 1}).status.\n"
        "emit w.html(200, \"<b>x</b>\").headers[\"content-type\"].\n"
        "emit w.text(500, \"boom\").status.\n"
        "let req := {\"method\": \"POST\", \"path\": \"/\", \"body\": json_encode({\"a\": 5})}.\n"
        "emit w.body_json(req).a.\n"
        "let empty := w.body_json({\"method\": \"POST\", \"path\": \"/\", \"body\": \"\"}).\n"
        "when has(empty, \"a\"):\n"
        "    emit \"bad\".\n"
        "else:\n"
        "    emit \"empty-ok\".\n"
        "done.\n"
        "let listb := w.body_json({\"method\": \"POST\", \"path\": \"/\", \"body\": \"[1, 2]\"}).\n"
        "when has(listb, \"a\"):\n"
        "    emit \"bad\".\n"
        "else:\n"
        "    emit \"not-a-map-ok\".\n"
        "done.\n"
    )
    assert out(src) == (
        "202\n"
        "text/html; charset=utf-8\n"
        "500\n"
        "5\n"
        "empty-ok\n"
        "not-a-map-ok\n"
    )


# --------------------------------------------------------------- over HTTP
def test_full_app_over_http():
    src = (
        "use packages/webapp as w.\n"
        "let app := w.new().\n"
        "var state := {\"hits\": 0}.\n"
        "w.route(app, \"GET\", \"/count\", fn(req: Map) -> Map:\n"
        "    let n := state.hits + 1.\n"
        "    state <- set(state, \"hits\", n).\n"
        "    give w.json(200, {\"n\": n}).\n"
        "done).\n"
        "w.route(app, \"GET\", \"/echo/:word\", \\req -> w.text(200, req.params.word)).\n"
        "let srv := w.start(app, 8126, \"127.0.0.1\", true).\n"
        "sleep(0.3).\n"
        "emit http_get(\"http://127.0.0.1:8126/count\").body.\n"
        "emit http_get(\"http://127.0.0.1:8126/count\").body.\n"
        "emit http_get(\"http://127.0.0.1:8126/echo/ai-lang\").body.\n"
        "emit http_get(\"http://127.0.0.1:8126/nope\").status.\n"
        "serve_stop(srv).\n"
        "emit \"stopped\".\n"
    )
    assert out(src) == (
        '{"n": 1}\n'
        '{"n": 2}\n'
        "ai-lang\n"
        "404\n"
        "stopped\n"
    )


# ----------------------------------------------------- project root imports
def test_use_packages_resolves_from_any_subdir(tmp_path):
    """`use packages/webapp` works from a script anywhere in the project:
    the module loader walks up to the project root (ailang.project.json)."""
    sub = tmp_path / "inner" / "deeper"
    sub.mkdir(parents=True)
    # put a probe inside a *copy* of the project root? No: use the real root,
    # but the script must live under it. Write under tests/ (in the repo).
    probe_dir = ROOT / "tests" / "webapp_probe"
    probe_dir.mkdir(exist_ok=True)
    try:
        probe = probe_dir / "probe.al"
        probe.write_text(
            "use packages/webapp as w.\n"
            "let app := w.new().\n"
            "w.route(app, \"GET\", \"/p\", \\req -> w.text(200, \"imported\")).\n"
            "emit w.dispatch(app, {\"method\": \"GET\", \"path\": \"/p\", \"body\": \"\"}).body.\n",
            encoding="utf-8",
        )
        r = subprocess.run(
            [sys.executable, str(ROOT / "ailang.py"), "run", str(probe)],
            capture_output=True, text=True, cwd=str(probe_dir),
        )
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "imported"
    finally:
        for f in probe_dir.iterdir():
            f.unlink()
        probe_dir.rmdir()


def test_webapp_package_typechecks():
    r = subprocess.run(
        [sys.executable, str(ROOT / "ailang.py"), "check", str(ROOT / "packages" / "webapp" / "main.al")],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
