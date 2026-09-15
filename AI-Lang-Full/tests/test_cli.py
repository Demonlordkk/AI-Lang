"""Tests for packages/cli (the command-line toolkit), the read_line/exit
builtins, and the neural<->app model persistence bridge."""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

try:
    import pytest
except ImportError:  # no pytest installed (air-gapped): use the bundled shim
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    import _pytest_stub as pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ailang.errors import AILangError, ProcessExit  # noqa: E402
from ailang.toolchain import run_source  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def out(source: str, check: bool = True) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_source(source, "<test>", [ROOT], check=check)
    return buf.getvalue().strip()


def fails(source: str, fragment: str = "") -> str:
    try:
        out(source)
    except AILangError as e:
        if fragment:
            assert fragment.lower() in str(e).lower(), f"expected {fragment!r} in {e!r}"
        return str(e)
    raise AssertionError(f"expected clean error for: {source!r}")


# ------------------------------------------------------------------ parse

CLI = "use packages/cli as c."


def test_parse_basic_flags_and_positionals():
    src = CLI + """
let p := c.parse(["--name", "x", "-n", "3", "--verbose", "go", "stop"],
    {"--name": "text", "-n": "int", "--verbose": "flag"}).
emit to Text(p.flags) + " " + to Text(p.positional).
"""
    assert out(src) == '{"--name": "x", "-n": 3, "--verbose": true} ["go", "stop"]'


def test_parse_equals_form_and_list_accumulation():
    src = CLI + """
let p := c.parse(["--a=0.5", "--tag", "x", "--tag", "y", "--", "raw -1"],
    {"--a": "real", "--tag": "list"}).
emit to Text(p.flags["--a"]) + " " + to Text(p.flags["--tag"]) + " " + to Text(p.positional).
"""
    assert out(src) == '0.5 ["x", "y"] ["raw -1"]' 


def test_parse_negative_numbers_are_positional_or_values():
    src = CLI + """
let p := c.parse(["run", "-5", "-0.25", "--rate", "-0.01"],
    {"--rate": "real"}).
emit to Text(p.positional) + " " + to Text(p.flags["--rate"]).
"""
    assert out(src) == '["run", "-5", "-0.25"] -0.01' 


def test_parse_unknown_flag_raises_cleanly():
    src = CLI + """
attempt:
    c.parse(["--nope"], {"--ok": "flag"}).
    emit "no error".
rescue e:
    emit e.message.
done.
"""
    assert out(src) == "unknown flag: --nope (try --help for usage)"


def test_parse_missing_value_raises_cleanly():
    src = CLI + """
attempt:
    c.parse(["--name"], {"--name": "text"}).
    emit "no error".
rescue e:
    emit e.message.
done.
"""
    assert out(src) == "flag --name needs a value"


def test_parse_bad_number_raises_cleanly():
    src = CLI + """
attempt:
    c.parse(["-n", "abc"], {"-n": "int"}).
    emit "no error".
rescue e:
    emit e.message.
done.
"""
    msg = out(src)
    assert msg != "no error" and "abc" in msg


# ------------------------------------------------------------------ output

def test_table_alignment_with_lists_and_maps():
    src = CLI + """
let t := c.table(["id", "name", "score"], [
    ["1", "short", "9"],
    ["22", "a much longer name", "123"],
    {"id": "3", "name": "map row", "score": "7"},
]).
emit t + "\\nEND".
"""
    result = out(src)
    lines = result.split("\n")
    assert lines[5].startswith("END")
    # every table line is the same width once padded
    widths = {len(l) for l in lines[:5]}
    assert len(widths) == 1, f"columns not aligned: {lines}"
    assert "a much longer name" in lines[3]


def test_bar_progression():
    src = CLI + """
emit c.bar(0, 4, "work").
emit c.bar(2, 4, "work").
emit c.bar(4, 4, "work").
emit c.bar(9, 4, "work").
"""
    result = out(src)
    lines = result.split("\n")
    assert "0%" in lines[0] and lines[0].startswith("[----")
    assert "50%" in lines[1] and lines[1].startswith("[####")
    assert "100%" in lines[2] and "#" * 24 in lines[2]
    assert "100%" in lines[3]  # clamped


def test_usage_text():
    src = CLI + """
emit c.usage("demo", {"--rate": "real", "-v": "flag"}, ["<cmd>  help"]).
"""
    result = out(src)
    assert "usage: demo [options]" in result
    assert "--rate" in result and "(boolean)" in result and "<cmd>  help" in result


def test_color_wraps_ansi():
    src = CLI + 'emit c.color("x", 31).'
    esc = chr(27)
    assert out(src) == esc + "[31mx" + esc + "[0m"


# ------------------------------------------------------------------ config

def test_config_defaults_and_override(tmp_path):
    cfg = tmp_path / "app.json"
    cfg.write_text('{"rate": 0.5, "extra": true}')
    src = CLI + f"""
let c2 := c.config("{cfg}", {{"rate": 0.1, "mode": "train"}}).
emit to Text(c2["rate"]) + " " + to Text(c2["mode"]) + " " + to Text(c2["extra"]).
"""
    assert out(src) == "0.5 train true"
    src2 = CLI + f"""
let c3 := c.config("{tmp_path / 'missing.json'}", {{"a": 1}}).
emit to Text(c3["a"]).
"""
    assert out(src2) == "1"


def test_confirm_reads_stdin():
    src = CLI + """
emit to Text(c.confirm("Sure?")).
"""
    # feed "y" via the host stdin is not possible in-process; check the
    # false path (empty stdin) instead
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_source(src, "<t>", [ROOT], check=True)
    assert buf.getvalue().strip().endswith("false")


# ------------------------------------------------------- read_line / exit

def test_read_line_reads_stdin(monkeypatch):
    fake = io.StringIO("hello\nworld\n")
    monkeypatch.setattr(sys, "stdin", fake)
    src = "emit read_line().\nemit read_line()."
    assert out(src) == "hello\nworld"


def test_exit_terminates_with_code(monkeypatch):
    fake = io.StringIO("")
    monkeypatch.setattr(sys, "stdin", fake)
    src = "emit 1.\nexit(7).\nemit 2."
    with pytest.raises(ProcessExit) as exc:
        out(src)
    assert exc.value.code == 7


def test_exit_cannot_be_rescued():
    src = """
attempt:
    exit(3).
rescue e:
    emit "caught".
done.
"""
    with pytest.raises(ProcessExit):
        out(src)


# ------------------------------------------------- neural <-> app bridge

def test_model_save_load_roundtrip(tmp_path):
    src = f"""
use packages/neural as nn.
seed(3).
let net := nn.mlp([2, 4, 3], 9).
nn.save(net, "{tmp_path / 'm.almodel'}").
let net2 := nn.load("{tmp_path / 'm.almodel'}").
let a := value_of(net.layers[1].w)[0].
let b := value_of(net2.layers[1].w)[0].
emit to Text(a == b).
let x := tensor([[0.5, -1.0]]).
let p1 := value_of(nn.forward(net, x))[0].
let p2 := value_of(nn.forward(net2, x))[0].
emit to Text(p1 == p2).
"""
    assert out(src) == "true\ntrue"


def test_saved_model_serves_identical_predictions(tmp_path):
    """Train -> save -> load: the loaded model predicts exactly the same."""
    src = f"""
use packages/neural as nn.
seed(11).
var xs := [].
var ys := [].
repeat i in range(20):
    let b := i % 2.
    append(xs, [b * 1.5, 1.0 - b]).
    append(ys, one_hot(b, 2)).
done.
let net := nn.mlp([2, 4, 2], 11).
nn.train(net, tensor(xs), \\out -> sum_t(ce_t(t_softmax(out), tensor(ys))), 120, 0.05, 0.0).
nn.save(net, "{tmp_path / 'net.almodel'}").
let net2 := nn.load("{tmp_path / 'net.almodel'}").
let probe := tensor([[1.5, 0.0], [0.0, 1.0]]).
let before := value_of(t_softmax(nn.forward(net, probe))).
let after := value_of(t_softmax(nn.forward(net2, probe))).
emit to Text(before == after).
"""
    assert out(src) == "true"


def test_webapp_error_handler_and_throttle():
    src = """
use packages/webapp as w.
let app := w.new().
w.route(app, "GET", "/boom", fn(req: Map) -> Map:
    raise "kaput".
done).
w.middleware(app, w.make_error_handler(false)).
emit to Text(w.dispatch(app, {"method": "GET", "path": "/boom", "body": ""}).status).
let app2 := w.new().
w.route(app2, "GET", "/hit", \\req -> w.text(200, "ok")).
w.middleware(app2, w.make_throttle(2, 60.0)).
emit to Text(w.dispatch(app2, {"method": "GET", "path": "/hit", "body": ""}).status).
emit to Text(w.dispatch(app2, {"method": "GET", "path": "/hit", "body": ""}).status).
emit to Text(w.dispatch(app2, {"method": "GET", "path": "/hit", "body": ""}).status).
"""
    assert out(src) == "500\n200\n200\n429"


def test_cli_classifier_end_to_end(tmp_path):
    """The full normal-app story: train, save, load, predict — subprocess,
    both backends, identical output."""
    import os
    import subprocess

    prog = ROOT / "examples" / "apps" / "cli_classifier.al"
    model = tmp_path / "m.almodel"

    def run(args, native):
        env = dict(os.environ, AILANG_NATIVE="1" if native else "0")
        r = subprocess.run(
            [sys.executable, str(ROOT / "ailang.py"), "run", str(prog), "--", *args,
             "--model", str(model)],
            capture_output=True, text=True, env=env, cwd=tmp_path,
            timeout=120,
        )
        return r.returncode, r.stdout.strip()

    code, trained_a = run(["train", "-e", "80"], True)
    assert code == 0, trained_a
    pa, out_a = run(["predict", "0.8", "0.1"], True)
    pb, out_b = run(["predict", "0.8", "0.1"], False)
    assert pa == 0 and pb == 0
    assert out_a == out_b and out_a.startswith("class")
    ra, rep_a = run(["report"], True)
    rb, rep_b = run(["report"], False)
    assert ra == 0 and rb == 0
    assert rep_a == rep_b and "probe" in rep_a
