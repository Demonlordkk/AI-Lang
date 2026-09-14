"""Tests for diagnostic quality: suggestions and runtime error locations."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent


def _run(src, tmp_path, cmd="run", native=True):
    p = tmp_path / "p.al"
    p.write_text(src, encoding="utf-8")
    env = dict(os.environ, AILANG_NATIVE="1" if native else "0")
    r = subprocess.run(
        [sys.executable, str(ROOT / "ailang.py"), cmd, str(p)],
        capture_output=True, text=True, env=env, cwd=tmp_path,
    )
    return r.stdout + r.stderr


# ------------------------------------------------------------- did you mean


def test_misspelled_builtin_is_suggested(tmp_path):
    assert "did you mean 'len'" in _run("emit lenn([1, 2]).", tmp_path, "check")


def test_misspelled_underscore_builtin(tmp_path):
    out = _run("emit grup_by([], \\x -> x).", tmp_path, "check")
    assert "group_by" in out


def test_local_name_outranks_a_builtin(tmp_path):
    """A mistyped local is likelier than a mistyped builtin."""
    out = _run("let counter := 1.\nemit countr.", tmp_path, "check")
    assert "did you mean one of 'counter'" in out


def test_user_function_is_suggested(tmp_path):
    src = "fn process_batch() -> Int:\n    give 1.\ndone.\nemit proces_batch().\n"
    assert "process_batch" in _run(src, tmp_path, "check")


def test_transposed_letters_are_caught(tmp_path):
    assert "did you mean" in _run("emit lenght([1]).", tmp_path, "check")


def test_nonsense_gets_no_suggestion(tmp_path):
    out = _run("emit zzzzqqqwwww.", tmp_path, "check")
    assert "undefined name" in out
    assert "did you mean" not in out


def test_assignment_to_unknown_name_suggests(tmp_path):
    src = "var total := 0.\ntotl <- 5.\n"
    assert "total" in _run(src, tmp_path, "check")


# --------------------------------------------------- runtime error locations


def test_runtime_error_reports_the_line(tmp_path):
    src = "let xs := [1, 2, 3].\nlet y := 5.\nemit xs[10].\n"
    out = _run(src, tmp_path)
    assert ":3:" in out
    assert "out of range" in out
    assert "emit xs[10]." in out


def test_runtime_error_inside_a_function_reports_the_inner_line(tmp_path):
    src = "fn f(xs: List) -> Any:\n    give xs[99].\ndone.\nlet d := [1].\nemit f(d).\n"
    out = _run(src, tmp_path)
    assert ":2:" in out, f"expected the error at the give line:\n{out}"


def test_missing_map_key_reports_the_line(tmp_path):
    src = 'let a := 1.\nlet m := {"k": 1}.\nemit m.missing.\n'
    out = _run(src, tmp_path)
    assert ":3:" in out and "missing" in out


def test_division_by_zero_reports_the_line(tmp_path):
    src = "let a := 1.\nlet xs := [].\nemit sum(xs) / len(xs).\n"
    out = _run(src, tmp_path)
    assert ":3:" in out and "zero" in out


@pytest.mark.parametrize("native", [True, False])
def test_error_location_is_the_same_under_both_backends(tmp_path, native):
    src = "fn f(xs: List) -> Any:\n    give xs[99].\ndone.\nlet d := [1].\nemit f(d).\n"
    out = _run(src, tmp_path, native=native)
    assert ":2:" in out


def test_error_excerpt_shows_the_source_line(tmp_path):
    src = "let xs := [1].\nemit xs[7].\n"
    out = _run(src, tmp_path)
    assert "emit xs[7]." in out
    assert "^" in out


# ------------------------------------------------- errors from imported code


def _module_project(tmp_path, module_src, main_src):
    (tmp_path / "helper.al").write_text(module_src, encoding="utf-8")
    (tmp_path / "main.al").write_text(main_src, encoding="utf-8")
    return tmp_path / "main.al"


def _run_path(path, native="1"):
    # contracts are pinned on: these tests assert on the checks themselves
    env = dict(os.environ, AILANG_NATIVE=native, AILANG_CONTRACTS="1")
    r = subprocess.run(
        [sys.executable, str(ROOT / "ailang.py"), "run", str(path)],
        capture_output=True, text=True, env=env, cwd=path.parent,
    )
    return r.stdout + r.stderr


@pytest.mark.parametrize("native", ["0", "1"])
def test_error_in_a_module_names_the_module_file(tmp_path, native):
    """A fault inside an import must not be blamed on the importer."""
    main = _module_project(
        tmp_path,
        "# pad\n# pad\n# pad\nfn boom(xs: List) -> Any:\n    give xs[99].\ndone.\n",
        "use helper as h.\nemit h.boom([1]).\n",
    )
    out = _run_path(main, native)
    assert "helper.al:5:" in out, out
    assert "main.al:" not in out.split("\n")[0]


@pytest.mark.parametrize("native", ["0", "1"])
def test_module_error_shows_the_module_source_line(tmp_path, native):
    main = _module_project(
        tmp_path,
        "fn boom(xs: List) -> Any:\n    give xs[42].\ndone.\n",
        "use helper as h.\nemit h.boom([]).\n",
    )
    out = _run_path(main, native)
    assert "give xs[42]." in out


@pytest.mark.parametrize("native", ["0", "1"])
def test_contract_failure_in_a_module_names_the_module(tmp_path, native):
    main = _module_project(
        tmp_path,
        "fn half(n: Int) -> Int:\n    needs n >= 0.\n    give n * 2.\ndone.\n",
        "use helper as h.\nemit h.half(-4).\n",
    )
    out = _run_path(main, native)
    assert "helper.al:2:" in out
    assert "half: precondition failed: n >= 0" in out


def test_error_in_the_main_file_still_names_the_main_file(tmp_path):
    main = _module_project(
        tmp_path,
        "fn ok(x: Int) -> Int:\n    give x.\ndone.\n",
        "use helper as h.\nlet xs := [1].\nemit xs[5].\n",
    )
    out = _run_path(main)
    assert "main.al:3:" in out
