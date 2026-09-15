"""Diagnostic quality, overridable step budget (fuel), and lint coverage.

Covers:
* near-miss name suggestions and runtime error locations
* `ailang lint` understanding contracts (needs/ensures) -- it used to
  report them as "unsupported statement"
* an unknown type used twice producing the *identical* diagnostic twice;
  each unknown name is now reported once, without derived bind noise
* the 50M-step fuel cap being overridable via `--fuel N` and
  `AILANG_FUEL=N`, with the exhaustion message saying so
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

from ailang.errors import AILangError  # noqa: E402
from ailang.format import lint  # noqa: E402
from ailang.parser import parse  # noqa: E402
from ailang.toolchain import run_source  # noqa: E402
from ailang.typecheck import TypeChecker  # noqa: E402

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

CONTRACT_SRC = """fn withdraw(balance: Real, amount: Real) -> Real:
    needs amount > 0.
    needs amount <= balance.
    ensures result >= 0.
    give balance - amount.
done.

emit withdraw(10.0, 4.0).
"""

# --------------------------------------------------------------- lint + contracts


def test_lint_understands_contracts():
    issues = lint(CONTRACT_SRC)
    assert issues == []


def test_lint_still_reports_real_errors():
    issues = lint(CONTRACT_SRC + "emit undefined_name_xyz.\n")
    text = "; ".join(msg for _ln, msg in issues)
    assert "undefined_name_xyz" in text
    assert not any("unsupported statement" in m for _l, m in issues)


def test_lint_rejects_syntax_errors():
    issues = lint("fn f(:\n")
    assert issues


def test_contracts_still_enforced_at_runtime():
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_source(CONTRACT_SRC, "<t>", [ROOT], check=True)
    assert "6.0" in buf.getvalue()


def test_contract_violation_message_unchanged():
    src = CONTRACT_SRC.replace('emit withdraw(10.0, 4.0).', 'emit withdraw(10.0, -1.0).')
    buf = io.StringIO()
    with pytest.raises(AILangError), redirect_stdout(buf):
        run_source(src, "<t>", [ROOT], check=True)


# --------------------------------------------------------------- unknown types


def _diagnostics(source: str):
    try:
        TypeChecker().check(parse(source))
        return []
    except AILangError as e:
        return list(getattr(e, "diagnostics", None) or [])


def test_unknown_type_reported_once_per_name():
    diags = _diagnostics("fn f(a: Poimt, b: Poimt) -> Int:\n    give 1.\ndone.\n")
    msgs = [m for _l, _c, m in diags if "unknown type" in m]
    assert len(msgs) == 1
    assert msgs[0] == "unknown type 'Poimt'"


def test_unknown_type_param_and_return_reported_once():
    diags = _diagnostics("fn f(a: Poimt) -> Poimt:\n    give a.\ndone.\n")
    msgs = [m for _l, _c, m in diags if "unknown type" in m]
    assert len(msgs) == 1


def test_unknown_declared_type_has_no_derived_bind_noise():
    diags = _diagnostics("let x: Poimt := 1.\n")
    assert len(diags) == 1
    assert diags[0][2] == "unknown type 'Poimt'"


def test_real_bind_mismatch_still_reported():
    diags = _diagnostics("let x: Int := 1.5.\n")
    assert any("cannot bind" in m for _l, _c, m in diags)


def test_two_different_unknown_types_both_reported():
    diags = _diagnostics("fn f(a: Alpha, b: Beta) -> Int:\n    give 1.\ndone.\n")
    msgs = sorted(m for _l, _c, m in diags if "unknown type" in m)
    assert msgs == ["unknown type 'Alpha'", "unknown type 'Beta'"]


# --------------------------------------------------------------------- fuel


def _run_cli(tmp_path: Path, *extra: str, env_extra=None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("AILANG_FUEL", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [PY, str(ROOT / "ailang.py"), "run", *extra, str(tmp_path / "loop.al")],
        capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)


LOOP = "var i := 0.\nwhile i < 1000:\n    i <- i + 1.\ndone.\nemit \"ok\".\n"


def test_fuel_flag_causes_clean_exhaustion(tmp_path):
    p = tmp_path / "loop.al"
    p.write_text(LOOP, encoding="utf-8")
    r = _run_cli(tmp_path, "--fuel", "200")
    assert r.returncode != 0
    assert "execution limit exceeded" in r.stderr
    assert "--fuel N" in r.stderr
    assert "AILANG_FUEL=N" in r.stderr


def test_fuel_flag_adequate_budget_succeeds(tmp_path):
    p = tmp_path / "loop.al"
    p.write_text(LOOP, encoding="utf-8")
    r = _run_cli(tmp_path, "--fuel", "10000000")
    assert r.returncode == 0
    assert "ok" in r.stdout


def test_fuel_env_override(tmp_path):
    p = tmp_path / "loop.al"
    p.write_text(LOOP, encoding="utf-8")
    low = _run_cli(tmp_path, env_extra={"AILANG_FUEL": "200"})
    assert low.returncode != 0
    assert "execution limit exceeded" in low.stderr
    high = _run_cli(tmp_path, env_extra={"AILANG_FUEL": "10000000"})
    assert high.returncode == 0


def test_fuel_env_beats_nothing_but_not_the_flag(tmp_path):
    p = tmp_path / "loop.al"
    p.write_text(LOOP, encoding="utf-8")
    # flag wins over env: a big flag with a tiny env still succeeds
    env = dict(os.environ, AILANG_FUEL="200")
    r = subprocess.run(
        [PY, str(ROOT / "ailang.py"), "run", "--fuel", "10000000", str(p)],
        capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)
    assert r.returncode == 0


def test_fuel_invalid_flag_rejected(tmp_path):
    p = tmp_path / "loop.al"
    p.write_text(LOOP, encoding="utf-8")
    r = _run_cli(tmp_path, "--fuel", "0")
    assert r.returncode == 2
    assert "positive" in r.stderr


def test_fuel_exhaustion_message_identical_both_backends(tmp_path):
    p = tmp_path / "loop.al"
    p.write_text(LOOP, encoding="utf-8")
    results = []
    for native in ("1", "0"):
        env = dict(os.environ, AILANG_NATIVE=native)
        env.pop("AILANG_FUEL", None)
        results.append(subprocess.run(
            [PY, str(ROOT / "ailang.py"), "run", "--fuel", "200", str(p)],
            capture_output=True, text=True, env=env, cwd=ROOT, timeout=300))
    assert results[0].returncode == results[1].returncode != 0
    # the message itself is byte-identical; the line:col prefix may differ
    # per engine (the interpreter names the failang instruction)
    def messages(stderr: str):
        out = []
        for l in stderr.splitlines():
            if "execution limit exceeded" in l:
                i = l.find("runtime error:")
                out.append(l[i:] if i >= 0 else l)
        return out

    assert messages(results[0].stderr) == messages(results[1].stderr)
    assert messages(results[0].stderr)


def test_fuel_default_is_50m_and_env_respected_in_process():
    from ailang import vm as _vm

    os.environ.pop("AILANG_FUEL", None)
    assert _vm.fuel_default() == 50_000_000
    try:
        os.environ["AILANG_FUEL"] = "777"
        assert _vm.fuel_default() == 777
    finally:
        os.environ.pop("AILANG_FUEL", None)
    assert _vm.fuel_default() == 50_000_000


def test_run_source_accepts_explicit_fuel():
    src = "var i := 0.\nwhile i < 1000:\n    i <- i + 1.\ndone.\nemit \"ok\".\n"
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_source(src, "<t>", [ROOT], check=True, fuel=5_000_000)
    assert "ok" in buf.getvalue()
    with pytest.raises(AILangError):
        run_source(src, "<t>", [ROOT], check=True, fuel=200)
