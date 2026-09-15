"""Tests for the package manager: versions, resolution, integrity, CLI."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import pytest
except ImportError:  # no pytest installed (air-gapped): use the bundled shim
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    import _pytest_stub as pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ailang.packages import (
    LOCKFILE,
    MODULES_DIR,
    PROJECT,
    SIGNATURE,
    PackageError,
    Registry,
    best_match,
    digest_dir,
    install,
    parse_version,
    resolve,
    satisfies,
    verify,
)

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------- versions


@pytest.mark.parametrize(
    "version,constraint,ok",
    [
        ("1.2.3", "1.2.3", True),
        ("1.2.4", "1.2.3", False),
        ("1.2.3", "*", True),
        ("1.9.0", "^1.2.3", True),
        ("2.0.0", "^1.2.3", False),
        ("1.2.9", "~1.2.3", True),
        ("1.3.0", "~1.2.3", False),
        ("1.2.0", "^1.2.3", False),
        ("2.0.0", ">=1.0.0", True),
    ],
)
def test_constraint_satisfaction(version, constraint, ok):
    assert satisfies(version, constraint) is ok


def test_parse_version_rejects_junk():
    with pytest.raises(PackageError):
        parse_version("1.2")
    with pytest.raises(PackageError):
        parse_version("a.b.c")


def test_best_match_picks_the_highest():
    assert best_match(["1.0.0", "1.4.0", "1.2.0"], "^1.0.0") == "1.4.0"
    assert best_match(["1.0.0", "2.0.0"], "^3.0.0") is None


def test_versions_sort_numerically_not_lexically():
    assert best_match(["1.9.0", "1.10.0"], "*") == "1.10.0"


# ------------------------------------------------------------------ integrity


def _pkg(root: Path, name: str, version: str, deps=None, body="fn f() -> Int:\n    give 1.\ndone.\n"):
    d = root / f"src_{name}_{version.replace('.', '_')}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "ailang.package.json").write_text(
        json.dumps({"name": name, "version": version, "dependencies": deps or {}}),
        encoding="utf-8",
    )
    (d / "main.al").write_text(body, encoding="utf-8")
    return d


def test_digest_is_stable_and_content_sensitive(tmp_path):
    a = _pkg(tmp_path, "x", "1.0.0")
    first = digest_dir(a)
    assert first == digest_dir(a), "digest must be deterministic"
    (a / "main.al").write_text("fn f() -> Int:\n    give 2.\ndone.\n", encoding="utf-8")
    assert digest_dir(a) != first, "digest must track content"


def test_digest_ignores_directory_listing_order(tmp_path):
    a = _pkg(tmp_path, "y", "1.0.0")
    (a / "b.al").write_text("fn b() -> Int:\n    give 1.\ndone.\n", encoding="utf-8")
    (a / "a.al").write_text("fn a() -> Int:\n    give 1.\ndone.\n", encoding="utf-8")
    assert digest_dir(a) == digest_dir(a)


# ------------------------------------------------------------------ registry


def test_publish_and_list_versions(tmp_path):
    reg = Registry(tmp_path / "reg")
    reg.publish(_pkg(tmp_path, "stats", "1.0.0"))
    reg.publish(_pkg(tmp_path, "stats", "1.1.0"))
    assert reg.versions("stats") == ["1.0.0", "1.1.0"]


def test_publishing_the_same_version_twice_is_refused(tmp_path):
    reg = Registry(tmp_path / "reg")
    src = _pkg(tmp_path, "stats", "1.0.0")
    reg.publish(src)
    with pytest.raises(PackageError, match="already published"):
        reg.publish(src)


def test_manifest_must_declare_name_and_version(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "ailang.package.json").write_text('{"name": "x"}', encoding="utf-8")
    with pytest.raises(PackageError, match="version"):
        Registry(tmp_path / "reg").publish(d)


def test_package_json_rejects_duplicate_keys(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "ailang.package.json").write_text(
        '{"name":"x","name":"y","version":"1.0.0"}', encoding="utf-8"
    )
    with pytest.raises(PackageError, match="duplicate"):
        Registry(tmp_path / "reg").publish(d)


def test_package_symlinks_are_refused(tmp_path):
    outside = tmp_path / "outside.al"
    outside.write_text("fn secret() -> Int:\n    give 1.\ndone.\n", encoding="utf-8")
    d = _pkg(tmp_path, "linked", "1.0.0")
    link = d / "linked.al"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(PackageError, match="symlink"):
        digest_dir(d)


# ------------------------------------------------------------------ resolver


def test_resolves_transitive_dependencies(tmp_path):
    reg = Registry(tmp_path / "reg")
    reg.publish(_pkg(tmp_path, "stats", "1.0.0"))
    reg.publish(_pkg(tmp_path, "report", "2.0.0", deps={"stats": "^1.0.0"}))
    got = resolve({"name": "app", "dependencies": {"report": "^2.0.0"}}, reg)
    assert set(got) == {"report", "stats"}
    assert got["stats"]["version"] == "1.0.0"


def test_resolution_is_deterministic(tmp_path):
    reg = Registry(tmp_path / "reg")
    reg.publish(_pkg(tmp_path, "a", "1.0.0"))
    reg.publish(_pkg(tmp_path, "b", "1.0.0", deps={"a": "*"}))
    m = {"name": "app", "dependencies": {"b": "*", "a": "*"}}
    assert resolve(m, reg) == resolve(m, reg)


def test_conflicting_constraints_are_reported_with_both_requesters(tmp_path):
    reg = Registry(tmp_path / "reg")
    reg.publish(_pkg(tmp_path, "core", "1.0.0"))
    reg.publish(_pkg(tmp_path, "core", "2.0.0"))
    reg.publish(_pkg(tmp_path, "old", "1.0.0", deps={"core": "^1.0.0"}))
    reg.publish(_pkg(tmp_path, "new", "1.0.0", deps={"core": "^2.0.0"}))
    with pytest.raises(PackageError) as e:
        resolve({"name": "app", "dependencies": {"old": "*", "new": "*"}}, reg)
    msg = str(e.value)
    assert "core" in msg and "old" in msg and "new" in msg


def test_missing_package_names_the_requester(tmp_path):
    reg = Registry(tmp_path / "reg")
    with pytest.raises(PackageError, match="myapp"):
        resolve({"name": "myapp", "dependencies": {"ghost": "*"}}, reg)


def test_highest_compatible_version_is_chosen(tmp_path):
    reg = Registry(tmp_path / "reg")
    for v in ("1.0.0", "1.5.0", "2.0.0"):
        reg.publish(_pkg(tmp_path, "lib", v))
    got = resolve({"name": "app", "dependencies": {"lib": "^1.0.0"}}, reg)
    assert got["lib"]["version"] == "1.5.0"


# ------------------------------------------------------------------ installer


def test_install_writes_lockfile_and_modules(tmp_path):
    reg = Registry(tmp_path / "reg")
    reg.publish(_pkg(tmp_path, "stats", "1.0.0"))
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / PROJECT).write_text(
        json.dumps({"name": "app", "version": "0.1.0", "dependencies": {"stats": "^1.0.0"}}),
        encoding="utf-8",
    )
    install(proj, reg)
    assert (proj / MODULES_DIR / "stats" / "main.al").is_file()
    lock = json.loads((proj / LOCKFILE).read_text(encoding="utf-8"))
    assert lock["packages"]["stats"]["version"] == "1.0.0"
    assert lock["packages"]["stats"]["digest"].startswith("sha256:")


def test_verify_detects_tampering(tmp_path):
    reg = Registry(tmp_path / "reg")
    reg.publish(_pkg(tmp_path, "stats", "1.0.0"))
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / PROJECT).write_text(
        json.dumps({"name": "app", "dependencies": {"stats": "*"}}), encoding="utf-8"
    )
    install(proj, reg)
    assert verify(proj) == []
    (proj / MODULES_DIR / "stats" / "main.al").write_text("fn evil() -> Int:\n    give 0.\ndone.\n", encoding="utf-8")
    problems = verify(proj)
    assert problems and "digest mismatch" in problems[0]


def test_verify_reports_a_missing_package(tmp_path):
    reg = Registry(tmp_path / "reg")
    reg.publish(_pkg(tmp_path, "stats", "1.0.0"))
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / PROJECT).write_text(
        json.dumps({"name": "app", "dependencies": {"stats": "*"}}), encoding="utf-8"
    )
    install(proj, reg)
    import shutil

    shutil.rmtree(proj / MODULES_DIR / "stats")
    assert "not installed" in verify(proj)[0]


def test_verify_without_a_lockfile():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        assert "run 'ailang install'" in verify(Path(d))[0]


def test_reinstall_is_idempotent(tmp_path):
    reg = Registry(tmp_path / "reg")
    reg.publish(_pkg(tmp_path, "stats", "1.0.0"))
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / PROJECT).write_text(
        json.dumps({"name": "app", "dependencies": {"stats": "*"}}), encoding="utf-8"
    )
    install(proj, reg)
    first = (proj / LOCKFILE).read_text(encoding="utf-8")
    install(proj, reg)
    assert (proj / LOCKFILE).read_text(encoding="utf-8") == first
    assert verify(proj) == []


# ----------------------------------------------------------- end-to-end usage


def _cli(args, cwd, registry):
    env = dict(os.environ, AILANG_REGISTRY=str(registry))
    return subprocess.run(
        [sys.executable, str(ROOT / "ailang.py"), *args],
        capture_output=True, text=True, cwd=cwd, env=env,
        timeout=120,
    )


def test_installed_packages_are_importable(tmp_path):
    """The whole point: publish, install, then `use` it from a program."""
    reg_dir = tmp_path / "reg"
    reg = Registry(reg_dir)
    reg.publish(
        _pkg(
            tmp_path, "stats", "1.0.0",
            body="fn mean(xs: List) -> Real:\n    give sum(xs) / len(xs).\ndone.\n",
        )
    )
    reg.publish(
        _pkg(
            tmp_path, "report", "2.0.0", deps={"stats": "^1.0.0"},
            body=(
                "use stats as stats.\n"
                "fn summarise(xs: List) -> Text:\n"
                '    give "mean={stats.mean(xs)}".\n'
                "done.\n"
            ),
        )
    )
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / PROJECT).write_text(
        json.dumps({"name": "app", "version": "0.1.0", "dependencies": {"report": "^2.0.0"}}),
        encoding="utf-8",
    )
    r = _cli(["install"], proj, reg_dir)
    assert r.returncode == 0, r.stderr

    (proj / "app.al").write_text(
        "use report as report.\nemit report.summarise([2, 4]).\n", encoding="utf-8"
    )
    r = _cli(["run", "app.al"], proj, reg_dir)
    assert r.returncode == 0, r.stderr
    assert "mean=3.0" in r.stdout


def test_module_can_export_a_name_that_shadows_a_builtin(tmp_path):
    """`mean` is a builtin; a module defining its own must still export it."""
    reg_dir = tmp_path / "reg"
    Registry(reg_dir).publish(
        _pkg(
            tmp_path, "mystats", "1.0.0",
            body="fn mean(xs: List) -> Real:\n    give 42.0.\ndone.\n",
        )
    )
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / PROJECT).write_text(
        json.dumps({"name": "app", "dependencies": {"mystats": "*"}}), encoding="utf-8"
    )
    assert _cli(["install"], proj, reg_dir).returncode == 0
    (proj / "app.al").write_text(
        "use mystats as m.\nemit m.mean([1, 2, 3]).\n", encoding="utf-8"
    )
    r = _cli(["run", "app.al"], proj, reg_dir)
    assert r.returncode == 0, r.stderr
    assert "42.0" in r.stdout


def test_cli_add_records_the_dependency(tmp_path):
    reg_dir = tmp_path / "reg"
    Registry(reg_dir).publish(_pkg(tmp_path, "stats", "1.0.0"))
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / PROJECT).write_text(
        json.dumps({"name": "app", "version": "0.1.0"}), encoding="utf-8"
    )
    r = _cli(["add", "stats", "^1.0.0"], proj, reg_dir)
    assert r.returncode == 0, r.stderr
    manifest = json.loads((proj / PROJECT).read_text(encoding="utf-8"))
    assert manifest["dependencies"]["stats"] == "^1.0.0"
    assert (proj / MODULES_DIR / "stats").is_dir()


def test_cli_reports_a_missing_project_manifest(tmp_path):
    r = _cli(["install"], tmp_path, tmp_path / "reg")
    assert r.returncode == 1
    assert "ailang.project.json" in r.stderr


def test_cli_verify_fails_on_tampering(tmp_path):
    reg_dir = tmp_path / "reg"
    Registry(reg_dir).publish(_pkg(tmp_path, "stats", "1.0.0"))
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / PROJECT).write_text(
        json.dumps({"name": "app", "dependencies": {"stats": "*"}}), encoding="utf-8"
    )
    _cli(["install"], proj, reg_dir)
    (proj / MODULES_DIR / "stats" / "main.al").write_text("fn x() -> Int:\n    give 0.\ndone.\n", encoding="utf-8")
    r = _cli(["verify"], proj, reg_dir)
    assert r.returncode == 1
    assert "mismatch" in r.stderr


# --------------------------------------------------- bundled standard packages
STD = sorted((ROOT / "packages").glob("*/main.al"))


@pytest.mark.parametrize("path", STD, ids=lambda p: p.parent.name)
def test_bundled_package_type_checks(path):
    r = subprocess.run(
        [sys.executable, str(ROOT / "ailang.py"), "check", str(path)],
        capture_output=True, text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("path", STD, ids=lambda p: p.parent.name)
def test_bundled_package_has_a_valid_manifest(path):
    from ailang.packages import read_manifest

    m = read_manifest(path.parent)
    assert m["name"] == path.parent.name
    parse_version(m["version"])


def test_bundled_packages_behave_correctly(tmp_path):
    """Publish the real packages and exercise every exported function."""
    reg_dir = tmp_path / "reg"
    reg = Registry(reg_dir)
    for p in STD:
        reg.publish(p.parent)

    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / PROJECT).write_text(
        json.dumps(
            {
                "name": "trial",
                "version": "0.1.0",
                "dependencies": {"text": "*", "collections": "*", "testing": "*"},
            }
        ),
        encoding="utf-8",
    )
    assert _cli(["install"], proj, reg_dir).returncode == 0

    (proj / "trial.al").write_text(
        "use text as t.\n"
        "use collections as c.\n"
        'emit t.title_case("hello wide world").\n'
        'emit t.word_count("  one two   three ").\n'
        'emit t.truncate("abcdefghij", 8).\n'
        'emit t.is_blank("   ").\n'
        'emit t.count_occurrences("a-b-c", "-").\n'
        "emit c.union([1, 2], [2, 3]).\n"
        "emit c.intersect([1, 2, 3], [2, 3, 4]).\n"
        "emit c.difference([1, 2, 3], [2]).\n"
        "emit c.is_subset([1, 2], [1, 2, 3]).\n"
        "emit c.rotate([1, 2, 3, 4], 1).\n",
        encoding="utf-8",
    )
    r = _cli(["run", "trial.al"], proj, reg_dir)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split("\n")[:11] == [
        "Hello Wide World",
        "3",
        "abcde...",
        "true",
        "2",
        "[1, 2, 3]",
        "[2, 3]",
        "[1, 3]",
        "true",
        "[2, 3, 4, 1]",
        "",
    ]


def test_testing_package_assertions_actually_fail(tmp_path):
    """An assertion helper that never fails would be worse than useless."""
    reg_dir = tmp_path / "reg"
    reg = Registry(reg_dir)
    for p in STD:
        reg.publish(p.parent)
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / PROJECT).write_text(
        json.dumps({"name": "t", "dependencies": {"testing": "*"}}), encoding="utf-8"
    )
    _cli(["install"], proj, reg_dir)

    for src, fragment in [
        ('use testing as c.\nc.equals(1, 2, "lbl").\n', "expected 2 but got 1"),
        ('use testing as c.\nc.near(1.0, 2.0, 0.1, "lbl").\n', "not within"),
        ('use testing as c.\nc.has_length([1], 5, "lbl").\n', "expected 5 items"),
        ('use testing as c.\nc.contains_item([1], 9, "lbl").\n', "is not in"),
    ]:
        (proj / "f.al").write_text(src, encoding="utf-8")
        r = _cli(["run", "f.al"], proj, reg_dir)
        assert r.returncode != 0, f"should have failed: {src}"
        assert fragment in (r.stdout + r.stderr), f"missing {fragment!r}"


def test_testing_package_assertions_pass_when_correct(tmp_path):
    reg_dir = tmp_path / "reg"
    reg = Registry(reg_dir)
    for p in STD:
        reg.publish(p.parent)
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / PROJECT).write_text(
        json.dumps({"name": "t", "dependencies": {"testing": "*"}}), encoding="utf-8"
    )
    _cli(["install"], proj, reg_dir)
    (proj / "ok.al").write_text(
        "use testing as c.\n"
        'c.equals(2 + 2, 4, "arithmetic").\n'
        'c.near(0.1 + 0.2, 0.3, 0.0001, "floats").\n'
        'c.is_true(true, "truth").\n'
        'c.contains_item([1, 2], 2, "member").\n'
        'c.has_length([1, 2, 3], 3, "length").\n'
        'emit "ok".\n',
        encoding="utf-8",
    )
    r = _cli(["run", "ok.al"], proj, reg_dir)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout


# ------------------------------------------------------------------ signing
def _project(tmp_path, deps):
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    (proj / PROJECT).write_text(
        json.dumps({"name": "app", "version": "0.1.0", "dependencies": deps}),
        encoding="utf-8",
    )
    return proj


def test_publish_signs_and_install_verifies(tmp_path):
    reg = Registry(tmp_path / "reg", key="k1")
    reg.publish(_pkg(tmp_path, "stats", "1.0.0"), publisher="ada")
    assert (tmp_path / "reg" / "stats" / "1.0.0" / SIGNATURE).is_file()
    proj = _project(tmp_path, {"stats": "^1.0.0"})
    resolved = install(proj, reg)
    assert resolved["stats"]["signature"] == "ok"
    lock = json.loads((proj / LOCKFILE).read_text(encoding="utf-8"))
    assert lock["packages"]["stats"].get("signature") == "ok"
    assert verify(proj, key="k1") == []


def test_install_with_wrong_key_is_refused(tmp_path):
    reg = Registry(tmp_path / "reg", key="right")
    reg.publish(_pkg(tmp_path, "stats", "1.0.0"))
    proj = _project(tmp_path, {"stats": "^1.0.0"})
    try:
        install(proj, Registry(tmp_path / "reg", key="wrong"))
        raise AssertionError("wrong key must be refused")
    except PackageError as e:
        assert "signature" in str(e).lower()


def test_tampered_registry_package_is_refused(tmp_path):
    reg = Registry(tmp_path / "reg", key="k1")
    reg.publish(_pkg(tmp_path, "stats", "1.0.0"))
    (tmp_path / "reg" / "stats" / "1.0.0" / "main.al").write_text(
        "fn f() -> Int:\n    give 999.\ndone.\n", encoding="utf-8"
    )
    proj = _project(tmp_path, {"stats": "^1.0.0"})
    try:
        install(proj, reg)
        raise AssertionError("tampered package must be refused")
    except PackageError as e:
        assert "signature" in str(e).lower()


def test_unsigned_packages_install_without_a_key(tmp_path):
    reg = Registry(tmp_path / "reg")
    reg.publish(_pkg(tmp_path, "stats", "1.0.0"))
    proj = _project(tmp_path, {"stats": "^1.0.0"})
    resolved = install(proj, reg)
    assert resolved["stats"].get("signature") == "unsigned"
    lock = json.loads((proj / LOCKFILE).read_text(encoding="utf-8"))
    assert "signature" not in lock["packages"]["stats"]


def test_signed_without_key_installs_but_is_flagged(tmp_path):
    reg = Registry(tmp_path / "reg", key="k1")
    reg.publish(_pkg(tmp_path, "stats", "1.0.0"), publisher="ada")
    proj = _project(tmp_path, {"stats": "^1.0.0"})
    resolved = install(proj, Registry(tmp_path / "reg"))  # no key
    assert resolved["stats"]["signature"] == "unverifiable"
    lock = json.loads((proj / LOCKFILE).read_text(encoding="utf-8"))
    assert lock["packages"]["stats"].get("signature") == "unverifiable"


def test_verify_with_key_catches_installed_tamper(tmp_path):
    reg = Registry(tmp_path / "reg", key="k1")
    reg.publish(_pkg(tmp_path, "stats", "1.0.0"))
    proj = _project(tmp_path, {"stats": "^1.0.0"})
    install(proj, reg)
    (proj / MODULES_DIR / "stats" / "main.al").write_text(
        "fn f() -> Int:\n    give 0.\ndone.\n", encoding="utf-8"
    )
    problems = verify(proj, key="k1")
    assert any("stats" in p for p in problems)
