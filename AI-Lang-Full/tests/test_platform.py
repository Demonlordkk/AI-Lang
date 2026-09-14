"""Tests for the platform capabilities: FFI, networking, storage, graphics, math."""
from __future__ import annotations

import io
import socket
import struct
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ailang.errors import AILangError  # noqa: E402
from ailang.toolchain import run_source  # noqa: E402


def out(source: str) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_source(source, "<test>", [ROOT], check=True)
    return buf.getvalue().strip()


def fails(source: str, fragment: str = ""):
    try:
        out(source)
    except AILangError as e:
        assert fragment in str(e), f"expected {fragment!r} in {e}"
        return
    raise AssertionError(f"expected a failure containing {fragment!r}")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# ------------------------------------------------------------------ maths


@pytest.mark.parametrize("expr,expected", [
    ("round(pi(), 5)", "3.14159"),
    ("round(sin(0.0), 5)", "0.0"),
    ("round(cos(0.0), 5)", "1.0"),
    ("round(atan2(1.0, 1.0) * 4.0, 4)", "3.1416"),
    ("round(log10(1000.0), 5)", "3.0"),
    ("round(log2(1024.0), 5)", "10.0"),
    ("round(hypot(3.0, 4.0), 5)", "5.0"),
    ("round(degrees(pi()), 2)", "180.0"),
    ("round(radians(180.0), 5)", "3.14159"),
    ("sign(-3.0)", "-1"),
    ("sign(0.0)", "0"),
    ("clamp(15, 1, 10)", "10"),
    ("clamp(-5, 1, 10)", "1"),
    ("gcd(48, 18)", "6"),
    ("lcm(4, 6)", "12"),
    ("factorial(10)", "3628800"),
    ("trunc(3.99)", "3"),
    ("is_finite(1.0)", "true"),
])
def test_math_builtin(expr, expected):
    assert out(f"emit {expr}.") == expected


def test_sin_and_asin_round_trip():
    assert out("emit round(asin(sin(0.5)), 6).") == "0.5"


@pytest.mark.parametrize("bad,msg", [
    ("asin(5.0)", "between -1 and 1"),
    ("acos(-2.0)", "between -1 and 1"),
    ("log10(0.0)", "must be positive"),
    ("log2(-1.0)", "must be positive"),
    ("factorial(-1)", "must not be negative"),
    ("clamp(5, 10, 1)", "is above high"),
])
def test_math_domain_errors(bad, msg):
    fails(f"emit {bad}.", msg)


# -------------------------------------------------------------------- ffi


def test_ffi_calls_a_real_c_function():
    src = """let m := ffi_open("m").
let c := ffi_fn(m, "cos", ["real"], "real").
emit ffi_call(c, [0.0]).
"""
    assert out(src) == "1.0"


def test_ffi_two_arguments():
    src = """let m := ffi_open("m").
let p := ffi_fn(m, "pow", ["real", "real"], "real").
emit ffi_call(p, [2.0, 10.0]).
"""
    assert out(src) == "1024.0"


def test_ffi_reports_the_library_path():
    src = 'let m := ffi_open("m").\nemit len(ffi_info(m).path) > 0.\n'
    assert out(src) == "true"


def test_ffi_symbol_presence():
    src = """let m := ffi_open("m").
emit ffi_symbol(m, "sqrt").
emit ffi_symbol(m, "definitely_not_here").
"""
    assert out(src).split() == ["true", "false"]


def test_ffi_unknown_library_is_reported():
    fails('let m := ffi_open("no_such_library_xyz").', "cannot load library")


def test_ffi_unknown_symbol_is_reported():
    fails('let m := ffi_open("m").\nlet f := ffi_fn(m, "nope_xyz", [], "void").',
          "has no symbol")


def test_ffi_unknown_type_is_reported():
    fails('let m := ffi_open("m").\nlet f := ffi_fn(m, "cos", ["wobble"], "real").',
          "unknown foreign type")


def test_ffi_wrong_argument_count_is_reported():
    src = """let m := ffi_open("m").
let c := ffi_fn(m, "cos", ["real"], "real").
emit ffi_call(c, [1.0, 2.0]).
"""
    fails(src, "expects 1 argument")


def test_ffi_wrong_argument_type_is_reported():
    src = """let m := ffi_open("m").
let c := ffi_fn(m, "cos", ["real"], "real").
emit ffi_call(c, ["text"]).
"""
    fails(src, "expects real")


def test_ffi_refuses_a_fabricated_pointer():
    src = """let m := ffi_open("m").
let c := ffi_fn(m, "cos", ["ptr"], "real").
emit ffi_call(c, [12345]).
"""
    fails(src, "pointers come from foreign calls")


# ---------------------------------------------------------------- storage


def test_db_create_insert_query(tmp_path):
    src = f"""let d := db_open("{tmp_path}/t.db").
db_exec(d, "create table u(id integer, name text)").
db_exec(d, "insert into u values(?, ?)", [1, "ann"]).
emit db_one(d, "select name from u where id = ?", [1]).name.
"""
    assert out(src) == "ann"


def test_db_rows_are_maps_usable_by_the_stdlib():
    src = """let d := db_open(":memory:").
db_exec(d, "create table n(v integer)").
db_many(d, "insert into n values(?)", [[1], [2], [3], [4]]).
let rows := db_query(d, "select v from n").
emit mean(map(rows, \\r -> r.v)).
"""
    assert out(src) == "2.5"


def test_db_parameters_are_bound_not_interpolated():
    """A value containing SQL must not be able to change the statement."""
    src = """let d := db_open(":memory:").
db_exec(d, "create table t(name text)").
db_exec(d, "insert into t values(?)", ["rob'); drop table t;--"]).
emit db_tables(d).
emit len(db_query(d, "select * from t")).
"""
    assert out(src).split("\n") == ['["t"]', "1"]


def test_db_one_gives_nothing_when_no_row():
    src = """let d := db_open(":memory:").
db_exec(d, "create table t(id integer)").
emit is_nothing(db_one(d, "select * from t where id = 9")).
"""
    assert out(src) == "true"


def test_db_transaction_commits_on_success():
    src = """let d := db_open(":memory:").
db_exec(d, "create table t(v integer)").
fn work(c: Any) -> Int:
    db_exec(c, "insert into t values(1)").
    db_exec(c, "insert into t values(2)").
    give 2.
done.
emit db_transaction(d, \\ -> work(d)).
emit len(db_query(d, "select * from t")).
"""
    assert out(src).split() == ["2", "2"]


def test_db_transaction_rolls_back_on_failure():
    src = """let d := db_open(":memory:").
db_exec(d, "create table t(v integer)").
fn work(c: Any) -> Void:
    db_exec(c, "insert into t values(1)").
    raise "abort".
done.
attempt:
    db_transaction(d, \\ -> work(d)).
rescue e:
    emit "caught".
done.
emit len(db_query(d, "select * from t")).
"""
    assert out(src).split() == ["caught", "0"]


def test_nested_transaction_rolls_back_with_the_outer():
    src = """let d := db_open(":memory:").
db_exec(d, "create table t(v integer)").
fn inner(c: Any) -> Void:
    db_exec(c, "insert into t values(2)").
done.
fn outer(c: Any) -> Void:
    db_exec(c, "insert into t values(1)").
    db_transaction(c, \\ -> inner(c)).
    raise "abort".
done.
attempt:
    db_transaction(d, \\ -> outer(d)).
rescue e:
    emit "caught".
done.
emit len(db_query(d, "select * from t")).
"""
    assert out(src).split() == ["caught", "0"]


def test_db_bad_sql_names_the_statement():
    fails('let d := db_open(":memory:").\nemit db_query(d, "select * from nope").',
          "no such table")


def test_db_survives_reopening(tmp_path):
    path = f"{tmp_path}/p.db"
    out(f'let d := db_open("{path}").\n'
        'db_exec(d, "create table t(v integer)").\n'
        'db_exec(d, "insert into t values(7)").\ndb_close(d).')
    src = f'let d := db_open("{path}").\nemit db_one(d, "select v from t").v.'
    assert out(src) == "7"


# --------------------------------------------------------- document store


def test_store_round_trips_a_document(tmp_path):
    src = f"""let s := store_open("{tmp_path}/s.db").
store_put(s, "k", {{"name": "ann", "tags": ["a", "b"]}}).
emit store_get(s, "k").name.
emit store_get(s, "k").tags.
"""
    assert out(src).split("\n") == ["ann", '["a", "b"]']


def test_store_default_when_absent(tmp_path):
    src = f'let s := store_open("{tmp_path}/s.db").\nemit store_get(s, "nope", "fallback").'
    assert out(src) == "fallback"


def test_store_keys_and_delete(tmp_path):
    src = f"""let s := store_open("{tmp_path}/s.db").
store_put(s, "user:1", 1).
store_put(s, "user:2", 2).
store_put(s, "other", 3).
emit store_keys(s, "user:").
emit store_delete(s, "other").
emit store_keys(s).
"""
    lines = out(src).split("\n")
    assert lines[0] == '["user:1", "user:2"]'
    assert lines[1] == "true"
    assert lines[2] == '["user:1", "user:2"]'


def test_store_overwrites_an_existing_key(tmp_path):
    src = f"""let s := store_open("{tmp_path}/s.db").
store_put(s, "k", 1).
store_put(s, "k", 2).
emit store_get(s, "k").
emit len(store_keys(s)).
"""
    assert out(src).split() == ["2", "1"]


# --------------------------------------------------------------- graphics


def test_canvas_writes_a_valid_png(tmp_path):
    path = tmp_path / "o.png"
    src = f"""let c := canvas(64, 48).
canvas_fill(c, "#ff0000").
emit canvas_save(c, "{path}") > 0.
"""
    assert out(src) == "true"
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", data[16:24])
    assert (w, h) == (64, 48)


def test_canvas_size_reports_dimensions():
    src = "let c := canvas(20, 10).\nemit canvas_size(c).width.\nemit canvas_size(c).height."
    assert out(src).split() == ["20", "10"]


@pytest.mark.parametrize("colour", ['"#f00"', '"#ff0000"', '{"r": 255, "g": 0, "b": 0}'])
def test_colour_formats_are_accepted(colour, tmp_path):
    src = f'let c := canvas(8, 8).\ncanvas_fill(c, {colour}).\nemit canvas_save(c, "{tmp_path}/c.png") > 0.'
    assert out(src) == "true"


def test_bad_colour_is_reported():
    fails('let c := canvas(8, 8).\ncanvas_fill(c, "not-a-colour").', "bad colour")


def test_drawing_off_canvas_is_clipped_not_an_error(tmp_path):
    """Generated drawings must not fail on a rounding error."""
    src = f"""let c := canvas(16, 16).
rect(c, -50, -50, 10, 10, "#fff").
circle(c, 100, 100, 20, "#fff").
line(c, -20, -20, 200, 200, "#fff").
pixel(c, 999, 999, "#fff").
emit canvas_save(c, "{tmp_path}/c.png") > 0.
"""
    assert out(src) == "true"


def test_plot_writes_a_chart(tmp_path):
    path = tmp_path / "chart.png"
    src = f'emit plot("{path}", map(range(50), \\i -> sin(real(i) / 5.0))) > 0.'
    assert out(src) == "true"
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_rejects_an_empty_series(tmp_path):
    fails(f'emit plot("{tmp_path}/c.png", []).', "non-empty")


def test_canvas_rejects_bad_dimensions():
    fails("let c := canvas(0, 10).", "must be positive")


# --------------------------------------------------------------- networking


def test_http_server_round_trip():
    port = free_port()
    src = f"""fn h(req: Map) -> Map:
    give {{"status": 200, "body": "hi " + req.path}}.
done.
let s := serve({port}, h, "127.0.0.1", true).
sleep(0.3).
emit http_get("http://127.0.0.1:{port}/world").body.
serve_stop(s).
"""
    assert out(src) == "hi /world"


def test_http_server_returns_json():
    port = free_port()
    src = f"""fn h(req: Map) -> Map:
    give {{"status": 200, "json": {{"n": 42}}}}.
done.
let s := serve({port}, h, "127.0.0.1", true).
sleep(0.3).
let body := json_decode(http_get("http://127.0.0.1:{port}/").body).
emit body.n.
serve_stop(s).
"""
    assert out(src) == "42"


def test_http_server_sees_method_and_body():
    port = free_port()
    src = f"""fn h(req: Map) -> Map:
    give {{"status": 200, "body": req.method + ":" + req.body}}.
done.
let s := serve({port}, h, "127.0.0.1", true).
sleep(0.3).
emit http_post("http://127.0.0.1:{port}/", "payload").body.
serve_stop(s).
"""
    assert out(src) == "POST:payload"


def test_http_server_status_codes():
    port = free_port()
    src = f"""fn h(req: Map) -> Map:
    when req.path == "/missing":
        give {{"status": 404, "body": "no"}}.
    done.
    give {{"status": 201, "body": "made"}}.
done.
let s := serve({port}, h, "127.0.0.1", true).
sleep(0.3).
emit http_get("http://127.0.0.1:{port}/missing").status.
emit http_get("http://127.0.0.1:{port}/other").status.
serve_stop(s).
"""
    assert out(src).split() == ["404", "201"]


def test_handler_fault_becomes_a_500_not_a_crash():
    port = free_port()
    src = f"""fn h(req: Map) -> Map:
    let xs := [].
    give {{"status": 200, "body": to Text(xs[9])}}.
done.
let s := serve({port}, h, "127.0.0.1", true).
sleep(0.3).
emit http_get("http://127.0.0.1:{port}/").status.
serve_stop(s).
"""
    assert out(src) == "500"


def test_server_reads_query_parameters():
    port = free_port()
    src = f"""fn h(req: Map) -> Map:
    give {{"status": 200, "body": get(req.query, "name", "none")}}.
done.
let s := serve({port}, h, "127.0.0.1", true).
sleep(0.3).
emit http_get("http://127.0.0.1:{port}/?name=ann").body.
serve_stop(s).
"""
    assert out(src) == "ann"


def test_tcp_socket_round_trip():
    port = free_port()
    src = f"""fn server() -> Void:
    let l := tcp_listen({port}, "127.0.0.1").
    let c := tcp_accept(l, 5.0).
    tcp_send(c, "echo:" + tcp_receive(c, 1024, 5.0)).
    tcp_close(c).
    tcp_close(l).
done.
let bg := spawn(server).
sleep(0.3).
let c := tcp_connect("127.0.0.1", {port}).
emit tcp_send(c, "ping").
emit tcp_receive(c, 1024, 5.0).
tcp_close(c).
"""
    assert out(src).split() == ["4", "echo:ping"]


def test_tcp_connect_to_nothing_is_reported():
    fails(f'let c := tcp_connect("127.0.0.1", {free_port()}, 1.0).', "cannot reach")


def test_serve_rejects_a_non_function_handler():
    """The checker catches this before the program ever runs."""
    fails(f'let s := serve({free_port()}, 5, "127.0.0.1", true).', "expects Function")
