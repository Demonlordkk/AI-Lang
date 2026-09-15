"""TLS/HTTPS tests: serve(..., ssl=...) servers and http_request verify/timeout.

Certs are self-signed, generated with openssl in a temp dir per test, so the
suite stays hermetic (no network, no committed key material).
"""
from __future__ import annotations

import shutil
import socket
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

try:
    import pytest
except ImportError:  # no pytest installed (air-gapped): use the bundled shim
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    import _pytest_stub as pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ailang import net as _net  # noqa: E402
from ailang.errors import VMError  # noqa: E402
from ailang.stdlib import _http_request  # noqa: E402
from ailang.toolchain import run_source  # noqa: E402

OPENSSL = shutil.which("openssl")
pytestmark = pytest.mark.skipif(
    OPENSSL is None, reason="openssl not available on this machine"
)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _make_cert(d: Path):
    cert, key = d / "cert.pem", d / "key.pem"
    r = subprocess.run(
        [OPENSSL, "req", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", str(key), "-out", str(cert), "-days", "2",
         "-subj", "/CN=127.0.0.1"],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, r.stderr
    # combined PEM (cert + key) for the single-path serve spec
    full = d / "full.pem"
    full.write_bytes(cert.read_bytes() + key.read_bytes())
    return cert, key, full


def _unverified_ctx() -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _wait_port(port: int, tries: int = 50) -> None:
    for _ in range(tries):
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=0.2)
            s.close()
            return
        except OSError:
            import time
            time.sleep(0.05)


def _get(url: str, ctx=None) -> tuple[int, str]:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10, context=ctx) as r:
        return r.status, r.read().decode("utf-8", "replace")


def test_serve_https_roundtrip(tmp_path):
    cert, key, full = _make_cert(tmp_path)
    port = _free_port()
    handler = lambda req: {"status": 200, "body": "hi " + req["path"]}  # noqa: E731
    srv = _net.serve(port, handler, "127.0.0.1", True, [str(cert), str(key)])
    try:
        status, body = _get(f"https://127.0.0.1:{port}/x", ctx=_unverified_ctx())
        assert status == 200
        assert body == "hi /x"
        # a verifying client must REJECT the self-signed cert — proof the
        # connection is really TLS, not plaintext with a TLS label
        try:
            _get(f"https://127.0.0.1:{port}/x")
            raise AssertionError("verifying client unexpectedly succeeded")
        except urllib.error.URLError as e:
            assert "CERTIFICATE_VERIFY_FAILED" in str(e)
        # plain http against the TLS listener must not parse
        try:
            _get(f"http://127.0.0.1:{port}/x")
            raise AssertionError("plaintext http unexpectedly succeeded")
        except Exception:
            pass
    finally:
        _net.serve_stop(srv)


def test_http_request_verify_false(tmp_path):
    cert, key, full = _make_cert(tmp_path)
    port = _free_port()
    handler = lambda req: {"status": 200, "body": "ok"}  # noqa: E731
    srv = _net.serve(port, handler, "127.0.0.1", True, [str(cert), str(key)])
    url = f"https://127.0.0.1:{port}/health"
    try:
        r = _http_request(url, "GET", None, None, False)
        assert r["status"] == 200
        assert r["body"] == "ok"
        # default (verifying) request to a self-signed server fails cleanly
        try:
            _http_request(url, "GET")
            raise AssertionError("verifying http_request unexpectedly succeeded")
        except VMError as e:
            assert "certificate" in str(e).lower()
    finally:
        _net.serve_stop(srv)


def test_serve_rejects_bad_ssl(tmp_path):
    cert, key, full = _make_cert(tmp_path)
    port = _free_port()
    handler = lambda req: {"status": 200, "body": "ok"}  # noqa: E731
    for bad in ("/nonexistent-cert.pem", [str(cert), "/nonexistent-key.pem"],
                [str(cert)]):
        try:
            _net.serve(port, handler, "127.0.0.1", False, bad)
            raise AssertionError(f"expected VMError for ssl spec {bad!r}")
        except VMError as e:
            assert "ssl" in str(e).lower()
    # a cert file that is not a certificate
    junk = tmp_path / "junk.pem"
    junk.write_text("not a pem file\n", encoding="utf-8")
    try:
        _net.serve(port, handler, "127.0.0.1", False, str(junk))
        raise AssertionError("expected VMError for a non-PEM cert")
    except VMError as e:
        assert "certificate" in str(e).lower()


def test_tls_vm_end_to_end(tmp_path):
    """A full AI-Lang program: TLS server + TLS client in one process."""
    cert, key, full = _make_cert(tmp_path)
    port = _free_port()
    source = f'''
fn router(req: Map) -> Map:
    when req.path == "/health":
        give {{"status": 200, "body": "ok"}}.
    done.
    give {{"status": 404, "body": "no"}}.
done.
let cert := "{full}".
let srv := serve({port}, router, "127.0.0.1", true, cert).
let url := "https://127.0.0.1:{port}/health".
let r := http_request(url, "GET", nothing, nothing, false).
emit r.status.
emit r.body.
serve_stop(srv).
'''
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        run_source(source, "<tls-test>", [tmp_path])
    assert buf.getvalue().strip() == "200\nok"


def test_serve_body_cap_returns_413(tmp_path):
    port = _free_port()
    calls = []
    handler = lambda req: (calls.append(1),
                           {"status": 200, "body": "got " + str(len(req["body"]))})[1]  # noqa: E731
    srv = _net.serve(port, handler, "127.0.0.1", True)
    _wait_port(port)
    try:
        # within the cap
        r = _http_request(f"http://127.0.0.1:{port}/", "POST",
                          "x" * 1000, {"Content-Type": "text/plain"})
        assert r["status"] == 200 and r["body"] == "got 1000"
        assert calls
        # over the cap: the server refuses before the body is read. Depending
        # on when the client finishes writing, the client sees a 413 or a
        # connection abort — both are rejections; the handler must not run.
        calls.clear()
        big = "x" * (_net.MAX_BODY_BYTES + 10)
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/", data=big.encode(), method="POST")
        try:
            urllib.request.urlopen(req, timeout=10)
            raise AssertionError("oversized body was not rejected")
        except urllib.error.HTTPError as e:
            assert e.code == 413
        except (urllib.error.URLError, ConnectionError, BrokenPipeError,
                TimeoutError):
            pass
        assert not calls, "handler must not run on an oversized body"
    finally:
        _net.serve_stop(srv)
