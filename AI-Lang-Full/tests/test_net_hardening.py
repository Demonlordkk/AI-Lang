"""Focused boundary tests for networking primitives and HTTP framing."""
from __future__ import annotations

import math
import sys
from pathlib import Path

try:
    import pytest
except ImportError:  # no pytest installed (air-gapped): use the bundled shim
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _pytest_stub as pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ailang.errors import VMError  # noqa: E402
from ailang.net import _normalise, _port, _timeout, tcp_listen  # noqa: E402
from ailang.stdlib import _http_request  # noqa: E402


def _raises(fn, fragment):
    with pytest.raises(VMError, match=fragment):
        fn()


def test_ports_are_exact_bounded_integers():
    for value in (True, False, -1, 65536, 1.5, "80"):
        _raises(lambda value=value: _port(value, "test"), "port must be")
    assert _port(0, "test") == 0
    assert _port(65535, "test") == 65535


def test_timeouts_reject_non_finite_zero_and_unbounded_values():
    for value in (0, -1, math.inf, -math.inf, math.nan, "soon"):
        _raises(lambda value=value: _timeout(value, "test"), "timeout must be")
    assert _timeout(None, "test", 3.0) == 3.0
    assert _timeout(0.25, "test") == 0.25


def test_listen_rejects_coercion_and_bad_backlog():
    for args in ((1.5,), (True,), (1, ""), (1, "127.0.0.1", 0),
                 (1, "127.0.0.1", 65536)):
        _raises(lambda args=args: tcp_listen(*args), "tcp_listen")


def test_http_client_rejects_local_file_urls():
    _raises(lambda: _http_request("file:///etc/passwd", "GET"), "http:// or https://")
    _raises(lambda: _http_request("http://[broken", "GET"), "invalid URL")


def test_http_client_rejects_bad_timeout_and_header_injection():
    _raises(lambda: _http_request("http://127.0.0.1/", "GET", timeout=math.nan),
            "finite positive")
    _raises(lambda: _http_request("http://127.0.0.1/", "GET",
                                  headers={"X-Bad\nName": "x"}),
            "header names")
    _raises(lambda: _http_request("http://127.0.0.1/", "GET",
                                  headers={"X-Good": "x\r\nInjected: yes"}),
            "header values")


def test_http_response_framing_headers_cannot_be_overridden():
    status, headers, body = _normalise({
        "status": 200,
        "headers": {
            "Content-Length": "999",
            "Transfer-Encoding": "chunked",
            "Connection": "upgrade",
            "X-Trace": "ok",
        },
        "body": "hello",
    })
    assert status == 200 and body == b"hello"
    assert "content-length" not in headers
    assert "transfer-encoding" not in headers
    assert "connection" not in headers
    assert headers["x-trace"] == "ok"


def test_http_response_rejects_bad_status_and_header_values():
    _raises(lambda: _normalise({"status": True, "body": "x"}), "status must be")
    _raises(lambda: _normalise({"status": 700, "body": "x"}), "between 100 and 599")
    _raises(lambda: _normalise({"headers": {"X-Test": "bad\nvalue"}}), "invalid header")
