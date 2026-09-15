"""Networking: TCP sockets and an HTTP server.

This closes the biggest gap in the language: a program could call a web API
but could not be one. With `serve` an AI-Lang program is a service.

    serve(8080, \\req -> {"status": 200, "body": "hello " + req.path}).

The handler is an ordinary function from a request map to a response map, so
routing, middleware and templating are written in AI-Lang rather than
configured. Requests are handled on a thread pool; the handler runs on the
same VM, so closures and globals behave exactly as they do anywhere else.

Sockets are exposed too, for protocols that are not HTTP:

    let s := tcp_listen(9000).
    let c := tcp_accept(s).
    tcp_send(c, "hi").
"""
from __future__ import annotations

import json
import socket
import ssl as _ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import atexit
from urllib.parse import parse_qs, urlparse

from pathlib import Path

from .errors import VMError

_servers = {}
_next = [1]


# --------------------------------------------------------------- raw sockets
class _Sock:
    __slots__ = ("sock", "id", "kind", "addr")

    def __init__(self, sock, kind, addr):
        self.sock = sock
        self.kind = kind
        self.addr = addr
        self.id = _next[0]
        _next[0] += 1


def tcp_listen(port, host="0.0.0.0", backlog=64):
    """Open a listening TCP socket."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((str(host), int(port)))
        s.listen(int(backlog))
    except OSError as e:
        raise VMError(f"tcp_listen: cannot listen on {host}:{port}: {e}") from e
    return _Sock(s, "listener", f"{host}:{port}")


def tcp_accept(listener, timeout=None):
    """Wait for a connection; returns a connected socket."""
    if not isinstance(listener, _Sock) or listener.kind != "listener":
        raise VMError("tcp_accept: argument must be a socket from tcp_listen")
    try:
        if timeout is not None:
            listener.sock.settimeout(float(timeout))
        conn, addr = listener.sock.accept()
        return _Sock(conn, "connection", f"{addr[0]}:{addr[1]}")
    except socket.timeout:
        raise VMError("tcp_accept: timed out waiting for a connection") from None
    except OSError as e:
        raise VMError(f"tcp_accept: {e}") from e


def tcp_connect(host, port, timeout=10.0):
    """Open a TCP connection to a remote host."""
    try:
        s = socket.create_connection((str(host), int(port)), timeout=float(timeout))
    except OSError as e:
        raise VMError(f"tcp_connect: cannot reach {host}:{port}: {e}") from e
    return _Sock(s, "connection", f"{host}:{port}")


def tcp_send(conn, data):
    """Send text over a connection; returns the byte count."""
    if not isinstance(conn, _Sock) or conn.kind != "connection":
        raise VMError("tcp_send: argument must be a connected socket")
    payload = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    try:
        conn.sock.sendall(payload)
    except OSError as e:
        raise VMError(f"tcp_send: {e}") from e
    return len(payload)


def tcp_receive(conn, limit=65536, timeout=None):
    """Read up to `limit` bytes as text; empty text means the peer closed."""
    if not isinstance(conn, _Sock) or conn.kind != "connection":
        raise VMError("tcp_receive: argument must be a connected socket")
    try:
        if timeout is not None:
            conn.sock.settimeout(float(timeout))
        return conn.sock.recv(int(limit)).decode("utf-8", "replace")
    except socket.timeout:
        raise VMError("tcp_receive: timed out") from None
    except OSError as e:
        raise VMError(f"tcp_receive: {e}") from e


def tcp_close(sock):
    """Close a socket."""
    if not isinstance(sock, _Sock):
        raise VMError("tcp_close: argument must be a socket")
    try:
        sock.sock.close()
    except OSError:
        pass
    return None


# ---------------------------------------------------------------- http server
def _request_map(handler, body):
    parsed = urlparse(handler.path)
    query = {k: v[0] if len(v) == 1 else v for k, v in parse_qs(parsed.query).items()}
    return {
        "method": handler.command,
        "path": parsed.path,
        "query": query,
        "headers": {k.lower(): v for k, v in handler.headers.items()},
        "body": body,
        "client": handler.client_address[0],
    }


def _normalise(response):
    """Accept a map, or bare text, and produce (status, headers, body)."""
    if isinstance(response, str):
        return 200, {"content-type": "text/plain; charset=utf-8"}, response
    if response is None:
        return 204, {}, ""
    if not isinstance(response, dict):
        raise VMError(
            "serve: the handler must return a Map like "
            '{"status": 200, "body": "..."} or Text'
        )
    status = int(response.get("status", 200))
    headers = {str(k).lower(): str(v) for k, v in (response.get("headers") or {}).items()}
    if "json" in response:
        body = json.dumps(response["json"])
        headers.setdefault("content-type", "application/json")
    else:
        body = response.get("body", "")
        if not isinstance(body, str):
            body = json.dumps(body)
            headers.setdefault("content-type", "application/json")
        else:
            headers.setdefault("content-type", "text/plain; charset=utf-8")
    return status, headers, body


MAX_BODY_BYTES = 1_000_000
"""Default request-body cap for `serve` (~1 MB); larger requests get 413."""


def _ssl_context(spec):
    """Build a server SSLContext from a cert path or a [cert, key] pair."""
    if isinstance(spec, str):
        cert, key = spec, None
    elif isinstance(spec, (list, tuple)) and len(spec) == 2:
        cert, key = spec
    else:
        raise VMError("serve: ssl must be a cert path or a [cert, key] pair")
    cert = str(cert)
    if not Path(cert).is_file():
        raise VMError(f"serve: ssl certificate not found: {cert}")
    if key is not None:
        key = str(key)
        if not Path(key).is_file():
            raise VMError(f"serve: ssl key not found: {key}")
    ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_SERVER)
    try:
        ctx.load_cert_chain(certfile=cert, keyfile=key)
    except Exception as e:
        raise VMError(f"serve: cannot load ssl certificate: {e}") from e
    return ctx


def serve(port, handler, host="0.0.0.0", background=False, ssl=None):
    """Run an HTTP (or HTTPS, when `ssl` is given) server.

    Every request calls `handler(request)`. `ssl` accepts a single PEM file
    holding both certificate and key, or a `[cert, key]` pair.
    """
    if not callable(handler):
        raise VMError("serve: second argument must be a function taking a request")
    ctx = _ssl_context(ssl) if ssl is not None else None

    class _H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 30  # idle keep-alive connections cannot pin a worker thread
        MAX_BODY = MAX_BODY_BYTES

        def log_message(self, *a):  # silence stderr access logs
            pass

        def _dispatch(self):
            try:
                length = int(self.headers.get("content-length") or 0)
                if length > self.MAX_BODY:
                    data = b"request body too large"
                    self.send_response(413)
                    self.send_header("content-length", str(len(data)))
                    self.send_header("connection", "close")
                    self.end_headers()
                    self.wfile.write(data)
                    self.close_connection = True
                    return
                body = self.rfile.read(length).decode("utf-8", "replace") if length else ""
                req = _request_map(self, body)
                status, headers, text = _normalise(handler(req))
            except Exception as e:  # noqa: BLE001 - a handler fault is a 500
                status, headers, text = 500, {"content-type": "text/plain"}, f"handler error: {e}"
            data = text.encode("utf-8")
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = _dispatch

    try:
        httpd = ThreadingHTTPServer((str(host), int(port)), _H)
    except OSError as e:
        raise VMError(f"serve: cannot bind {host}:{port}: {e}") from e
    httpd.daemon_threads = True
    if ctx is not None:
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

    ident = _next[0]
    _next[0] += 1
    _servers[ident] = httpd

    if background:
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        return {"port": int(port), "host": str(host), "id": ident}

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        httpd.server_close()
        _servers.pop(ident, None)
    return None


def _shutdown_all():
    """Stop any servers still running at interpreter exit so a suite or
    long-running host process never keeps orphaned threads alive."""
    for ident, httpd in list(_servers.items()):
        try:
            httpd.shutdown()
        except Exception:
            pass
        try:
            httpd.server_close()
        except Exception:
            pass
        _servers.pop(ident, None)


atexit.register(_shutdown_all)


def serve_stop(server):
    """Stop a background server started by serve(..., background: true)."""
    ident = server.get("id") if isinstance(server, dict) else server
    httpd = _servers.pop(ident, None)
    if httpd is None:
        raise VMError("serve_stop: no such server")
    httpd.shutdown()
    httpd.server_close()
    return None
