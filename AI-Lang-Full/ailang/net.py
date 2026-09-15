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

import contextvars
import json
import math
import socket
import ssl as _ssl
import threading
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import atexit
from urllib.parse import parse_qs, urlparse

from pathlib import Path

from .capabilities import require, resource_host, resource_path
from .errors import VMError
from .resources import register as register_resource

_servers = {}
_next = [1]
_MAX_SOCKET_READ = 16 * 1024 * 1024


def _port(value, where):
    if type(value) is not int or not 0 <= value <= 65535:
        raise VMError(f"{where}: port must be an Int between 0 and 65535")
    return value


def _endpoint(host, port):
    return resource_host(f"{host}:{port}")


def _timeout(value, where, default=None):
    if value is None:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise VMError(f"{where}: timeout must be a finite positive number") from None
    if not math.isfinite(value) or value <= 0 or value > 24 * 60 * 60:
        raise VMError(f"{where}: timeout must be a finite positive number")
    return value


# --------------------------------------------------------------- raw sockets
class _Sock:
    __slots__ = ("sock", "id", "kind", "addr")

    def __init__(self, sock, kind, addr):
        self.sock = sock
        self.kind = kind
        self.addr = addr
        self.id = _next[0]
        _next[0] += 1
        register_resource(lambda sock=self: _close_quiet(sock))


def _close_quiet(sock):
    try:
        sock.sock.close()
    except (AttributeError, OSError):
        pass


def tcp_listen(port, host="0.0.0.0", backlog=64):
    """Open a listening TCP socket."""
    port = _port(port, "tcp_listen")
    if not isinstance(host, str) or not host:
        raise VMError("tcp_listen: host must be non-empty Text")
    if type(backlog) is not int or not 1 <= backlog <= 65535:
        raise VMError("tcp_listen: backlog must be an Int between 1 and 65535")
    require("net.listen", _endpoint(host, port), "tcp_listen")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(backlog)
    except OSError as e:
        raise VMError(f"tcp_listen: cannot listen on {host}:{port}: {e}") from e
    return _Sock(s, "listener", f"{host}:{port}")


def tcp_accept(listener, timeout=None):
    """Wait for a connection; returns a connected socket."""
    if not isinstance(listener, _Sock) or listener.kind != "listener":
        raise VMError("tcp_accept: argument must be a socket from tcp_listen")
    require("net.listen", resource_host(listener.addr), "tcp_accept")
    timeout = _timeout(timeout, "tcp_accept")
    previous_timeout = listener.sock.gettimeout()
    try:
        if timeout is not None:
            listener.sock.settimeout(timeout)
        conn, addr = listener.sock.accept()
        conn.settimeout(None)
        return _Sock(conn, "connection", f"{addr[0]}:{addr[1]}")
    except socket.timeout:
        raise VMError("tcp_accept: timed out waiting for a connection") from None
    except (OSError, ValueError) as e:
        raise VMError(f"tcp_accept: {e}") from None
    finally:
        if timeout is not None:
            try:
                listener.sock.settimeout(previous_timeout)
            except OSError:
                pass


def tcp_connect(host, port, timeout=10.0):
    """Open a TCP connection to a remote host."""
    if not isinstance(host, str) or not host:
        raise VMError("tcp_connect: host must be non-empty Text")
    port = _port(port, "tcp_connect")
    timeout = _timeout(timeout, "tcp_connect", 10.0)
    require("net.connect", _endpoint(host, port), "tcp_connect")
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.settimeout(None)
    except (OSError, ValueError) as e:
        raise VMError(f"tcp_connect: cannot reach {host}:{port}: {e}") from None
    return _Sock(s, "connection", f"{host}:{port}")


def _payload(data, where):
    if isinstance(data, str):
        return data.encode("utf-8")
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    if isinstance(data, list):
        if any(type(value) is not int or not 0 <= value <= 255 for value in data):
            raise VMError(f"{where}: byte List values must be Ints between 0 and 255")
        return bytes(data)
    raise VMError(f"{where}: data must be Text, bytes, or a List of bytes")


def tcp_send(conn, data):
    """Send UTF-8 text or bytes over a connection; returns the byte count."""
    if not isinstance(conn, _Sock) or conn.kind != "connection":
        raise VMError("tcp_send: argument must be a connected socket")
    require("net.connect", resource_host(conn.addr), "tcp_send")
    payload = _payload(data, "tcp_send")
    try:
        conn.sock.sendall(payload)
    except OSError as e:
        raise VMError(f"tcp_send: {e}") from None
    return len(payload)


def tcp_receive(conn, limit=65536, timeout=None):
    """Read up to `limit` bytes as UTF-8 text; empty text means peer closed."""
    if not isinstance(conn, _Sock) or conn.kind != "connection":
        raise VMError("tcp_receive: argument must be a connected socket")
    require("net.connect", resource_host(conn.addr), "tcp_receive")
    if type(limit) is not int or not 1 <= limit <= _MAX_SOCKET_READ:
        raise VMError(
            f"tcp_receive: limit must be an Int between 1 and {_MAX_SOCKET_READ}"
        )
    timeout = _timeout(timeout, "tcp_receive")
    previous_timeout = conn.sock.gettimeout()
    try:
        if timeout is not None:
            conn.sock.settimeout(timeout)
        return conn.sock.recv(limit).decode("utf-8", "replace")
    except socket.timeout:
        raise VMError("tcp_receive: timed out") from None
    except (OSError, ValueError) as e:
        raise VMError(f"tcp_receive: {e}") from None
    finally:
        if timeout is not None:
            try:
                conn.sock.settimeout(previous_timeout)
            except OSError:
                pass


def tcp_close(sock):
    """Close a socket."""
    if not isinstance(sock, _Sock):
        raise VMError("tcp_close: argument must be a socket")
    require("net.listen" if sock.kind == "listener" else "net.connect",
            resource_host(sock.addr), "tcp_close")
    try:
        sock.sock.close()
    except OSError:
        pass
    return None


# ---------------------------------------------------------------- http server
# HTTP is intentionally conservative: request bodies and headers are bounded,
# response framing is owned by the runtime, and every request receives an
# isolated VM execution context.  This keeps a slow or faulty handler from
# corrupting the next request on a keep-alive connection.
MAX_BODY_BYTES = 1_000_000
"""Default request-body cap for `serve` (~1 MB); larger requests get 413."""

MAX_HEADER_BYTES = 64 * 1024
"""Maximum combined HTTP request-header size accepted by `serve`."""

MAX_RESPONSE_BYTES = 16 * 1024 * 1024
"""Maximum encoded response body emitted by `serve`."""

_MAX_REQUEST_TARGET = 16 * 1024
_SERVERS = {}
_SERVERS_LOCK = threading.RLock()
_NEXT_SERVER_ID = 1
_MAX_SERVER_WORKERS = 128


class _HTTPBadRequest(Exception):
    """Internal marker for malformed client input (not a handler failure)."""


def _next_server_id():
    global _NEXT_SERVER_ID
    with _SERVERS_LOCK:
        ident = _NEXT_SERVER_ID
        _NEXT_SERVER_ID += 1
        return ident


def _request_map(handler, body):
    if len(handler.path) > _MAX_REQUEST_TARGET:
        raise _HTTPBadRequest("request target is too long")
    try:
        parsed = urlparse(handler.path)
        query = {
            k: v[0] if len(v) == 1 else v
            for k, v in parse_qs(parsed.query, max_num_fields=1000).items()
        }
    except ValueError as e:
        raise _HTTPBadRequest(f"invalid request target: {e}") from e
    return {
        "method": handler.command,
        "path": parsed.path,
        "query": query,
        "headers": {k.lower(): v for k, v in handler.headers.items()},
        "body": body,
        "client": handler.client_address[0],
    }


_HEADER_NAME = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


def _header_is_safe(name, value):
    # RFC 7230 field-names are tokens. Rejecting control characters and
    # newlines prevents response splitting even when headers originate in an
    # AI-Lang map supplied by an untrusted handler.
    if not isinstance(name, str) or not _HEADER_NAME.fullmatch(name):
        return False
    return not any(ord(ch) < 32 or ord(ch) == 127 for ch in value)


def _normalise(response):
    """Accept a map or bare text and produce ``(status, headers, body)``.

    Body framing headers are controlled by the server.  A handler may set
    content type and application headers but cannot create a conflicting
    Content-Length, Transfer-Encoding, or Connection header.
    """
    if isinstance(response, str):
        status, headers, body = 200, {"content-type": "text/plain; charset=utf-8"}, response.encode("utf-8")
    elif response is None:
        status, headers, body = 204, {}, b""
    elif isinstance(response, (bytes, bytearray, memoryview)):
        status, headers, body = 200, {"content-type": "application/octet-stream"}, bytes(response)
    elif not isinstance(response, dict):
        raise VMError(
            "serve: the handler must return a Map like "
            '{"status": 200, "body": "..."} or Text'
        )
    else:
        raw_status = response.get("status", 200)
        if type(raw_status) is not int:
            raise VMError("serve: response status must be an integer")
        status = raw_status
        raw_headers = response.get("headers") or {}
        if not isinstance(raw_headers, dict):
            raise VMError("serve: response headers must be a map")
        headers = {}
        for key, value in raw_headers.items():
            if not isinstance(key, str):
                raise VMError("serve: response header names must be Text")
            name = key.lower()
            text_value = str(value)
            if name in {"content-length", "transfer-encoding", "connection"}:
                continue
            if not _header_is_safe(name, text_value):
                raise VMError("serve: response contains an invalid header")
            headers[name] = text_value
        try:
            if "json" in response:
                body = json.dumps(response["json"], ensure_ascii=False, allow_nan=False).encode("utf-8")
                headers.setdefault("content-type", "application/json; charset=utf-8")
            else:
                raw_body = response.get("body", "")
                if isinstance(raw_body, (bytes, bytearray, memoryview)):
                    body = bytes(raw_body)
                    headers.setdefault("content-type", "application/octet-stream")
                elif isinstance(raw_body, str):
                    body = raw_body.encode("utf-8")
                    headers.setdefault("content-type", "text/plain; charset=utf-8")
                else:
                    body = json.dumps(raw_body, ensure_ascii=False, allow_nan=False).encode("utf-8")
                    headers.setdefault("content-type", "application/json; charset=utf-8")
        except (TypeError, ValueError, OverflowError) as e:
            raise VMError(f"serve: response cannot be encoded as JSON: {e}") from e

    if status < 100 or status > 599:
        raise VMError("serve: response status must be between 100 and 599")
    if len(body) > MAX_RESPONSE_BYTES:
        raise VMError("serve: response body is too large")
    return status, headers, body


def _ssl_context(spec):
    """Build a server SSLContext from a cert path or a [cert, key] pair."""
    if isinstance(spec, str):
        cert, key = spec, None
    elif isinstance(spec, (list, tuple)) and len(spec) == 2:
        cert, key = spec
    else:
        raise VMError("serve: ssl must be a cert path or a [cert, key] pair")
    cert = str(cert)
    require("fs.read", resource_path(cert), "serve ssl certificate")
    if not Path(cert).is_file():
        raise VMError(f"serve: ssl certificate not found: {cert}")
    if key is not None:
        key = str(key)
        require("fs.read", resource_path(key), "serve ssl key")
        if not Path(key).is_file():
            raise VMError(f"serve: ssl key not found: {key}")
    ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_SERVER)
    try:
        ctx.load_cert_chain(certfile=cert, keyfile=key)
    except Exception as e:
        raise VMError(f"serve: cannot load ssl certificate: {e}") from e
    return ctx


def _send_error(handler, status, message, close=True):
    try:
        handler.send_error(status, message)
    finally:
        if close:
            handler.close_connection = True


def _read_request_body(handler):
    values = handler.headers.get_all("content-length", [])
    if len(values) > 1:
        try:
            lengths = {int(value.strip()) for value in values}
        except (TypeError, ValueError):
            raise _HTTPBadRequest("invalid Content-Length") from None
        if len(lengths) != 1:
            raise _HTTPBadRequest("conflicting Content-Length headers")
        raw_length = next(iter(lengths))
    else:
        raw_length = values[0].strip() if values else "0"
        try:
            raw_length = int(raw_length)
        except (TypeError, ValueError):
            raise _HTTPBadRequest("invalid Content-Length") from None
    if raw_length < 0:
        raise _HTTPBadRequest("invalid Content-Length")
    if raw_length > MAX_BODY_BYTES:
        raise OverflowError("request body too large")

    transfer = (handler.headers.get("transfer-encoding") or "").strip().lower()
    if transfer and transfer != "identity":
        raise NotImplementedError("Transfer-Encoding is not supported")
    if not raw_length:
        return ""

    chunks = []
    remaining = raw_length
    while remaining:
        chunk = handler.rfile.read(min(64 * 1024, remaining))
        if not chunk:
            raise _HTTPBadRequest("request body ended before Content-Length")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks).decode("utf-8", "replace")


def _call_in_request_context(handler_fn, request):
    """Run a language closure on a child VM, never on its parent VM."""
    try:
        from .vm import Closure
    except ImportError:
        Closure = ()
    if isinstance(handler_fn, Closure):
        child_vm = handler_fn.vm.fork()
        handler_fn = Closure(handler_fn.code, handler_fn.env, child_vm, handler_fn.program)
    return handler_fn(request)


def serve(port, handler, host="0.0.0.0", background=False, ssl=None):
    """Run an HTTP (or HTTPS, when `ssl` is given) server.

    Every request calls ``handler(request)``. ``ssl`` accepts a single PEM
    file holding both certificate and key, or a ``[cert, key]`` pair.
    """
    if not callable(handler):
        raise VMError("serve: second argument must be a function taking a request")
    port = _port(port, "serve")
    require("net.listen", _endpoint(host, port), "serve")
    ctx = _ssl_context(ssl) if ssl is not None else None
    request_context = contextvars.copy_context()
    request_slots = threading.BoundedSemaphore(_MAX_SERVER_WORKERS)

    class _H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 30
        server_version = "AI-Lang"
        sys_version = ""

        def setup(self):
            super().setup()
            self.connection.settimeout(self.timeout)

        def version_string(self):
            return self.server_version

        def log_message(self, *args):  # silence access logs and user data
            pass

        def parse_request(self):
            if not super().parse_request():
                return False
            header_bytes = sum(len(k) + len(v) + 4 for k, v in self.headers.items())
            if header_bytes > MAX_HEADER_BYTES:
                _send_error(self, 431, "request headers too large")
                return False
            if len(self.path) > _MAX_REQUEST_TARGET:
                _send_error(self, 414, "request target too long")
                return False
            transfer = (self.headers.get("transfer-encoding") or "").strip().lower()
            if transfer and transfer != "identity":
                _send_error(self, 501, "Transfer-Encoding is not supported")
                return False
            return True

        def _write_response(self, status, headers, body):
            # RFC 9110 forbids bodies for these statuses; HEAD also keeps the
            # computed Content-Length while suppressing the bytes themselves.
            no_body_status = 100 <= status < 200 or status in {204, 304}
            send_body = self.command != "HEAD" and not no_body_status
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("content-length", str(0 if no_body_status else len(body)))
            if not send_body:
                self.send_header("connection", "close" if self.command == "HEAD" else "keep-alive")
            self.end_headers()
            if send_body:
                self.wfile.write(body)

        def _dispatch(self):
            if not request_slots.acquire(blocking=False):
                _send_error(self, 503, "server is busy")
                return
            try:
                body = _read_request_body(self)
                req = _request_map(self, body)
                # Context.run requires a fresh Context object per request.
                result = request_context.copy().run(_call_in_request_context, handler, req)
                status, headers, data = _normalise(result)
                self._write_response(status, headers, data)
            except _HTTPBadRequest as e:
                _send_error(self, 400, str(e))
            except OverflowError:
                _send_error(self, 413, "request body too large")
            except NotImplementedError as e:
                _send_error(self, 501, str(e))
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except Exception as e:  # noqa: BLE001 - a handler fault is a 500
                try:
                    self._write_response(
                        500,
                        {"content-type": "text/plain; charset=utf-8", "connection": "close"},
                        f"handler error: {e}".encode("utf-8", "replace"),
                    )
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                self.close_connection = True
            finally:
                request_slots.release()

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = _dispatch

    try:
        httpd = ThreadingHTTPServer((str(host), port), _H)
    except OSError as e:
        raise VMError(f"serve: cannot bind {host}:{port}: {e}") from e
    httpd.daemon_threads = True
    httpd.block_on_close = False
    if ctx is not None:
        try:
            httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        except Exception:
            httpd.server_close()
            raise

    ident = _next_server_id()
    with _SERVERS_LOCK:
        _SERVERS[ident] = httpd
    register_resource(lambda httpd=httpd: _stop_server_quiet(httpd))
    actual_port = int(httpd.server_address[1])

    if background:
        thread = threading.Thread(target=httpd.serve_forever, name=f"ailang-http-{ident}", daemon=True)
        thread.start()
        return {"port": actual_port, "host": str(host), "id": ident}

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        httpd.server_close()
        with _SERVERS_LOCK:
            _SERVERS.pop(ident, None)
    return None


def _stop_server_quiet(httpd):
    try:
        httpd.shutdown()
    except Exception:
        pass
    try:
        httpd.server_close()
    except Exception:
        pass


def _shutdown_all():
    """Stop any servers still running at interpreter exit."""
    with _SERVERS_LOCK:
        servers = list(_SERVERS.items())
        _SERVERS.clear()
    for _, httpd in servers:
        try:
            httpd.shutdown()
        except Exception:
            pass
        try:
            httpd.server_close()
        except Exception:
            pass


atexit.register(_shutdown_all)


def serve_stop(server):
    """Stop a background server started by ``serve(..., background: true)``."""
    if isinstance(server, dict):
        target = _endpoint(server.get("host", "0.0.0.0"), server.get("port", "*"))
    else:
        target = "*"
    require("net.listen", target, "serve_stop")
    ident = server.get("id") if isinstance(server, dict) else server
    with _SERVERS_LOCK:
        httpd = _SERVERS.pop(ident, None)
    if httpd is None:
        raise VMError("serve_stop: no such server")
    httpd.shutdown()
    httpd.server_close()
    return None
