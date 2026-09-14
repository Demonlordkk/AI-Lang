# A web-app toolkit written in AI-Lang itself.
#
# It sits one level above serve(): you declare routes once (with :params),
# plug in static files and middleware, and start the app. Everything is
# ordinary AI-Lang, so routing logic is testable with app.dispatch() and a
# plain request map, no server required.
#
#     use webapp as w.
#     let app := w.new().
#     w.route(app, "GET", "/notes/:id", \req -> w.html(200, "note " + req.params.id)).
#     w.start(app, 8080, "127.0.0.1", true).

record App:
    routes: List.
    middlewares: List.
    statics: List.
    vars: Map.
done.

record Route:
    method: Text.
    segments: List.
    names: List.
    handler: Function.
done.

fn new() -> App:
    give App([], [], [], {}).
done.

# Split a URL path into segments: "/notes/42/" -> ["notes", "42"].
fn split_path(path: Text) -> List:
    var parts := split(path, "/").
    when starts_with(path, "/"):
        parts <- slice(parts, 1, len(parts)).
    done.
    when len(parts) > 1 and last(parts) == "":
        parts <- slice(parts, 0, len(parts) - 1).
    done.
    give parts.
done.

# Register a route. `:name` in the pattern captures that path segment into
# req.params.name when the request matches.
fn route(app: App, method: Text, pattern: Text, handler: Function) -> App:
    let segments := split_path(pattern).
    var names := [].
    var i := 0.
    while i < len(segments):
        when starts_with(segments[i], ":"):
            append(names, slice(segments[i], 1, len(segments[i]))).
        done.
        i <- i + 1.
    done.
    append(app.routes, Route(upper(method), segments, names, handler)).
    give app.
done.

# Serve files from `dir` for URLs under `prefix`.
fn static(app: App, prefix: Text, dir: Text) -> App:
    append(app.statics, [prefix, dir]).
    give app.
done.

# Middleware runs in registration order: each gets the request and a `next`
# function that continues the chain. Call next exactly once to continue.
fn middleware(app: App, fn_req: Function) -> App:
    append(app.middlewares, fn_req).
    give app.
done.

# Match one route against a path; give the params map, or nothing.
fn match_route(route: Route, path: Text) -> Any:
    let segs := split_path(path).
    when len(segs) != len(route.segments):
        give nothing.
    done.
    var params := {}.
    var i := 0.
    while i < len(segs):
        let pat := route.segments[i].
        let val := segs[i].
        when starts_with(pat, ":"):
            when val == "":
                give nothing.
            done.
            params <- set(params, slice(pat, 1, len(pat)), val).
        elif pat != val:
            give nothing.
        done.
        i <- i + 1.
    done.
    give params.
done.

# Run the middleware chain, ending at the route handler.
fn call_chain(app: App, handler: Function, i: Int, req: Map) -> Map:
    when i >= len(app.middlewares):
        give handler(req).
    done.
    let mw := app.middlewares[i].
    let rest := fn(r: Map) -> Map:
        give call_chain(app, handler, i + 1, r).
    done.
    give mw(req, rest).
done.

# Find the route for a request; give [handler, request-with-params] or nothing.
fn find_handler(app: App, req: Map) -> Any:
    var i := 0.
    while i < len(app.routes):
        let r := app.routes[i].
        when r.method == upper(req.method):
            let params := match_route(r, req.path).
            when not is_nothing(params):
                give [r.handler, set(req, "params", params)].
            done.
        done.
        i <- i + 1.
    done.
    give nothing.
done.

# Dispatch a request map to the app. This is the whole of the router: no
# sockets involved, so tests can drive it directly. The request flows
# through the middleware chain and on to the route (or static serving),
# so middleware can decorate it — req.params is present on route matches.
fn dispatch(app: App, req: Map) -> Map:
    let hit := find_handler(app, req).
    var cur := req.
    when not is_nothing(hit):
        let handler := hit[0].
        cur <- hit[1].
        let route_call := fn(r: Map) -> Map:
            give handler(r).
        done.
        give call_chain(app, route_call, 0, cur).
    done.
    let static_call := fn(r: Map) -> Map:
        give serve_static(app, r).
    done.
    give call_chain(app, static_call, 0, cur).
done.

fn serve_static(app: App, req: Map) -> Map:
    var i := 0.
    while i < len(app.statics):
        let entry := app.statics[i].
        let prefix := entry[0].
        var hit := req.path == prefix.
        when not hit and prefix == "/":
            hit <- true.
        done.
        when not hit and prefix != "/":
            hit <- starts_with(req.path, prefix + "/").
        done.
        when hit:
            give serve_file(app, req, entry[1], prefix).
        done.
        i <- i + 1.
    done.
    give {"status": 404, "json": {"error": "no route for " + req.path}}.
done.

fn serve_file(app: App, req: Map, dir: Text, prefix: Text) -> Map:
    var rel := req.path.
    when req.path != prefix:
        rel <- replace(rel, prefix, "").
    done.
    when starts_with(rel, "/"):
        rel <- slice(rel, 1, len(rel)).
    done.
    when contains(rel, ".."):
        give {"status": 404, "json": {"error": "not found"}}.
    done.
    when rel == "":
        when path_exists(dir + "/index.html"):
            rel <- "index.html".
        else:
            give {"status": 404, "json": {"error": "not found"}}.
        done.
    done.
    let full := dir + "/" + rel.
    when not path_exists(full):
        give {"status": 404, "json": {"error": "no file " + rel}}.
    done.
    let body := read_file(full).
    give {"status": 200, "headers": {"content-type": content_type(rel)}, "body": body}.
done.

fn content_type(name: Text) -> Text:
    let dot := index_of(name, ".").
    var ext := "txt".
    when dot >= 0:
        ext <- slice(name, dot + 1, len(name)).
    done.
    given ext:
    is "html", "htm":
        give "text/html; charset=utf-8".
    is "css":
        give "text/css".
    is "js":
        give "application/javascript".
    is "json":
        give "application/json".
    is "png":
        give "image/png".
    is "svg":
        give "image/svg+xml".
    is "csv":
        give "text/csv".
    else:
        give "text/plain; charset=utf-8".
    done.
done.

# --------------------------------------------------------------- responses
fn json(status: Int, payload: Any) -> Map:
    give {"status": status, "json": payload}.
done.

fn text(status: Int, body: Text) -> Map:
    give {"status": status, "headers": {"content-type": "text/plain; charset=utf-8"}, "body": body}.
done.

fn html(status: Int, body: Text) -> Map:
    give {"status": status, "headers": {"content-type": "text/html; charset=utf-8"}, "body": body}.
done.

# Decode the request body as a JSON object; give {} when there is none.
fn body_json(req: Map) -> Map:
    when req.body == "":
        give {}.
    done.
    let parsed := json_decode(req.body).
    when type_of(parsed) != "Map":
        give {}.
    done.
    give parsed.
done.

# Start serving the app. Returns the server info (pass it to serve_stop).
fn start(app: App, port: Int, host: Text, background: Bool) -> Any:
    let router := fn(req: Map) -> Map:
        give dispatch(app, req).
    done.
    give serve(port, router, host, background).
done.

# --------------------------------------------------------------- professional
# Standard middleware factories for production-style apps: request logging,
# a JSON error handler that turns exceptions into 500s, and a simple
# in-memory rate limiter.

# Request logging: prints "METHOD /path -> STATUS" for every dispatched
# request. Works under w.dispatch() too, so tests can see the log.
fn make_logger() -> Function:
    give fn(req: Map, cont: Function) -> Map:
        let res := cont(req).
        emit req.method + " " + req.path + " -> " + to Text(res.status).
        give res.
    done.
done.

# Error handler: any exception raised by a route becomes a clean JSON 500
# (and a one-line log when log is true) instead of a failed request.
fn make_error_handler(log: Bool) -> Function:
    give fn(req: Map, cont: Function) -> Map:
        attempt:
            give cont(req).
        rescue e:
            when log:
                emit "error on " + req.method + " " + req.path + ": " + e.message.
            done.
            give {"status": 500, "headers": {"content-type": "application/json"}, "body": json_encode({"error": e.message})}.
        done.
    done.
done.

# In-memory rate limit: at most `limit` requests per `window_s` seconds,
# tracked per "METHOD /path". Over-limit requests get a clean 429 JSON.
fn make_throttle(limit: Int, window_s: Real) -> Function:
    var hits := {}.
    give fn(req: Map, cont: Function) -> Map:
        let t := now().
        let key := req.method + " " + req.path.
        var start := get(hits, key, 0.0).
        var count := get(hits, key + "#n", 0).
        when t - start >= window_s:
            start <- t.
            count <- 0.
        done.
        when count >= limit:
            give {"status": 429, "headers": {"content-type": "application/json"}, "body": json_encode({"error": "rate limit exceeded"})}.
        done.
        hits <- set(hits, key, start).
        hits <- set(hits, key + "#n", count + 1).
        give cont(req).
    done.
done.
