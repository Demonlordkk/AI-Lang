# A complete notes web app in AI-Lang: SQLite storage, a JSON API, and a
# single-page HTML UI, wired together by the webapp package.
#
# The router is a plain function from request to response, so the final
# block drives the finished app over HTTP and proves every endpoint —
# this file is its own integration test.

use packages/webapp as w.

let db := db_open("./notes_demo.db").
# start from a clean table so the self-test prints the same thing every run
db_exec(db, "drop table if exists notes").
db_exec(db, "create table notes(id integer primary key autoincrement, title text, body text, created text)").

let app := w.new().
w.static(app, "/", "static").

fn list_notes(req: Map) -> Map:
    give w.json(200, db_query(db, "select * from notes order by id")).
done.

fn create_note(req: Map) -> Map:
    let data := w.body_json(req).
    when not has(data, "title"):
        give w.json(400, {"error": "title is required"}).
    done.
    db_exec(db, "insert into notes(title, body, created) values(?, ?, ?)",
        [data.title, get(data, "body", ""), "2026-01-01"]).
    let row := db_one(db, "select * from notes order by id desc limit 1").
    give w.json(201, row).
done.

fn get_note(req: Map) -> Map:
    let row := db_one(db, "select * from notes where id = ?", [int(req.params.id)]).
    when is_nothing(row):
        give w.json(404, {"error": "no note " + req.params.id}).
    done.
    give w.json(200, row).
done.

fn delete_note(req: Map) -> Map:
    db_exec(db, "delete from notes where id = ?", [int(req.params.id)]).
    give w.json(200, {"deleted": int(req.params.id)}).
done.

w.route(app, "GET", "/notes", list_notes).
w.route(app, "POST", "/notes", create_note).
w.route(app, "GET", "/notes/:id", get_note).
w.route(app, "DELETE", "/notes/:id", delete_note).
w.route(app, "GET", "/health", \req -> w.text(200, "ok")).

# Middleware demo: count every request in the app's shared vars. The
# handler gets the request and a `cont` function; calling cont continues
# the chain to the route.
w.middleware(app, fn(req: Map, cont: Function) -> Map:
    let out := cont(req).
    app.vars <- set(app.vars, "count", get(app.vars, "count", 0) + 1).
    give out.
done).

let srv := w.start(app, 8124, "127.0.0.1", true).
emit "notes app serving on port {srv.port}".
sleep(0.4).

emit "health:  " + to Text(http_get("http://127.0.0.1:8124/health").status).
emit "create:  " + to Text(http_post("http://127.0.0.1:8124/notes", json_encode({"title": "first", "body": "hello"})).status).
emit "create:  " + to Text(http_post("http://127.0.0.1:8124/notes", json_encode({"title": "second", "body": "world"})).status).
emit "bad:     " + to Text(http_post("http://127.0.0.1:8124/notes", json_encode({"body": "no title"})).status).
emit "get 1:   " + http_get("http://127.0.0.1:8124/notes/1").body.
emit "get 99:  " + to Text(http_get("http://127.0.0.1:8124/notes/99").status).
emit "delete:  " + to Text(http_request("http://127.0.0.1:8124/notes/1", "DELETE", nothing, nothing).status).
emit "list:    " + http_get("http://127.0.0.1:8124/notes").body.
let page := http_get("http://127.0.0.1:8124/").
emit "page:    {page.status} ({len(page.body)} bytes of HTML)".
let served := get(app.vars, "count", 0).
emit "served {served} requests through middleware".

serve_stop(srv).
db_close(db).
delete_file("./notes_demo.db").
emit "done".
