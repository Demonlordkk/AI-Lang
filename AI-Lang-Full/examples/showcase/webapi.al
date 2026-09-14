# A complete REST service: database-backed, JSON API, with a chart endpoint.
let db := db_open("./notes.db").
# start from a clean table so the example prints the same thing every run
db_exec(db, "drop table if exists notes").
db_exec(db, "create table notes(id integer primary key autoincrement, title text, body text, created text)").

fn json_ok(payload: Any) -> Map:
    give {"status": 200, "json": payload}.
done.

fn list_notes(req: Map) -> Map:
    give json_ok(db_query(db, "select * from notes order by id desc")).
done.

fn create_note(req: Map) -> Map:
    let data := json_decode(req.body).
    when not has(data, "title"):
        give {"status": 400, "json": {"error": "title is required"}}.
    done.
    db_exec(db, "insert into notes(title, body, created) values(?, ?, ?)",
        [get(data, "title"), get(data, "body", ""), timestamp()]).
    let row := db_one(db, "select * from notes order by id desc limit 1").
    give {"status": 201, "json": row}.
done.

fn get_note(req: Map) -> Map:
    let id := int(replace(req.path, "/notes/", "")).
    let row := db_one(db, "select * from notes where id = ?", [id]).
    when is_nothing(row):
        give {"status": 404, "json": {"error": "no note " + to Text(id)}}.
    done.
    give json_ok(row).
done.

fn stats(req: Map) -> Map:
    let rows := db_query(db, "select title, length(body) as size from notes").
    when len(rows) == 0:
        give json_ok({"count": 0}).
    done.
    let sizes := map(rows, \r -> r.size).
    plot("./sizes.png", map(sizes, \s -> real(s))).
    give json_ok({
        "count": len(rows),
        "mean_size": round(mean(sizes), 2),
        "largest": max(sizes),
        "chart": "/stats.png"
    }).
done.

fn router(req: Map) -> Map:
    when req.path == "/notes" and req.method == "GET":
        give list_notes(req).
    elif req.path == "/notes" and req.method == "POST":
        give create_note(req).
    elif starts_with(req.path, "/notes/"):
        give get_note(req).
    elif req.path == "/stats":
        give stats(req).
    elif req.path == "/health":
        give {"status": 200, "body": "ok"}.
    done.
    give {"status": 404, "json": {"error": "no route for " + req.path}}.
done.

let srv := serve(8123, router, "127.0.0.1", true).
emit "serving on " + to Text(srv.port).
sleep(0.4).

emit http_post("http://127.0.0.1:8123/notes", json_encode({"title": "first", "body": "hello world"})).status.
emit http_post("http://127.0.0.1:8123/notes", json_encode({"title": "second", "body": "a much longer body here"})).status.
emit http_post("http://127.0.0.1:8123/notes", json_encode({"body": "no title"})).status.
emit http_get("http://127.0.0.1:8123/notes/1").status.
emit http_get("http://127.0.0.1:8123/notes/99").status.
emit http_get("http://127.0.0.1:8123/stats").body.
emit "count: " + to Text(len(json_decode(http_get("http://127.0.0.1:8123/notes").body))).
serve_stop(srv).
db_close(db).
emit "done".
