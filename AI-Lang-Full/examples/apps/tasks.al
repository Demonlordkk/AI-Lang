# A professional command-line task manager, written entirely in AI-Lang.
#
# This is what "normal app building" looks like in the language: a real
# terminal product with declarative argument parsing, generated usage,
# aligned tables, colored output, interactive confirmation, and JSON
# persistence -- all from the cli package plus ordinary AI-Lang.
#
#   tasks.al add "ship the release"
#   tasks.al add fix -- crash -- in -- parser      # multi-word titles
#   tasks.al list                 # open tasks
#   tasks.al list --all --limit 5 # with limit
#   tasks.al done 3
#   tasks.al rm 2
#   tasks.al stats
#   tasks.al clear
#
# State lives in tasks.json next to the script (override with --file).

use packages/cli as c.

let spec := {
    "--all": "flag",
    "--limit": "int",
    "--file": "text"
}.

let usage_text := c.usage("tasks", spec, [
    "<command>  one of: add | list | done | rm | stats | clear",
    "add <title...>     add a task",
    "done <id>          mark a task done",
    "rm <id>            remove a task",
    "list [--all] [--limit N]   show tasks",
    "stats              counts and completion",
    "clear              delete all tasks (asks)"
]).

# ------------------------------------------------------------------- storage
fn default_db() -> Map:
    give {"tasks": [], "next_id": 1}.
done.

fn load(path: Text) -> Map:
    when not path_exists(path):
        give default_db().
    done.
    give json_decode(read_file(path)).
done.

fn save(path: Text, db: Map) -> Any:
    let r := write_file(path, json_encode(db)).
    give r.
done.

fn tasks_of(db: Map) -> List:
    give get(db, "tasks", []).
done.

fn id_from(arg_list: List, command: Text) -> Int:
    when len(arg_list) < 2:
        raise "usage: tasks " + command + " <id>".
    done.
    give int(arg_list[1]).
done.

fn is_known(command: Text) -> Bool:
    when command == "add":
        give true.
    done.
    when command == "list":
        give true.
    done.
    when command == "done":
        give true.
    done.
    when command == "rm":
        give true.
    done.
    when command == "stats":
        give true.
    done.
    when command == "clear":
        give true.
    done.
    give false.
done.

fn find_index(tasks: List, id: Int) -> Int:
    var i := 0.
    repeat t in tasks:
        when get(t, "id", -1) == id:
            give i.
        done.
        i <- i + 1.
    done.
    give -1.
done.

# --------------------------------------------------------------------- main
let argv := args().
let p := c.parse(argv, spec).
let path := get(p.flags, "--file", script_dir() + "/tasks.json").
let command := get(p.positional, 0, "").

when len(p.positional) == 0:
    emit usage_text.
    exit(0).
done.

# -- add ---------------------------------------------------------------------
when command == "add":
    when len(p.positional) < 2:
        emit c.color("error: add needs a title", 31).
        exit(2).
    done.
    var title := p.positional[1].
    var i := 2.
    while i < len(p.positional):
        title <- title + " " + p.positional[i].
        i <- i + 1.
    done.
    let db := load(path).
    let id := get(db, "next_id", 1).
    var tasks := tasks_of(db).
    append(tasks, {"id": id, "title": title, "done": false, "at": clock()}).
    db.tasks <- tasks.
    db.next_id <- id + 1.
    save(path, db).
    emit c.color("added #" + to Text(id) + ": " + title, 32).
done.

# -- list --------------------------------------------------------------------
when command == "list":
    let db := load(path).
    var tasks := tasks_of(db).
    let limit := get(p.flags, "--limit", -1).
    when limit >= 0:
        var first := [].
        var i := 0.
        while i < len(tasks) and i < limit:
            append(first, tasks[i]).
            i <- i + 1.
        done.
        tasks <- first.
    done.
    var rows := [].
    repeat t in tasks:
        let is_done := get(t, "done", false).
        let show := not is_done or has(p.flags, "--all").
        when show:
            var mark := " ".
            when is_done:
                mark <- "x".
            done.
            append(rows, {"id": get(t, "id", 0), "status": mark, "title": get(t, "title", "")}).
        done.
    done.
    when len(rows) == 0:
        emit "nothing here".
    done.
    when len(rows) > 0:
        emit c.table(["id", "status", "title"], rows).
    done.
done.

# -- done --------------------------------------------------------------------
when command == "done":
    let id := id_from(p.positional, "done").
    let db := load(path).
    var tasks := tasks_of(db).
    let idx := find_index(tasks, id).
    when idx < 0:
        emit c.color("error: no task #" + to Text(id), 31).
        exit(2).
    done.
    var t := tasks[idx].
    tasks[idx] <- set(t, "done", true).
    db.tasks <- tasks.
    save(path, db).
    emit c.color("done #" + to Text(id) + ": " + to Text(get(t, "title", "")), 32).
done.

# -- rm ----------------------------------------------------------------------
when command == "rm":
    let id := id_from(p.positional, "rm").
    let db := load(path).
    var tasks := tasks_of(db).
    let idx := find_index(tasks, id).
    when idx < 0:
        emit c.color("error: no task #" + to Text(id), 31).
        exit(2).
    done.
    var kept := [].
    var i := 0.
    while i < len(tasks):
        when i != idx:
            append(kept, tasks[i]).
        done.
        i <- i + 1.
    done.
    db.tasks <- kept.
    save(path, db).
    emit c.color("removed #" + to Text(id), 33).
done.

# -- stats -------------------------------------------------------------------
when command == "stats":
    let db := load(path).
    var total := 0.
    var done_n := 0.
    repeat t in tasks_of(db):
        total <- total + 1.
        when get(t, "done", false):
            done_n <- done_n + 1.
        done.
    done.
    var pct := 0.
    when total > 0:
        pct <- to Int(to Real(done_n) / to Real(total) * 100.0 + 0.5).
    done.
    emit c.table(["total", "open", "done", "percent"], [[total, total - done_n, done_n, pct]]).
done.

# -- clear -------------------------------------------------------------------
when command == "clear":
    var keep := true.
    when not c.confirm("delete all tasks?"):
        keep <- false.
    done.
    when keep and not c.confirm("really delete?"):
        keep <- false.
    done.
    when not keep:
        emit "kept them".
    done.
    when keep:
        save(path, default_db()).
        emit c.color("cleared.", 33).
    done.
done.

# -- unknown -----------------------------------------------------------------
when not is_known(command):
    emit c.color("error: unknown command: " + command, 31).
    emit usage_text.
    exit(2).
done.
