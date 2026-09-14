use lib/store as store.

var db := store.new_store().
db <- store.add(db, "Write the parser", 3).
db <- store.add(db, "Fix the VM stack bug", 5).
db <- store.add(db, "Ship v2.0", 4).
db <- store.add(db, "Update the docs", 1).

db <- store.complete(db, 2).

emit "=== All tasks by priority ===".
repeat t in store.by_priority(db):
    var mark := "[ ]".
    when t.done:
        mark <- "[x]".
    done.
    emit "{mark} P{t.priority}  {t.title}".
done.

emit "".
emit "Pending: " + to Text(len(store.pending(db))) + " of " + to Text(len(db["tasks"])).
emit "JSON: {store.to_json(db)}".
