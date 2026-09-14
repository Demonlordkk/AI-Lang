# In-memory task store with a JSON persistence layer.

record Task:
    id: Int.
    title: Text.
    done: Bool.
    priority: Int.
done.

fn new_store() -> Map:
    give {"tasks": [], "next_id": 1}.
done.

fn add(store: Map, title: Text, priority: Int) -> Map:
    let id := store["next_id"].
    let task := Task(id: id, title: title, done: false, priority: priority).
    give {
        "tasks": push(store["tasks"], task),
        "next_id": id + 1
    }.
done.

fn complete(store: Map, id: Int) -> Map:
    let updated := map(store["tasks"], fn(t: Any) -> Any:
        when t.id == id:
            give Task(id: t.id, title: t.title, done: true, priority: t.priority).
        done.
        give t.
    done).
    give {"tasks": updated, "next_id": store["next_id"]}.
done.

fn pending(store: Map) -> List:
    give filter(store["tasks"], \t -> not t.done).
done.

fn by_priority(store: Map) -> List:
    give sort_by(store["tasks"], \t -> 0 - t.priority).
done.

fn to_json(store: Map) -> Text:
    give json_encode(map(store["tasks"], \t -> {
        "id": t.id,
        "title": t.title,
        "done": t.done,
        "priority": t.priority
    })).
done.
