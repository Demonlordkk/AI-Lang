# AI-Lang data transformations.
#
# These helpers intentionally use ordinary Lists and Maps. They have no
# hidden I/O, global state, or random choices, which makes a pipeline easy to
# test and replay.

fn chunked(xs: List, size: Int) -> List:
    needs size > 0.
    var out := [].
    var i := 0.
    while i < len(xs):
        append(out, slice(xs, i, min(i + size, len(xs)))).
        i <- i + size.
    done.
    give out.
done.

fn windows(xs: List, size: Int) -> List:
    needs size > 0.
    var out := [].
    var i := 0.
    while i + size <= len(xs):
        append(out, slice(xs, i, i + size)).
        i <- i + 1.
    done.
    give out.
done.

fn scan(xs: List, initial: Any, step: Function) -> List:
    var out := [initial].
    var state := initial.
    repeat item in xs:
        state <- step(state, item).
        append(out, state).
    done.
    give out.
done.

fn flatten_once(xs: List) -> List:
    var out := [].
    repeat item in xs:
        when type_of(item) == "List":
            extend(out, item).
        else:
            append(out, item).
        done.
    done.
    give out.
done.

fn flatten_deep(xs: List) -> List:
    var out := [].
    fn visit(item: Any) -> Void:
        when type_of(item) == "List":
            repeat child in item:
                visit(child).
            done.
        else:
            append(out, item).
        done.
    done.
    repeat item in xs:
        visit(item).
    done.
    give out.
done.

fn frequencies(xs: List) -> Map:
    var out := {}.
    repeat item in xs:
        let key := to Text(item).
        out <- set(out, key, get(out, key, 0) + 1).
    done.
    give out.
done.

fn partition_at(xs: List, predicate: Function) -> List:
    var yes := [].
    var no := [].
    repeat item in xs:
        when predicate(item):
            append(yes, item).
        else:
            append(no, item).
        done.
    done.
    give [yes, no].
done.

fn transpose(rows: List) -> List:
    when len(rows) == 0:
        give [].
    done.
    let width := len(rows[0]).
    give map(range(width), \i -> map(rows, \row -> row[i])).
done.

fn distinct_by(xs: List, key: Function) -> List:
    var seen := {}.
    var out := [].
    repeat item in xs:
        let k := to Text(key(item)).
        when not has(seen, k):
            seen <- set(seen, k, true).
            append(out, item).
        done.
    done.
    give out.
done.
