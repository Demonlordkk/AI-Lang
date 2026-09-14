# Log analyzer: parse, aggregate, and report. ~50 lines total.

record Entry:
    time: Text.
    method: Text.
    path: Text.
    status: Int.
    ms: Int.
done.

fn parse_line(line: Text) -> Any:
    let p := split(trim(line), " ").
    when len(p) != 5:
        give nothing.
    done.
    give Entry(p[0], p[1], p[2], to Int(p[3]), to Int(p[4])).
done.

fn group_by(items: List, key: Function) -> Map:
    var out := {}.
    repeat it in items:
        let k := key(it).
        when has(out, k):
            out[k] <- push(out[k], it).
        else:
            out[k] <- [it].
        done.
    done.
    give out.
done.

let raw := split(trim(read_file("access.log")), "\n").
let entries := raw |> map(parse_line) |> filter(\e -> not is_nothing(e)).

emit "Parsed {len(entries)} of {len(raw)} lines".
emit "".

emit "=== Traffic by endpoint ===".
let by_path := group_by(entries, \e -> e.path).
repeat path in sort(keys(by_path)):
    let hits := by_path[path].
    let avg := (hits |> map(\e -> e.ms) |> sum()) / len(hits).
    emit "{pad(path, 16)}{pad_left(to Text(len(hits)), 3)} hits   avg {round(avg, 1)}ms".
done.

emit "".
emit "=== Errors ===".
let errors := filter(entries, \e -> e.status >= 400).
when len(errors) == 0:
    emit "none".
else:
    repeat s in sort(unique(map(errors, \e -> e.status))):
        let n := count(errors, \e -> e.status == s).
        emit "  HTTP {s}: {n}".
    done.
done.

let slowest := first(sort_by(entries, \e -> 0 - e.ms)).
emit "".
emit "Slowest: {slowest.path} ({slowest.ms}ms)".
emit "Error rate: {round(100.0 * len(errors) / len(entries), 1)}%".
