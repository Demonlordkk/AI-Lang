# A small end-to-end program: parse, aggregate, report.
let raw := [
    {"user": "ada",   "action": "login",  "ms": 120},
    {"user": "grace", "action": "query",  "ms": 880},
    {"user": "ada",   "action": "query",  "ms": 240},
    {"user": "alan",  "action": "login",  "ms": 95},
    {"user": "grace", "action": "logout", "ms": 60}
].

emit "events: {len(raw)}  users: {len(counts(pluck(raw, "user")))}".

let slow := raw |> filter(\e -> e.ms > 200) |> sort_by(\e -> 0 - e.ms).
repeat e at i in slow:
    emit "  {i + 1}. {e.user} {e.action} {e.ms}ms".
done.

let by_user := group_by(raw, \e -> e.user).
repeat name in sort(keys(by_user)):
    let events := by_user[name].
    emit "{name}: {len(events)} events, {sum_by(events, \e -> e.ms)}ms total".
done.

let [fastest, slowest] := [min_by(raw, \e -> e.ms), max_by(raw, \e -> e.ms)].
emit "fastest {fastest.user} ({fastest.ms}ms), slowest {slowest.user} ({slowest.ms}ms)".

repeat e in raw:
    given e.action:
        is "login":
            next.
            is "query":
                when e.ms > 500:
                    emit "slow query by {e.user}".
                done.
            done.
        done.
