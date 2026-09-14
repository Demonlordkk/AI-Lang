# AI-Lang advanced features — closures, pipelines, and data processing.

# Lambdas: `\x -> expr` for short forms, `fn(...) -> T: ... done` for blocks.
let double := \x -> x * 2.
let is_even := \x -> x % 2 == 0.
emit double(21).

# Closures capture their defining environment.
fn make_counter() -> Function:
    var total := 0.
    give fn() -> Int:
        total <- total + 1.
        give total.
    done.
done.

let tick := make_counter().
emit tick().
emit tick().
emit tick().

# The pipeline operator threads a value through a chain of calls.
# `x |> f(a)` is exactly `f(x, a)`, which keeps data flow left to right.
let result := range_from(1, 21, 1)
    |> filter(is_even)
    |> map(\x -> x * x)
    |> sum().
emit "sum of squares of evens 1..20: " + to Text(result).

# Higher-order functions over records.
record Employee:
    name: Text.
    team: Text.
    salary: Int.
done.

let staff := [
    Employee("Ada", "platform", 165000),
    Employee("Grace", "platform", 172000),
    Employee("Alan", "research", 158000),
    Employee("Katherine", "research", 169000)
].

# Group by an arbitrary key.
fn group_by(items: List, key: Function) -> Map:
    var out := {}.
    repeat item in items:
        let k := key(item).
        when has(out, k):
            out[k] <- push(out[k], item).
        else:
            out[k] <- [item].
        done.
    done.
    give out.
done.

let by_team := group_by(staff, \e -> e.team).

repeat team in sort(keys(by_team)):
    let members := by_team[team].
    let total := members |> map(\e -> e.salary) |> sum().
    emit team + ": " + to Text(len(members)) + " people, budget " + to Text(total).
done.

# Sorting by a computed key.
let ranked := sort_by(staff, \e -> 0 - e.salary).
emit "highest paid: " + first(ranked).name.

# Maps as lightweight structs, with safe lookup and defaulting.
let config := {"retries": 3, "verbose": true}.
let timeout := get(config, "timeout", nothing) ?? 30.
emit "timeout falls back to " + to Text(timeout).

# JSON round-trips for real interchange.
let encoded := json_encode({"ok": true, "items": [1, 2, 3]}).
emit encoded.
emit json_decode(encoded).items.

# Errors carry structured information.
fn parse_port(text: Text) -> Int:
    attempt:
        give to Int(text).
    rescue e:
        give 8080.
    done.
done.

emit parse_port("9090").
emit parse_port("not-a-number").
