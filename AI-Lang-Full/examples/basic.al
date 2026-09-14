# AI-Lang basics — every core construct in one file.

# Bindings: `let` is immutable, `var` is mutable.
let name := "AI-Lang".
var count := 0.
count <- count + 12.
emit "Hello from " + name + "!".
emit count.

# Functions with typed parameters and return types.
fn add(a: Int, b: Int) -> Int:
    give a + b.
done.

emit add(10, 20).

# Named arguments read well at the call site.
fn greet(greeting: Text, name: Text) -> Text:
    give greeting + ", " + name + "!".
done.

emit greet(name: "Ada", greeting: "Welcome").

# Conditionals: when / elif / else.
fn classify(n: Int) -> Text:
    when n < 0:
        give "negative".
    elif n == 0:
        give "zero".
    else:
        give "positive".
    done.
done.

emit classify(-5).
emit classify(0).
emit classify(17).

# Loops: repeat over a collection, optionally with an index.
repeat item at i in ["red", "green", "blue"]:
    emit to Text(i) + ". " + item.
done.

# while with stop (break) and next (continue).
var n := 0.
while true:
    n <- n + 1.
    when n % 2 == 0:
        next.
    done.
    when n > 7:
        stop.
    done.
    emit "odd: " + to Text(n).
done.

# Collections.
let nums := [8, 3, 5, 1, 9].
emit sort(nums).
emit sum(nums).
emit first(nums).

let user := {"name": "Grace", "role": "engineer"}.
emit user.name.
emit keys(user).

# Records give you structured data.
record Point:
    x: Real.
    y: Real.
done.

let p := Point(3.0, 4.0).
emit "distance: " + to Text(sqrt(p.x * p.x + p.y * p.y)).

# Errors are values you can recover from.
attempt:
    let bad := 1 / 0.
    emit bad.
rescue e:
    emit "recovered from: " + e.message.
done.
