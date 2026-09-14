# AI-Lang

A concise, general-purpose programming language with a static checker, a
bytecode VM, a module system, and a complete toolchain.

AI-Lang is designed around one idea: **the shortest program that is still
unambiguous**. Common operations that take a loop and a temporary variable in
most languages are single expressions here, and the static checker catches
mistakes before anything runs.

Use it for anything you would use a general-purpose language for — services,
scripts, data work, automation, simulations, tools, and the models and AI
systems you build with them. Nothing in the language is specialised to one
domain: machine learning is a library, not the language.

It runs anywhere. AI-Lang depends on **no third-party packages at all** — only
a host Python runtime — and ships as a single 191 KB file you can copy to a
device and run with no install step.

```text
let users := [
    {"name": "Ada",   "team": "platform", "salary": 165000},
    {"name": "Grace", "team": "platform", "salary": 172000},
    {"name": "Alan",  "team": "research", "salary": 158000}
].

emit users
    |> filter(\u -> u.salary > 160000)
    |> map(\u -> u.name)
    |> join(", ").
```

```
Ada, Grace
```

---

## Install

```bash
pip install -e .
ailang version
```

Or run directly from the repository without installing:

```bash
python3 ailang.py run examples/basic.al
```

## The toolchain

| Command | Purpose |
| --- | --- |
| `ailang run FILE [args...]` | Type-check, compile and execute a program |
| `ailang check FILE` | Static analysis only, no execution |
| `ailang build FILE [-o OUT]` | Emit a deterministic bytecode artifact |
| `ailang test FILE` | Run every `fn test_*()` and report results |
| `ailang fmt FILE [--check]` | Canonical formatting |
| `ailang lint FILE` | Style and correctness diagnostics |
| `ailang repl` | Interactive session with persistent state |

---

## Language tour

### Bindings

`let` is immutable, `var` is mutable. Reassignment uses `<-`, never `=`, so
binding and mutation can never be confused.

```text
let limit := 100.        # immutable
var count := 0.          # mutable
count <- count + 1.
```

Statements end with `.` — this is what lets blocks nest without needing
semicolons or significant whitespace.

### Types

Annotations are optional. Anything unannotated is `Any` and checked at
runtime; anything annotated is checked statically.

```text
let n: Int := 42.
fn area(w: Real, h: Real) -> Real:
    give w * h.
done.
```

Built-in types: `Int`, `Real`, `Bool`, `Text`, `Byte`, `List`, `Map`, `Any`.

### Functions

```text
fn add(a: Int, b: Int) -> Int:
    give a + b.
done.

emit add(2, 3).
emit add(a: 2, b: 3).        # named arguments
```

Functions are hoisted, so definition order does not matter and mutual
recursion works out of the box.

Anonymous functions come in two forms:

```text
let double := \x -> x * 2.              # short
let adder := fn(a: Int, b: Int) -> Int: # block
    give a + b.
done.
```

Closures capture their environment:

```text
fn make_counter() -> Function:
    var total := 0.
    give fn() -> Int:
        total <- total + 1.
        give total.
    done.
done.
```

### Control flow

```text
when score >= 90:
    emit "A".
elif score >= 80:
    emit "B".
else:
    emit "C".
done.

repeat item at index in items:
    emit to Text(index) + ": " + item.
done.

while running:
    when should_skip:
        next.        # continue
    done.
    when should_quit:
        stop.        # break
    done.
done.
```

`and` and `or` short-circuit properly, so this is safe:

```text
when count != 0 and total / count > 10:
    emit "high average".
done.
```

### Data

```text
let nums := [3, 1, 2].
let user := {"name": "Ada", "age": 36}.

emit user.name.          # field syntax on maps
emit user["name"].       # index syntax
emit sort(nums).
```

Records are structured, named data:

```text
record Point:
    x: Real.
    y: Real.
done.

let p := Point(3.0, 4.0).
emit sqrt(p.x * p.x + p.y * p.y).
```

### The pipeline operator

`x |> f(a)` is exactly `f(x, a)`. It keeps data flow left-to-right and removes
nested-call soup.

```text
emit range(20)
    |> filter(\x -> x % 2 == 0)
    |> map(\x -> x * x)
    |> sum().
```

### Errors

Errors are recoverable values, not crashes:

```text
attempt:
    let config := json_decode(read_file("config.json")).
    emit config.port.
rescue e:
    emit "using defaults: " + e.message.
done.
```

`raise` throws any value; `e` in the rescue block is a map with `kind`,
`message` and `value`.

The `??` operator supplies a fallback for `nothing`:

```text
let port := get(config, "port", nothing) ?? 8080.
```

### Modules

```text
# lib/stats.al
fn mean(xs: List) -> Real:
    give sum(xs) / len(xs).
done.
```

```text
# main.al
use lib/stats as stats.
emit stats.mean([1, 2, 3, 4]).
```

Modules resolve relative to the importing file, then the project root. Imports
are cached, and circular imports are detected and reported.

---

### Text interpolation

Any `{...}` inside text is an expression, converted and spliced in place. This
is resolved at parse time into ordinary concatenation, so it costs nothing at
runtime.

```text
let name := "Ada".
let scores := [90, 84, 77].

emit "{name} averaged {round(mean(scores), 1)} across {len(scores)} runs".
```

```
Ada averaged 83.7 across 3 runs
```

Write `{{` and `}}` for literal braces. A bare `{}` is left alone, so the
`format("{} and {}", [...])` helper still works.

### Compound assignment

`+<-`, `-<-`, `*<-` and `/<-` update a binding in place, including fields and
elements:

```text
var total := 0.
repeat i in range(5):
    total +<- i.
done.

var counts := [0, 0].
counts[0] +<- 10.
```

Each form means exactly its longhand (`total <- total + i`) and is typed
identically — `/<-` yields `Real`, matching AI-Lang's division rule.

### Destructuring

Pull several values out of a list or map in one binding. The subject is
evaluated once.

```text
let [first, second] := pair().
let {name, age} := person.
```

### Multi-way branching

`given` tests one subject against several values. It replaces a ladder of
`elif` comparisons and evaluates the subject exactly once.

```text
given response.status:
is 200:
    emit "ok".
is 301, 302:
    emit "redirect".
else:
    emit "error {response.status}".
done.
```

### Membership

`in` and `not in` read as prose and work on lists, maps and text:

```text
when user not in banned and "admin" in user.roles:
    grant(user).
done.
```

### Records as types

A record name is a type name, usable anywhere a builtin type is:

```text
record Point:
    x: Int.
    y: Int.
done.

fn shift(p: Point, by: Int) -> Point:
    give Point(p.x + by, p.y + by).
done.
```

A type that was never declared is rejected before the program runs, with the
same near-miss suggestions as any other name:

```
error: unknown type 'Poimt'; did you mean 'Point'?
```

### Contracts

A function can state what it requires and what it guarantees. The conditions
are part of the function, not a separate test file:

```text
fn withdraw(balance: Real, amount: Real) -> Real:
    needs amount > 0.
    needs amount <= balance.
    ensures result >= 0.
    give balance - amount.
done.
```

`needs` is checked on entry, `ensures` on every path out with `result` bound
to the value being returned -- including a `give` nested inside a `when`, so
a branch added later cannot quietly escape the guarantee. A violation names
the function, which half of the contract broke, and the condition as written:

```
program.al:2:0: raised: withdraw: precondition failed: amount > 0
     2 |     needs amount > 0.
       | ^
```

Failures are ordinary errors, so `attempt`/`rescue` can catch them. Contracts
are compiled away entirely -- not merely skipped -- under
`AILANG_CONTRACTS=0`, so a released build pays nothing for the checks it
developed against.

## Standard library

Available everywhere without imports.

**Core** — `len` `type_of` `abs` `floor` `ceil` `round` `sqrt` `pow` `min`
`max` `sum` `exp` `log` `clock` `now` `sleep` `print` `assert` `is_nothing`

**Convert** — `str` `int` `real` `bool`

**Text** — `join` `split` `upper` `lower` `trim` `replace` `contains`
`starts_with` `ends_with` `format` `pad` `pad_left` `repeat_text` `chars`
`code_of` `text_of` `index_of`

**Lists (pure)** — `range` `range_from` `first` `last` `push` `concat`
`slice` `reverse` `sort` `sort_by` `map` `filter` `reduce` `any` `all` `find`
`count` `unique` `zip` `enumerate` `flatten`

**Lists (in-place, O(1) append)** — `append` `extend` `insert` `pop` `clear`

> `push` returns a new list; `append` mutates. In an accumulation loop
> `append` is ~13x faster because it avoids copying on every iteration.

**Maps** — `keys` `values` `entries` `has` `get` `set` `remove` `merge`

**Data** — `json_encode` `json_decode` `hash_text` `uuid` `random`
`random_int`

**I/O** — `read_file` `write_file` `append_file` `env` `args` `input`

**Network** — `http_get` `http_post`

**Concurrency** — `spawn` `await_all` `parallel_map`

**Statistics** — `mean` `median` `variance` `stddev` `percentile`
`normalize` `standardize` `correlation` `bincount`

**Vectors & matrices** — `dot` `vadd` `vsub` `vmul` `vdiv` `matmul`
`transpose` `shape` `identity` `zeros`

**Machine learning** — `sigmoid` `relu` `tanh` `softmax` `argmax` `argmin`
`mse` `mae` `cross_entropy` `accuracy` `linear_fit` `one_hot`
`train_test_split` `shuffle` `sample`

**Automation** — `read_csv` `write_csv` `read_lines` `write_lines`
`list_dir` `find_files` `path_exists` `is_dir` `make_dir` `delete_file`
`run` `timestamp` `retry` `timed`

---

### Working with collections

Each of these replaces a loop and a temporary variable:

| Operation | Result |
| --- | --- |
| `group_by(xs, f)` | map of key to the items sharing it |
| `count_by(xs, f)` / `counts(xs)` | how many fall in each bucket |
| `sum_by(xs, f)` | total of a projection |
| `max_by(xs, f)` / `min_by(xs, f)` | the extreme item, not just its value |
| `partition(xs, f)` | `[matching, rest]` in one pass |
| `chunk(xs, n)` / `windows(xs, n)` | fixed blocks / sliding runs |
| `take`, `drop`, `take_while`, `drop_while` | prefixes and suffixes |
| `zip_with(a, b, f)` | element-wise combine |
| `flat_map(xs, f)` | map then flatten one level |
| `pluck(xs, "field")` | one field from every item |
| `index_where(xs, f)` | first matching index, or `-1` |
| `sort_desc(xs)` | descending sort |

```text
let by_team := group_by(people, \p -> p.team).
let payroll := sum_by(people, \p -> p.salary).
let top     := max_by(people, \p -> p.salary).name.
```

`examples/analytics.al` puts these together in a short end-to-end report.

## Building models: automatic differentiation

AI-Lang is built for *writing* learning algorithms, not just calling them. The
language computes derivatives for you, so you never hand-derive calculus.

Mark the values you want trained with `param`, write the forward pass with
ordinary operators, then call `backward`:

```text
let w := param(3.0).
backward(t_mul(w, w)).      # d(w^2)/dw
emit grad_of(w).            # 6.0
```

That scales to a full network. Here is XOR — a problem no linear model can
solve — as a 2-layer net:

```text
let w1 := param(randn(2, 4)).
let b1 := param(zeros(4)).
let w2 := param(randn(4, 1)).
let b2 := param(zeros(1)).
let weights := [w1, b1, w2, b2].
let opt := adam(weights).

fn forward(x: Any) -> Any:
    let hidden := t_tanh(t_add(t_matmul(x, w1), b1)).
    give t_sigmoid(t_add(t_matmul(hidden, w2), b2)).
done.

var epoch := 0.
while epoch < 2000:
    zero_grad(weights).
    backward(bce_t(forward(inputs), targets)).
    adam_step(opt, 0.05).
    epoch <- epoch + 1.
done.
```

```
epoch    0   loss 0.74488
epoch 1500   loss 0.00024
accuracy: 4/4
```

There is no derivative anywhere in that program. Compare
`examples/ml/logistic.al` (6 lines of hand-derived gradient math) with
`examples/ml/logistic_autodiff.al` (the same model, 0 lines) — the training
loop collapses from 20 lines to 6.

**Autodiff vocabulary**

| Purpose | Functions |
| --- | --- |
| Create | `param` `tensor` `randn` `zeros` |
| Inspect | `value_of` `grad_of` `shape_of` `is_tensor` |
| Math | `t_add` `t_sub` `t_mul` `t_div` `t_pow` `t_neg` `t_exp` `t_log` `t_sqrt` `t_abs` |
| Layers | `t_matmul` `t_transpose` `t_reshape` |
| Activations | `t_sigmoid` `t_relu` `t_tanh` `t_softmax` |
| Reduce | `sum_t` `mean_t` |
| Losses | `mse_t` `mae_t` `bce_t` `ce_t` |
| Train | `backward` `zero_grad` `sgd_step` `adam` `adam_step` |

Tensors broadcast (a bias vector adds across every row), shapes are checked
with readable errors, and `backward` is iterative so network depth is not
limited by recursion.

Every gradient is verified two ways in the test suite: against closed-form
derivatives, and against central-difference numerical gradients (agreement to
~1e-10) for matmul chains, softmax/cross-entropy, sigmoid/BCE, broadcasting,
and a 2-layer network.

## Fixed-form models

For standard tasks the closed-form helpers are still there, no gradients
needed:

```text
fn train(rows: List, labels: List, epochs: Int, rate: Real) -> Any:
    var w := zeros(len(rows[0])).
    var b := 0.0.
    var epoch := 0.
    while epoch < epochs:
        var grad_w := zeros(len(rows[0])).
        var grad_b := 0.0.
        repeat i in range(len(rows)):
            let error := sigmoid(dot(w, rows[i]) + b) - labels[i].
            repeat j in range(len(w)):
                grad_w[j] <- grad_w[j] + error * rows[i][j].
            done.
            grad_b <- grad_b + error.
        done.
        repeat j in range(len(w)):
            w[j] <- w[j] - rate * grad_w[j] / to Real(len(rows)).
        done.
        b <- b - rate * grad_b / to Real(len(rows)).
        epoch <- epoch + 1.
    done.
    give Model(w, b).
done.
```

See `examples/ml/logistic.al` for the hand-written version and
`examples/ml/neural_net.al` for the autodiff one.

Data analysis is a one-liner:

```text
let rows := read_csv("data.csv").
emit mean(map(rows, \r -> r.salary)).
emit linear_fit(map(rows, \r -> r.age), map(rows, \r -> r.salary)).r2.
```

`read_csv` infers types per cell, so `r.salary` is an `Int` and `r.active` is
a `Bool` without any conversion step.

## Automation

```text
repeat path in find_files("./logs", ".txt"):
    let errors := read_lines(path) |> filter(\l -> contains(l, "ERROR")).
    when len(errors) > 0:
        emit path + ": " + to Text(len(errors)) + " errors".
    done.
done.

let results := parallel_map(urls, \u -> http_get(u).status).
let output := retry(\ -> run("deploy.sh"), 3, 1.0).
```

## Services

An AI-Lang program can be a web service. The handler is an ordinary function
from a request Map to a response Map, so routing and middleware are written
in the language rather than configured:

```text
fn router(req: Map) -> Map:
    when req.path == "/health":
        give {"status": 200, "body": "ok"}.
    elif req.path == "/notes" and req.method == "POST":
        give create_note(req).
    done.
    give {"status": 404, "json": {"error": "no route for " + req.path}}.
done.

serve(8080, router).
```

A request carries `method`, `path`, `query`, `headers`, `body` and `client`.
A response may set `status`, `headers`, and either `body` or `json`. Requests
are served on a thread pool, and a fault inside a handler becomes a 500
rather than taking the process down. Pass `background: true` to keep running
and stop later with `serve_stop`.

Raw TCP is available for protocols that are not HTTP: `tcp_listen`,
`tcp_accept`, `tcp_connect`, `tcp_send`, `tcp_receive`, `tcp_close`.

## Storage

Durable storage with transactions, backed by SQLite:

```text
let db := db_open("app.db").
db_exec(db, "create table if not exists users(id integer, name text)").
db_exec(db, "insert into users values(?, ?)", [1, "ann"]).
emit db_query(db, "select * from users where id = ?", [1]).
```

Rows come back as Maps keyed by column name, so a result flows straight into
`map`, `filter` and `group_by`. Parameters are always bound, never
interpolated, so a value containing a quote cannot alter the statement.
`db_transaction` commits on success and rolls back on any error, and nesting
behaves correctly -- an inner block joins the outer transaction.

For documents rather than tables there is a key/value store:

```text
let s := store_open("state.db").
store_put(s, "config", {"theme": "dark", "retries": 3}).
emit store_get(s, "config").theme.
```

## Graphics

Draw and write real PNG files, with no dependency:

```text
let c := canvas(480, 320).
canvas_fill(c, "#101820").
circle(c, 140, 160, 70, "#ffcc00").
text(c, 60, 60, "AI-LANG", "#ffffff", 3).
canvas_save(c, "out.png").
```

`pixel`, `line`, `rect`, `circle` and `text` all clip at the edges rather
than raising, so a generated drawing never fails on a rounding error. For the
common case there is a one-call chart:

```text
plot("wave.png", map(range(200), \i -> sin(real(i) / 12.0))).
```

## Foreign functions

Anything the standard library does not cover can be reached in a C library,
so a missing capability is a binding away rather than a wall:

```text
let m := ffi_open("m").
let cosine := ffi_fn(m, "cos", ["real"], "real").
emit ffi_call(cosine, [0.0]).
```

Foreign types are named with AI-Lang words -- `int`, `real`, `text`, `bool`,
`ptr`, `byte`, `void` -- so a binding reads the same everywhere. Bindings are
validated when declared: an unknown symbol or type fails at `ffi_fn`, not at
the first call. Argument counts and types are checked before crossing the
boundary, and a pointer cannot be fabricated from an integer.

## Packages

A package is a directory with `ailang.package.json` and `.al` sources.

```bash
ailang publish ./my-package      # add it to the registry
ailang add stats ^1.0.0          # record a dependency and install
ailang install                   # resolve, install, write the lockfile
ailang verify                    # recheck every digest
```

Dependencies install into `ai_modules/`, where the module loader already
looks, so `use stats as stats.` just works. Resolution handles transitive
dependencies, picks the highest version satisfying every constraint, and
reports a conflict with both requesters named rather than guessing. Versions
support `1.2.3`, `^1.2.3`, `~1.2.3`, `>=1.2.3` and `*`.

`ailang.lock.json` pins an exact version and SHA-256 digest for every package,
so an install is reproducible and tampering is detected:

```
  stats: digest mismatch (expected sha256:521cfc2dc981..., got sha256:a96c18a8b34a...)
```

Three packages ship in `packages/`: `text` (casing, padding, word counts),
`collections` (set operations, rotation, frequency tables) and `testing`
(assertions that report what actually differed).

## Running anywhere

AI-Lang imports only the host runtime's standard library. There is no numpy,
no build step, and no native extension anywhere in the implementation —
including the automatic-differentiation engine, which is written from scratch.
A test in the suite walks every import in the source and fails if a
third-party package ever appears.

Build the standalone interpreter:

```bash
python3 tools/make_bundle.py
```

That writes `ailang-bundle.pyz`, a single 191 KB file:

```bash
python3 ailang-bundle.pyz run program.al
```

Copy it to a server, a container, a Raspberry Pi, or a locked-down machine with
no package manager, and it runs as-is.

## Diagnostics

Mistyped names are matched against everything in scope, with your own names
ranked ahead of builtins:

```
error: undefined name 'grup_by'; did you mean 'group_by'?
error: undefined name 'countr'; did you mean one of 'counter', 'counts', 'count'?
```

Runtime errors carry the source line and an excerpt, and point inside the
function that actually failed rather than at the call site:

```
program.al:2:0: runtime error: index 99 is out of range for a list of 1
     2 |     give xs[99].
       | ^
```

## Performance

The VM dispatches on integer opcodes through a frequency-ordered chain, and
the compiler emits specialised instructions when it can prove operand types.
Measured on this machine:

| Workload | Before | After | Gain |
| --- | --- | --- | --- |
| fib 21 + 300k arithmetic loop | 1.118s | 0.159s | 7.0x |
| Call-heavy recursion (fib 24) | 0.543s | 0.200s | 2.7x |
| 50k element accumulation (`append`) | 3.395s | 0.243s | 14.0x |

Key optimisations:

* Integer opcodes with a dispatch chain ordered by measured frequency
* Specialised `ADD_NN`/`LT_NN`/... opcodes that skip generic type dispatch
* `INC_FAST` — `i <- i + 1` compiles to one instruction with no stack traffic
* `repeat i in range(n)` iterates lazily instead of building a list
* In-place `append` removes the O(n²) copy from accumulation loops
* Calls bind parameters in one `dict(zip(...))` and share a precomputed
  immutable-parameter set instead of rebuilding it per invocation
* Loop bodies that declare no bindings skip their per-iteration scope
  entirely: a 300k-iteration loop went from 300,001 environment allocations
  to one
* A bytecode peephole pass fuses adjacent instruction pairs (`LOAD;LOAD`,
  `LOAD;PUSH`, `LOAD;ADD_NN`, `LOAD;FIELD`, ...) so the hot path makes one
  dispatch instead of two. Fusion never crosses a jump target

### The native backend

Functions and top-level loops are compiled to host bytecode on first use, so
their locals become real slots instead of scope-chain lookups. Operations
whose AI-Lang meaning differs from the host's are emitted as a guarded fast
path with the interpreter's own operator as the fallback, so `true == 1` is
still `false` and `6 / 3` is still `2.0`.

The backend refuses anything it cannot model exactly -- closures that capture
or mutate an enclosing scope, named arguments, records -- and those functions
keep running on the VM. Set `AILANG_NATIVE=0` to disable it entirely; the test
suite runs both ways and 45 differential tests assert the two backends produce
identical output, exit codes and error text.

## Testing

Name a function `test_*` and run the file:

```text
fn test_addition():
    assert(2 + 2 == 4, "arithmetic works").
done.
```

```bash
ailang test mytests.al
```

## Architecture

```
source → lexer → parser → AST → type checker → optimizer → compiler → bytecode → VM
```

| Module | Role |
| --- | --- |
| `ailang/lexer.py` | Tokenizer with position tracking |
| `ailang/parser.py` | Recursive-descent parser |
| `ailang/typecheck.py` | Scope resolution, mutability, gradual types |
| `ailang/optimizer.py` | Constant folding, dead-branch elimination |
| `ailang/compiler.py` | Bytecode generation, closure conversion |
| `ailang/vm.py` | Stack machine with lexical scope chain |
| `ailang/stdlib.py` | Built-in functions |
| `ailang/modules.py` | Module resolution and caching |
| `ailang/bytecode.py` | Deterministic artifact serialization |
| `ailang/cli.py` | Command-line interface |

Build artifacts are canonical JSON: the same source always produces a
byte-identical file and the same `artifact_sha256`.

## Running the test suite

```bash
python3 tests/test_language.py     # standalone
python3 -m pytest tests/ -q        # via pytest
```

## Status

The reference implementation runs on Python. Semantics, diagnostics and the
bytecode format are stable; raw execution speed is not the current priority.
