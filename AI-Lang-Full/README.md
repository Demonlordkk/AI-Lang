# AI-Lang

A concise, general-purpose programming language with a static checker, a
bytecode VM, a module system, and a complete toolchain.

AI-Lang is designed around one idea: **the shortest program that is still
unambiguous**. Common operations that take a loop and a temporary variable in
most languages are single expressions here, and the static checker catches
mistakes before anything runs.

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

## Machine learning

The numeric layer is built into the language, with no dependencies. A complete
logistic regression trained by gradient descent:

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

See `examples/ml/logistic.al` — it converges to 100% accuracy on separable
data in under 0.2s.

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

## Performance

The VM dispatches on integer opcodes through a frequency-ordered chain, and
the compiler emits specialised instructions when it can prove operand types.
Measured on this machine:

| Workload | Before | After | Gain |
| --- | --- | --- | --- |
| Mixed benchmark (fib 22, 200k loop, 20k list) | 1.485s | 1.037s | 1.43x |
| 50k element accumulation (`append`) | 3.395s | 0.243s | 14.0x |

Key optimisations:

* Integer opcodes with a dispatch chain ordered by measured frequency
* Specialised `ADD_NN`/`LT_NN`/... opcodes that skip generic type dispatch
* `INC_FAST` — `i <- i + 1` compiles to one instruction with no stack traffic
* `repeat i in range(n)` iterates lazily instead of building a list
* In-place `append` removes the O(n²) copy from accumulation loops

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
