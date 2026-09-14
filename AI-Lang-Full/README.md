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
`max` `sum` `clock` `now` `sleep` `print` `assert` `is_nothing`

**Convert** — `str` `int` `real` `bool`

**Text** — `join` `split` `upper` `lower` `trim` `replace` `contains`
`starts_with` `ends_with` `format` `pad` `pad_left` `repeat_text` `chars`
`code_of` `text_of` `index_of`

**Lists** — `range` `range_from` `first` `last` `push` `concat` `slice`
`reverse` `sort` `sort_by` `map` `filter` `reduce` `any` `all` `find` `count`
`unique` `zip` `enumerate` `flatten`

**Maps** — `keys` `values` `entries` `has` `get` `set` `remove` `merge`

**Data** — `json_encode` `json_decode` `hash_text` `uuid` `random`
`random_int`

**I/O** — `read_file` `write_file` `append_file` `env` `args` `input`

**Network** — `http_get` `http_post`

**Concurrency** — `spawn` `await_all`

---

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
