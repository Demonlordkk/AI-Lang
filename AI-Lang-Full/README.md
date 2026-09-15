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

It runs anywhere. AI-Lang depends on **no third-party packages** — only a
host Python runtime — and ships as a compact single-file interpreter you can copy to a
device and run with no install step. One exception, deliberately optional: if
numpy is already installed (Colab, a scientific Python, a server), the tensor
engine transparently accelerates on it; without numpy the reference engine
runs the exact same program, just slower. Either way nothing needs to be
installed for the language itself.

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
| `ailang run FILE [args...]` | Type-check, compile and execute a program (`.al` or a built `.albc.json`) |
| `ailang run FILE --profile` | Run and print a top-15 cProfile time table |
| `ailang check FILE` | Static analysis only, no execution |
| `ailang build FILE [-o OUT]` | Emit a deterministic bytecode artifact |
| `ailang trace FILE [args...]` | Run, printing each source line as it executes |
| `ailang dis FILE` | Disassemble to opcodes without running |

A built `.albc.json` artifact runs directly with `ailang run`, exactly like
its source — same output, same `args`, no re-parse, no re-check — so a
shipped program needs only the interpreter and one file. Artifacts use the
AILBC-4 format: they contain validated VM data only, never generated Python
source. Loading checks the schema, opcode operands, jump bounds, function
references, resource limits, and a canonical SHA-256 integrity field. The
hash detects accidental tampering or truncation; authenticate artifacts from
untrusted publishers separately before executing them.
| `ailang test FILE` | Run every `fn test_*()` and report results |
| `ailang fmt FILE [--check]` | Canonical formatting |
| `ailang lint FILE` | Style and correctness diagnostics |
| `ailang repl` | Interactive session with persistent state |

### The step budget

Every run has a step budget (fuel): a hard ceiling on how many interpreter
steps a program may take, so a genuine infinite loop fails with a clean
error instead of hanging the machine. The default is 50,000,000 steps, and
it is overridable when a program legitimately needs more — a
multi-million-iteration numerical loop, for instance:

```bash
ailang run heavy.al --fuel 200000000      # or:
AILANG_FUEL=200000000 ailang run heavy.al
```

Exhaustion is a normal, catchable error that says exactly what to do:

```
program.al:4:0: runtime error: execution limit exceeded (raise the step budget with --fuel N or AILANG_FUEL=N)
```

### Debugging

Three tools for when a program misbehaves:

* **`ailang trace FILE`** prints each source line as it executes, numbered
  — the last line printed is the line about to misbehave. `ailang run FILE
  --trace` does the same for a normal run. Tracing never changes the
  program's output; it only slows it down.
* **`ailang dis FILE`** prints the bytecode without running: per-function
  headers, instruction numbers, opcode names and inline arguments. It
  type-checks first, so a `dis` that completes is a program the checker
  accepts.
* **`panic(msg)`** is the fatal sibling of `raise`. Unlike an error, a
  panic cannot be caught by `attempt` / `rescue` — like `exit()`, it is a
  process-level event, in both engines. Use it for states where continuing
  is never correct: corrupt configuration, missing credentials, an
  invariant that should be impossible. The run stops with `ailang:
  panic: msg` on stderr and exit code 1.

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
nested-call soup. Binding: the pipe sits below arithmetic (so `1 - 6 |> abs()`
is `abs(1 - 6)`), but above comparisons and equality — those apply to the
*result of the whole chain*: `xs |> filter(\x -> x > 1) |> sum() == 14` parses
as `((xs |> filter(\x -> x > 1)) |> sum()) == 14`, never as a call on a Bool.
`10 |> add(4) |> double()` applies left to right, and pipes work inside call
arguments, loop conditions, and lambdas exactly like any other expression.

```text
emit range(20)
    |> filter(\x -> x % 2 == 0)
    |> map(\x -> x * x)
    |> sum().
```

Named arguments work through pipes too, with a hard guarantee: the checker
reconciles every builtin's declared parameter names with its real
implementation at startup, so `5 |> max(a: 3)` can never reach a Python
function it was never going to match — it is a clean check error (`max` takes
positional arguments only), not a crash.

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

**Function caching** — `memo` `memo_rec`

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

**I/O** — `read_file` `write_file` `append_file` `env` `args` `script_dir`
`input` `read_line` `exit`

**Network** — `http_get` `http_post` `http_request` `serve` `serve_stop`

> HTTPS, both directions: `serve(port, handler, host, background, cert)`
> takes a combined cert+key PEM path (or a `[cert, key]` pair) as its fifth
> argument and serves TLS; `http_request(url, method, body, headers, verify,
> timeout)` speaks `https://` natively, with an opt-out of certificate
> verification for self-signed endpoints. Servers cap request bodies at ~1 MB
> (413) and drop idle connections after 30 s.

**Concurrency** — `spawn` `await_all` `parallel_map` `await` `mutex` `lock`
`unlock` `channel` `channel_send` `channel_recv` `channel_try_recv`
`channel_close`

> `spawn` runs a function on a shared thread pool; `await(task)` joins one
> task, `await_all` joins many. `mutex()`/`lock`/`unlock` guard shared state;
> `channel()` gives threads a queue with `send`/`recv`/`try_recv`/`close`.

**Statistics** — `mean` `median` `variance` `stddev` `percentile`
`normalize` `standardize` `correlation` `bincount`

**Vectors & matrices** — `dot` `vadd` `vsub` `vmul` `vdiv` `matmul`
`transpose` `shape` `identity` `zeros`

**Machine learning** — `sigmoid` `relu` `tanh` `softmax` `argmax` `argmin`
`mse` `mae` `cross_entropy` `accuracy` `linear_fit` `one_hot`
`train_test_split` `shuffle` `sample`

**Tensors & autodiff** — `tensor` `param` `zeros` `randn` `seed` `value_of`
`grad_of` `shape_of` `is_tensor` `ml_backend` `backward` `zero_grad`
`sgd_step` `momentum` `momentum_step` `adam` `adam_step` `adamw` `adamw_step`
`clip_grad` and the differentiable `t_*` operators, which also accept the
plain `+` `-` `*` `/` and unary `-` on tensors. Full table in
[Building models](#building-models-automatic-differentiation).

**Automation** — `read_csv` `write_csv` `read_lines` `write_lines`
`list_dir` `find_files` `path_exists` `is_dir` `make_dir` `delete_file`
`run` `timestamp` `retry` `timed`

**Packages** — `ailang publish` (optionally HMAC-SHA256 signed with
`--key`/`AILANG_REGISTRY_KEY`), `ailang add`, `ailang install` (enforces
publisher signatures when a key is set), `ailang verify` (digests +
signatures)

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

## Beyond the usual

A few capabilities that are uncommon even in established languages — built
into the language's standard library or shipped packages, so every program
gets them:

**Lazy infinite streams.** `packages/stream` (itself written in AI-Lang)
wraps a zero-arg next-value function in an infinite sequence: the state
lives in the closure, nothing computes until you ask for a value, and a
stream of a million values costs the memory of one.

```text
use packages/stream as s.

var a := 0.
var b := 1.
let fib := s.stream(fn() -> Int:
    let t := a.
    a <- b.
    b <- t + b.
    give t.
done).

emit s.stream_take(s.stream_map(fib, \x -> x * x), 5).
```

`stream_take`, `stream_map`, `stream_filter`, `stream_find` and `stream_zip`
compose the way pipelines do. The ML side uses the same primitive:
`examples/ml/sine_regression.al` generates its training points as a stream
slice, so model code and app code share one lazy-data story.
`examples/streams.al` shows the whole package end to end.

**Memoization as a builtin.** `memo(f)` caches a one-argument function;
`memo_rec(f)` passes the wrapper to the body as `self`, so *recursive*
functions get cached recursion for free:

```text
let fib := memo_rec(fn(self: Function, n: Int) -> Int:
    when n < 2:
        give n.
    done.
    give self(n - 1) + self(n - 2).
done).
emit fib(30).   # 832040, instantly
```

**Property-based testing.** `packages/testing` adds `prop_test(check, n,
seed)`: run a property on `n` random inputs, report the first counterexample
with its value, and keep the seed so a failure reproduces exactly.

---

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

**Operators.** Tensor math uses the ordinary operators, not a parallel
vocabulary: `+`, `-`, `*`, `/` and unary `-` on a tensor take the same
differentiable path as the `t_*` functions, and broadcast like them (`a * 2.0`
scales every element). Exponentiation is `t_pow`. So a squared-error loss is
just `mean_t((t_matmul(x, w) - y) * (t_matmul(x, w) - y))`.

**Autodiff vocabulary**

| Purpose | Functions |
| --- | --- |
| Create | `param` `tensor` `zeros` `randn` `seed` |
| Inspect | `value_of` `grad_of` `shape_of` `is_tensor` `ml_backend` |
| Math | `t_add` `t_sub` `t_mul` `t_div` `t_pow` `t_neg` `t_exp` `t_log` `t_sqrt` `t_abs` |
| Layers | `t_matmul` `t_transpose` `t_reshape` `t_slice` `t_gather` `t_concat` |
| Activations | `t_sigmoid` `t_relu` `t_tanh` `t_softmax` |
| Element-wise | `t_clip` `where_t` `l2_norm_t` |
| Reduce | `sum_t` `mean_t` |
| Losses | `mse_t` `mae_t` `bce_t` `bce_logits_t` `ce_t` `huber_t` `l2_penalty_t` `ce_softmax_t` |
| Fused | `t_matmul_bias` `ce_softmax_t` `t_dropout` |
| Init | `xavier` `he_init` |
| Schedules | `lr_step_decay` `lr_cosine` |
| Guardrails | `early_stop` `early_stop_step` `gradcheck` |
| Metrics | `accuracy` `f1` `train_test_split` |
| Train | `backward` `zero_grad` `sgd_step` `momentum` `momentum_step` `adam` `adam_step` `adamw` `adamw_step` `clip_grad` |

Tensors broadcast (a bias vector adds across every row), shapes are checked
with readable errors, and `backward` is iterative so network depth is not
limited by recursion. Optimizer state lives in an opaque handle returned by
`momentum` / `adam` / `adamw`, so the training loop is the same six lines no
matter which optimizer you pick, and `clip_grad` clamps before any step.

**Sequence models.** Anything the language can loop over, it can unroll.
`examples/ml/char_lm.al` trains a character-level RNN — the recurrent
connection `h <- t_tanh(t_add(t_matmul(h, w_hh), ...))` written as an
ordinary assignment inside an explicit unrolling loop — and it learns to
continue `the dog ...` with `barks and the fox jumps and the fox jumps...`.
No `LSTM` class, no `Sequence`: the recurrence, the loss and the greedy
decoding loop are all plain AI-Lang, which is also why it runs on the
reference engine with nothing installed.

### A training guide

**Choose the loss to match the target.**

| Task | Loss | Note |
| --- | --- | --- |
| Regression (predict a number) | `mse_t`, `mae_t`, `huber_t` | `huber_t` resists outlier spikes |
| Binary classification, probabilities | `bce_t` | predictions in (0,1) — compose with `t_sigmoid` |
| Binary classification, raw logits | `bce_logits_t` | numerically stable, no `t_sigmoid` needed |
| Multiclass, one-hot labels | `ce_t` | compose with `t_softmax` |
| Regularisation | `l2_penalty_t` | add a small multiple to the loss |

**Choose the optimizer.** `sgd_step` is the baseline; `momentum(params, rate, mu)`
smooths it; `adam(params, rate)` adapts the rate per parameter; `adamw(params,
rate, wd)` — Adam with decoupled weight decay — is the modern default for
nets, and what the examples use. All three are created the same way and
stepped with the same `*_step(state)` call, so switching is a one-line change.
`clip_grad(params, 1.0)` before a step bounds the update when one bad batch
sends a gradient wild.

**The loop.** Seed first so runs are reproducible, keep the rate in the
0.01–0.05 neighbourhood for `adamw`, and print the loss now and then so you
can see it fall:

```text
seed(7).
let opt := adamw(weights, 0.02, 0.0).
var epoch := 0.
while epoch < epochs:
    zero_grad(weights).
    let loss := ce_t(t_softmax(logits(x)), y).
    backward(loss).
    adamw_step(opt).
    epoch <- epoch + 1.
    when epoch % 100 == 0:
        emit "epoch {epoch}  loss {round(value_of(loss), 5)}".
    done.
done.
```

**When it doesn't learn**, in order of frequency: the rate is too high (loss
jumps — divide by 10) or too low (loss barely moves — multiply by 10); labels
and outputs misaligned (spot-check `argmax(probs[i]) == labels[i]` on a few
rows); a broadcastable-but-wrong shape (`(3,)` where `(3,1)` is expected —
ask `shape_of(...)`); the net has no hidden layer (linear models can't
separate non-linear data — XOR is the canonical case); or the data itself has
no signal.

### Production toolkit

Small examples teach the mechanics; a real training run needs a few more
things, and they are all standard-library functions — still no framework:

| Concern | Functions | Notes |
| --- | --- | --- |
| Regularisation | `t_dropout(x, rate, training)` | Inverted dropout: scaled pass-through while training, identity at evaluation — one net serves both |
| Weight init | `xavier(n, m, seed)` `he_init(n, m, seed)` | Glorot for tanh, Kaiming for relu; `nn.mlp_init(sizes, seed, init)` builds a net with either |
| Learning-rate schedule | `lr_step_decay(step, rate, factor, every)` `lr_cosine(step, total, rate, floor)` | Pure functions of the epoch — pass the result to the optimizer step |
| Early stopping | `early_stop(patience)` `early_stop_step(state, val_loss)` | Tracks the best validation loss and reports when to stop |
| Metrics | `f1(y_true, y_pred, threshold)` | Joins `accuracy` for per-class reporting |
| Self-verification | `gradcheck(params, loss_fn)` | Analytic vs numerical gradients; around 1e-9..1e-10 means the engine is sound |
| Fast layers | `t_matmul_bias(x, w, b)` `ce_softmax_t(logits, y)` | Fused forward+backward primitives — about 1.5× faster training, same gradients |

`examples/ml/production_classifier.al` is the reference production run on
these primitives: a 3-arm spiral split 120/30 into train and validation, a
`[2, 24, 24, 3]` net with He init, dropout 0.1, AdamW under a cosine
schedule, early stopping on validation loss, a gradcheck on the
deterministic path, per-class F1, and a save/reload round-trip. It is
seeded end to end, so it is reproducible word for word — on every repeated
run and on both backends:

```
spiral: 150 points (120 train / 30 validation)...
gradcheck: 2.93e-10
  ...
stopped at epoch 1374 (best val 0.09124)
train accuracy:   1.0
validation acc:   1.0
validation f1:    1.0
  class 0: f1 1.0
  class 1: f1 1.0
  class 2: f1 1.0
reload identical: true
```

That is the numpy engine, in about 3 s; the reference engine runs the same
program in 22 s and lands at train 1.0 / validation 0.967 — the engines
share one API and one result quality, and each one is reproducible.

### From data to deployment

A model in AI-Lang is data plus functions, so it flows through the rest of
the language like any other value:

1. **Load.** `read_csv` gives typed rows — numbers stay numbers, booleans stay
   booleans; `one_hot` encodes categorical targets; `train_test_split` and
   `shuffle` keep an honest holdout out of the loop.
2. **Train.** The loop above.
3. **Evaluate.** `accuracy(probs, labels)` over the holdout, or `argmax` by
   hand for anything custom.
4. **Save.** `nn.save(net, "model.almodel")` writes the weights as a JSON
   file — an exact float round-trip, so a loaded model predicts
   bit-identically to the trained one.
5. **Serve.** `examples/apps/predictor.al` is the whole story in one file:
   train a classifier with the autodiff engine, save it, and start a web app
   in the same process — `POST /predict` runs the forward pass on the request
   and answers with the class and a confidence. The second run (and every
   later run) skips training and loads the saved model instead, so the web
   app starts in milliseconds. The same file powers a terminal app:
   `examples/apps/cli_classifier.al` reads the same model format for its
   `train` / `predict` / `report` commands. Train in AI-Lang, deploy from
   AI-Lang — the model file is the only format in between.

### Two engines, one language

The autodiff engine has two backends with identical semantics:

* **Reference** — pure Python, zero dependencies. This is what Termux, a
  Raspberry Pi, or any locked-down machine gets.
* **numpy accelerator** — if numpy is importable at startup, the same
  differentiable operations run on numpy arrays. No API change, no separate
  program: `AILANG_NUMPY=0` turns it off, and `ml_backend()` reports which
  engine is live.

Measured on this machine (`tools/bench_ml.py`): the same spiral-classification
training program takes **6.3 s on the reference engine and 0.68 s on the
numpy engine — about a 9× speedup**. Deterministic programs produce
bit-identical output on both — the test suite runs every example and
differential test on both engines and asserts it. Seeded random programs
(dropout is the one place the training loop draws) are reproducible word
for word *within* an engine: `seed(n)` fixes every stream the engine owns.

A word about GPUs, said plainly: AI-Lang has **no built-in GPU path**. A
Colab notebook or CUDA server gets the numpy accelerator, which is a large
constant factor but not a device migration. The intended path to real
device-speed training is the language's foreign-function interface — bind a
host library that already owns the GPU, and keep writing the model in
AI-Lang. That is a binding away, not a language change, and it is a
deliberate design decision: the core stays dependency-free rather than
shipping a CUDA build of the interpreter.

### Where it runs

| Machine | What the model gets |
| --- | --- |
| Termux, Raspberry Pi, locked-down box | The reference engine — pure Python, nothing installed |
| Desktop Python with numpy | The numpy engine, picked up automatically (~9× on training) |
| Colab / Jupyter | Same as desktop, in a single file |
| CUDA server | The numpy engine for now; device speed is a binding away via FFI |

The program is the same on every row. `ml_backend()` tells you which engine
is live, `AILANG_NUMPY=0` forces the reference engine, and `seed(...)` makes
any two of them agree bit for bit.

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

## Building apps

A service is one shape of app. The language covers the others the same way:
a **web app** is a function from request to response, a **terminal app** is a
loop, a **CLI app** is a function of `args`. The language provides the
plumbing — HTTP, sockets, SQLite, the file system, JSON, text — and
`packages/webapp` provides the structure on top of it. Like ML, apps are a
capability of the language, not a separate framework.

### Web apps with the webapp package

`packages/webapp` adds routing with path parameters, static files and
middleware on top of `serve`:

```text
use packages/webapp as w.

let app := w.new().

w.route(app, "GET", "/notes/:id", fn(req: Map) -> Map:
    give w.json(200, {"id": req.params.id}).
done).
w.static(app, "/", "static").       # static/index.html becomes the homepage
w.middleware(app, fn(req: Map, cont: Function) -> Map:
    give cont(req).                  # runs around every request
done).

w.start(app, 8080, "127.0.0.1", true).
```

| Piece | Function |
| --- | --- |
| `w.new()` | an empty app |
| `w.route(app, method, pattern, handler)` | a route; `:name` in the pattern lands in `req.params.name` |
| `w.static(app, prefix, dir)` | serve files under `prefix`, `index.html` as the prefix fallback, `..` rejected |
| `w.middleware(app, fn)` | `fn(req, cont)` around every request — routes, static files, 404s; call `cont` to continue |
| `w.dispatch(app, req)` | the whole router as a plain function — no server, so tests drive it directly |
| `w.json` / `w.html` / `w.text` | response builders |
| `w.body_json(req)` | the request body decoded as a Map (`{}` when absent) |
| `w.start(app, port, host, background)` | serve; pass the result to `serve_stop` |
| `w.make_logger()` | request-log middleware: prints `METHOD /path -> STATUS` per request |
| `w.make_error_handler(log)` | turns any route exception into a clean JSON 500 (and logs it) |
| `w.make_throttle(limit, window_s)` | per-path rate limiter; over-limit requests get a clean JSON 429 |

Because `dispatch` is just a function, routing logic is unit-testable: build
the app, call `dispatch` with a request map, assert on the response map —
that is exactly what `tests/test_webapp.py` does, plus one test that drives
the finished app over real HTTP.

The three `w.make_*` factories are the professional middleware layer: a
production route gets logging, a 500 handler, and a rate limit with three
lines of app setup, and each one is testable through `dispatch` without a
server.

Four complete apps ship in `examples/apps/`:

* **`notes.al`** — a full notes app: SQLite storage, a JSON API
  (`GET/POST /notes`, `GET/DELETE /notes/:id`), a single-page HTML UI served
  from `static/`, and a middleware that counts requests. Its final block
  drives the finished app over HTTP — create, fetch, 404, delete — so the
  example is its own integration test, and it passes the two-engine parity
  gate.
* **`predictor.al`** — trains a classifier with the autodiff engine, saves
  it as a model file, and serves the live model as a `POST /predict` API in
  the same file; later runs load the saved model and skip training. The
  shortest distance in this language between "training" and "deployment" is
  one `nn.save` and one `w.start`.
* **`cli_classifier.al`** — the same model, as a normal terminal app:
  `train` / `predict x y` / `report` built on the cli package.
* **`tasks.al`** — a professional *normal* app with no ML at all: a terminal
  task manager (`add`, `list --all --limit N`, `done`, `rm`, `stats`,
  `clear` with two-step confirmation) pairing the cli package's tables,
  colors and argument parsing with JSON persistence in a file next to the
  script. State, parsing and presentation are all ordinary AI-Lang.

### Terminal apps

A terminal app is a loop with `input`. End-of-input raises, so apps exit
cleanly by turning that into a value:

```text
while true:
    var line := "".
    attempt:
        line <- input("> ").
    rescue e:
        stop.            # EOF: a normal exit, not a crash
    done.
    ...
done.
```

`examples/apps/calc_tui.al` is a calculator REPL built exactly this way:
interactive in a terminal (type `3 + 4`, errors like division by zero are
rescued and shown), and safe to run unattended — the same `attempt`/`rescue`
turns closed stdin into a clean stop, which is how it passes the parity gate.

### CLI apps

`args` carries the command line; `store_open` and `db_open` carry state
between runs. A todo CLI is a `given args[0]` over `is "add"`, `is "list"`,
`is "done"` — a module-based todo app ships in `examples/todo_app`, and the
web `notes.al` above is the same idea with an HTTP front end.

`packages/cli` is the professional layer on top: declarative argument
parsing, generated usage, aligned tables, a progress bar, ANSI color,
interactive confirmation, and JSON config with defaults — all written in
AI-Lang.

```text
use packages/cli as c.

let spec := {"--rate": "real", "-e": "int", "--verbose": "flag", "--model": "text"}.
attempt:
    let args := c.parse(args(), spec).
rescue e:
    emit c.color(e.message, 31).
    emit c.usage("classifier", spec, ["<command>  train | predict | report"]).
    exit(2).
done.
emit c.table(["id", "score"], [["1", "0.97"]]).
emit c.bar(7, 10, "training").
```

`c.parse` understands `--flag value`, `--flag=value`, `-f value`, boolean
`--flag`, accumulating `--tag` lists, a `--` separator, and negative numbers
(`predict -0.5 0.9` stays positional); unknown flags and bad numbers raise
clean, descriptive errors instead of crashing. `c.config(path, defaults)`
reads a JSON config over defaults; `c.confirm("Sure?")` reads a y/N from
`read_line()`; and `exit(code)` terminates with a real status code.

`examples/apps/cli_classifier.al` is a complete normal app built on it:
`train` trains and saves a neural model, `predict x y` loads the model and
answers, `report` prints a table — the same model file `predictor.al` the
web app uses, so the ML and the apps genuinely share one artifact.
`examples/apps/tasks.al` is the other complete normal app: a task manager
that pairs the same toolkit with JSON persistence, so "normal app building"
has a reference on both sides — apps that talk to a model and apps that
just manage data.

### Anything else

Anything that is not HTTP is a socket away (`tcp_listen` and friends, shown
above), and anything that is not in the standard library is a foreign
function away. The app is the program; the language never gets in the way.

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

Seven packages ship in `packages/`: `text` (casing, padding, word counts),
`collections` (set operations, rotation, frequency tables), `testing`
(assertions that report what actually differed, plus `prop_test`, a
property-based test runner), `neural` (a feed-forward net built from the
tensor operators, plus model save/load — see
[Building models](#building-models-automatic-differentiation)), `webapp`
(routing, static files, middleware and the professional `make_*` helpers for
web apps — see [Building apps](#building-apps)), `cli` (argument parsing,
usage text, tables, progress bars, color and config for terminal apps — see
[Building apps](#building-apps)) and `stream` (lazy infinite streams — see
[Beyond the usual](#beyond-the-usual)).

## Running anywhere

AI-Lang imports only the host runtime's standard library. There is no build
step and no native extension anywhere in the implementation — including the
automatic-differentiation engine, which is written from scratch. A test in
the suite walks every import in the source and fails if a third-party package
ever appears, with exactly one whitelisted exception: **numpy is an optional
accelerator, not a dependency**. It is imported lazily and only when present;
`AILANG_NUMPY=0` — or simply not having numpy installed — runs the language
on the pure-Python reference engine, which is the entire story on Termux, a
Raspberry Pi, or any machine without pip.

Build the standalone interpreter:

```bash
python3 tools/make_bundle.py
```

That writes `ailang-bundle.pyz`, a compact single-file interpreter:

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
Benchmark results are workload- and host-dependent. The reproducible local
benchmark is `python3 tools/bench_ml.py`; the latest verification run completed
the pure-Python two-layer spiral workload in 4.93 s with final accuracy 1.0.
The optional numpy accelerator was unavailable in that environment, so no
numpy speedup is claimed here. See [Two engines, one language](#two-engines-one-language).

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
suite runs both ways, and every differential scenario in the suite — plus
the shipped examples covered by the conformance tests — is executed on both
backends and asserted to produce identical output, exit codes and error text.

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
| `ailang/autodiff.py` | Reference tensor engine + reverse-mode autodiff |
| `ailang/accel.py` | Optional numpy backend (same operations, numpy arrays) |
| `ailang/modules.py` | Module resolution and caching |
| `ailang/bytecode.py` | Deterministic artifact serialization |
| `ailang/cli.py` | Command-line interface |

Build artifacts are canonical JSON: the same source always produces a
byte-identical file and the same `artifact_sha256`.

## Running the test suite

```bash
python3 tools/run_tests.py         # whole suite, zero dependencies (no pytest)
python3 tests/test_language.py     # the conformance file, standalone
python3 -m pytest tests/ -q        # via pytest, when it is installed
```

`tools/run_tests.py` needs only the Python standard library: when `pytest`
is missing it substitutes a small bundled shim (`tests/_pytest_stub.py`)
covering the subset the suite uses (`raises`, `mark.parametrize`,
`mark.skipif`, `tmp_path`/`monkeypatch` fixtures). Every test runs with a
wall-clock timeout (120 s, `RUN_TESTS_TEST_TIMEOUT`), the whole run has a
total cap (900 s, `RUN_TESTS_TOTAL_TIMEOUT`), and a file filter is accepted
(`python3 tools/run_tests.py contracts ml`). With pytest installed the same
suite runs through it unchanged.

## Status

The reference implementation runs on Python. Semantics, diagnostics and the
AILBC-4 bytecode format are versioned and tested. Speed is handled in layers:
specialized VM opcodes and peephole fusion are always available, the native
backend is an opt-in optimization for eligible source programs, and the
optional numpy engine accelerates tensor workloads. Measurements are
workload- and host-dependent; use `ailang run --profile` or
`tools/bench_ml.py` instead of treating a single benchmark as a guarantee.
The core remains pure Python with no required dependencies.
