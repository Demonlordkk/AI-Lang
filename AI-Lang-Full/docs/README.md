# AI-Lang design and implementation documents

The numbered documents preserve the original v0.1 design goals and syntax.
This page records the parts that are intentionally stable and the current
implementation boundary, so the documentation does not overstate guarantees.

## How the v0.1 spec maps to the current implementation

The grammar in `05_syntax.md` works today: `.al` files, `.` statement
terminators, `:`/`done.` blocks, `let`/`var`/`<-`, `emit`, `when`/`else`,
`fn`/`give`, `repeat`, `record`, `use`, and `to Int(...)` conversions are all
implemented.

Two names evolved during implementation:

| v0.1 spec | Current | Why |
| --- | --- | --- |
| `Float` | `Real` | The syntax document already used `Real` in its `record Point` example. |
| `Null` | `nothing` | The current spelling matches the language's word-like keywords. |

The current type system includes `Int`, `Real`, `Bool`, `Text`, `Byte`,
`List`, `Map`, `Any`, `Void`, `Function`, and user-defined records. Explicit
conversion, runtime type information, and invalid-operation diagnostics are
implemented by the reference VM and checked by the conformance suite.

## Design goals and current status

| Goal | Status |
| --- | --- |
| Simplicity | Met. The grammar is small, regular, and formatter-supported. |
| Precision | Met for the gradual type system: scope, mutability, arity, records, modules, and annotated operations are checked before execution. |
| Reliability | Met for language errors: structured diagnostics carry source positions; `attempt`/`rescue` handles recoverable failures, while `panic` and `exit` remain uncatchable. |
| Determinism | Met for builds: AILBC-4 artifacts use canonical JSON and an `artifact_sha256`; package lockfiles and package signatures use sorted content digests. |
| Portability | Met for the reference runtime: the interpreter uses Python's standard library and supports a dependency-free zipapp; numpy is optional acceleration only. |
| Extensibility | Met: modules, ten source packages (including data, metrics, learning, and neural), records, autodiff, networking, storage, FFI, and the CLI are separate APIs. |
| Maintainability | Met in practice: lexer, parser, checker, optimizer, compiler, VM, stdlib, packages, and platform modules are separated and exercised by a dependency-free test runner. |
| Scalability | Partial: lazy ranges, streams, bounded concurrency, native loop/function lowering, optional tensor acceleration, and incremental-safe module caching exist; a distributed compiler/runtime does not. |
| Performance | Improved and measured with specialized VM opcodes, peephole fusion, allocation-aware loops, native lowering, and optional numpy tensors. Absolute timings remain host- and workload-dependent; use `tools/bench_ml.py` and `ailang run --profile` for local measurements. |
| Capability-based security | Partial: artifact loading rejects host-code loops and validates integrity/schema; module/package paths are constrained and packages can be signed. File, process, network, database, and FFI builtins are still explicit escape hatches rather than a complete capability-typed security model. |

## v3.0.0 adaptive training

See [07_active_training.md](07_active_training.md) for the device-aware
active-training scheduler, atomic sleep checkpoints, model/optimizer
persistence, and the neural/data/metrics/learning packages. The reference
runtime remains dependency-free; acceleration is optional. The README's
[design lessons](../README.md#design-lessons-from-other-ecosystems) records
which recurring costs in other language, framework, and platform ecosystems
informed these choices and which tradeoffs remain.

## Artifact safety boundary

`ailang build` emits AILBC-4 JSON with no embedded Python source. `ailang run`
validates the format, language/version, schema, opcodes, operand arity, jump
bounds, function references, metadata, resource limits, and canonical digest
before rehydrating it. The digest detects accidental modification or
truncation; it is not a cryptographic signature. Authenticate artifacts from
untrusted publishers at the distribution layer before executing them.
