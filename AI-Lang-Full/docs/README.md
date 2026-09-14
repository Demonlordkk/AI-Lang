# AI-Lang design documents

These are the original v0.1 design documents that defined the language. They
are preserved as written — they are the specification the implementation was
built to satisfy.

## How the v0.1 spec maps to the current implementation

Everything in `05_syntax.md` works today, unchanged. The `.al` file extension,
`.` statement terminators, `:`/`done.` blocks, `let`/`var`/`<-`, `emit`,
`when`/`else`, `fn`/`give`, `repeat`, `record`, `use`, and `to Int(...)`
conversions are all implemented exactly as specified.

Two details evolved during implementation:

| v0.1 spec | Current | Why |
| --- | --- | --- |
| `Float` | `Real` | `05_syntax.md` already used `Real` in the `record Point` example; the implementation follows the syntax doc. |
| `Null` | `nothing` | Reads better in a language that spells its keywords as words (`give`, `emit`, `done`). |

Everything else in the type system doc — `Int`, `Bool`, `Text`, `List`, `Map`,
explicit numeric conversion, runtime type information, rejection of invalid
operations — is implemented as written.

## Design goals and current status

`01_philosophy.md` lists ten goals. Honest assessment of each:

| Goal | Status |
| --- | --- |
| Simplicity | Met. The grammar is small and regular. |
| Precision | Met. Static checker catches type, scope and mutability errors before execution. |
| Reliability | Met. Errors are structured values with source positions, recoverable via `attempt`/`rescue`. |
| Determinism | Met. Build artifacts are canonical JSON; identical source always yields an identical `artifact_sha256`. |
| Portability | Met for the reference runtime — pure Python, no dependencies, any platform with Python 3.10+. |
| Extensibility | Met. Module system with path resolution, caching and cycle detection. |
| Maintainability | Met. Separated lexer/parser/checker/optimizer/compiler/VM, 83 tests. |
| Scalability | Partial. Multi-module projects work; no incremental or parallel compilation yet. |
| Performance | Not yet. Tree-walking bytecode VM on a Python host. Correct, not fast. |
| Capability-based security | Not yet. `02_programming_model.md` specifies explicit capabilities for resource access; the current stdlib exposes `read_file`/`write_file`/`http_get` as ordinary globals with no capability gating. |

The last two are the honest gaps between the specification and the
implementation.
