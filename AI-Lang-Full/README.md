# AI-Lang — Full Reference Build (Phases 1–25)

AI-Lang is an original general-purpose programming language. It is not an AI model and is not Python/C++ syntax with a new name. The reference implementation uses Python only as a host.

## Authoritative syntax

```text
let score := 42.
var score := 42.
score <- 43.
emit "hello".
when score > 40:
    emit "high".
else:
    emit "low".
done.
fn add(a: Int, b: Int) -> Int:
    give a + b.
done.
repeat item in [1, 2, 3]:
    emit item.
done.
```

## Phases
1–5: language identity, programming model, types, memory/resources, syntax.
6–10: lexer, parser, runtime/semantic execution, IR, toolchain.
11–15: standard library, packages, FFI boundary, build, distribution.
16–20: structured diagnostics, collections, modules, concurrency, bytecode boundary.
21–25: optimizer, CLI, portability, capability security, conformance/release.

## Pipeline
`source → lexer → parser/AST → runtime/IR → libraries/modules → bytecode/build → package/distribution → conformance`

This is a reference implementation, not a claim of production compiler performance.
