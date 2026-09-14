# AI-Lang 1.0.0 — Production-Grade Core

AI-Lang is an original general-purpose programming language with its own syntax, AST, static-first type checking, compiler, bytecode format, stack VM, standard library boundary, capability boundary, package metadata, formatter/linter hooks, and conformance/release artifacts.

## Release pipeline

`source -> lexer -> parser/AST -> type checker -> optimizer -> bytecode compiler -> AILBC-2 artifact -> stack VM`

## Production guarantees in this release

- deterministic source and artifact hashing
- versioned bytecode container (`AILBC-2`)
- static checks before execution
- immutable `let` bindings and mutable `var` bindings
- function arity/type checking
- required returns for explicitly typed non-void functions
- runtime execution fuel limit
- explicit filesystem capability boundary
- UTF-8 source handling
- reproducible JSON serialization
- CLI commands: `run`, `check`, `build`, `fmt`, `lint`, `version`
- automated compiler/runtime/type/security/release tests

## Scope

This is the production-grade reference implementation of the currently implemented AI-Lang core. Future native compilation, richer ownership/borrowing, concurrency, networking, database, GPU, WASM, LSP/debugger, and larger standard-library components remain separate engineering layers rather than being claimed as complete merely because their architecture is documented.
