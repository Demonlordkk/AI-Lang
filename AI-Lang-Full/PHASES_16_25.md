# AI-Lang Phases 16–25 v0.2

## 16 — Structured Diagnostics
Defines a common diagnostic object and error boundary so lexer/parser/runtime/build failures can be reported as language errors without exposing host implementation details.

## 17 — Core Collections
Adds portable collection operations (`first`, `last`, `push`, `join`, `range`) to the standard runtime surface.

## 18 — Modules
Adds explicit local module loading through `.al` files and search paths. Source cannot arbitrarily import host-language modules.

## 19 — Concurrency
Adds a runtime scheduler boundary with `spawn`, allowing concurrent work without making the source language depend on a specific host threading API.

## 20 — Bytecode Boundary
Defines versioned AI-Lang bytecode records and a VM boundary. This is the stable target between language semantics and future native/JIT implementations.

## 21 — Optimization
Adds semantics-preserving constant folding for expressions. Optimization is optional and cannot change language meaning.

## 22 — CLI
Adds `run`, `check`, and `version` commands for a consistent command-line interface.

## 23 — Portability
Adds target reporting and a portability contract for the reference runtime. The language specification remains host-independent.

## 24 — Capability Security
Adds explicit capability policies for host integrations. Unapproved capabilities are denied by default at the FFI boundary.

## 25 — Conformance & Release
Adds executable conformance cases, release hashing, and a reproducible release manifest. This phase defines the current v0.2 release boundary.

## End-to-end architecture

`AI-Lang source → lexer → parser/AST → optimizer → runtime/IR → stdlib/collections → modules/FFI → bytecode/build → package/distribution → conformance`
