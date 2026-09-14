# AI-Lang Phases 6–10 v0.1

## Phase 6 — Lexer
Converts original AI-Lang source into typed tokens while preserving source locations.

## Phase 7 — Parser / AST
Converts tokens into an AI-Lang AST using the Phase 5 frame syntax (`:`, `done.`, `.`) and original bindings (`let`, `var`, `<-`, `emit`, `give`).

## Phase 8 — Semantic / Type Analysis
Checks bindings, mutability, function contracts, conditions, iteration, conversions, and core expression types against the Phase 3 rules.

## Phase 9 — Intermediate Representation
Transforms validated AST nodes into a platform-neutral IR. The IR is an implementation boundary; it does not redefine AI-Lang semantics.

## Phase 10 — Toolchain
Connects lexing, parsing, semantic analysis, IR generation, and a reference runtime behind one AI-Lang command entry point.
