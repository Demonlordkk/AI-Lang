# Showcase programs

Four complete programs, each exercising a different part of the language.
Run any of them with `python3 ../../ailang.py run <file>`.

| File | What it demonstrates |
|---|---|
| `calc.al` | A recursive-descent tokenizer, parser and evaluator for arithmetic with precedence and parentheses. Records as types, mutual recursion, `while` with state. |
| `inventory.al` | Stateful business logic: records, contracts guarding invariants, map updates, error handling with `attempt`/`rescue`. |
| `pipeline.al` | A data pipeline end to end: CSV write/read, `group_by`, statistics, JSON encode/decode, report output. |
| `nn.al` | A two-layer neural network trained on XOR with the autodiff library, converging to `[0.02, 0.98, 0.98, 0.02]`. |
