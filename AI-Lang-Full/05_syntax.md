# AI-Lang Phase 5 — Syntax

AI-Lang uses `.al` source files. Statements end with `.`, blocks use `:` and close with `done.`.

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

repeat item in values:
    emit item.
done.

record Point:
    x: Real.
    y: Real.
done.

use math.
let count := to Int(real_count).
```
