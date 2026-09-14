#!/usr/bin/env python3
"""Benchmark AI-Lang's two autodiff engines on one fixed training workload.

The same AI-Lang program -- a 2-layer net untangling the 3-arm spiral,
1200 epochs, AdamW -- runs once on the reference (pure Python) engine and
once on the numpy engine. Wall time and the backend each run reported are
printed; with no numpy installed the numpy leg is skipped.

Usage:  python3 tools/bench_ml.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROGRAM = """
seed(7).
let arms := 3.
let per_arm := 40.
fn arm_point(arm: Int, t: Real) -> List:
    let r := 0.4 + 1.6 * t.
    let ang := t * 4.712389 + to Real(arm) * 2.094395.
    give [r * cos(ang), r * sin(ang)].
done.
var xs := [].
var ys := [].
var labels := [].
repeat arm in range(arms):
    repeat i in range(per_arm):
        let t := to Real(i) / to Real(per_arm - 1).
        append(xs, arm_point(arm, t)).
        append(ys, one_hot(arm, arms)).
        append(labels, arm).
    done.
done.
let n := len(xs).
let w1 := param(randn(2, 32, nothing, 1)).
let b1 := param(zeros(32)).
let w2 := param(randn(32, arms, nothing, 2)).
let b2 := param(zeros(arms)).
let weights := [w1, b1, w2, b2].
let opt := adamw(weights, 0.02, 0.0).
fn logits(x: Any) -> Any:
    let h := t_relu(t_add(t_matmul(x, w1), b1)).
    give t_add(t_matmul(h, w2), b2).
done.
var epoch := 0.
while epoch < 1200:
    zero_grad(weights).
    let loss := ce_t(logits(tensor(xs)), tensor(ys)).
    backward(loss).
    adamw_step(opt).
    epoch <- epoch + 1.
done.
let probs := value_of(logits(tensor(xs))).
var correct := 0.
repeat i in range(n):
    when argmax(probs[i]) == labels[i]:
        correct <- correct + 1.
    done.
done.
emit "backend: {ml_backend()}".
emit "final accuracy: {round(to Real(correct) / to Real(n), 4)}".
"""

RUNS = 2


def run_once(numpy: bool) -> tuple[float, str, str]:
    env = dict(os.environ)
    env["AILANG_NUMPY"] = "1" if numpy else "0"
    best = float("inf")
    out = ""
    with tempfile.TemporaryDirectory() as d:
        prog = Path(d) / "bench.al"
        prog.write_text(PROGRAM, encoding="utf-8")
        for _ in range(RUNS):
            t0 = time.perf_counter()
            r = subprocess.run(
                [sys.executable, str(ROOT / "ailang.py"), "run", str(prog)],
                capture_output=True, text=True, env=env, cwd=d,
            )
            if r.returncode != 0:
                raise RuntimeError(r.stderr or r.stdout)
            best = min(best, time.perf_counter() - t0)
            out = r.stdout
    return best, out


def main() -> int:
    try:
        import numpy  # noqa: F401
        have_numpy = True
    except Exception:
        have_numpy = False

    print("AI-Lang ML benchmark: 2-layer net, 3-arm spiral, 1200 epochs (AdamW)")
    print(f"runs per engine: {RUNS} (best shown)\n")

    t_pure, out_pure = run_once(False)
    line = [l for l in out_pure.splitlines() if l.startswith("backend")]
    print(f"reference (pure Python)  {t_pure:6.2f}s   {line[0] if line else ''}")

    speedup = ""
    if have_numpy:
        t_np, out_np = run_once(True)
        line = [l for l in out_np.splitlines() if l.startswith("backend")]
        print(f"numpy accelerator          {t_np:6.2f}s   {line[0] if line else ''}")
        speedup = f"   speedup: {t_pure / t_np:.2f}x"
    else:
        print("numpy accelerator          skipped  (numpy not installed)")

    acc = [l for l in out_pure.splitlines() if "accuracy" in l]
    if acc:
        print(f"\n{acc[0]}")
    print(speedup)
    print("\nBoth engines compute the same gradients; numpy is a speed path,")
    print("not a different language. GPU acceleration would go through the FFI")
    print("to a host library -- it is not part of the core language.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
