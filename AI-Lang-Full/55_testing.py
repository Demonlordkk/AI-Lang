"""Phase 55: language-level testing primitives."""
import time

def expect(condition, message="expectation failed"):
    if not condition: raise AssertionError(message)

def benchmark(fn, iterations=1000):
    start=time.perf_counter()
    for _ in range(iterations): fn()
    return time.perf_counter()-start
