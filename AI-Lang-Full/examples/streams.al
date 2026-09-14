# Lazy infinite streams, end to end.
#
# A stream is an infinite sequence you can only move forward through. You
# build one from a zero-arg function that returns the next value -- state
# lives in the closure -- then take finite slices of it or derive new
# streams. Nothing is computed until asked, so a stream of a million values
# costs the memory of one.
#
#   * stream / stream_take / stream_map / stream_filter / stream_find /
#     stream_zip -- the package, itself written in AI-Lang
#   * memo / memo_rec -- cached function values, including self-referential
#   * prop_test     -- property-based testing over random inputs

use packages/stream as s.
use packages/testing as t.

# --- the Fibonacci stream: two captured vars hold the position -----------
var a := 0.
var b := 1.
let fib := s.stream(fn() -> Int:
    let tt := a.
    a <- b.
    b <- tt + b.
    give tt.
done).

emit "first 10: " + to Text(s.stream_take(fib, 10)).
emit "mapped  : " + to Text(s.stream_take(s.stream_map(fib, \x -> x * x), 5)).
emit "filtered: " + to Text(s.stream_take(s.stream_filter(fib, \x -> x % 3 == 0), 5)).
emit "zipped  : " + to Text(s.stream_take(s.stream_zip(fib, fib), 3)).
let big := s.stream_find(fib, \x -> x > 100 and x % 7 == 0).
emit "found   : " + to Text(big).

# --- memoization -----------------------------------------------------------
let fib2 := memo_rec(fn(self: Function, n: Int) -> Int:
    when n < 2:
        give n.
    done.
    give self(n - 1) + self(n - 2).
done).
emit "memo_rec fib(30) = " + to Text(fib2(30)).

let sq := memo(fn(x: Int) -> Int:
    give x * x.
done).
emit "memo sq(9) twice = " + to Text([sq(9), sq(9)]).

# --- property-based testing ------------------------------------------------
emit to Text(t.prop_test(\x -> (x + 0) == x, 100, 7)).
emit to Text(t.prop_test(\x -> abs(x) >= 0, 100, 8)).
attempt:
    t.prop_test(\x -> abs(x) >= 1, 500, 42).
rescue e:
    emit "caught: " + e.message.
done.

# --- streams feed the ML side too ------------------------------------------
# A synthetic dataset is just a stream slice: pairs of (x, sin x) from an
# infinite source, 8 taken for a quick fit demo.
var i := 0.
let pairs := s.stream(fn() -> List:
    let x := 0.5 * to Real(i).
    i <- i + 1.
    give [x, sin(x)].
done).
emit "dataset head: " + to Text(s.stream_take(pairs, 3)).
