# Lazy infinite streams, written in AI-Lang itself.
#
# A stream is an infinite sequence you can only move forward through:
# create one from a zero-argument function that hands back the next value
# (state lives in the closure), then take finite slices of it or derive new
# streams by mapping/filtering. Nothing is computed until you ask for it, so
# a stream of a million values costs the memory of one.
#
#     use stream as s.
#
#     # the Fibonacci stream: state kept in two captured variables
#     var a := 0.
#     var b := 1.
#     let fib := s.stream(fn() -> Int:
#         let t := a.
#         a <- b.
#         b <- t + b.
#         give t.
#     done).
#
#     emit s.stream_take(fib, 10).            # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
#     emit s.stream_take(s.stream_map(fib, \x -> x * x), 5).
#     emit s.stream_take(s.stream_filter(fib, \x -> x % 2 == 0), 5).
#
# Pairs with the ML tooling naturally: a training dataset is just
# stream_take(dataset_stream, batch), and a synthetic data generator is a
# stream, so models and apps share the same lazy-data primitive.

record Stream:
    next_fn: Function.
done.

# Wrap a zero-arg "next value" function as a stream. The function may close
# over `var`s to keep its position — that is how infinite generators work.
fn stream(next_fn: Function) -> Stream:
    give Stream(next_fn).
done.

# The next n values of a stream, as a plain list.
fn stream_take(s: Stream, n: Int) -> List:
    var out := [].
    var i := 0.
    while i < n:
        append(out, s.next_fn()).
        i <- i + 1.
    done.
    give out.
done.

# A stream of f applied to each value. Lazily: f runs once per taken value.
fn stream_map(s: Stream, f: Function) -> Stream:
    give Stream(fn() -> Any:
        give f(s.next_fn()).
    done).
done.

# A stream of the values that pass the predicate. Lazily: the source is
# pulled only as far as needed for each kept value.
fn stream_filter(s: Stream, p: Function) -> Stream:
    give Stream(fn() -> Any:
        var v := s.next_fn().
        while not p(v):
            v <- s.next_fn().
        done.
        give v.
    done).
done.

# The first value of a stream that passes the predicate; nothing if the
# stream runs out before one does (only finite sources can run out).
fn stream_find(s: Stream, p: Function) -> Any:
    var v := s.next_fn().
    while not p(v):
        v <- s.next_fn().
    done.
    give v.
done.

# Zip two streams into a stream of pairs. Both advance together.
fn stream_zip(a: Stream, b: Stream) -> Stream:
    give Stream(fn() -> List:
        give [a.next_fn(), b.next_fn()].
    done).
done.
