# Assertion helpers that report what actually differed.

fn equals(actual: Any, expected: Any, label: Text):
    assert(actual == expected, "{label}: expected {expected} but got {actual}").
done.

fn near(actual: Real, expected: Real, tolerance: Real, label: Text):
    assert(abs(actual - expected) <= tolerance,
        "{label}: {actual} is not within {tolerance} of {expected}").
done.

fn is_true(value: Bool, label: Text):
    assert(value, "{label}: expected true").
done.

fn contains_item(xs: List, item: Any, label: Text):
    assert(item in xs, "{label}: {item} is not in {xs}").
done.

fn has_length(xs: List, n: Int, label: Text):
    assert(len(xs) == n, "{label}: expected {n} items but got {len(xs)}").
done.

# Property-based testing: run `check` on `n` random inputs and fail the
# moment one breaks the property. `check` takes one Int in [-1000, 1000]
# and must return true (or raise). The first counterexample is reported,
# and the seed makes a failure reproducible.
#
#     prop_test(\x -> (x + 0) == x, 200, 7).
#     prop_test(\x -> abs(x) >= 0, 500, 42).
fn prop_test(check: Function, n: Int, seed_v: Int) -> Bool:
    seed(seed_v).
    var i := 0.
    while i < n:
        let x := random_int(-1000, 1000).
        var ok := false.
        attempt:
            ok <- check(x).
        rescue e:
            raise "check threw at x = " + to Text(x) + ": " + e.message.
        done.
        when not ok:
            raise "counterexample: x = " + to Text(x).
        done.
        i <- i + 1.
    done.
    give true.
done.
