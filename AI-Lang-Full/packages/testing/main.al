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
