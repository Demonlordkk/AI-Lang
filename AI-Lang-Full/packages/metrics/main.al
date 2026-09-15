# Small, transparent metrics for experiments and production checks.
# Every function is pure: callers can inspect the intermediate counts and
# choose their own threshold rather than inheriting hidden global state.

fn _safe_div(n: Real, d: Real) -> Real:
    when d == 0.0:
        give 0.0.
    done.
    give n / d.
done.

fn accuracy(actual: List, predicted: List) -> Real:
    needs len(actual) == len(predicted).
    when len(actual) == 0:
        give 0.0.
    done.
    let matches := count(zip(actual, predicted), \pair -> pair[0] == pair[1]).
    give to Real(matches) / to Real(len(actual)).
done.

fn binary_counts(actual: List, predicted: List, positive: Any) -> Map:
    needs len(actual) == len(predicted).
    var tp := 0.
    var fp := 0.
    var tn := 0.
    var fn_count := 0.
    var i := 0.
    while i < len(actual):
        let a := actual[i] == positive.
        let p := predicted[i] == positive.
        when a and p:
            tp <- tp + 1.
        elif not a and p:
            fp <- fp + 1.
        elif a and not p:
            fn_count <- fn_count + 1.
        else:
            tn <- tn + 1.
        done.
        i <- i + 1.
    done.
    give {"tp": tp, "fp": fp, "tn": tn, "fn": fn_count}.
done.

fn precision(actual: List, predicted: List, positive: Any) -> Real:
    let c := binary_counts(actual, predicted, positive).
    give _safe_div(to Real(c.tp), to Real(c.tp + c.fp)).
done.

fn recall(actual: List, predicted: List, positive: Any) -> Real:
    let c := binary_counts(actual, predicted, positive).
    give _safe_div(to Real(c.tp), to Real(c.tp + c.fn)).
done.

fn f1(actual: List, predicted: List, positive: Any) -> Real:
    let p := precision(actual, predicted, positive).
    let r := recall(actual, predicted, positive).
    give _safe_div(2.0 * p * r, p + r).
done.

fn mae(actual: List, predicted: List) -> Real:
    needs len(actual) == len(predicted).
    when len(actual) == 0:
        give 0.0.
    done.
    let errors := map(zip(actual, predicted), \pair -> abs(pair[0] - pair[1])).
    give to Real(sum(errors)) / to Real(len(errors)).
done.

fn rmse(actual: List, predicted: List) -> Real:
    needs len(actual) == len(predicted).
    when len(actual) == 0:
        give 0.0.
    done.
    let errors := map(zip(actual, predicted), \pair -> (pair[0] - pair[1]) * (pair[0] - pair[1])).
    give sqrt(to Real(sum(errors)) / to Real(len(errors))).
done.

fn threshold(scores: List, cutoff: Real) -> List:
    give map(scores, \score -> score >= cutoff).
done.

fn top_k(scores: List, k: Int) -> List:
    needs k >= 0.
    let indexed := enumerate(scores).
    let ordered := sort_by(indexed, \pair -> -pair[1]).
    give map(slice(ordered, 0, min(k, len(ordered))), \pair -> pair[0]).
done.
