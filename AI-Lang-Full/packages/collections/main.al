# Set-style operations over lists.

fn union(a: List, b: List) -> List:
    give unique(concat(a, b)).
done.

fn intersect(a: List, b: List) -> List:
    ensures len(result) <= len(a).
    give unique(filter(a, \x -> x in b)).
done.

fn difference(a: List, b: List) -> List:
    give unique(filter(a, \x -> x not in b)).
done.

fn is_subset(a: List, b: List) -> Bool:
    give all(a, \x -> x in b).
done.

fn frequencies_desc(xs: List) -> List:
    let table := counts(xs).
    give sort_by(entries(table), \e -> 0 - e[1]).
done.

fn rotate(xs: List, n: Int) -> List:
    ensures len(result) == len(xs).
    when len(xs) == 0:
        give xs.
    done.
    let k := n % len(xs).
    give concat(drop(xs, k), take(xs, k)).
done.
