# Data pipeline: file IO, CSV, JSON, grouping, stats, report writing.
write_csv("./sales.csv", [
    {"region": "north", "rep": "ann", "amount": 120, "month": "jan"},
    {"region": "south", "rep": "bob", "amount": 80,  "month": "jan"},
    {"region": "north", "rep": "cy",  "amount": 200, "month": "feb"},
    {"region": "south", "rep": "dee", "amount": 95,  "month": "feb"},
    {"region": "north", "rep": "ann", "amount": 150, "month": "feb"}
]).

let rows := read_csv("./sales.csv").
emit "rows: " + to Text(len(rows)).

let by_region := group_by(rows, \r -> r.region).
repeat region in sort(keys(by_region)):
    let group := get(by_region, region).
    let amounts := map(group, \r -> r.amount).
    emit region + ": n=" + to Text(len(group))
        + " total=" + to Text(sum(amounts))
        + " mean=" + to Text(round(mean(amounts), 1))
        + " max=" + to Text(max(amounts)).
done.

let top := first(sort_by(rows, \r -> 0 - r.amount)).
emit "top sale: " + top.rep + " " + to Text(top.amount).

let summary := {
    "total": sum(map(rows, \r -> r.amount)),
    "regions": len(by_region),
    "reps": len(unique(map(rows, \r -> r.rep)))
}.
write_file("./summary.json", json_encode(summary)).
emit read_file("./summary.json").

let back := json_decode(read_file("./summary.json")).
emit "roundtrip total = " + to Text(get(back, "total")).
