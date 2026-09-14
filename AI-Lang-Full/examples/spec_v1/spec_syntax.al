use math.

let score := 42.
var counter := 42.
counter <- 43.
emit "hello".

when score > 40:
    emit "high".
else:
    emit "low".
done.

fn add(a: Int, b: Int) -> Int:
    give a + b.
done.

let values := [1, 2, 3].
repeat item in values:
    emit item.
done.

record Point:
    x: Real.
    y: Real.
done.

let real_count := 7.9.
let count := to Int(real_count).
emit count.
emit add(1, 2).
emit math.square(5).
