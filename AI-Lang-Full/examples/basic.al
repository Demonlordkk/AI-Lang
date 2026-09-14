let a := 5.
var b := 7.
b <- b + a.
emit b.
fn add(x: Int, y: Int) -> Int:
    give x + y.
done.
emit add(10, 20).
when b > 10:
    emit "high".
else:
    emit "low".
done.
repeat item in [1, 2, 3]:
    emit item.
done.
