# A stateful business app: inventory with transactions and reporting.
record Item:
    sku: Text.
    name: Text.
    qty: Int.
    price: Real.
done.

fn restock(inv: Map, sku: Text, n: Int) -> Map:
    needs n > 0.
    ensures len(result) == len(inv).
    when not has(inv, sku):
        raise "unknown sku: " + sku.
    done.
    let it := get(inv, sku).
    let updated := Item(it.sku, it.name, it.qty + n, it.price).
    give merge(inv, {sku: updated}).
done.

fn sell(inv: Map, sku: Text, n: Int) -> Map:
    needs n > 0.
    let it := get(inv, sku).
    when it.qty < n:
        raise "insufficient stock for " + sku.
    done.
    give merge(inv, {sku: Item(it.sku, it.name, it.qty - n, it.price)}).
done.

fn total_value(inv: Map) -> Real:
    give sum(map(values(inv), \i -> real(i.qty) * i.price)).
done.

var inv := {
    "A1": Item("A1", "widget", 10, 2.50),
    "B2": Item("B2", "gadget", 4, 15.00)
}.

inv <- restock(inv, "A1", 5).
inv <- sell(inv, "B2", 2).
emit "value: " + to Text(total_value(inv)).
emit map(values(inv), \i -> i.name + "=" + to Text(i.qty)).

attempt:
    inv <- sell(inv, "B2", 99).
rescue e:
    emit "rejected: " + get(e, "message").
done.
