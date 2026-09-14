# The same logistic regression as logistic.al — but with autodiff.
#
# logistic.al  : 6 lines of hand-derived gradient math inside a 20-line loop.
# this file    : 0 lines. Write the forward pass; backward() does the calculus.

let rows := [
    [1.0, 1.0], [1.5, 2.0], [2.0, 1.5], [1.2, 1.8], [0.8, 1.3],
    [6.0, 6.5], [6.5, 7.0], [7.0, 6.0], [6.2, 6.8], [7.1, 7.2]
].
let labels := [[0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0]].

let w := param(zeros(2, 1)).
let b := param(zeros(1)).
let opt := adam([w, b]).

fn predict(x: Any) -> Any:
    give t_sigmoid(t_add(t_matmul(x, w), b)).
done.

var epoch := 0.
while epoch < 600:
    zero_grad([w, b]).
    backward(bce_t(predict(rows), labels)).
    adam_step(opt, 0.1).
    epoch <- epoch + 1.
done.

emit "weights: {map(flatten(value_of(w)), \v -> round(v, 3))}".
emit "bias:    {round(first(value_of(b)), 3)}".
emit "loss:    {round(value_of(bce_t(predict(rows), labels)), 6)}".

let probs := map(value_of(predict(rows)), \r -> first(r)).
let classes := map(probs, \p -> to Real(to Int(p >= 0.5))).
emit "accuracy: {accuracy(classes, flatten(labels))}".

emit "".
emit "new points:".
repeat p in [[1.0, 1.2], [6.8, 6.9], [4.0, 4.0]]:
    let prob := first(first(value_of(predict([p])))).
    var verdict := "cluster A".
    when prob >= 0.5:
        verdict <- "cluster B".
    done.
    emit "  {p} -> {verdict} ({round(prob, 3)})".
done.
