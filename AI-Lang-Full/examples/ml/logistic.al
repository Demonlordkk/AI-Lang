# Logistic regression trained with gradient descent — pure AI-Lang, no libraries.

record Model:
    weights: List.
    bias: Real.
done.

fn predict_one(model: Any, x: List) -> Real:
    give sigmoid(dot(model.weights, x) + model.bias).
done.

fn predict(model: Any, rows: List) -> List:
    give map(rows, \x -> predict_one(model, x)).
done.

fn train(rows: List, labels: List, epochs: Int, rate: Real) -> Any:
    let features := len(rows[0]).
    var w := zeros(features).
    var b := 0.0.
    var epoch := 0.

    while epoch < epochs:
        var grad_w := zeros(features).
        var grad_b := 0.0.

        repeat i in range(len(rows)):
            let x := rows[i].
            let target := labels[i].
            let guess := sigmoid(dot(w, x) + b).
            let error := guess - target.
            repeat j in range(features):
                grad_w[j] <- grad_w[j] + error * x[j].
            done.
            grad_b <- grad_b + error.
        done.

        let n := to Real(len(rows)).
        repeat j in range(features):
            w[j] <- w[j] - rate * grad_w[j] / n.
        done.
        b <- b - rate * grad_b / n.
        epoch <- epoch + 1.
    done.

    give Model(w, b).
done.

# Two clearly separable clusters.
let rows := [
    [1.0, 1.0], [1.5, 2.0], [2.0, 1.5], [1.2, 1.8], [0.8, 1.3],
    [6.0, 6.5], [6.5, 7.0], [7.0, 6.0], [6.2, 6.8], [7.1, 7.2]
].
let labels := [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0].

let model := train(rows, labels, 400, 0.5).

emit "weights: {map(model.weights, \v -> round(v, 3))}".
emit "bias:    {round(model.bias, 3)}".

let probs := predict(model, rows).
let classes := map(probs, \p -> when_gt(p)).

fn when_gt(p: Real) -> Real:
    when p >= 0.5:
        give 1.0.
    done.
    give 0.0.
done.

emit "accuracy: {accuracy(classes, labels)}".
emit "loss:     {round(mse(probs, labels), 5)}".

emit "".
emit "predictions on new points:".
repeat p in [[1.0, 1.2], [6.8, 6.9], [4.0, 4.0]]:
    let prob := predict_one(model, p).
    var verdict := "cluster A".
    when prob >= 0.5:
        verdict <- "cluster B".
    done.
    emit "  {p} -> {verdict} ({round(prob, 3)})".
done.
