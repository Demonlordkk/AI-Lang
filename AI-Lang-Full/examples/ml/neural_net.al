# A 2-layer neural network solving XOR — the classic non-linear problem.
#
# Note what is NOT in this file: any derivative. You write the forward pass
# with ordinary AI-Lang operators, call backward(loss), and the gradients are
# computed for you. This is the difference between using a model and building one.

let inputs  := [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]].
let targets := [[0.0],      [1.0],      [1.0],      [0.0]].

# Two layers: 2 -> 4 -> 1, randomly initialised.
let w1 := param(randn(2, 4, nothing, 1)).
let b1 := param(zeros(4)).
let w2 := param(randn(4, 1, nothing, 2)).
let b2 := param(zeros(1)).

let weights := [w1, b1, w2, b2].
let opt := adam(weights).

fn forward(x: Any) -> Any:
    let hidden := t_tanh(t_add(t_matmul(x, w1), b1)).
    give t_sigmoid(t_add(t_matmul(hidden, w2), b2)).
done.

emit "training a 2-layer network on XOR...".
var epoch := 0.
while epoch < 2000:
    zero_grad(weights).
    let loss := bce_t(forward(inputs), targets).
    backward(loss).
    adam_step(opt, 0.05).

    when epoch % 500 == 0:
        emit "  epoch " + pad_left(to Text(epoch), 4) + "   loss " + to Text(round(value_of(loss), 5)).
    done.
    epoch <- epoch + 1.
done.

let final_loss := bce_t(forward(inputs), targets).
emit "  final      loss " + to Text(round(value_of(final_loss), 5)).
emit "".

emit "predictions:".
let preds := value_of(forward(inputs)).
var correct := 0.
repeat row at i in preds:
    let p := row[0].
    var label := 0.
    when p >= 0.5:
        label <- 1.
    done.
    let want := to Int(targets[i][0]).
    when label == want:
        correct <- correct + 1.
    done.
    emit "  " + to Text(inputs[i]) + " -> " + to Text(round(p, 4)) + "   expected " + to Text(want).
done.

emit "".
emit "accuracy: " + to Text(correct) + "/4".
