# A 2-layer network untangling the classic 3-arm spiral: a multiclass
# problem that linear models cannot separate.
#
# What it exercises that XOR does not:
#   * ce_t      -- softmax cross-entropy straight from logits (stable)
#   * t_softmax -- the network's own output distribution
#   * momentum  -- the momentum optimizer + clip_grad for stable steps
#
# Everything is generated and trained here, in AI-Lang, from a fixed seed so
# the run is reproducible word-for-word.

seed(7).

let arms := 3.
let per_arm := 40.

fn arm_point(arm: Int, t: Real) -> List:
    let r := 0.4 + 1.6 * t.
    let ang := t * 4.712389 + to Real(arm) * 2.094395.
    give [r * cos(ang), r * sin(ang)].
done.

var xs := [].
var ys := [].
var labels := [].
repeat arm in range(arms):
    repeat i in range(per_arm):
        let t := to Real(i) / to Real(per_arm - 1).
        append(xs, arm_point(arm, t)).
        append(ys, one_hot(arm, arms)).
        append(labels, arm).
    done.
done.

let n := len(xs).
let w1 := param(randn(2, 32, nothing, 1)).
let b1 := param(zeros(32)).
let w2 := param(randn(32, arms, nothing, 2)).
let b2 := param(zeros(arms)).
let weights := [w1, b1, w2, b2].
let opt := momentum(weights, 0.05, 0.9).

fn logits(x: Any) -> Any:
    let h := t_relu(t_matmul_bias(x, w1, b1)).
    give t_matmul_bias(h, w2, b2).
done.

# the data tensors are built once, outside the loop
let X := tensor(xs).
let Y := tensor(ys).

emit "training a 2-layer net on the {arms}-arm spiral ({n} points)...".
var epoch := 0.
while epoch < 1200:
    zero_grad(weights).
    let loss := ce_softmax_t(logits(X), Y).
    backward(loss).
    clip_grad(weights, 5.0).
    momentum_step(opt).

    when epoch % 300 == 0:
        emit "  epoch {pad_left(to Text(epoch), 4)}  loss {round(value_of(loss), 5)}".
    done.
    epoch <- epoch + 1.
done.

let probs := value_of(logits(X)).
var correct := 0.
repeat i in range(n):
    when argmax(probs[i]) == labels[i]:
        correct <- correct + 1.
    done.
done.

emit "  final      loss {round(value_of(ce_t(logits(tensor(xs)), tensor(ys))), 5)}".
emit "accuracy: {round(to Real(correct) / to Real(n), 4)}".
