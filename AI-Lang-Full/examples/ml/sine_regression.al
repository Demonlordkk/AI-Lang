# Fitting y = sin(x) with a 2-layer net and Huber loss -- the robust one.
#
# Two of the labels are corrupted by +/-1.5. Squared error would let those
# two points pull the whole curve around; Huber (delta 0.5) scores large
# errors linearly, so the fit shrugs them off.
#
#   * huber_t      -- robust differentiable regression loss
#   * l2_penalty_t -- weight decay folded into the loss
#   * adamw        -- the modern default optimizer

seed(7).

let n := 60.
let flat := map(range(n), \i -> -1.5 + 4.5 * to Real(i) / to Real(n - 1)).
let xs := map(flat, \x -> [x]).
var ys := map(flat, \x -> sin(x)).
ys[7] <- ys[7] + 1.5.
ys[29] <- ys[29] - 1.5.

let w1 := param(randn(1, 24, nothing, 3)).
let b1 := param(zeros(24)).
let w2 := param(randn(24, 1, nothing, 4)).
let b2 := param(zeros(1)).
let weights := [w1, b1, w2, b2].
let opt := adamw(weights, 0.02, 0.0).

fn predict(x: Any) -> Any:
    let h := t_tanh(t_add(t_matmul(x, w1), b1)).
    give t_add(t_matmul(h, w2), b2).
done.

fn full_loss(x: Any, y: Any) -> Any:
    # task loss plus 0.5% weight decay on the two weight matrices
    let decay := t_add(l2_penalty_t(w1), l2_penalty_t(w2)).
    give t_add(huber_t(predict(x), y, 0.5), t_mul(0.005, decay)).
done.

emit "fitting {n} points, two of them corrupted by +/-1.5...".
var epoch := 0.
var last := 0.0.
while epoch < 1500:
    zero_grad(weights).
    let loss := full_loss(tensor(xs), tensor(ys)).
    backward(loss).
    adamw_step(opt).
    last <- value_of(loss).

    when epoch % 500 == 0:
        emit "  epoch {pad_left(to Text(epoch), 4)}  loss {round(last, 5)}".
    done.
    epoch <- epoch + 1.
done.
emit "  final                loss {round(last, 5)}".

# How well does it know the *uncorrupted* truth at probe points?
var bad := 0.0.
repeat x in [-1.5, -0.86, -0.23, 0.4, 1.03, 1.66, 2.29, 3.0]:
    let p := value_of(predict(tensor([[x]])))[0][0].
    bad <- bad + abs(p - sin(x)).
done.
emit "mean abs error on clean probes: {round(bad / 8.0, 4)}".
