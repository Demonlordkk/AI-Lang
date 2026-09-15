# A production-grade training run, in pure AI-Lang, on nothing but the
# standard library:
#
#   * He init            — he_init, the right scale for relu nets
#   * dropout            — t_dropout (inverted), off at evaluation
#   * AdamW              — adamw + adamw_step with per-step learning rate
#   * cosine schedule    — lr_cosine anneals the rate over the run
#   * early stopping     — early_stop / early_stop_step on validation loss
#   * metrics            — accuracy and per-class F1
#   * gradcheck          — numerical vs analytic gradients, self-verified
#   * save / reload      — the trained net round-trips through a file
#
# The problem is the 3-arm spiral (a nonlinear multiclass task), split
# 80/20 into train and validation. Everything is seeded, so the run is
# reproducible word for word.

seed(1234).

use packages/neural as nn.

# ---------------------------------------------------------------- the data
let arms := 3.
let per_arm := 50.

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
let split := train_test_split(range(n), 0.8, 7).
let Xtr := tensor(map(get(split, "train"), \i -> xs[i])).
let Ytr := tensor(map(get(split, "train"), \i -> ys[i])).
let Xv := tensor(map(get(split, "test"), \i -> xs[i])).
let Yv := tensor(map(get(split, "test"), \i -> ys[i])).
let val_labels := map(get(split, "test"), \i -> labels[i]).
emit "spiral: {n} points ({len(get(split, "train"))} train / {len(get(split, "test"))} validation)...".

# ------------------------------------------------------------------ the net
let net := nn.mlp([2, 24, 24, 3], 42, "he").
let ps := nn.params(net).
let opt := adamw(ps, 0.003, 0.01).
let es := early_stop(100).
let dropout_rate := 0.1.

fn train_loss() -> Any:
    give ce_softmax_t(nn.forward_drop(net, Xtr, dropout_rate, true), Ytr).
done.
fn val_loss() -> Any:
    give ce_softmax_t(nn.forward_drop(net, Xv, 0.0, false), Yv).
done.

# a healthy autodiff engine reports around 1e-9..1e-12 here.
# dropout is off: its mask is random per evaluation, so a derivative
# check must run on the deterministic path
fn check_loss() -> Any:
    give ce_softmax_t(nn.forward_drop(net, Xtr, 0.0, false), Ytr).
done.
emit "gradcheck: " + to Text(round(gradcheck(ps, check_loss), 12)).

# ---------------------------------------------------------------- training
let total := 1500.
var epoch := 0.
var stopping := false.
while epoch < total and not stopping:
    let lr := lr_cosine(epoch, total, 0.003, 0.0005).
    zero_grad(ps).
    backward(train_loss()).
    adamw_step(opt, lr).
    stopping <- get(early_stop_step(es, value_of(val_loss())), "stop").
    when epoch % 250 == 0:
        let best := get(es, "best").
        emit "  epoch {pad_left(to Text(epoch), 4)}  train {round(value_of(train_loss()), 5)}  best-val {round(best, 5)}".
    done.
    epoch <- epoch + 1.
done.
emit "stopped at epoch {epoch} (best val {round(get(es, "best"), 5)})".

# ----------------------------------------------------------------- results
let tr_probs := nn.forward_drop(net, Xtr, 0.0, false).
let tr_labels := map(get(split, "train"), \i -> labels[i]).
let val_probs := nn.forward_drop(net, Xv, 0.0, false).
let val_pred := nn.labels(val_probs).
let train_acc := nn.accuracy(tr_probs, tr_labels).
let val_acc := nn.accuracy(val_probs, val_labels).

emit "train accuracy:   {round(train_acc, 4)}".
emit "validation acc:   {round(val_acc, 4)}".
emit "validation f1:    {round(f1(val_labels, val_pred), 4)}".
fn class_f1(yt: List, yp: List, k: Int) -> Real:
    give f1(map(yt, \v -> v == k), map(yp, \v -> v == k), 0.5).
done.
repeat k in range(arms):
    emit "  class {k}: f1 {round(class_f1(val_labels, val_pred, k), 4)}".
done.

# ------------------------------------------------------------ save / reload
let model := "/tmp/spiral_prod.almodel".
nn.save(net, model).
let reloaded := nn.load(model).
let again := nn.labels(nn.forward_drop(reloaded, Xv, 0.0, false)).
emit "reload identical: {again == val_pred}".
