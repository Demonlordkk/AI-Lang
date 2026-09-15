# A small feed-forward network, built from AI-Lang's own tensor operators.
#
# This is deliberately written in the language rather than shipped as a
# builtin: it is the reference for how to compose param(), the t_* ops,
# backward() and an optimizer into a trainable model. Import it and reuse
# the pieces, or copy them into your own program.
#
#     use neural as nn.
#     let net := nn.mlp([2, 8, 3], 11).
#     nn.train(net, xs, \out -> ce_t(t_softmax(out), ys), 2000, 0.05).

record Net:
    layers: List.
done.

# One layer: a weight matrix and a bias vector, both trainable.
record Layer:
    w: Any.
    b: Any.
done.

# A multi-layer network from a list of layer widths, e.g. [2, 8, 3] is
# 2 inputs -> 8 hidden -> 3 outputs. Hidden layers are tanh; the final
# layer is linear so the caller chooses softmax/sigmoid/identity.
#
# `init` picks the weight scheme: "randn" (historical default), "xavier"
# (Glorot, good for tanh) or "he" (Kaiming, good for relu).
fn mlp(sizes: List, seed: Int, init: Text) -> Any:
    var layers := [].
    var i := 0.
    while i < len(sizes) - 1:
        let n_in := sizes[i].
        let n_out := sizes[i + 1].
        when init == "xavier":
            append(layers, Layer(param(xavier(n_in, n_out, seed + i)), param(zeros(n_out)))).
        elif init == "he":
            append(layers, Layer(param(he_init(n_in, n_out, seed + i)), param(zeros(n_out)))).
        else:
            append(layers, Layer(param(randn(n_in, n_out, nothing, seed + i)), param(zeros(n_out)))).
        done.
        i <- i + 1.
    done.
    give Net(layers).
done.

# Forward pass: x is a 2-D tensor of shape (batch, sizes[0]).
fn forward(net: Any, x: Any) -> Any:
    var a := x.
    repeat layer at i in net.layers:
        a <- t_matmul_bias(a, layer.w, layer.b).
        when i < len(net.layers) - 1:
            a <- t_tanh(a).
        done.
    done.
    give a.
done.

# Forward pass with inverted dropout between the hidden layers. `rate` 0
# (or training false) makes it identical to forward().
fn forward_drop(net: Any, x: Any, rate: Real, training: Bool) -> Any:
    var a := x.
    var i := 0.
    while i < len(net.layers):
        a <- t_matmul_bias(a, get(net.layers, i).w, get(net.layers, i).b).
        when i < len(net.layers) - 1:
            a <- t_dropout(t_tanh(a), rate, training).
        done.
        i <- i + 1.
    done.
    give a.
done.

# Every trainable tensor in the net, in order.
fn params(net: Any) -> List:
    var out := [].
    repeat layer in net.layers:
        append(out, layer.w).
        append(out, layer.b).
    done.
    give out.
done.

# One full-batch training step. `loss` takes the forward output and gives a
# scalar tensor. Returns that loss.
fn step(net: Any, x: Any, loss: Any) -> Any:
    let ps := params(net).
    zero_grad(ps).
    let l := loss(forward(net, x)).
    backward(l).
    give l.
done.

# Train for `epochs` full batches with AdamW at `rate` and `wd` decay.
fn train(net: Any, x: Any, loss: Any, epochs: Int, rate: Real, wd: Real) -> Any:
    let ps := params(net).
    let opt := adamw(ps, rate, wd).
    var epoch := 0.
    var last := step(net, x, loss).
    while epoch < epochs:
        last <- step(net, x, loss).
        adamw_step(opt).
        epoch <- epoch + 1.
    done.
    give last.
done.

# Argmax over the rows of a probability matrix, as a flat list of labels.
fn labels(probs: Any) -> List:
    var out := [].
    repeat row in value_of(probs):
        append(out, argmax(row)).
    done.
    give out.
done.

# Fraction of rows whose argmax matches the given flat list of class ints.
fn accuracy(probs: Any, expected: List) -> Real:
    let got := labels(probs).
    var correct := 0.
    repeat i in range(len(got)):
        when got[i] == expected[i]:
            correct <- correct + 1.
        done.
    done.
    give to Real(correct) / to Real(len(got)).
done.

# ---------------------------------------------------------------- persistence
# A trained net as a JSON file: weights and biases as nested lists. Train
# once, then load the file from a web app or a CLI without retraining —
# this is the bridge between the neural package and app building.

# Save `net` to `path` (JSON, exact float round-trip).
fn save(net: Any, path: Text):
    var ws := [].
    var bs := [].
    repeat layer in net.layers:
        append(ws, value_of(layer.w)).
        append(bs, value_of(layer.b)).
    done.
    write_file(path, json_encode({"weights": ws, "biases": bs})).
done.

# Load a net saved by save(); the weights are fresh trainable params, so the
# loaded model can also be fine-tuned.
fn load(path: Text) -> Any:
    let doc := json_decode(read_file(path)).
    var layers := [].
    var i := 0.
    while i < len(doc.weights):
        append(layers, Layer(param(tensor(doc.weights[i])), param(tensor(doc.biases[i])))).
        i <- i + 1.
    done.
    give Net(layers).
done.
