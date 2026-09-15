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
fn mlp(sizes: List, seed: Int) -> Any:
    give mlp_init(sizes, seed, "randn").
done.

fn mlp_init(sizes: List, seed: Int, init: Text) -> Any:
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

# --------------------------------------------------------- active training
# Train data rows incrementally with the universal sleep/resume scheduler.
# Each row is [features, target], for example:
#   [[0.0, 1.0], [1, 0]]
# `loss_fn` receives (logits, target_tensor). The model and optimizer are
# included in the checkpoint state, so a later invocation continues with the
# next chunk even after a mobile process was stopped.
fn active(net: Any, rows: List, loss_fn: Function, checkpoint: Text, options: Map) -> Map:
    let ps := params(net).
    let rate := get(options, "rate", 0.001).
    let wd := get(options, "wd", 0.01).
    let optimizer := adamw(ps, rate, wd).
    let initial := {"net": net, "optimizer": optimizer}.
    let configured := set(options, "state", initial).
    fn run_batch(batch: List, saved: Any) -> Map:
        let current_net := saved.net.
        let current_optimizer := set(saved.optimizer, "params", params(current_net)).
        let xs := tensor(map(batch, \row -> row[0])).
        let ys := tensor(map(batch, \row -> row[1])).
        let current_params := params(current_net).
        zero_grad(current_params).
        let loss := loss_fn(forward(current_net, xs), ys).
        backward(loss).
        when has(options, "clip_norm"):
            clip_grad(current_params, get(options, "clip_norm", 1.0)).
        done.
        adamw_step(current_optimizer).
        # The model tensors are already persisted in `net`; omitting the
        # optimizer's duplicate parameter references keeps mobile checkpoints
        # smaller. The callback rebinds them before the next step.
        let checkpoint_optimizer := set(current_optimizer, "params", []).
        give {"state": {"net": current_net, "optimizer": checkpoint_optimizer},
            "loss": value_of(loss)}.
    done.
    give active_train(rows, run_batch, checkpoint, configured).
done.

# ---------------------------------------------------------------- persistence
# A model checkpoint keeps the network structure, trainable tensors, and
# integrity metadata in the shared atomic persistence format. It is safe to
# use at an active-training sleep boundary and remains loadable on a host that
# has no numpy accelerator.

# Save `net` to `path`. The older JSON weights/biases format is still accepted
# by load() for backwards compatibility with existing applications.
fn save(net: Any, path: Text):
    model_save(path, net, {"package": "neural"}).
done.

# Load a model saved by save(); the template-free loader returns records as
# field-compatible maps, so forward(), params(), and fine-tuning continue to
# work without requiring a Python pickle or a desktop-only runtime.
fn load(path: Text) -> Any:
    attempt:
        let loaded := model_load(path).
        when has(loaded, "layers"):
            give loaded.
        done.
        raise "model checkpoint does not contain neural layers".
    rescue e:
        # Legacy neural files were plain {weights, biases} JSON. Keep them
        # deployable while new saves use authenticated atomic checkpoints.
        let doc := json_decode(read_file(path)).
        var layers := [].
        var i := 0.
        while i < len(doc.weights):
            append(layers, Layer(param(tensor(doc.weights[i])), param(tensor(doc.biases[i])))).
            i <- i + 1.
        done.
        give Net(layers).
    done.
done.

# Save and restore a model plus its optimizer moments as two independently
# replaceable files. Rebinding the restored optimizer to the restored model's
# fresh parameters makes this suitable for a later active_train window.
fn save_bundle(net: Any, optimizer: Map, model_path: Text, optimizer_path: Text) -> Map:
    let m := model_save(model_path, net, {"package": "neural", "bundle": true}).
    let o := optimizer_save(optimizer_path, optimizer, {"package": "neural", "bundle": true}).
    give {"model": m, "optimizer": o}.
done.

fn load_bundle(model_path: Text, optimizer_path: Text, template: Any) -> Map:
    let net := model_load(model_path, template).
    let optimizer := optimizer_load(optimizer_path, params(net)).
    give {"model": net, "optimizer": optimizer}.
done.
