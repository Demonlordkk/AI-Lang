# A character-level RNN language model, built from primitives.
#
# This is a *sequence* model: at every position it reads the previous hidden
# state and the current character, and writes a new hidden state. The whole
# unrolled recurrence is a plain `while` loop -- no special "RNN" primitive.
# Every arrow in the computation graph is recorded by the t_* ops, so
# backward() flows the cross-entropy loss back through all 200+ unrolled
# steps (135 of them) to the weight matrices, and adamw updates them.
#
#   * ce_t       -- softmax cross-entropy, summed over the sequence
#   * adamw      -- the modern default optimizer

seed(3).

let base := "the dog barks and the fox jumps and the dog jumps and the fox barks ".
let text := base + base.

var vocab := [].
repeat ch in chars(text):
    when not contains(vocab, ch):
        append(vocab, ch).
    done.
done.
let vsize := len(vocab).

fn enc(s: Text) -> List:
    var out := [].
    repeat ch in chars(s):
        append(out, index_of(vocab, ch)).
    done.
    give out.
done.

let ids := enc(text).
let L := len(ids).

# The three weight matrices and two biases of a vanilla RNN.
let H := 12.
let w_ih := param(randn(vsize, H, nothing, 5)).
let w_hh := param(randn(H, H, nothing, 6)).
let w_ho := param(randn(H, vsize, nothing, 7)).
let b_h := param(zeros(H)).
let b_o := param(zeros(vsize)).
let params := [w_ih, w_hh, w_ho, b_h, b_o].
let opt := adamw(params, 0.02, 0.0).

# One full pass over the sequence: h <- tanh(h W_hh + x W_ih + b_h),
# emitting a distribution over the next character at each step.
fn seq_loss() -> Any:
    var h := tensor([zeros(H)]).
    var acc := tensor(0.0).
    var t := 0.
    while t < L - 1:
        let x := tensor([one_hot(ids[t], vsize)]).
        let h_new := t_tanh(t_add(t_add(t_matmul(h, w_hh), t_matmul(x, w_ih)), b_h)).
        let o := t_add(t_matmul(h_new, w_ho), b_o).
        acc <- t_add(acc, ce_t(o, tensor([one_hot(ids[t + 1], vsize)]))).
        h <- h_new.
        t <- t + 1.
    done.
    give t_div(acc, to Real(L - 1)).
done.

emit "RNN char LM: {L} characters, {vsize} symbols, hidden {H}".
var epoch := 0.
var ls := 0.0.
while epoch < 80:
    zero_grad(params).
    let loss := seq_loss().
    backward(loss).
    adamw_step(opt).
    ls <- value_of(loss).

    when epoch % 20 == 0:
        emit "  step {pad_left(to Text(epoch), 3)}  loss {round(ls, 5)}".
    done.
    epoch <- epoch + 1.
done.
emit "  final          loss {round(ls, 5)}".

# Greedy generation: warm the hidden state over a seed, then keep sampling
# the most likely next character and feeding it back in.
var gids := enc("the dog ").
var h := tensor([zeros(H)]).
repeat c in gids:
    let x := tensor([one_hot(c, vsize)]).
    h <- t_tanh(t_add(t_add(t_matmul(h, w_hh), t_matmul(x, w_ih)), b_h)).
done.
repeat i in range(52):
    let o := t_add(t_matmul(h, w_ho), b_o).
    let c := argmax(value_of(o)[0]).
    gids <- append(gids, c).
    let x := tensor([one_hot(c, vsize)]).
    h <- t_tanh(t_add(t_add(t_matmul(h, w_hh), t_matmul(x, w_ih)), b_h)).
done.
let gen := join(map(gids, \i -> vocab[i]), "").
emit "generated: {gen}".
