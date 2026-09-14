# Train a small network on XOR using the autodiff library.
let xs := tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).
let ys := tensor([[0.0], [1.0], [1.0], [0.0]]).

let w1 := param(randn(2, 8, 1.0, 42)).
let b1 := param(zeros(1, 8)).
let w2 := param(randn(8, 1, 1.0, 43)).
let b2 := param(zeros(1, 1)).
let ps := [w1, b1, w2, b2].

fn forward(x: Any) -> Any:
    let h := t_tanh(t_add(t_matmul(x, w1), b1)).
    give t_sigmoid(t_add(t_matmul(h, w2), b2)).
done.

var i := 0.
while i < 3000:
    zero_grad(ps).
    let out := forward(xs).
    let loss := mse_t(out, ys).
    backward(loss).
    sgd_step(ps, 0.5).
    when i % 1000 == 0:
        emit "step " + to Text(i) + " loss " + to Text(round(value_of(loss), 4)).
    done.
    i <- i + 1.
done.

let final := forward(xs).
emit "predictions: " + to Text(map(value_of(final), \r -> round(r[0], 2))).
