# Train a classifier in AI-Lang, then serve it as a web app in AI-Lang.
#
# The model is a small 2-layer net trained from scratch on a seeded
# synthetic dataset (two blobs). After the training loop, the same
# program becomes a service: POST /predict with {"x": ..., "y": ...}
# returns the model's class and confidence.

use packages/webapp as w.

seed(7).

# ------------------------------------------------------------- dataset
# Two overlapping-ish blobs around (0.8, 0.1) and (-0.5, 0.9).
fn make_point(blob: Int) -> List:
    when blob == 0:
        give [0.8 + random() - 0.5, 0.1 + random() - 0.5].
    done.
    give [-0.5 + random() - 0.5, 0.9 + random() - 0.5].
done.

var xs := [].
var ys := [].
repeat i in range(40):
    let blob := i % 2.
    append(xs, make_point(blob)).
    append(ys, one_hot(blob, 2)).
done.

let data := tensor(xs).
let labels := tensor(ys).

# ------------------------------------------------------------- train
let w1 := param(randn(2, 4, nothing, 7)).
let b1 := param(zeros(4)).
let w2 := param(randn(4, 2, nothing, 11)).
let b2 := param(zeros(2)).
let weights := [w1, b1, w2, b2].
let opt := adamw(weights, 0.05, 0.0).

fn logits(x: Any) -> Any:
    let h := t_tanh(t_add(t_matmul(x, w1), b1)).
    give t_add(t_matmul(h, w2), b2).
done.

var epoch := 0.
while epoch < 150:
    zero_grad(weights).
    backward(sum_t(ce_t(t_softmax(logits(data)), labels))).
    adamw_step(opt).
    epoch <- epoch + 1.
done.

let probs := value_of(t_softmax(logits(data))).
var correct := 0.
repeat p at i in probs:
    when argmax(p) == i % 2:
        correct <- correct + 1.
    done.
done.
emit "trained: accuracy {round(to Real(correct) / 40.0, 3)} on 40 points".

# ------------------------------------------------------------- serve
let app := w.new().

w.route(app, "POST", "/predict", fn(req: Map) -> Map:
    let point := w.body_json(req).
    let single := tensor([[point.x, point.y]]).
    let p := value_of(t_softmax(logits(single)))[0].
    let cls := argmax(p).
    give w.json(200, {"class": cls, "confidence": round(p[cls], 4)}).
done).
w.route(app, "GET", "/health", \req -> w.text(200, "ok")).

let srv := w.start(app, 8125, "127.0.0.1", true).
sleep(0.4).

emit "predict [0.8, 0.1]:   " + http_post("http://127.0.0.1:8125/predict", json_encode({"x": 0.8, "y": 0.1})).body.
emit "predict [-0.5, 0.9]:  " + http_post("http://127.0.0.1:8125/predict", json_encode({"x": -0.5, "y": 0.9})).body.
emit "predict [0.0, 0.5]:   " + http_post("http://127.0.0.1:8125/predict", json_encode({"x": 0.0, "y": 0.5})).body.
emit "health:               " + to Text(http_get("http://127.0.0.1:8125/health").status).

serve_stop(srv).
emit "done".
