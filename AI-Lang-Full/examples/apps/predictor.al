# A trained model, served as a web app — with the model persisted between
# runs. This is the ML/app bridge: the neural package trains a classifier
# from a seeded synthetic dataset (two blobs), saves it to a JSON file, and
# on every later run loads the saved model and serves predictions without
# retraining. POST /predict with {"x": ..., "y": ...} returns the class and
# confidence.

use packages/webapp as w.
use packages/neural as nn.

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

# ------------------------------------------------- train once, reuse forever
let model_path := "predictor.almodel".
var net := nn.mlp([2, 4, 2], 7).
when path_exists(model_path):
    net <- nn.load(model_path).
else:
    nn.train(net, data, \out -> sum_t(ce_t(t_softmax(out), labels)), 150, 0.05, 0.0).
    nn.save(net, model_path).
done.

fn accuracy_of(net: Any) -> Real:
    let probs := value_of(t_softmax(nn.forward(net, data))).
    var correct := 0.
    repeat p at i in probs:
        when argmax(p) == i % 2:
            correct <- correct + 1.
        done.
    done.
    give round(to Real(correct) / 40.0, 3).
done.

emit "model ready: accuracy {accuracy_of(net)} on 40 points (file: {model_path})".

# ------------------------------------------------------------- serve
let app := w.new().

w.route(app, "POST", "/predict", fn(req: Map) -> Map:
    let point := w.body_json(req).
    let single := tensor([[point.x, point.y]]).
    let p := value_of(t_softmax(nn.forward(net, single)))[0].
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
