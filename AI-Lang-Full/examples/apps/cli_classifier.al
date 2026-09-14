# A professional "normal app" (no web server): a command-line classifier
# that wraps a neural model. It shows the ML/app bridge from the CLI side —
# `train` trains and saves a model file, `predict` and `report` load it.
#
#     ailang run cli_classifier.al -- train -e 200
#     ailang run cli_classifier.al -- predict 0.8 0.1
#     ailang run cli_classifier.al -- report
#
# (The `--` separates ailang's own flags from the program's arguments.)

use packages/cli as c.
use packages/neural as nn.

let usage_text := c.usage("classifier", {"--rate": "real", "-e": "int", "--seed": "int", "--model": "text", "--help": "flag", "-h": "flag"}, ["<command>  one of: train | predict | report"]).

let spec := {"--rate": "real", "-e": "int", "--seed": "int", "--model": "text", "--help": "flag", "-h": "flag"}.
attempt:
    let args := c.parse(args(), spec).
rescue e:
    emit c.color(e.message, 31).
    emit usage_text.
    exit(2).
done.

when has(args.flags, "help"):
    emit usage_text.
    exit(0).
done.

let model_path := get(args.flags, "--model", "classifier.almodel").
var cmd := "help".
when len(args.positional) > 0:
    cmd <- args.positional[0].
done.

fn make_point(blob: Int) -> List:
    when blob == 0:
        give [0.8 + random() - 0.5, 0.1 + random() - 0.5].
    done.
    give [-0.5 + random() - 0.5, 0.9 + random() - 0.5].
done.

when cmd == "train":
    let seed_n := get(args.flags, "--seed", 7).
    seed(seed_n).
    var xs := [].
    var ys := [].
    repeat i in range(40):
        let blob := i % 2.
        append(xs, make_point(blob)).
        append(ys, one_hot(blob, 2)).
    done.
    let net := nn.mlp([2, 4, 2], seed_n).
    let rate := get(args.flags, "--rate", 0.05).
    let epochs := get(args.flags, "-e", 150).
    let data := tensor(xs).
    let labels := tensor(ys).
    emit c.bar(0, epochs, "training").
    nn.train(net, data, \out -> sum_t(ce_t(t_softmax(out), labels)), epochs, rate, 0.0).
    let probs := value_of(t_softmax(nn.forward(net, data))).
    var correct := 0.
    repeat p at i in probs:
        when argmax(p) == i % 2:
            correct <- correct + 1.
        done.
    done.
    emit c.bar(epochs, epochs, "training").
    nn.save(net, model_path).
    emit c.color("saved " + model_path + " (accuracy " + to Text(round(to Real(correct) / 40.0, 3)) + ")", 32).
    exit(0).
done.

when cmd == "predict":
    when not path_exists(model_path):
        emit c.color("no model at " + model_path + " — run the train command first", 31).
        exit(1).
    done.
    when len(args.positional) < 3:
        emit c.color("usage: classifier predict <x> <y>", 31).
        exit(2).
    done.
    let net := nn.load(model_path).
    let x := real(args.positional[1]).
    let y := real(args.positional[2]).
    let out := nn.forward(net, tensor([[x, y]])).
    let p := value_of(t_softmax(out))[0].
    let cls := argmax(p).
    emit "class " + to Text(cls) + " (confidence " + to Text(round(p[cls], 3)) + ")".
    exit(0).
done.

when cmd == "report":
    when not path_exists(model_path):
        emit c.color("no model at " + model_path + " — run the train command first", 31).
        exit(1).
    done.
    let net := nn.load(model_path).
    let probes := [[0.8, 0.1], [-0.5, 0.9], [0.0, 0.5], [1.2, 0.3], [-0.9, 0.7]].
    var rows := [].
    repeat point in probes:
        let out := nn.forward(net, tensor([point])).
        let p := value_of(t_softmax(out))[0].
        let cls := argmax(p).
        append(rows, [to Text(point) + " -> " + to Text(cls), cls, round(p[cls], 3)]).
    done.
    emit c.table(["probe", "class", "confidence"], rows).
    exit(0).
done.

when cmd == "help":
    emit usage_text.
    exit(0).
done.
emit c.color("unknown command: " + cmd, 31).
emit usage_text.
exit(2).
