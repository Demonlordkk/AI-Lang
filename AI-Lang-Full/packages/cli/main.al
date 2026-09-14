# A command-line application toolkit, written in AI-Lang itself.
#
# Everything a professional terminal program needs: argument parsing with a
# declarative spec, generated usage text, aligned table output, a progress
# bar, interactive confirmation, ANSI color, and JSON config files with
# defaults. Pairs with packages/neural when a CLI wraps a trained model.
#
#     use cli as c.
#     let spec := {"--rate": "real", "-r": "real", "--verbose": "flag",
#                  "--name": "text", "--tag": "list"}.
#     attempt:
#         let args := c.parse(args(), spec).
#     rescue e:
#         emit c.color(e.message, 31).
#         emit c.usage("myapp", spec, ["<command>  one of: train | predict"]).
#         exit(2).
#     done.

record Parsed:
    flags: Map.
    positional: List.
done.

# ---------------------------------------------------------------- arguments
# True for "-5", "-0.25", "-1e3" — negative numbers must stay positional
# (and be accepted as values for int/real flags), never misread as flags.
fn _looks_negative_number(s: Text) -> Bool:
    when not starts_with(s, "-") or len(s) < 2:
        give false.
    done.
    attempt:
        real(s).
        give true.
    rescue e:
        give false.
    done.
done.

# True when an argument token starts a flag (and is not a negative number).
fn _is_flag(a: Text) -> Bool:
    when not starts_with(a, "-"):
        give false.
    done.
    when _looks_negative_number(a):
        give false.
    done.
    give true.
done.

# Parse `args` (a List of Text; pass the program's own `args()`) against
# `spec`, a Map from flag string ("--name" or "-n") to kind: "flag", "text",
# "int", "real" or "list".
#     --name value   /   --name=value   /   -n value
#     --verbose               -> boolean flag (value true)
#     --tag a --tag b         -> "list" flags accumulate values
#     --                      -> everything after is positional
# Unknown flags and bad numbers raise clean, descriptive errors.
fn parse(args: List, spec: Map) -> Parsed:
    var flags := {}.
    var positional := [].
    var i := 0.
    while i < len(args):
        let a := args[i].
        when a == "--":
            var j := i + 1.
            while j < len(args):
                append(positional, args[j]).
                j <- j + 1.
            done.
            i <- len(args).
        done.
        when i < len(args) and _is_flag(a):
            let eq := index_of(a, "=").
            var name := a.
            when eq > 0:
                name <- slice(a, 0, eq).
            done.
            when not has(spec, name):
                raise "unknown flag: " + a + " (try --help for usage)".
            done.
            let kind := spec[name].
            when kind == "flag":
                flags <- set(flags, name, true).
                i <- i + 1.
            done.
            when eq > 0 and kind != "flag":
                flags <- set(flags, name, _coerce(kind, slice(a, eq + 1, len(a)), name)).
                i <- i + 1.
            done.
            when kind != "flag" and eq <= 0:
                when i + 1 >= len(args):
                    raise "flag " + name + " needs a value".
                done.
                let val := args[i + 1].
                when starts_with(val, "-") and len(val) > 1 and not _looks_negative_number(val) and kind != "text":
                    raise "flag " + name + " needs a value (got " + val + ")".
                done.
                when kind == "list":
                    let cur := get(flags, name, []).
                    append(cur, val).
                    flags <- set(flags, name, cur).
                else:
                    flags <- set(flags, name, _coerce(kind, val, name)).
                done.
                i <- i + 2.
            done.
        done.
        when i < len(args) and not _is_flag(a):
            append(positional, a).
            i <- i + 1.
        done.
    done.
    give Parsed(flags, positional).
done.

# Convert a raw value to its declared kind; bad numbers raise cleanly.
fn _coerce(kind: Text, raw: Text, flag: Text) -> Any:
    when kind == "int":
        give int(raw).
    done.
    when kind == "real":
        give real(raw).
    done.
    when kind == "flag":
        give true.
    done.
    give raw.
done.

# Generated usage text for a spec, e.g. for a program called "train".
fn usage(prog: Text, spec: Map, positional_help: List) -> Text:
    var lines := ["usage: " + prog + " [options]"].
    let flags := sort(keys(spec)).
    repeat name in flags:
        let kind := spec[name].
        var note := to Text(kind).
        when kind == "flag":
            note <- "(boolean)".
        done.
        append(lines, "  " + pad(name, 14) + note).
    done.
    repeat ph in positional_help:
        append(lines, "  " + ph).
    done.
    give join(lines, "\n").
done.

# ------------------------------------------------------------------- output
# Aligned column table. `headers` is a List of Text; each row is either a
# List of Any (positionally aligned) or a Map (columns taken by header name).
fn table(headers: List, rows: List) -> Text:
    var cells := [].
    repeat row in rows:
        var line := [].
        when type_of(row) == "List":
            repeat v in row:
                append(line, to Text(v)).
            done.
        else:
            repeat h in headers:
                append(line, to Text(get(row, h, ""))).
            done.
        done.
        append(cells, line).
    done.
    var widths := [].
    repeat h in headers:
        append(widths, len(h)).
    done.
    repeat line in cells:
        var wi := 0.
        repeat cell in line:
            when len(cell) > widths[wi]:
                widths[wi] <- len(cell).
            done.
            wi <- wi + 1.
        done.
    done.
    var out := [_format_row(headers, widths)].
    var rule := [].
    repeat w in widths:
        append(rule, _dashes(w)).
    done.
    append(out, join(rule, "  ")).
    repeat line in cells:
        append(out, _format_row(line, widths)).
    done.
    give join(out, "\n").
done.

fn _format_row(cells: List, widths: List) -> Text:
    var parts := [].
    var i := 0.
    while i < len(cells):
        append(parts, pad(to Text(cells[i]), widths[i])).
        i <- i + 1.
    done.
    give join(parts, "  ").
done.

fn _dashes(n: Int) -> Text:
    var s := "".
    var i := 0.
    while i < n:
        s <- s + "-".
        i <- i + 1.
    done.
    give s.
done.

fn _hashes(n: Int) -> Text:
    var s := "".
    var i := 0.
    while i < n:
        s <- s + "#".
        i <- i + 1.
    done.
    give s.
done.

# Progress bar string: [#####-----------------------]  17%  label
fn bar(count: Int, total: Int, label: Text) -> Text:
    let width := 24.
    var clamped := count.
    when count < 0:
        clamped <- 0.
    done.
    when count > total:
        clamped <- total.
    done.
    var frac := 0.0.
    when total > 0:
        frac <- to Real(clamped) / to Real(total).
    done.
    let filled := to Int(frac * to Real(width) + 0.5).
    let pct := pad(to Text(to Int(frac * 100.0)) + "%", 4).
    give "[" + _hashes(filled) + _dashes(width - filled) + "] " + pct + "  " + label.
done.

# ANSI color: 31 red, 32 green, 33 yellow, 34 blue, 36 cyan, 1 bold, 90 dim.
fn color(s: Text, code: Int) -> Text:
    give "\u001b[" + to Text(code) + "m" + s + "\u001b[0m".
done.

# ---------------------------------------------------------------- interactive
# Ask "label [y/N]" on standard input; true for y/yes, false otherwise.
fn confirm(label: Text) -> Bool:
    emit label + " [y/N]".
    let ans := trim(lower(read_line())).
    when ans == "y" or ans == "yes":
        give true.
    done.
    give false.
done.

# ------------------------------------------------------------------- config
# Load a JSON config file over `defaults`; a missing file is not an error,
# defaults are used. Explicit file values win over defaults.
fn config(path: Text, defaults: Map) -> Map:
    when not path_exists(path):
        give defaults.
    done.
    let doc := json_decode(read_file(path)).
    var out := defaults.
    repeat k in keys(doc):
        out <- set(out, k, doc[k]).
    done.
    give out.
done.
