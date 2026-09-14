# A recursive-descent expression evaluator: parsing, recursion, state.
fn tokenize(s: Text) -> List:
    var out := [].
    var i := 0.
    let cs := chars(s).
    while i < len(cs):
        let c := cs[i].
        when c == " ":
            i <- i + 1.
        elif contains("0123456789", c):
            var num := "".
            while i < len(cs) and contains("0123456789", cs[i]):
                num <- num + cs[i].
                i <- i + 1.
            done.
            append(out, num).
        else:
            append(out, c).
            i <- i + 1.
        done.
    done.
    give out.
done.

record P:
    toks: List.
    pos: Int.
done.

fn peek(p: P) -> Text:
    when p.pos >= len(p.toks):
        give "".
    done.
    give p.toks[p.pos].
done.

fn parse_expr(p: P) -> List:
    var res := parse_term(p).
    var node := res[0].
    var cur := res[1].
    while peek(cur) == "+" or peek(cur) == "-":
        let op := peek(cur).
        let nxt := P(cur.toks, cur.pos + 1).
        let r2 := parse_term(nxt).
        when op == "+":
            node <- node + r2[0].
        else:
            node <- node - r2[0].
        done.
        cur <- r2[1].
    done.
    give [node, cur].
done.

fn parse_term(p: P) -> List:
    var res := parse_atom(p).
    var node := res[0].
    var cur := res[1].
    while peek(cur) == "*":
        let nxt := P(cur.toks, cur.pos + 1).
        let r2 := parse_atom(nxt).
        node <- node * r2[0].
        cur <- r2[1].
    done.
    give [node, cur].
done.

fn parse_atom(p: P) -> List:
    let t := peek(p).
    when t == "(":
        let inner := parse_expr(P(p.toks, p.pos + 1)).
        let after := inner[1].
        give [inner[0], P(after.toks, after.pos + 1)].
    done.
    give [int(t), P(p.toks, p.pos + 1)].
done.

fn eval(src: Text) -> Int:
    let r := parse_expr(P(tokenize(src), 0)).
    give r[0].
done.

emit eval("2 + 3 * 4").
emit eval("(2 + 3) * 4").
emit eval("10 - 2 - 3").
emit eval("((1 + 2) * (3 + 4)) - 5").
