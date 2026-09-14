# An interactive terminal app: a little calculator REPL.
#
# In a terminal, type expressions like "3 + 4" and press enter; type
# "quit" to leave. On end-of-input the EOF error becomes a normal exit
# through attempt/rescue, so the app is also safe to run unattended
# (piped input, CI, the test harness) — it processes whatever it is
# given and stops cleanly.

fn eval_expr(line: Text) -> Real:
    let parts := split(trim(line), " ").
    when len(parts) != 3:
        raise "expected 'a op b', e.g. 3 + 4".
    done.
    let a := real(parts[0]).
    let b := real(parts[2]).
    given parts[1]:
    is "+":
        give a + b.
    is "-":
        give a - b.
    is "*":
        give a * b.
    is "/":
        when b == 0.0:
            raise "division by zero".
        done.
        give a / b.
    else:
        raise "unknown operator " + parts[1].
    done.
done.

emit "AI-Lang calculator — type 'a op b' (3 + 4), or 'quit'".
while true:
    var line := "".
    attempt:
        line <- input("> ").
    rescue e:
        emit "(end of input — bye)".
        stop.
    done.
    line <- trim(line).
    when line == "":
        next.
    done.
    when line == "quit" or line == "exit":
        emit "bye".
        stop.
    done.
    attempt:
        emit "  = {eval_expr(line)}".
    rescue e:
        emit "  error: {e.message}".
    done.
done.
