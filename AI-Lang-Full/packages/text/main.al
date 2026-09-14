# Text utilities beyond the builtins.

fn title_case(s: Text) -> Text:
    var out := "".
    var capitalise := true.
    repeat ch in chars(s):
        when ch == " ":
            out <- out + ch.
            capitalise <- true.
        else:
            when capitalise:
                out <- out + upper(ch).
                capitalise <- false.
            else:
                out <- out + lower(ch).
            done.
        done.
    done.
    give out.
done.

fn word_count(s: Text) -> Int:
    give len(filter(split(trim(s), " "), \w -> len(w) > 0)).
done.

fn truncate(s: Text, limit: Int) -> Text:
    needs limit >= 0.
    when len(s) <= limit:
        give s.
    done.
    when limit <= 3:
        give slice(s, 0, limit).
    done.
    give slice(s, 0, limit - 3) + "...".
done.

fn is_blank(s: Text) -> Bool:
    give len(trim(s)) == 0.
done.

fn count_occurrences(s: Text, needle: Text) -> Int:
    when len(needle) == 0:
        give 0.
    done.
    give len(split(s, needle)) - 1.
done.
