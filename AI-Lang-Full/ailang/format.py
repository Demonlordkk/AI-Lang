"""Canonical formatter and linter for AI-Lang source.

The formatter re-indents by block structure. Two things make this more than a
line-by-line pass:

*   A statement can span several lines while a bracket is open, or while the
    next line continues a pipeline with `|>`. Those continuation lines are
    indented one level past the statement that owns them, instead of being
    forced back to block depth.
*   Whitespace is normalised only outside text literals, so interpolation
    slots and string contents are never rewritten.
"""

from __future__ import annotations

import re

INDENT = "    "
_OPENS = re.compile(r":\s*$")
_CLOSERS = ("done", "else", "elif", "rescue")
# two-plus spaces after a comma or `:=` means the author is aligning columns
_ALIGNED = re.compile(r'(,|:=)  +\S|\S  +:=')


def _is_closer(line: str) -> bool:
    """True for a line that closes or chains a block.

    Matches `done.`, `else:`, `elif cond:` and `rescue e:`. Deliberately does
    NOT match a record field that happens to be called `done`, which appears
    as `done: Bool.` -- a type annotation, not a block end.
    """
    for kw in _CLOSERS:
        if line == kw or line == kw + ".":
            return True
        if line.startswith(kw):
            rest = line[len(kw):]
            # `done.` and also `done).` / `done]).` -- a block end that also
            # closes the brackets of the call it was passed to
            if rest[:1] == "." or (rest and set(rest) <= set(").]}") | {"."}):
                return True
            if kw in ("else", "elif", "rescue") and rest[:1] in (":", " "):
                return True
    return False


def _scan(line: str):
    """Return (bracket_delta, ends_in_text) for one line, ignoring literals."""
    depth = 0
    in_str = False
    esc = False
    for c in line:
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
    return depth, in_str


def format_source(source: str) -> str:
    """Re-indent by block structure and normalise whitespace."""
    out = []
    depth = 0          # block nesting from `:` ... `done`
    bracket = 0        # unclosed ( [ { carried across lines
    continued = False  # previous line left a statement unfinished
    cont_depth = 0     # block nesting opened *inside* a continuation

    for raw in source.splitlines():
        line = raw.strip()
        if not line:
            out.append("")
            continue
        if line.startswith("#"):
            out.append(INDENT * depth + line)
            continue

        is_continuation = bracket > 0 or continued

        closer = _is_closer(line)
        if closer:
            if is_continuation and cont_depth > 0:
                # closing a block that was opened inside the continuation,
                # e.g. the `done).` of a block lambda passed to a call
                cont_depth -= 1
            elif not is_continuation:
                depth = max(depth - 1, 0)

        normalised = _normalise(line)

        if is_continuation:
            # A continuation hangs one level in from its owning statement,
            # except a line that opens by closing the bracket it continues --
            # `].` lines up with the `let` that opened the list. Blocks opened
            # inside the continuation add their own levels.
            closing_first = normalised[:1] in (")", "]", "}")
            if closing_first:
                # `].` lines up with the statement that opened the bracket
                extra = 0
            elif closer and cont_depth == 0:
                # `done).` ends the last block *and* the bracket, so it sits
                # back at the owning statement's level
                extra = 0
            elif cont_depth > 0 or closer:
                # inside a block opened by the continuation itself
                extra = cont_depth
            else:
                # a plain wrapped line: one level in
                extra = 1
            out.append(INDENT * (depth + extra) + normalised)
        else:
            out.append(INDENT * depth + normalised)

        delta, _ = _scan(normalised)
        bracket = max(bracket + delta, 0)

        # A statement is finished when brackets are balanced and it ends with
        # `.` (statement terminator) or opens a block with `:`.
        if bracket > 0:
            if _OPENS.search(normalised):
                cont_depth += 1
            continued = True
        elif _OPENS.search(normalised):
            if not is_continuation:
                depth += 1
            else:
                cont_depth += 1
            continued = False
        elif normalised.endswith("."):
            continued = False
            cont_depth = 0
        elif normalised.endswith(":"):
            continued = False
        else:
            # e.g. a line ending in `|>` or a dangling operator
            continued = True

    while out and not out[-1]:
        out.pop()
    return "\n".join(out) + ("\n" if out else "")


def _normalise(line: str) -> str:
    """Collapse duplicate spaces and tidy separators outside text literals.

    Runs of two or more spaces that are used to line values up into columns
    are left alone: deliberate alignment is a readability choice the
    formatter has no business undoing.
    """
    if _ALIGNED.search(line):
        return line.rstrip()
    result = []
    in_str = False
    esc = False
    for c in line:
        if in_str:
            result.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            result.append(c)
        elif c == " " and result and result[-1] == " ":
            # collapse runs of spaces (alignment is handled by the caller)
            pass
        else:
            result.append(c)
    text = "".join(result).rstrip()
    if not in_str:
        text = _outside_strings(text, lambda seg: re.sub(r"\s+([,.])", r"\1", seg))
        text = _outside_strings(text, lambda seg: re.sub(r",(\S)", r", \1", seg))
    return text


def _outside_strings(text: str, fn):
    """Apply `fn` to the parts of `text` that sit outside double quotes."""
    parts = []
    buf = []
    in_str = False
    esc = False
    for c in text:
        if in_str:
            buf.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
                parts.append(("str", "".join(buf)))
                buf = []
            continue
        if c == '"':
            if buf:
                parts.append(("code", "".join(buf)))
                buf = []
            in_str = True
            buf.append(c)
        else:
            buf.append(c)
    if buf:
        parts.append(("str" if in_str else "code", "".join(buf)))
    return "".join(fn(s) if kind == "code" else s for kind, s in parts)


RULES = [
    (re.compile(r"\t"), "tabs are not allowed; use four spaces"),
    (re.compile(r"[ \t]+$"), "trailing whitespace"),
]


def lint(source: str):
    """Return a list of (line, message) diagnostics."""
    issues = []
    for n, line in enumerate(source.splitlines(), 1):
        if len(line) > 120:
            issues.append((n, f"line is {len(line)} characters; limit is 120"))
        for pattern, message in RULES:
            if pattern.search(line):
                issues.append((n, message))
    # structural checks via the parser
    try:
        from .parser import parse
        from .typecheck import TypeChecker
        from .errors import AILangError

        program = parse(source)
        try:
            TypeChecker().check(program)
        except AILangError as e:
            for ln, col, msg in getattr(e, "diagnostics", []) or [(e.line, e.col, e.message)]:
                issues.append((ln, msg))
    except Exception as e:  # lex/parse failure
        line = getattr(e, "line", 0)
        issues.append((line, str(e)))
    issues.sort(key=lambda x: x[0])
    return issues
