"""Canonical formatter and linter for AI-Lang source."""

from __future__ import annotations

import re

INDENT = "    "
_OPENS = re.compile(r":\s*$")


def format_source(source: str) -> str:
    """Re-indent by block structure and normalise whitespace."""
    out = []
    depth = 0
    for raw in source.splitlines():
        line = raw.strip()
        if not line:
            out.append("")
            continue
        if line.startswith("#"):
            out.append(INDENT * depth + line)
            continue

        # dedent before writing for block-closing keywords
        if line.startswith("done") or line.startswith("else") or line.startswith("elif") \
                or line.startswith("rescue"):
            depth = max(depth - 1, 0)

        line = _normalise(line)
        out.append(INDENT * depth + line)

        # indent after a block opener
        if _OPENS.search(line):
            depth += 1

    # collapse trailing blank lines
    while out and not out[-1]:
        out.pop()
    return "\n".join(out) + ("\n" if out else "")


def _normalise(line: str) -> str:
    # collapse runs of spaces outside of string literals
    result = []
    in_str = False
    i = 0
    while i < len(line):
        c = line[i]
        if c == '"' and (i == 0 or line[i - 1] != "\\"):
            in_str = not in_str
            result.append(c)
        elif not in_str and c == " " and result and result[-1] == " ":
            pass
        else:
            result.append(c)
        i += 1
    text = "".join(result).rstrip()
    if not in_str:
        text = re.sub(r"\s+([,.])", r"\1", text)
        text = re.sub(r",(\S)", r", \1", text)
    return text


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
        stripped = line.strip()
        if stripped.startswith("var ") and "<-" not in source:
            pass
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
