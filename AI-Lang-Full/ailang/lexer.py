"""AI-Lang lexer.

Produces a flat token stream with accurate line/column info for diagnostics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List

from .errors import LexError


@dataclass(frozen=True)
class Token:
    kind: str
    value: Any
    line: int
    col: int
    # True only for a '.' written tight against both neighbours (`point.x`),
    # which distinguishes field access from the statement terminator (`x.`).
    tight: bool = False

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Token({self.kind},{self.value!r},{self.line}:{self.col})"


KEYWORDS = {
    "let": "LET",
    "var": "VAR",
    "emit": "EMIT",
    "fn": "FN",
    "give": "GIVE",
    "when": "WHEN",
    "elif": "ELIF",
    "else": "ELSE",
    "repeat": "REPEAT",
    "while": "WHILE",
    "in": "IN",
    "at": "AT",
    "record": "RECORD",
    "use": "USE",
    "as": "AS",
    "to": "TO",
    "done": "DONE",
    "and": "AND",
    "or": "OR",
    "not": "NOT",
    "stop": "STOP",
    "next": "NEXT",
    "raise": "RAISE",
    "attempt": "ATTEMPT",
    "rescue": "RESCUE",
    "true": "BOOL",
    "false": "BOOL",
    "nothing": "NOTHING",
}

TYPE_NAMES = {"Int", "Real", "Bool", "Text", "Byte", "List", "Map", "Any", "Void", "Function"}

THREE = {"...": "ELLIPSIS"}
TWO = {
    ":=": "DEFINE",
    "<-": "ASSIGN",
    "->": "ARROW",
    "==": "EQEQ",
    "!=": "NE",
    "<=": "LE",
    ">=": "GE",
    "|>": "PIPE",
    "??": "COALESCE",
}
ONE = {
    "+": "PLUS",
    "-": "MINUS",
    "*": "STAR",
    "/": "SLASH",
    "%": "PERCENT",
    "<": "LT",
    ">": "GT",
    "(": "LPAREN",
    ")": "RPAREN",
    "[": "LBRACKET",
    "]": "RBRACKET",
    "{": "LBRACE",
    "}": "RBRACE",
    ",": "COMMA",
    ":": "COLON",
    ".": "DOT",
    "?": "QUESTION",
    "\\": "BACKSLASH",
}

_NUMBER = re.compile(r"\d(?:[\d_]*\d)?(?:\.\d(?:[\d_]*\d)?)?(?:[eE][+-]?\d+)?")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "0": "\0"}


def lex(source: str) -> List[Token]:
    out: List[Token] = []
    i = 0
    line = 1
    col = 1
    n = len(source)

    while i < n:
        c = source[i]

        if c in " \t\r":
            i += 1
            col += 1
            continue
        if c == "\n":
            i += 1
            line += 1
            col = 1
            continue
        # line comment
        if c == "#":
            while i < n and source[i] != "\n":
                i += 1
            continue

        start_line, start_col = line, col

        # string literal (supports escapes and interpolation-free text)
        if c == '"':
            i += 1
            col += 1
            chars: List[str] = []
            while i < n and source[i] != '"':
                ch = source[i]
                if ch == "\\":
                    if i + 1 >= n:
                        raise LexError("unterminated string escape", start_line, start_col)
                    esc = source[i + 1]
                    if esc == "u":
                        hexdigits = source[i + 2 : i + 6]
                        if len(hexdigits) < 4 or not all(x in "0123456789abcdefABCDEF" for x in hexdigits):
                            raise LexError("invalid \\u escape, expected 4 hex digits", line, col)
                        chars.append(chr(int(hexdigits, 16)))
                        i += 6
                        col += 6
                        continue
                    if esc == "\n":
                        i += 2
                        line += 1
                        col = 1
                        continue
                    chars.append(_ESCAPES.get(esc, esc))
                    i += 2
                    col += 2
                    continue
                if ch == "\n":
                    raise LexError("unterminated string (newline in literal)", start_line, start_col)
                chars.append(ch)
                i += 1
                col += 1
            if i >= n:
                raise LexError("unterminated string", start_line, start_col)
            i += 1
            col += 1
            out.append(Token("TEXT", "".join(chars), start_line, start_col))
            continue

        # numbers, before operators so `.` handling is unambiguous
        if c.isdigit():
            m = _NUMBER.match(source, i)
            raw = m.group()
            # `5.` is an Int followed by a statement terminator, not a Real.
            if raw.endswith(".") or (raw.count(".") == 1 and raw.split(".")[1] == ""):
                raw = raw[:-1]
            text = raw.replace("_", "")
            if "." in text or "e" in text or "E" in text:
                value: Any = float(text)
                kind = "REAL"
            else:
                value = int(text)
                kind = "INT"
            out.append(Token(kind, value, line, col))
            i += len(raw)
            col += len(raw)
            continue

        three = source[i : i + 3]
        if three in THREE:
            out.append(Token(THREE[three], three, line, col))
            i += 3
            col += 3
            continue

        two = source[i : i + 2]
        if two in TWO:
            out.append(Token(TWO[two], two, line, col))
            i += 2
            col += 2
            continue

        if c in ONE:
            tight = False
            if c == ".":
                prev_ch = source[i - 1] if i > 0 else ""
                next_ch = source[i + 1] if i + 1 < n else ""
                # `a.b` is field access; `a. b`, `a.\n` and `a.` are terminators
                tight = bool(
                    prev_ch
                    and (prev_ch.isalnum() or prev_ch in "_)]}\"")
                    and next_ch
                    and (next_ch.isalpha() or next_ch == "_")
                )
            out.append(Token(ONE[c], c, line, col, tight))
            i += 1
            col += 1
            continue

        m = _IDENT.match(source, i)
        if m:
            word = m.group()
            if word in KEYWORDS:
                kind = KEYWORDS[word]
                value = (word == "true") if kind == "BOOL" else word
            elif word in TYPE_NAMES:
                kind, value = "TYPE", word
            else:
                kind, value = "IDENT", word
            out.append(Token(kind, value, line, col))
            i += len(word)
            col += len(word)
            continue

        raise LexError(f"unexpected character {c!r}", line, col)

    out.append(Token("EOF", None, line, col))
    return out
