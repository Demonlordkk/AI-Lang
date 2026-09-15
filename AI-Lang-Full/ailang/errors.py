"""Structured diagnostics for AI-Lang.

Every error carries a source position so the toolchain can render a caret
diagnostic instead of a bare Python traceback.
"""

from __future__ import annotations


class AILangError(Exception):
    """Base class for every AI-Lang error surfaced to a user."""

    stage = "error"

    def __init__(self, message: str, line: int = 0, col: int = 0):
        super().__init__(message)
        self.message = message
        self.line = line
        self.col = col
        # set when the error came from an imported module, so the diagnostic
        # names the file that actually contains the faulty line
        self.origin = ""

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.line:
            return f"{self.message} (line {self.line}, column {self.col})"
        return self.message

    def render(self, source: str | None = None, filename: str = "<source>") -> str:
        """Return a human readable diagnostic, with a source excerpt if possible."""
        if self.origin and self.origin != filename:
            filename = self.origin
            try:
                import pathlib as _p

                source = _p.Path(self.origin).read_text(encoding="utf-8")
            except OSError:
                source = None
        where = f"{filename}:{self.line}:{self.col}" if self.line else filename
        out = [f"{where}: {self.stage}: {self.message}"]
        if source and self.line:
            lines = source.splitlines()
            if 0 < self.line <= len(lines):
                text = lines[self.line - 1]
                out.append(f"  {self.line:>4} | {text}")
                caret = " " * max(self.col - 1, 0) + "^"
                out.append(f"       | {caret}")
        return "\n".join(out)


class LexError(AILangError):
    stage = "lex error"


class ParseError(AILangError):
    stage = "parse error"


class CheckError(AILangError):
    stage = "type error"

    def __init__(self, message, line=0, col=0, diagnostics=None):
        super().__init__(message, line, col)
        self.diagnostics = diagnostics or []


class CompileError(AILangError):
    stage = "compile error"


class VMError(AILangError):
    stage = "runtime error"


class ImportError_(AILangError):
    stage = "import error"


class ProcessExit(BaseException):
    """Raised by the `exit` builtin to terminate the host process with a
    status code. Deliberately not an AILangError: attempt/rescue must not
    be able to catch a process termination."""

    def __init__(self, code: int = 0):
        super().__init__(f"exit({code})")
        self.code = int(code)


class AILangRaise(AILangError):
    """A value raised by user code via `raise`. Catchable with attempt/rescue."""

    stage = "raised"

    def __init__(self, value, line: int = 0, col: int = 0):
        from .values import display

        super().__init__(display(value), line, col)
        self.value = value


class Panic(BaseException):
    """Raised by the `panic` builtin: a fatal, unrecoverable abort.

    Deliberately not an AILangError (like ProcessExit): attempt/rescue
    must not be able to catch a process panic.
    """

    stage = "panic"

    def __init__(self, message, line: int = 0, col: int = 0):
        super().__init__(str(message))
        self.message = str(message)
        self.line = line
        self.col = col


__all__ = [
    "AILangError",
    "LexError",
    "ParseError",
    "CheckError",
    "CompileError",
    "VMError",
    "ImportError_",
    "AILangRaise",
    "Panic",
]
