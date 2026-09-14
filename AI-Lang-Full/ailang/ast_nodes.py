"""AST node definitions for AI-Lang."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


@dataclass
class Node:
    pass


# ---------------------------------------------------------------- expressions
@dataclass
class Literal(Node):
    value: Any
    line: int = 0
    col: int = 0


@dataclass
class Name(Node):
    value: str
    line: int = 0
    col: int = 0


@dataclass
class ListExpr(Node):
    items: List[Any] = field(default_factory=list)
    line: int = 0
    col: int = 0


@dataclass
class MapExpr(Node):
    # list of (key_expr, value_expr)
    items: List[Tuple[Any, Any]] = field(default_factory=list)
    line: int = 0
    col: int = 0


@dataclass
class Unary(Node):
    op: str
    expr: Any
    line: int = 0
    col: int = 0


@dataclass
class Binary(Node):
    left: Any
    op: str
    right: Any
    line: int = 0
    col: int = 0


@dataclass
class Call(Node):
    fn: Any
    # list of (name_or_None, expr)
    args: List[Tuple[Optional[str], Any]] = field(default_factory=list)
    line: int = 0
    col: int = 0


@dataclass
class Index(Node):
    obj: Any
    index: Any
    line: int = 0
    col: int = 0


@dataclass
class Field(Node):
    obj: Any
    name: str
    line: int = 0
    col: int = 0


@dataclass
class Convert(Node):
    type_name: str
    expr: Any
    line: int = 0
    col: int = 0


@dataclass
class FnExpr(Node):
    """Anonymous function: `fn(a, b) -> Int: ... done` or `\\a, b -> expr`."""
    params: List[Tuple[str, Optional[str]]] = field(default_factory=list)
    return_type: Optional[str] = None
    body: List[Any] = field(default_factory=list)
    name: str = "<anonymous>"
    line: int = 0
    col: int = 0


# ----------------------------------------------------------------- statements
@dataclass
class Let(Node):
    name: str
    expr: Any
    declared_type: Optional[str] = None
    line: int = 0
    col: int = 0


@dataclass
class Var(Node):
    name: str
    expr: Any
    declared_type: Optional[str] = None
    line: int = 0
    col: int = 0


@dataclass
class Assign(Node):
    target: Any  # Name | Index | Field
    expr: Any
    line: int = 0
    col: int = 0


@dataclass
class Emit(Node):
    expr: Any
    line: int = 0
    col: int = 0


@dataclass
class ExprStmt(Node):
    expr: Any
    line: int = 0
    col: int = 0


@dataclass
class Give(Node):
    expr: Any
    line: int = 0
    col: int = 0


@dataclass
class Fn(Node):
    name: str
    params: List[Tuple[str, Optional[str]]]
    return_type: Optional[str]
    body: List[Any]
    line: int = 0
    col: int = 0


@dataclass
class Branch(Node):
    cond: Any
    body: List[Any]


@dataclass
class When(Node):
    branches: List[Branch] = field(default_factory=list)
    else_body: Optional[List[Any]] = None
    line: int = 0
    col: int = 0


@dataclass
class Repeat(Node):
    name: str
    iterable: Any
    body: List[Any]
    index_name: Optional[str] = None
    line: int = 0
    col: int = 0


@dataclass
class While(Node):
    cond: Any
    body: List[Any]
    line: int = 0
    col: int = 0


@dataclass
class Stop(Node):
    line: int = 0
    col: int = 0


@dataclass
class Next(Node):
    line: int = 0
    col: int = 0


@dataclass
class Record(Node):
    name: str
    fields: List[Tuple[str, Optional[str]]]
    line: int = 0
    col: int = 0


@dataclass
class Use(Node):
    path: str
    alias: str
    line: int = 0
    col: int = 0


@dataclass
class Needs(Node):
    """A precondition: `needs amount > 0.` at the top of a function body."""
    expr: Any
    text: str = ""
    line: int = 0
    col: int = 0


@dataclass
class Ensures(Node):
    """A postcondition checked against `result` on every path out."""
    expr: Any
    text: str = ""
    line: int = 0
    col: int = 0


@dataclass
class Raise(Node):
    expr: Any
    line: int = 0
    col: int = 0


@dataclass
class Attempt(Node):
    body: List[Any]
    error_name: str
    rescue_body: List[Any]
    line: int = 0
    col: int = 0


@dataclass
class Program(Node):
    statements: List[Any] = field(default_factory=list)
