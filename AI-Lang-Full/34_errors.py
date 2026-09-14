"""Phase 34: explicit Result/Error algebra."""
from dataclasses import dataclass
@dataclass(frozen=True)
class Ok:
    value: object
@dataclass(frozen=True)
class Err:
    error: object
Result = Ok | Err
