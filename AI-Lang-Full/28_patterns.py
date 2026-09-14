"""Phase 28: enum and pattern metadata."""
from dataclasses import dataclass
@dataclass(frozen=True)
class Variant:
    name: str
    fields: tuple = ()
@dataclass
class EnumDef:
    name: str
    variants: tuple
