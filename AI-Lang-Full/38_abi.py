"""Phase 38: stable native ABI boundary specification primitives."""
from dataclasses import dataclass
@dataclass(frozen=True)
class ABI:
    name: str
    version: int = 1
    calling_convention: str = "platform-default"
