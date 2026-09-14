"""Phase 26: generic type model primitives for the reference implementation."""
from dataclasses import dataclass
@dataclass(frozen=True)
class TypeVar:
    name: str
    def __str__(self): return self.name
@dataclass(frozen=True)
class GenericType:
    name: str
    args: tuple = ()
    def __str__(self):
        return self.name if not self.args else f"{self.name}[{', '.join(map(str,self.args))}]"
