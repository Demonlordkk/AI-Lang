"""Phase 27: trait/interface metadata."""
from dataclasses import dataclass, field
@dataclass
class Trait:
    name: str
    methods: dict = field(default_factory=dict)
@dataclass
class Impl:
    trait: str
    target: str
    methods: dict = field(default_factory=dict)
