"""Phase 41: capability-oriented OS access facade."""
from pathlib import Path
class FileCapability:
    def __init__(self, root): self.root = Path(root).resolve()
    def resolve(self, relative):
        p=(self.root / relative).resolve()
        if self.root != p and self.root not in p.parents: raise PermissionError("path escapes capability root")
        return p
    def read_text(self, relative): return self.resolve(relative).read_text()
