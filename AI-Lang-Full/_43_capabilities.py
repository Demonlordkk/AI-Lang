from pathlib import Path
class CapabilityError(PermissionError): pass
class FileCapability:
 def __init__(self,root): self.root=Path(root).resolve()
 def _safe(self,n):
  p=(self.root/n).resolve()
  if p!=self.root and self.root not in p.parents: raise CapabilityError('path escapes capability root')
  return p
 def read_text(self,n): return self._safe(n).read_text(encoding='utf-8')
 def write_text(self,n,t):
  p=self._safe(n);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(t,encoding='utf-8')
 def exists(self,n): return self._safe(n).exists()
