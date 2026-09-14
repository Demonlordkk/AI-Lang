"""Phase 65: reproducible build metadata."""
import hashlib,json
def source_digest(source): return hashlib.sha256(source.encode()).hexdigest()
def manifest_digest(manifest): return hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest()
