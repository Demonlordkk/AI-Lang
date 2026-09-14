"""Phase 95: package integrity primitives."""
import hashlib
def sha256_bytes(data): return hashlib.sha256(data).hexdigest()
