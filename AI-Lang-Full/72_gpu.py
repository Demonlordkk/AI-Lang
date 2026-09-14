"""Phase 72: portable GPU capability contract."""
class GPUBackend:
    name="generic"
    def available(self): return False
