"""Phase 48: deterministic JSON serialization facade."""
import json
class JSON:
    @staticmethod
    def encode(value): return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    @staticmethod
    def decode(text): return json.loads(text)
