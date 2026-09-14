"""Phase 46: dependency-light HTTP client facade using the host runtime."""
from urllib.request import Request, urlopen
class HTTP:
    @staticmethod
    def get(url, timeout=15):
        with urlopen(Request(url, method="GET"), timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
