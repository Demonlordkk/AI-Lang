#!/usr/bin/env python3
"""Entry point shim so `python3 ailang.py ...` works from the repo root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ailang.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
