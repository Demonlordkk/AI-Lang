"""Phase 62: documentation extraction foundation."""
import re
def extract_doc_comments(source):
    return [line[2:].strip() for line in source.splitlines() if line.lstrip().startswith("##")]
