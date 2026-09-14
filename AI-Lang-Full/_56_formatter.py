"""Phase 56: conservative formatter foundation for .al files."""
def format_source(source):
    lines=[]
    for raw in source.splitlines():
        line=raw.rstrip()
        if line.strip(): lines.append(line)
    return "\n".join(lines) + ("\n" if lines else "")
