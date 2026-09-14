"""Phase 57: baseline source linter."""
import re
def lint(source):
    issues=[]
    for n,line in enumerate(source.splitlines(),1):
        if len(line)>120: issues.append((n,"line exceeds 120 columns"))
        if '\t' in line: issues.append((n,"tabs are not allowed; use spaces"))
    return issues
