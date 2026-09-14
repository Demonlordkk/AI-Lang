
from pathlib import Path
import json
def load_manifest(path):
    d=json.loads(Path(path).read_text());deps=d.get("dependencies",{})
    if not isinstance(deps,dict):raise ValueError("dependencies must be an object")
    return {str(k):str(v) for k,v in deps.items()}
def resolve_lock(manifest,path):
    d=json.loads(Path(path).read_text());pk=d.get("packages",{});out={}
    for n,w in manifest.items():
        if n not in pk:raise ValueError(f"dependency {n} is not locked")
        out[n]=str(pk[n])
    return out
