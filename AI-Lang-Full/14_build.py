
from pathlib import Path
import json,hashlib
def build_project(d):
    r=Path(d);m=json.loads((r/"ailang.project.json").read_text());e=r/m.get("entry","main.al");s=e.read_text()
    out=r/"build";out.mkdir(exist_ok=True);a=out/(e.stem+".albc")
    a.write_text(json.dumps({"format":"AILANG-IR-BUNDLE-0.1","entry":str(e.relative_to(r)),"sha256":hashlib.sha256(s.encode()).hexdigest(),"source":s},indent=2))
    return a
