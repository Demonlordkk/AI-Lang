
from pathlib import Path
import json,zipfile
def package_project(d,out=None):
    r=Path(d);m=json.loads((r/"ailang.project.json").read_text());out=Path(out or r/(m.get("name","app")+"-bundle.zip"))
    with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as z:
        z.write(r/"ailang.project.json","ailang.project.json");z.write(r/m.get("entry","main.al"),m.get("entry","main.al"))
    return out
