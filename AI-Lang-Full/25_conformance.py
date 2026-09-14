from pathlib import Path
import hashlib,json
VERSION='0.2.0'
def conformance_cases():
    return [
      ('arithmetic','let x := 2 + 3. emit x.'),
      ('branch','when true: emit "yes". done.'),
      ('loop','repeat x in [1,2,3]: emit x. done.'),
      ('function','fn add(a: Int, b: Int) -> Int: give a+b. done. emit add(2,3).')]
def release_manifest(root):
    root=Path(root);files=[]
    for p in sorted(root.glob('*.py')):
        if p.name.startswith('__'):continue
        files.append({'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    return {'language':'AI-Lang','version':VERSION,'files':files}
def write_manifest(root):
    p=Path(root)/'release-manifest.json';p.write_text(json.dumps(release_manifest(root),indent=2));return p
