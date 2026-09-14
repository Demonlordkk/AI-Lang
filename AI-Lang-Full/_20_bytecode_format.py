import hashlib,json
from pathlib import Path
from _00_version import BYTECODE_FORMAT,LANGUAGE,VERSION
def safe(v):
 if v is None or isinstance(v,(str,int,float,bool)): return v
 if isinstance(v,(list,tuple)): return [safe(x) for x in v]
 if isinstance(v,dict): return {str(k):safe(x) for k,x in sorted(v.items(),key=lambda z:str(z[0]))}
 return repr(v)
def artifact(p,source=None):
 o={'format':BYTECODE_FORMAT,'language':LANGUAGE,'version':VERSION,'source_sha256':hashlib.sha256(source.encode()).hexdigest() if source is not None else None,'main':{'name':p.main.name,'params':list(p.main.params),'code':[list(x) for x in p.main.code],'constants':safe(p.main.constants)},'functions':{n:{'params':list(f.params),'code':[list(x) for x in f.code],'constants':safe(f.constants)} for n,f in sorted(p.functions.items())}}
 raw=json.dumps(o,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode();o['artifact_sha256']=hashlib.sha256(raw).hexdigest();return o
def write(p,path,source=None):
 o=artifact(p,source);Path(path).write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');return o
def read(path):
 o=json.loads(Path(path).read_text(encoding='utf-8'))
 if o.get('format')!=BYTECODE_FORMAT: raise ValueError('unsupported bytecode format')
 return o
