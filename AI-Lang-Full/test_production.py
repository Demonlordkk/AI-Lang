import contextlib,io,tempfile
from pathlib import Path
from _58_toolchain import compile_source
from _20_bytecode_format import write,read
from _43_capabilities import FileCapability,CapabilityError
from vm import VM

def run(src):
 out=io.StringIO();e={'len':len,'join':lambda xs,s:s.join(map(str,xs))}
 with contextlib.redirect_stdout(out):VM(e).run(compile_source(src))
 return out.getvalue()
def main():
 assert run('let x := 2 + 3 * 4. emit x.')=='14\n'
 assert run('var x := 1. x <- x + 4. emit x.')=='5\n'
 assert run('fn fact(n: Int) -> Int: when n <= 1: give 1. else: give n * fact(n - 1). done. done. emit fact(6).')=='720\n'
 assert run('let xs := [1, 2, 3]. emit len(xs). emit join(xs, ":").')=='3\n1:2:3\n'
 for src,msg in [('let x := 1. x <- 2.','immutable'),('fn f(x: Int) -> Int: emit x. done. emit f("bad").','expects Int')]:
  try:compile_source(src)
  except Exception as e:assert msg in str(e)
  else:raise AssertionError('type safety failure')
 with tempfile.TemporaryDirectory() as d:
  src='emit "release".';p=compile_source(src);f=Path(d)/'x.albc.json';o=write(p,f,src);assert read(f)['artifact_sha256']==o['artifact_sha256']
  c=FileCapability(d);c.write_text('ok.txt','hello');assert c.read_text('ok.txt')=='hello'
  try:c.read_text('../escape')
  except CapabilityError:pass
  else:raise AssertionError('capability escape')
 print('AI-Lang production validation: PASS')
if __name__=='__main__':main()
