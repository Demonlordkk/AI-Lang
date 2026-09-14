from pathlib import Path
import contextlib,io,tempfile
from _07_parser import Parser
from compiler import Compiler
from vm import VM,VMError
from _11_stdlib import install
from _17_collections import install as install_collections

def run(src,fuel=100000):
 p=Compiler().compile(Parser(src).program()); out=io.StringIO()
 e={};install(e);install_collections(e)
 with contextlib.redirect_stdout(out): VM(e,fuel=fuel).run(p)
 return out.getvalue()

def main():
 assert run('let a := 5. var b := 7. b <- b + a. emit b.')=='12\n'
 assert run('fn add(x: Int, y: Int) -> Int:\n give x + y.\ndone.\nemit add(10, 20).')=='30\n'
 assert run('when 2 < 3:\n emit "yes".\nelse:\n emit "no".\ndone.')=='yes\n'
 assert run('repeat x in range(4):\n emit x.\ndone.')=='0\n1\n2\n3\n'
 assert run('let xs := [4, 5, 6]. emit xs[1]. emit join(xs, ":").')=='5\n4:5:6\n'
 assert run('fn fact(n: Int) -> Int:\n when n <= 1:\n  give 1.\n else:\n  give n * fact(n - 1).\n done.\ndone.\nemit fact(6).')=='720\n'
 try: run('emit 1 / 0.')
 except VMError as e: assert 'division by zero' in str(e)
 else: raise AssertionError('division by zero did not fail')
 try: run('fn loop(x: Int) -> Int:\n give loop(x).\ndone.\nemit loop(1).',fuel=100)
 except VMError as e: assert 'execution limit exceeded' in str(e)
 else: raise AssertionError('fuel limit did not fail')
 print('AI-Lang compiler/VM integration: PASS')
if __name__=='__main__':main()
