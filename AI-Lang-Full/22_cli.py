from pathlib import Path
import argparse,sys
from _07_parser import Parser
from _08_typecheck import TypeChecker
from _58_toolchain import compile_source
from vm import VM
from _11_stdlib import install
from _17_collections import install as install_collections
from _35_stdlib_math import install as install_math
from _47_data import install as install_data
from _20_bytecode_format import write as write_bytecode
from _00_version import LANGUAGE,VERSION
def env():
 e={};install(e);install_collections(e);install_math(e);install_data(e);return e
def main(argv=None):
 ap=argparse.ArgumentParser(prog='ailang');sub=ap.add_subparsers(dest='cmd',required=True)
 for n in ('run','check','build','fmt','lint'):
  p=sub.add_parser(n);p.add_argument('file')
 sub.add_parser('version');a=ap.parse_args(argv)
 try:
  if a.cmd=='version':print(f'{LANGUAGE} {VERSION}');return 0
  source=Path(a.file).read_text(encoding='utf-8')
  if a.cmd=='fmt':
   from _56_formatter import format_source;Path(a.file).write_text(format_source(source),encoding='utf-8');return 0
  if a.cmd=='lint':
   from _57_linter import lint;issues=lint(source);[print(x) for x in issues];return 1 if issues else 0
  if a.cmd=='check':TypeChecker().check(Parser(source).program());print('AI-Lang check: PASS');return 0
  program=compile_source(source)
  if a.cmd=='build':
   out=Path(a.file).with_suffix('.albc.json');write_bytecode(program,out,source);print(out);return 0
  VM(env()).run(program);return 0
 except Exception as e:print(f'AI-Lang error: {e}',file=sys.stderr);return 1
if __name__=='__main__':raise SystemExit(main())
