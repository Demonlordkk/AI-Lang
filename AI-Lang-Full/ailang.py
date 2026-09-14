#!/usr/bin/env python3
from pathlib import Path
import sys,importlib.util
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT))
from _06_lexer import lex
from _07_parser import Parser
from _58_toolchain import compile_source

def check(source):
    t=lex(source);p=Parser(source).program();return t,p

def compile(source): return compile_source(source)

def run(source,globals=None):
    from vm import VM
    from _11_stdlib import install
    from _17_collections import install as ci
    from _35_stdlib_math import install as mi
    from _47_data import install as di
    e={};install(e);ci(e);mi(e);di(e)
    if globals:e.update(globals)
    return VM(e).run(compile_source(source))

spec=importlib.util.spec_from_file_location('ailang_cli',ROOT/'22_cli.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
if __name__=='__main__':raise SystemExit(m.main())
