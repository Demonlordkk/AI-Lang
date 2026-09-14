from _07_parser import Parser
from _08_typecheck import TypeChecker
from _21_optimizer import optimize
from compiler import Compiler
class ToolchainError(Exception): pass
def compile_source(source):
 try:
  p=Parser(source).program();TypeChecker().check(p);optimize(p);return Compiler().compile(p)
 except Exception as e: raise ToolchainError(str(e)) from e
