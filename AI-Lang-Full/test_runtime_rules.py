from _07_parser import Parser
from compiler import Compiler
from vm import VM,VMError

def run(src): return VM().run(Compiler().compile(Parser(src).program()))
try:
    run('let x := 1.\nx <- 2.\n')
except VMError as e:
    assert 'immutable' in str(e)
else: raise AssertionError('immutable binding was mutable')
run('var x := 1.\nx <- 2.\nemit x.\n')
print('AI-Lang runtime binding rules: PASS')
