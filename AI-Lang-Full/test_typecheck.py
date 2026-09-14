from _07_parser import Parser
from _08_typecheck import TypeChecker,TypeCheckError

def ok(src): TypeChecker().check(Parser(src).program())
def bad(src):
    try: ok(src)
    except TypeCheckError:return
    raise AssertionError('expected type error')

ok('let x := 1.\nemit x.\n')
ok('var x := 1.\nx <- 2.\nemit x.\n')
ok('fn add(a: Int, b: Int) -> Int:\n    give a + b.\ndone.\nemit add(2, 3).\n')
bad('let x := 1.\nx <- 2.\n')
bad('let x := 1.\nemit x + "bad".\n')
bad('fn add(a: Int, b: Int) -> Int:\n    give a + b.\ndone.\nemit add(2).\n')
bad('when 1:\n    emit 2.\ndone.\n')
print('AI-Lang type checker: PASS')
