from compiler import ProgramCode,FunctionCode
class VMError(Exception): pass
class Function:
 def __init__(self,code,vm):self.code=code;self.vm=vm
 def __call__(self,*args):return self.vm.call(self.code,list(args))
class VM:
 def __init__(self,globals=None,fuel=1_000_000):
  self.globals=dict(globals or {});self.fuel=fuel
  self.globals.setdefault('true',True);self.globals.setdefault('false',False)
 def run(self,program):
  for n,f in program.functions.items(): self.globals[n]=Function(f,self)
  return self.exec_code(program.main,self.globals,[])
 def call(self,fn,args):
  if len(args)!=len(fn.params):raise VMError(f'{fn.name} expects {len(fn.params)} args, got {len(args)}')
  env=dict(self.globals);env.update(zip(fn.params,args));return self.exec_code(fn,env,[])
 def exec_code(self,fn,env,stack):
  ip=0;iters=[]; immutable=set()
  code=fn.code; const=fn.constants
  while ip<len(code):
   self.fuel-=1
   if self.fuel<0: raise VMError('execution limit exceeded')
   ins=code[ip];op=ins[0];a=ins[1:];ip+=1
   try:
    if op=='PUSH':stack.append(a[0])
    elif op=='LOAD':
     if a[0] not in env: raise VMError(f'undefined name: {a[0]}')
     stack.append(env[a[0]])
    elif op=='STORE':
     if a[0] in env: raise VMError(f'binding already exists: {a[0]}')
     env[a[0]]=stack.pop()
     if not a[1]: immutable.add(a[0])
    elif op=='SET':
     if a[0] not in env: raise VMError(f'undefined name: {a[0]}')
     if a[0] in immutable: raise VMError(f'cannot assign to immutable binding: {a[0]}')
     env[a[0]]=stack.pop()
    elif op=='PRINT':print(stack.pop())
    elif op=='POP':stack.pop()
    elif op=='UNARY':
     v=stack.pop();stack.append((not bool(v)) if a[0]=='not' else -v)
    elif op=='BINARY':
     b=stack.pop();x=stack.pop();o=a[0]
     if o=='and':r=bool(x) and bool(b)
     elif o=='or':r=bool(x) or bool(b)
     elif o=='==':r=type(x)==type(b) and x==b
     elif o=='!=':r=not(type(x)==type(b) and x==b)
     elif o in ('+','-','*','/','%'):
      if o=='+' and isinstance(x,str) and isinstance(b,str):r=x+b
      elif o=='+':r=x+b
      elif o=='-':r=x-b
      elif o=='*':r=x*b
      elif o=='/':
       if b==0:raise VMError('division by zero')
       r=x/b
      else:
       if b==0:raise VMError('modulo by zero')
       r=x%b
     elif o in ('<','<=','>','>='):r={'<':x<b,'<=':x<=b,'>':x>b,'>=':x>=b}[o]
     else:raise VMError(f'unknown operator {o}')
     stack.append(r)
    elif op=='CONVERT':
     v=stack.pop();t=a[0];f={'Int':int,'Real':float,'Text':str,'Bool':bool,'Byte':lambda z:int(z)%256}.get(t)
     if not f:raise VMError(f'unknown conversion {t}')
     stack.append(f(v))
    elif op=='MAKE_LIST':
     n=a[0];stack[-n:]=[stack[-n:]] if n else [[]]
    elif op=='INDEX':
     idx=stack.pop();obj=stack.pop()
     try:stack.append(obj[idx])
     except Exception as e:raise VMError(f'index error: {e}')
    elif op=='CALL':
     n=a[0];args=stack[-n:] if n else []; del stack[-n:]
     fnv=stack.pop()
     if not callable(fnv):raise VMError('value is not callable')
     stack.append(fnv(*args))
    elif op=='DEF_FUNC': stack.append(self.globals[a[0]])
    elif op=='JUMP':ip=a[0]
    elif op=='JUMP_IF_FALSE':
     if not stack.pop():ip=a[0]
    elif op=='ITER_INIT':
     it=stack.pop()
     try:iters.append(iter(it))
     except TypeError:raise VMError('repeat requires an iterable')
    elif op=='ITER_NEXT':
     try:v=next(iters[-1]);env[a[1]]=v
     except StopIteration:iters.pop();ip=a[0]
    elif op=='ITER_END':pass
    elif op=='RETURN':return stack.pop() if stack else None
    elif op=='HALT':return None
    elif op in ('RECORD','USE'):pass
    else:raise VMError(f'unknown opcode {op}')
   except VMError: raise
   except Exception as e: raise VMError(str(e)) from e
  return None
