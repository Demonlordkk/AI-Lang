from _07_parser import *
class LangRuntimeError(Exception): pass
class ReturnSignal(Exception):
 def __init__(self,v): self.value=v
class Env:
 def __init__(self,parent=None): self.parent=parent;self.values={};self.mutable=set()
 def define(self,n,v,mutable=False): self.values[n]=v; self.mutable.add(n) if mutable else None
 def get(self,n):
  if n in self.values:return self.values[n]
  if self.parent:return self.parent.get(n)
  raise LangRuntimeError(f'undefined name: {n}')
 def set(self,n,v):
  if n in self.values:
   if n not in self.mutable: raise LangRuntimeError(f'cannot assign immutable binding: {n}')
   self.values[n]=v;return
  if self.parent:self.parent.set(n,v);return
  raise LangRuntimeError(f'undefined name: {n}')
class UserFunction:
 def __init__(self,node,closure):self.node=node;self.closure=closure
 def __call__(self,*args):
  if len(args)!=len(self.node.params):raise LangRuntimeError(f'{self.node.name} expects {len(self.node.params)} args, got {len(args)}')
  e=Env(self.closure)
  for (n,_),v in zip(self.node.params,args):e.define(n,v,True)
  try:execute(self.node.body,e)
  except ReturnSignal as r:return r.value
  return None
def numeric(a,b=None):
 if isinstance(a,bool) or not isinstance(a,(int,float)) or (b is not None and (isinstance(b,bool) or not isinstance(b,(int,float)))): raise LangRuntimeError('arithmetic requires numeric values')
def ev(n,e):
 if isinstance(n,Literal):return n.value
 if isinstance(n,Name):return e.get(n.value)
 if isinstance(n,ListExpr):return [ev(x,e) for x in n.items]
 if isinstance(n,Convert):
  v=ev(n.expr,e)
  try:
   return {'Int':int,'Real':float,'Text':str,'Bool':bool,'Byte':lambda x:int(x)%256}[n.type_name](v)
  except Exception as ex: raise LangRuntimeError(f'invalid conversion to {n.type_name}: {ex}')
 if isinstance(n,Unary):
  v=ev(n.expr,e)
  if n.op=='not':return not bool(v)
  numeric(v);return -v
 if isinstance(n,Binary):
  a=ev(n.left,e);b=ev(n.right,e);o=n.op
  if o=='and':return bool(a) and bool(b)
  if o=='or':return bool(a) or bool(b)
  if o=='==':return type(a)==type(b) and a==b
  if o=='!=':return not(type(a)==type(b) and a==b)
  if o=='+' and isinstance(a,str) and isinstance(b,str):return a+b
  if o in '+-*/%':
   numeric(a,b)
   if o=='+':return a+b
   if o=='-':return a-b
   if o=='*':return a*b
   if o=='/':
    if b==0:raise LangRuntimeError('division by zero')
    return a/b
   if o=='%':
    if b==0:raise LangRuntimeError('modulo by zero')
    return a%b
  if o in ('<','<=','>','>='):
   if type(a)!=type(b) or not isinstance(a,(int,float,str)):raise LangRuntimeError('comparison requires matching ordered types')
   return {'<':a<b,'<=':a<=b,'>':a>b,'>=':a>=b}[o]
  raise LangRuntimeError(f'unknown operator: {o}')
 if isinstance(n,Call):
  f=ev(n.fn,e);args=[ev(x,e) for x in n.args]
  if not callable(f):raise LangRuntimeError('value is not callable')
  return f(*args)
 raise LangRuntimeError(f'invalid expression: {type(n).__name__}')
def execute(ss,e):
 for n in ss:
  if isinstance(n,Let):e.define(n.name,ev(n.expr,e),False)
  elif isinstance(n,Var):e.define(n.name,ev(n.expr,e),True)
  elif isinstance(n,Assign):e.set(n.name,ev(n.expr,e))
  elif isinstance(n,Emit):print(ev(n.expr,e))
  elif isinstance(n,Fn):e.define(n.name,UserFunction(n,e),False)
  elif isinstance(n,Give):raise ReturnSignal(ev(n.expr,e))
  elif isinstance(n,When):execute(n.then_body if bool(ev(n.cond,e)) else n.else_body,e)
  elif isinstance(n,Repeat):
   it=ev(n.iterable,e)
   if not isinstance(it,(list,tuple,str)):raise LangRuntimeError('repeat requires an iterable')
   for v in it:
    child=Env(e);child.define(n.name,v,False);execute(n.body,child)
  elif isinstance(n,Use): pass
  elif isinstance(n,Record): e.define(n.name, {'record':n.name,'fields':n.fields},False)
  else:raise LangRuntimeError(f'unknown statement: {type(n).__name__}')
