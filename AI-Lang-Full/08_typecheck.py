from _07_parser import *
from dataclasses import dataclass
class TypeCheckError(Exception): pass
@dataclass(frozen=True)
class Ty: name:str; elem:object=None
ANY=Ty('Any');VOID=Ty('Void')
BUILTINS={'len':([('value',None)],'Int'),'type_of':([('value',None)],'Text'),'abs':([('value',None)],'Any'),'floor':([('value',None)],'Int'),'ceil':([('value',None)],'Int'),'sqrt':([('value',None)],'Real'),'clock':([],'Real'),'str':([('value',None)],'Text'),'int':([('value',None)],'Int'),'real':([('value',None)],'Real'),'bool':([('value',None)],'Bool'),'print':([('value',None)],'Void'),'join':([('items','List'),('sep','Text')],'Text'),'range':([('n','Int')],'List'),'first':([('items','List')],'Any'),'last':([('items','List')],'Any'),'push':([('items','List'),('value',None)],'List')}
class TypeChecker:
 def __init__(self): self.errors=[];self.scopes=[{}];self.functions={};self.records={};self.current_return=VOID;self.current_fn=None;self._builtins()
 def _builtins(self):
  for n,(p,r) in BUILTINS.items(): self.functions[n]=(p,r);self.scopes[0][n]=(Ty('Function'),False)
 def error(self,m): self.errors.append(m)
 def declare(self,n,t,mut=False):
  if n in self.scopes[-1]: self.error(f'duplicate binding: {n}')
  self.scopes[-1][n]=(t,mut)
 def lookup(self,n):
  for s in reversed(self.scopes):
   if n in s:return s[n]
  return None
 def compatible(self,a,b): return a==b or a==ANY or b==ANY or (a.name=='Real' and b.name=='Int')
 def check(self,p):
  for s in p.statements:
   if isinstance(s,Fn):
    if s.name in self.functions:self.error(f'duplicate function: {s.name}')
    self.functions[s.name]=(s.params,s.return_type or 'Void');self.scopes[0][s.name]=(Ty('Function'),False)
   elif isinstance(s,Record):
    if s.name in self.records:self.error(f'duplicate record: {s.name}')
    self.records[s.name]=s.fields
  for s in p.statements:self.stmt(s)
  if self.errors:raise TypeCheckError('\n'.join(self.errors))
  return True
 def stmt(self,s):
  if isinstance(s,(Let,Var)):self.declare(s.name,self.expr(s.expr),isinstance(s,Var));return
  if isinstance(s,Assign):
   x=self.lookup(s.name)
   if not x:self.error(f'undefined name: {s.name}');return
   t,m=x
   if not m:self.error(f'cannot assign to immutable binding: {s.name}')
   et=self.expr(s.expr)
   if not self.compatible(t,et):self.error(f'type mismatch assigning {et.name} to {t.name}: {s.name}')
   return
  if isinstance(s,(Emit,ExprStmt)):self.expr(s.expr);return
  if isinstance(s,Give):
   if self.current_fn is None:self.error('give is only valid inside a function');return
   t=self.expr(s.expr)
   if not self.compatible(self.current_return,t):self.error(f'return type mismatch: expected {self.current_return.name}, got {t.name}')
   return
  if isinstance(s,Use):return
  if isinstance(s,Fn):
   oldr,oldf=self.current_return,self.current_fn;self.current_return=Ty(s.return_type or 'Void');self.current_fn=s;self.scopes.append({})
   for n,t in s.params:self.declare(n,Ty(t) if t else ANY)
   for x in s.body:self.stmt(x)
   if s.return_type and s.return_type!='Void' and not self.returns(s.body):self.error(f'function {s.name} may exit without returning {s.return_type}')
   self.scopes.pop();self.current_return=oldr;self.current_fn=oldf;return
  if isinstance(s,When):
   if self.expr(s.cond)!=Ty('Bool'):self.error('when condition must be Bool')
   for body in (s.then_body,s.else_body):self.scopes.append({});[self.stmt(x) for x in body];self.scopes.pop()
   return
  if isinstance(s,Repeat):
   it=self.expr(s.iterable);elem=it.elem if it.name=='List' else ANY;self.scopes.append({});self.declare(s.name,elem)
   for x in s.body:self.stmt(x)
   self.scopes.pop();return
  if isinstance(s,Record):return
  self.error(f'unsupported statement: {type(s).__name__}')
 def returns(self,b):
  for s in b:
   if isinstance(s,Give):return True
   if isinstance(s,When) and s.else_body and self.returns(s.then_body) and self.returns(s.else_body):return True
  return False
 def expr(self,n):
  if isinstance(n,Literal):
   v=n.value
   if isinstance(v,bool):return Ty('Bool')
   if isinstance(v,int):return Ty('Int')
   if isinstance(v,float):return Ty('Real')
   if isinstance(v,str):return Ty('Text')
   return VOID
  if isinstance(n,Name):
   x=self.lookup(n.value)
   if x:return x[0]
   self.error(f'undefined name: {n.value}');return ANY
  if isinstance(n,ListExpr):
   ts=[self.expr(x) for x in n.items];e=ANY if not ts else ts[0]
   if ts and not all(self.compatible(e,t) for t in ts[1:]):e=ANY
   return Ty('List',e)
  if isinstance(n,Convert):self.expr(n.expr);return Ty(n.type_name)
  if isinstance(n,Unary):
   t=self.expr(n.expr)
   if n.op=='not' and t!=Ty('Bool'):self.error('not requires Bool')
   if n.op=='-' and t.name not in ('Int','Real'):self.error('unary - requires Int or Real')
   return Ty('Bool') if n.op=='not' else t
  if isinstance(n,Binary):
   a,b=self.expr(n.left),self.expr(n.right);o=n.op
   if o in ('and','or'):
    if a!=Ty('Bool') or b!=Ty('Bool'):self.error(f'{o} requires Bool operands')
    return Ty('Bool')
   if o in ('==','!='):return Ty('Bool')
   if o in ('<','<=','>','>='):
    if a.name not in ('Int','Real') or b.name not in ('Int','Real'):self.error(f'{o} requires numeric operands')
    return Ty('Bool')
   if o in ('+','-','*','/','%'):
    if o=='+' and a==Ty('Text') and b==Ty('Text'):return Ty('Text')
    if a.name not in ('Int','Real') or b.name not in ('Int','Real'):self.error(f'{o} requires numeric operands');return ANY
    return Ty('Real') if 'Real' in (a.name,b.name) or o=='/' else Ty('Int')
  if isinstance(n,Index):
   o,i=self.expr(n.obj),self.expr(n.index)
   if o.name!='List':self.error('indexing requires List');return ANY
   if i!=Ty('Int'):self.error('list index must be Int')
   return o.elem or ANY
  if isinstance(n,Call):
   args=[self.expr(x) for x in n.args]
   if isinstance(n.fn,Name) and n.fn.value in self.functions:
    ps,r=self.functions[n.fn.value]
    if len(ps)!=len(args):self.error(f'{n.fn.value} expects {len(ps)} args, got {len(args)}')
    for spec,at in zip(ps,args):
     pn,pt=spec
     if pt=='List' and at.name!='List':self.error(f'argument {pn} expects List, got {at.name}')
     elif pt and pt in ('Int','Real','Bool','Text','Byte') and not self.compatible(Ty(pt),at):self.error(f'argument {pn} expects {pt}, got {at.name}')
    return Ty(r)
   ft=self.expr(n.fn)
   if ft!=Ty('Function') and ft!=ANY:self.error('value is not callable')
   return ANY
  self.error(f'unsupported expression: {type(n).__name__}');return ANY
