from dataclasses import dataclass
from _07_parser import *
class CompileError(Exception): pass
@dataclass
class FunctionCode: name:str; params:list; code:list; constants:list
@dataclass
class ProgramCode: main:FunctionCode; functions:dict

class Compiler:
 def __init__(self): self.code=[];self.const=[];self.functions={};self.labels=[]
 def emit(self,op,*args): self.code.append((op,*args)); return len(self.code)-1
 def patch(self,pos,target): self.code[pos]=(*self.code[pos][:1],target,*self.code[pos][2:])
 def const_id(self,v):
  for i,x in enumerate(self.const):
   if type(x)==type(v) and x==v:return i
  self.const.append(v);return len(self.const)-1
 def compile(self,p):
  for s in p.statements:self.stmt(s)
  self.emit('HALT')
  main=FunctionCode('<main>',[],self.code,self.const)
  return ProgramCode(main,self.functions)
 def stmt(self,s):
  if isinstance(s,(Let,Var)):
   self.expr(s.expr);self.emit('STORE',s.name, isinstance(s,Var));return
  if isinstance(s,Assign): self.expr(s.expr);self.emit('SET',s.name);return
  if isinstance(s,Emit): self.expr(s.expr);self.emit('PRINT');return
  if isinstance(s,ExprStmt): self.expr(s.expr);self.emit('POP');return
  if isinstance(s,Give): self.expr(s.expr);self.emit('RETURN');return
  if isinstance(s,Fn):
   sub=Compiler()
   for x in s.body: sub.stmt(x)
   sub.emit('PUSH',None);sub.emit('RETURN')
   self.functions[s.name]=FunctionCode(s.name,[n for n,_ in s.params],sub.code,sub.const);self.emit('DEF_FUNC',s.name);return
  if isinstance(s,When):
   self.expr(s.cond);jf=self.emit('JUMP_IF_FALSE',None)
   for x in s.then_body:self.stmt(x)
   if s.else_body:
    je=self.emit('JUMP',None);self.patch(jf,len(self.code))
    for x in s.else_body:self.stmt(x)
    self.patch(je,len(self.code))
   else:self.patch(jf,len(self.code))
   return
  if isinstance(s,Repeat):
   self.expr(s.iterable);self.emit('ITER_INIT');start=len(self.code);nxt=self.emit('ITER_NEXT',None,s.name)
   for x in s.body:self.stmt(x)
   self.emit('JUMP',start);self.patch(nxt,len(self.code));self.emit('ITER_END');return
  if isinstance(s,Record):
   self.emit('RECORD',s.name,[x for x,_ in s.fields]);return
  if isinstance(s,Use): self.emit('USE',s.name);return
  raise CompileError(f'unsupported statement {type(s).__name__}')
 def expr(self,n):
  if isinstance(n,Literal):self.emit('PUSH',n.value);return
  if isinstance(n,Name):self.emit('LOAD',n.value);return
  if isinstance(n,ListExpr):
   for x in n.items:self.expr(x)
   self.emit('MAKE_LIST',len(n.items));return
  if isinstance(n,Convert):self.expr(n.expr);self.emit('CONVERT',n.type_name);return
  if isinstance(n,Unary):self.expr(n.expr);self.emit('UNARY',n.op);return
  if isinstance(n,Binary):
   self.expr(n.left);self.expr(n.right);self.emit('BINARY',n.op);return
  if isinstance(n,Call):
   self.expr(n.fn)
   for x in n.args:self.expr(x)
   self.emit('CALL',len(n.args));return
  if isinstance(n,Index):
   self.expr(n.obj);self.expr(n.index);self.emit('INDEX');return
  raise CompileError(f'unsupported expression {type(n).__name__}')
