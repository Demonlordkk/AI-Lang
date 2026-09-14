from dataclasses import dataclass
from _06_lexer import lex
class ParseError(Exception): pass
@dataclass
class Program: statements:list
@dataclass
class Let: name:str;expr:object
@dataclass
class Var: name:str;expr:object
@dataclass
class Assign: name:str;expr:object
@dataclass
class Emit: expr:object
@dataclass
class ExprStmt: expr:object
@dataclass
class Fn: name:str;params:list;return_type:object;body:list
@dataclass
class Give: expr:object
@dataclass
class When: cond:object;then_body:list;else_body:list
@dataclass
class Repeat: name:str;iterable:object;body:list
@dataclass
class Record: name:str;fields:list
@dataclass
class Use: name:str
@dataclass
class Literal: value:object
@dataclass
class Name: value:str
@dataclass
class ListExpr: items:list
@dataclass
class MapExpr: items:list
@dataclass
class Unary: op:str;expr:object
@dataclass
class Binary: left:object;op:str;right:object
@dataclass
class Call: fn:object;args:list
@dataclass
class Index: obj:object;index:object
@dataclass
class Convert: type_name:str;expr:object

class Parser:
 def __init__(self,s): self.ts=lex(s);self.i=0
 def cur(self): return self.ts[self.i]
 def at(self,k): return self.cur().kind==k
 def take(self,k):
  t=self.cur()
  if t.kind!=k: raise ParseError(f'expected {k}, got {t.kind} at {t.line}:{t.column}')
  self.i+=1;return t
 def match(self,k):
  if self.at(k): self.i+=1;return True
  return False
 def dot(self): self.take('DOT')
 def program(self):
  a=[]
  while not self.at('EOF'): a.append(self.statement())
  return Program(a)
 def statement(self):
  if self.match('LET'):
   n=self.take('IDENT').value;self.take('DEFINE');e=self.expr();self.dot();return Let(n,e)
  if self.match('VAR'):
   n=self.take('IDENT').value;self.take('DEFINE');e=self.expr();self.dot();return Var(n,e)
  if self.at('IDENT') and self.ts[self.i+1].kind=='ASSIGN':
   n=self.take('IDENT').value;self.take('ASSIGN');e=self.expr();self.dot();return Assign(n,e)
  if self.match('EMIT'): e=self.expr();self.dot();return Emit(e)
  if self.match('GIVE'): e=self.expr();self.dot();return Give(e)
  if self.match('USE'): n=self.take('IDENT').value;self.dot();return Use(n)
  if self.match('FN'):
   n=self.take('IDENT').value;self.take('LPAREN');p=[]
   if not self.at('RPAREN'):
    while True:
     pn=self.take('IDENT').value;pt=None
     if self.match('COLON'): pt=self.take('TYPE').value
     p.append((pn,pt))
     if not self.match('COMMA'): break
   self.take('RPAREN');rt=None
   if self.match('ARROW'): rt=self.take('TYPE').value
   self.take('COLON');body=self.block();return Fn(n,p,rt,body)
  if self.match('WHEN'):
   c=self.expr();self.take('COLON');tb=self.block(stop_else=True);eb=[]
   if self.match('ELSE'): self.take('COLON');eb=self.block()
   return When(c,tb,eb)
  if self.match('REPEAT'):
   n=self.take('IDENT').value;self.take('IN');it=self.expr();self.take('COLON');return Repeat(n,it,self.block())
  if self.match('RECORD'):
   n=self.take('IDENT').value;self.take('COLON');f=[]
   while not self.at('DONE'):
    fn=self.take('IDENT').value;self.take('COLON');ft=self.take('TYPE').value;self.dot();f.append((fn,ft))
   self.take('DONE');self.dot();return Record(n,f)
  # Expression statements allow function calls and future APIs.
  e=self.expr();self.dot();return ExprStmt(e)
 def block(self,stop_else=False):
  a=[]
  while not self.at('DONE') and (not stop_else or not self.at('ELSE')):
   if self.at('EOF'): raise ParseError('unterminated block')
   a.append(self.statement())
  if stop_else and self.at('ELSE'): return a
  self.take('DONE');self.dot();return a
 def expr(self): return self.orx()
 def orx(self):
  x=self.andx()
  while self.match('OR'): x=Binary(x,'or',self.andx())
  return x
 def andx(self):
  x=self.eq()
  while self.match('AND'): x=Binary(x,'and',self.eq())
  return x
 def eq(self):
  x=self.cmp()
  while self.at('EQEQ') or self.at('NE'):
   op=self.cur().value;self.i+=1;x=Binary(x,op,self.cmp())
  return x
 def cmp(self):
  x=self.term()
  while self.at('LT') or self.at('LE') or self.at('GT') or self.at('GE'):
   op=self.cur().value;self.i+=1;x=Binary(x,op,self.term())
  return x
 def term(self):
  x=self.factor()
  while self.at('PLUS') or self.at('MINUS'):
   op=self.cur().value;self.i+=1;x=Binary(x,op,self.factor())
  return x
 def factor(self):
  x=self.unary()
  while self.at('STAR') or self.at('SLASH') or self.at('PERCENT'):
   op=self.cur().value;self.i+=1;x=Binary(x,op,self.unary())
  return x
 def unary(self):
  if self.match('NOT'): return Unary('not',self.unary())
  if self.match('MINUS'): return Unary('-',self.unary())
  return self.call()
 def call(self):
  x=self.primary()
  while True:
   if self.match('LPAREN'):
    a=[]
    if not self.at('RPAREN'):
     while True:
      a.append(self.expr())
      if not self.match('COMMA'): break
    self.take('RPAREN');x=Call(x,a)
   elif self.match('LBRACKET'):
    idx=self.expr();self.take('RBRACKET');x=Index(x,idx)
   else: break
  return x
 def primary(self):
  t=self.cur()
  if t.kind in ('INT','REAL','TEXT','BOOL'): self.i+=1;return Literal(t.value)
  if t.kind=='IDENT': self.i+=1;return Name(t.value)
  if self.match('LPAREN'): x=self.expr();self.take('RPAREN');return x
  if self.match('LBRACKET'):
   a=[]
   if not self.at('RBRACKET'):
    while True:
     a.append(self.expr())
     if not self.match('COMMA'): break
   self.take('RBRACKET');return ListExpr(a)
  if self.match('TO'):
   typ=self.take('TYPE').value;return Convert(typ,self.primary())
  raise ParseError(f'expected expression, got {t.kind} at {t.line}:{t.column}')
