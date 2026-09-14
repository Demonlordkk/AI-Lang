from _07_parser import *
def fold(n):
    if isinstance(n,Binary):
        n.left=fold(n.left);n.right=fold(n.right)
        if isinstance(n.left,Literal) and isinstance(n.right,Literal):
            a,b,o=n.left.value,n.right.value,n.op
            try:
                if o=='+':v=a+b
                elif o=='-':v=a-b
                elif o=='*':v=a*b
                elif o=='/':v=a/b
                elif o=='%':v=a%b
                elif o=='==':v=type(a)==type(b) and a==b
                elif o=='!=':v=not(type(a)==type(b) and a==b)
                elif o in ('<','<=','>','>='):v={'<':a<b,'<=':a<=b,'>':a>b,'>=':a>=b}[o]
                elif o=='and':v=bool(a) and bool(b)
                elif o=='or':v=bool(a) or bool(b)
                else:return n
                return Literal(v)
            except Exception:return n
    if isinstance(n,Unary):
        n.expr=fold(n.expr)
        if isinstance(n.expr,Literal):
            if n.op=='-': return Literal(-n.expr.value)
            if n.op=='not': return Literal(not bool(n.expr.value))
    if isinstance(n,ListExpr):n.items=[fold(x) for x in n.items]
    if isinstance(n,Call):n.fn=fold(n.fn);n.args=[fold(x) for x in n.args]
    return n
def optimize(program):
    for s in program.statements:
        for attr in ('expr','cond','iterable'):
            if hasattr(s,attr): setattr(s,attr,fold(getattr(s,attr)))
    return program
