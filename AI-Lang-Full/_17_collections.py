def _put(e,n,v):
    e.define(n,v) if hasattr(e,'define') else e.__setitem__(n,v)
def install(e):
    vals={'first':lambda xs:xs[0] if xs else None,'last':lambda xs:xs[-1] if xs else None,'push':lambda xs,x:xs+[x],'join':lambda xs,sep:sep.join(map(str,xs)),'range':lambda n:list(range(int(n)))}
    for n,v in vals.items():_put(e,n,v)
    return e
