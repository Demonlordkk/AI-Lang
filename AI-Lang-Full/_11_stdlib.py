import math,time
def _put(e,n,v):
    e.define(n,v) if hasattr(e,'define') else e.__setitem__(n,v)
def install(e):
    for n,v in {'len':len,'type_of':lambda x:type(x).__name__,'abs':abs,'floor':math.floor,'ceil':math.ceil,'sqrt':math.sqrt,'clock':time.time}.items():_put(e,n,v)
    return e
