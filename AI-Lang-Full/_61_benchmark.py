import time
def measure(fn,iterations=1):
 if iterations<1: raise ValueError('iterations must be positive')
 t=time.perf_counter();r=None
 for _ in range(iterations): r=fn()
 s=time.perf_counter()-t
 return {'seconds':s,'iterations':iterations,'ops_per_second':iterations/s if s else float('inf'),'result':r}
