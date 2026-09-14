"""Phase 63: lightweight profiler facade."""
import cProfile,pstats,io
def profile(fn,*args,**kwargs):
    p=cProfile.Profile();p.enable()
    result=fn(*args,**kwargs)
    p.disable();s=io.StringIO();pstats.Stats(p,stream=s).sort_stats("cumtime").print_stats()
    return result,s.getvalue()
