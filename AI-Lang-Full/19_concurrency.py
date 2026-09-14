from concurrent.futures import ThreadPoolExecutor
class Scheduler:
    def __init__(self,max_workers=4):self.pool=ThreadPoolExecutor(max_workers=max_workers)
    def spawn(self,fn,*args):return self.pool.submit(fn,*args)
    def close(self):self.pool.shutdown(wait=True)
def install(env,scheduler=None):
    s=scheduler or Scheduler()
    env.define('spawn',s.spawn)
    return s
