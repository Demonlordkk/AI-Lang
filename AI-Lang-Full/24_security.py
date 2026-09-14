class CapabilityPolicy:
    def __init__(self,allowed=None):self.allowed=set(allowed or [])
    def require(self,name):
        if name not in self.allowed: raise PermissionError(f'capability denied: {name}')
class SafeBindings(dict):
    def __init__(self,policy):self.policy=policy
    def register(self,name,value,capability):
        self.policy.require(capability);self[name]=value
