
class FFIError(Exception):pass
class FFIBridge:
    def __init__(self):self.functions={}
    def register(self,n,f):
        if not isinstance(n,str) or not n.isidentifier():raise FFIError("invalid FFI name")
        if not callable(f):raise FFIError("FFI target must be callable")
        self.functions[n]=f
    def bindings(self):return dict(self.functions)
