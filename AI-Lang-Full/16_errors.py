from dataclasses import dataclass
@dataclass
class Diagnostic:
    phase:str; message:str; line:int|None=None; column:int|None=None
class AILangError(Exception):
    def __init__(self,message,phase='runtime',line=None,column=None):
        super().__init__(message);self.diagnostic=Diagnostic(phase,message,line,column)
def wrap_error(exc,phase):
    if isinstance(exc,AILangError): return exc
    return AILangError(str(exc),phase)
