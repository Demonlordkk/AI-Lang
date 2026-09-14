
from dataclasses import dataclass
@dataclass
class IRInstruction:op:str;args:tuple=()
@dataclass
class IRProgram:instructions:list
class IRBuilder:
    def __init__(self):self.code=[]
    def emit(self,o,*a):self.code.append(IRInstruction(o,a))
    def build(self,p):
        for s in p.statements:self.emit("STMT",type(s).__name__)
        self.emit("HALT");return IRProgram(self.code)
