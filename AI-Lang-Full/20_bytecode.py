from dataclasses import dataclass,asdict
import json
@dataclass
class Instruction:
    op:str; arg:object=None
@dataclass
class Bytecode:
    version:str
    instructions:list
class BytecodeCompiler:
    VERSION='AILANG-BC-1'
    def compile(self,program):
        return Bytecode(self.VERSION,[Instruction('EXEC',type(s).__name__) for s in program.statements]+[Instruction('HALT')])
    def dump(self,bc,path):
        with open(path,'w') as f: json.dump({'version':bc.version,'instructions':[asdict(x) for x in bc.instructions]},f,indent=2)
class BytecodeVM:
    def run(self,bc):
        if bc.version!=BytecodeCompiler.VERSION: raise ValueError('unsupported bytecode version')
        return [i.op for i in bc.instructions]
