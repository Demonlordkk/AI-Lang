from pathlib import Path
from _07_parser import Parser
from _08_runtime import Env,execute
class ModuleError(Exception):pass
def load_module(name,search_paths,base_env):
    for root in map(Path,search_paths):
        p=root/(name.replace('.','/')+'.al')
        if p.exists():
            e=Env(base_env);execute(Parser(p.read_text()).program().statements,e);return e
    raise ModuleError(f'module not found: {name}')
