import subprocess,sys
from pathlib import Path
R=Path(__file__).parent
for f in ('test_compiler.py','test_typecheck.py','test_runtime_rules.py','test_production.py'):
    subprocess.run([sys.executable,str(R/f)],check=True)
subprocess.run([sys.executable,'ailang.py','version'],check=True,cwd=R)
subprocess.run([sys.executable,'ailang.py','check','examples/basic.al'],check=True,cwd=R)
print('AI-Lang FULL PRODUCTION SUITE: PASS')
