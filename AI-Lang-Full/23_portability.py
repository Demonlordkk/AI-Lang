import platform
SUPPORTED={'linux','android','darwin','win32','freebsd'}
def target_report():
    return {'platform':platform.system().lower(),'machine':platform.machine(),'python_host':platform.python_version(),'reference_runtime':'portable'}
def supported(): return platform.system().lower() in SUPPORTED
