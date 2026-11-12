import sys, re
src = open('../src/CMPC_NB05.py').read()
src = src[:src.index('def task_check')]
src = src.replace('QUICK = True', 'QUICK = False').replace('WEIGHTS_DIR = "."', 'WEIGHTS_DIR = "../weights"')
src = src.replace('OUT_DIR = "results"', 'OUT_DIR = "out"')
src = re.sub(r'(?m)^(\s*)print\(', r'\1(lambda *a, **k: None)(', src)
G = {"__name__": "certmod"}
exec(compile(src, "nb05", "exec"), G)
