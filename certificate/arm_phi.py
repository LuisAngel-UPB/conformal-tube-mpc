import os, re, numpy as np, pickle
from joblib import Parallel, delayed
S = os.path.dirname(os.path.abspath(__file__)); WD = os.path.join(S, '..', 'weights')
def load():
    src = open('../src/CMPC_NB03.py').read()
    a = src[:src.index('t0 = time.time(); FMODELS')]; b = src[src.index('def jac_fd'):src.index('def task_check')]
    a = a.replace('QUICK = True', 'QUICK = False').replace('OUT_DIR = "results"', f'OUT_DIR = "{S}/out3"')
    G = {"__name__": "armcert"}; s = re.sub(r'(?m)^(\s*)print\(', r'\1(lambda *a, **k: None)(', a + "\nFMODELS = {}\n" + b)
    exec(compile(s, 'nb03', 'exec'), G); return G
def model(G, kind):
    if kind == 'NOMINAL': return G['F_nominal']
    z = np.load(f'{WD}/arm_{kind}_35traj_seed0.npz'); net = G['ArmNet'](kind, 9, np.random.default_rng(0))
    net.W = [z[f'arr_{i}'] for i in range(6)]
    for k in ('x_mean', 'x_std', 'y_mean', 'y_std', 'r_std', 'th_scale'): setattr(net, k, z[k])
    return G['ArmSurrogate'](net)
def job(kind, sc, m):
    G = load(); F = model(G, kind)
    L = G['ltv_along_reference'](F, G['REFS'][sc], m); Kg = G['mpc_gains'](L)
    Phi = L['A'] + np.einsum('kij,kjl->kil', L['B'], Kg)
    return kind, sc, m, Phi, L['A'][-1], L['B'][-1], float(np.abs(L['U']).max())
G = load()
tasks = [(k, sc, m) for k in ('NOMINAL', 'MLP', 'HYBRID', 'PINN') for sc in G['SCALES'] for m in G['PAYLOADS']]
res = Parallel(n_jobs=4, verbose=5)(delayed(job)(*t) for t in tasks)
pickle.dump(res, open(f'{S}/arm_phi.pkl', 'wb')); print('done', len(res))
