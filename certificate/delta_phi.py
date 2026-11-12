import sys, os, numpy as np, pickle
from joblib import Parallel, delayed
S = os.path.dirname(os.path.abspath(__file__))
def job(label, f, sc, m):
    exec(open(os.path.join(S, 'load.py')).read(), globals())
    g = G
    F = g['F_nominal'] if f is None else g['NumpyFF'](os.path.join(S, "..", "weights", f)).as_F()
    L = g['ltv_along_reference'](F, g['REFS'][sc], m); Kg = g['mpc_gains'](L)
    Phi = L['A'] + np.einsum('kij,kjl->kil', L['B'], Kg)
    return label, sc, m, Phi.astype(np.float64), L['A'][-1], L['B'][-1], float(np.abs(L['U']).max())
MODELS = [("Nominal", None), ("MLP-35", "ff18_MLP_35traj_seed0_e150.npz"), ("HYBRID-35", "ff18_HYBRID_35traj_seed0_e150.npz"), ("PINN-175", "ff18_PINN_175traj_seed0_e150.npz")]
tasks = [(lab, f, sc, m) for lab, f in MODELS for sc in (2.0, 2.4, 3.2, 4.8, 6.4) for m in range(7)]
res = Parallel(n_jobs=4, verbose=5)(delayed(job)(*t) for t in tasks)
pickle.dump(res, open(os.path.join(S, 'delta_phi.pkl'), 'wb'))
print('done', len(res))
