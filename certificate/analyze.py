import sys, pickle, numpy as np
from scipy.linalg import solve_discrete_are, sqrtm
f, Qd, Rt, Ms = sys.argv[1], [float(x) for x in sys.argv[2].split(',')], float(sys.argv[3]), [int(x) for x in sys.argv[4].split(',')]
res = pickle.load(open(f, 'rb'))
labs = sorted(set(r[0] for r in res), key=[r[0] for r in res].index)
for lab in labs:
    rr = [r for r in res if r[0] == lab]
    ref = max(rr, key=lambda r: (r[1], r[2] == sorted(set(x[2] for x in rr))[len(set(x[2] for x in rr))//2]))
    A, B = ref[4], ref[5]; nx = A.shape[0]; nu = B.shape[1]
    Q = np.diag(np.repeat(Qd, nu)) if len(Qd) == 3 else np.diag(Qd)
    P = solve_discrete_are(A, B, Q, Rt*np.eye(nu)); Ph = np.real(sqrtm(P)); Pi = np.linalg.inv(Ph)
    out = {M: [0.0, 0.0] for M in Ms}
    for r in rr:
        Phi = r[3]; K = len(Phi); T = np.einsum('ij,kjl,lm->kim', Ph, Phi, Pi)   # transformed one-step maps
        for M in Ms:
            if K <= M: continue
            prod = np.repeat(np.eye(nx)[None], K - M, axis=0); cmax = 1.0
            for j in range(M):
                prod = np.einsum('kij,kjl->kil', T[j:K - M + j], prod)
                if j < M - 1: cmax = max(cmax, np.linalg.norm(prod, 2, axis=(1, 2)).max())
            rho = np.linalg.norm(prod, 2, axis=(1, 2)).max()
            out[M][0] = max(out[M][0], rho); out[M][1] = max(out[M][1], cmax)
    print(lab, ' '.join(f'M={M}: rho={v[0]:.3f} c={v[1]:.2f}' for M, v in out.items()), flush=True)
