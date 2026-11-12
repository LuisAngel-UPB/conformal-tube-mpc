# %% [markdown]
# # CMPC-NB04 — Baselines for the conformal-calibrated tube MPC (Delta robot)
#
# **Paper:** *Conformal-calibrated tube MPC with physics-informed learned dynamics for Lagrangian robots* (JAS draft, `paper/main.tex`, Sec. V-B, Tables III and IV).
#
# **What this notebook does.** It compares the proposed calibration with the baselines of Sec. IV-C on the same task distribution, the same controller and the same test episodes (paired comparison). Every method is the same LTV tube MPC; the methods differ only in the prediction model, in the normalizers `σ_k` that shape the tube and in the tube size `q`:
#
# | method | prediction model | `σ_k` | `q` |
# |---|---|---|---|
# | `nominal` | hybrid | – | 0 (no tightening) |
# | `oracle` | exact plant model | – | 0 (lower bound) |
# | `proposed` | hybrid | reference-dependent (NB02) | Algorithm 1 in closed loop, with extra rounds if `q̂ > q̃` |
# | `sample_max` | hybrid | reference-dependent | largest calibration score (no quantile) |
# | `ol_traj` | hybrid | reference-dependent | conformal quantile of trajectory scores of the *nominal* MPC (`q = 0`): open-loop calibration |
# | `ol_union` | hybrid | reference-dependent | one-step conformal quantile at level `δ/K` (union bound over the `K` steps) from nominal-MPC data |
# | `worst_case` | hybrid | constant box: largest `|w_i|` over all steps of the pilot episodes (data-driven worst-case disturbance bound) | 1 |
# | `gp` | hybrid | GP of each component of `w` on the reference state, from the pilot data: `σ_k = s_k + |μ_k|/z` (posterior mean `μ`, standard deviation `s`) | Gaussian quantile `z = Φ⁻¹(1−δ/2)`, i.e. tube `|μ| + z·s` |
# | `ensemble` | mean of the hybrid models of seeds 0, 1, 2 | `|B_k|·std_ens(x_r,k)`, the torque spread of the members along the reference (plus actuator noise) | Gaussian quantile `z` |
#
# For each method the score of an episode is `s = max_k max_i |w_{k,i}|/σ_{k,i}` with that method's `σ`, so **coverage** is the fraction of test episodes in which the mismatch stayed inside the method's own tube for the whole task (the event of Theorem 1). **Step coverage** is the mean fraction of control steps inside the tube, which is what a per-step Gaussian quantile targets. The pilot episodes (nominal MPC, `q = 0`) that fit the normalizers of the proposed method also provide the data of `worst_case` and `gp`, so all data-driven methods use the same data budget before calibration.
#
# **Outputs.** CSV files and figures in `OUT_DIR/CMPC_NB04_<tag>/`, packed into a zip at the end of every run.
#
# **Interrupted runs.** After each method the run saves `checkpoint.pkl` in its folder. If the run stops (power cut, crash), set `RESUME_DIR` to that folder (for example `"results/CMPC_NB04_FULL_20261020_090000"`) and run the notebook again from the top: finished methods are loaded and skipped; only the one in progress is repeated. Keep the other settings unchanged.
#
# **Modes.** `QUICK = True` is a smoke test (two speeds, two payloads, few episodes). `QUICK = False` is the long run for the paper.
#
# **Requirements.** numpy, scipy, pandas, matplotlib, osqp, joblib. Weights of notebook 18 in `WEIGHTS_DIR`: `ff18_HYBRID_35traj_seed{0,1,2}_e150.npz` (seeds 1 and 2 only for the ensemble).

# %%
import os, sys, types, time, json, shutil, datetime, math, pickle
import numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import expm, solve_discrete_are, cho_factor, cho_solve
from scipy.optimize import nnls, minimize
from scipy.ndimage import maximum_filter1d
from scipy import sparse
from scipy.stats import beta as beta_dist, norm
import osqp

QUICK = True                       # True: smoke test (minutes); False: full run for the paper (hours)
WEIGHTS_DIR = "."                  # folder with ff18_HYBRID_35traj_seed{0,1,2}_e150.npz (notebook 18)
OUT_DIR = "results"
RESUME_DIR = None                  # to continue an interrupted run: its folder, e.g. "results/CMPC_NB04_FULL_20261020_090000"
N_JOBS = max(1, (os.cpu_count() or 2) - 1)   # worker processes (1 = serial)

# ---- plant / controller (identical to NB02)
SEVERITY = 1.0
FS = 2000; dt = 1.0 / FS
T_C = 0.002
N_H = 25
TAU_MAX = 5.0
Q_POS, Q_VEL, Q_INT, R_TAU = 1e4, 1.0, 1e5, 1e-2

# ---- tube MPC
EPS_Q = 1.0e-3                     # [rad] joint tracking constraint |dq_i| <= EPS_Q
N_TAIL = N_H
TORQUE_MARGIN = 0.5

# ---- task distribution D (identical to NB02)
REF_C2 = True
SCALES = (2.0, 2.4, 3.2, 4.8, 6.4)
PAYLOADS = (0, 1, 2, 3, 4, 5, 6)
X0_BOX = 2.0e-4
SIGMA_TAU = 0.005

# ---- surrogate and baselines
BASE = ("HYBRID-35", "HYBRID", 35)     # surrogate of the proposed method and of most baselines
ENS_SEEDS = (0, 1, 2)                 # ensemble members (same architecture and data, different seeds)
METHODS = ("nominal", "oracle", "proposed", "sample_max", "ol_traj", "ol_union", "worst_case", "gp", "ensemble")
N_GP = 500                            # GP training points per component of w (random steps of the pilot episodes)

# ---- Algorithm 1
DELTA = 0.10
N_PILOT, N_CAL, N_TEST = 50, 200, 400
PILOT_ITERS, INFLATE = 1, 1.05
MAX_ROUNDS = 3                     # step 4 of Algorithm 1: if q_hat > q_tilde, q_tilde <- INFLATE*q_hat and recalibrate on fresh episodes
DELTAS_CHECK = (0.05, 0.10, 0.20)
N_SPLITS = 500
SEED = 2026                        # same seed and ids as NB02: the test episodes are those of NB02

if QUICK:
    SCALES, PAYLOADS = (SCALES[0], 3.2), (0, 6)
    N_PILOT, N_CAL, N_TEST, N_SPLITS = 16, 24, 24, 100
    N_GP = 300

STEP = int(round(T_C / dt))
TAG = ("QUICK_" if QUICK else "FULL_") + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESUME_DIR or os.path.join(OUT_DIR, f"CMPC_NB04_{TAG}"); os.makedirs(RUN_DIR, exist_ok=True)
def out(name): return os.path.join(RUN_DIR, name)
CONFIG = {k: v for k, v in dict(globals()).items() if k.isupper() and isinstance(v, (int, float, str, tuple, list, bool))}
json.dump(CONFIG, open(out("config.json"), "w"), indent=1, default=str)
print("run folder:", RUN_DIR, "| workers:", N_JOBS)

# %% [markdown]
# ## Robot model, unmodeled dynamics and plant (unchanged from the PINN-MPC notebooks)

# %%
R_BASE, R_PLAT, L_A, L_B = 0.114, 0.035, 0.200, 0.460
THETA0 = np.array([0.070, 0.023, 1e-5, 0.006, 0.253, 1.12, 0.008, 0.006, 0.006, 1.0, 0.008, 0.006, 0.006, 1.0])
M_BASE = 6.0
UNMODELED = dict(b_v=0.05, f_c=0.04, eps=0.02, c_cor=0.06, k_pos=0.04, w_pos=3.0)

def gamma_payload(m): return (M_BASE + float(m)) / M_BASE

def fk(q):
    rd = R_BASE - R_PLAT; C = np.zeros((3, 3))
    for i in range(3):
        a = 2*i*np.pi/3; r = rd + L_A*np.cos(q[i])
        C[i] = [r*np.cos(a), r*np.sin(a), L_A*np.sin(q[i])]
    A1, B1, C1 = 2*(C[1]-C[0]); D1 = 2*L_A*rd*(np.cos(q[0])-np.cos(q[1]))
    A2, B2, C2 = 2*(C[2]-C[0]); D2 = 2*L_A*rd*(np.cos(q[0])-np.cos(q[2]))
    den = A2*B1 - A1*B2
    L1 = (A1*D2-A2*D1)/den; L2 = (A1*C2-A2*C1)/den; L3 = (B2*D1-B1*D2)/den; L4 = (B2*C1-B1*C2)/den
    u = 1 + L2**2 + L4**2
    v = 2*(L3*L4 + L1*L2 - C[0,0]*L4 - C[0,1]*L2 - C[0,2])
    w = C[0,0]**2 + C[0,1]**2 + C[0,2]**2 + L3**2 + L1**2 - 2*C[0,0]*L3 - 2*C[0,1]*L1 - L_B**2
    z = (-v + np.sqrt(v**2 - 4*u*w)) / (2*u)
    return np.array([L3 + L4*z, L1 + L2*z, z])

def ik_vec(P):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]; q = np.zeros((len(P), 3)); rd = R_BASE - R_PLAT
    for i in range(3):
        al = 2*i*np.pi/3; u = -2*L_A*z; aux = x*np.cos(al) + y*np.sin(al)
        v = 2*L_A*(rd - aux); w = L_B**2 - (rd**2 - 2*rd*aux + x**2 + y**2 + z**2 + L_A**2)
        r = np.sqrt(u**2 + v**2 - w**2); den = u**2 + v**2
        s1 = (u*w + v*r)/den; c1 = (v*w - u*r)/den; t1 = np.where(c1 >= 0, np.arcsin(s1), np.pi - np.arcsin(s1))
        s2 = (u*w - v*r)/den; c2 = (v*w + u*r)/den; t2 = np.where(c2 >= 0, np.arcsin(s2), np.pi - np.arcsin(s2))
        q[:, i] = np.where(np.abs(t1) < np.abs(t2), t1, t2)
    return q

def regressor(q, a, p):
    x, y, z = p; s, c = np.sin(q), np.cos(q); Y = np.zeros((3, 14))
    Y[:, 0] = a; Y[:, 4] = c
    for i, (cx, cy, cz, cs) in enumerate([(1, 2, 3, 5), (6, 7, 8, 9), (10, 11, 12, 13)]):
        Y[i, cx] = -x*s[i]; Y[i, cy] = -y*s[i]; Y[i, cz] = z*c[i]; Y[i, cs] = s[i]
    return Y

def tau_unmodeled(q, qd, sev):
    u = UNMODELED
    return sev*(u["b_v"]*qd + u["f_c"]*np.tanh(qd/u["eps"])
                + u["c_cor"]*(qd**2*np.cos(q) + qd[[1, 2, 0]]*qd[[2, 0, 1]])
                + u["k_pos"]*np.sin(u["w_pos"]*q))

def plant_qdd(q, qd, tau, p, m, sev):
    g = gamma_payload(m)
    return (tau - g*(regressor(q, np.zeros(3), p) @ THETA0) - tau_unmodeled(q, qd, sev)) / (g*THETA0[0])

def F_nominal(q, qd, qdd, m):
    return gamma_payload(m) * (regressor(q, qdd, fk(q)) @ THETA0)

# %% [markdown]
# ## Surrogates of notebook 18

# %%
def silu(z): return z / (1.0 + np.exp(-z))

class NumpyFF:
    """Same forward pass as notebook 18. kind in {'MLP', 'PINN', 'HYBRID'}."""
    def __init__(self, path):
        d = np.load(path, allow_pickle=False); self.kind = str(d["kind"])
        self.W = [d[k].astype(float) for k in ("W1", "b1", "W2", "b2", "W3", "b3")]
        self.x_mean, self.x_std = d["x_mean"].astype(float), d["x_std"].astype(float)
        self.y_mean, self.y_std, self.r_std = d["y_mean"].astype(float), d["y_std"].astype(float), d["r_std"].astype(float)
    def net(self, xs):
        W1, b1, W2, b2, W3, b3 = self.W
        return W3 @ silu(W2 @ silu(W1 @ xs + b1) + b2) + b3
    def ff(self, q, qd, qdd, m, p):
        out_ = self.net((np.concatenate([q, qd, qdd, [m], p]) - self.x_mean) / self.x_std)
        if self.kind == "MLP":    return out_*self.y_std + self.y_mean
        if self.kind == "HYBRID": return gamma_payload(m)*(regressor(q, qdd, p) @ THETA0) + out_*self.r_std
        return regressor(q, qdd, p) @ out_
    def as_F(self):
        return lambda q, qd, qdd, m: self.ff(q, qd, qdd, m, fk(q))

def load_model(kind, size, seed):
    if kind == "NOMINAL": return F_nominal
    f = os.path.join(WEIGHTS_DIR, f"ff18_{kind}_{size}traj_seed{seed}_e150.npz")
    net = NumpyFF(f); assert net.kind == kind, f"{f} holds a {net.kind} network"
    return net.as_F()

FMODELS = {}                       # filled below: label -> F(q, qd, qdd, m)

# %% [markdown]
# ## References: Adept pick-and-place path at time scaling `sc`
#
# `REF_C2 = False` reproduces NB01: the vertical dip `H_P·sin²(πτ)` has a non-zero second derivative at the segment ends, so the reference acceleration jumps where segments of different duration meet. With `REF_C2 = True` the dip is `H_P·sin⁴(πτ)`, whose first three derivatives vanish at both segment ends, so the reference acceleration and jerk are continuous; the path, the timing and the dip depth are unchanged.

# %%
P_READY = np.array([0.0, 0.0, 0.5])
WAYPOINTS = [P_READY, np.array([.10, .10, .55]), np.array([.10, -.10, .55]),
             np.array([-.10, -.10, .55]), np.array([-.10, .10, .55]), np.array([.10, .10, .55]), P_READY]
BASE_T = np.array([0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.50]); H_P = 0.05

def adept_reference(scale):
    nt = BASE_T*scale; n = int(round(nt[-1]*FS)) + 1; t = np.arange(n)*dt
    seg = np.clip(np.searchsorted(nt[1:], t, side="right"), 0, len(WAYPOINTS) - 2)
    tau = np.clip((t - nt[seg]) / (nt[seg+1] - nt[seg]), 0, 1); s = 6*tau**5 - 15*tau**4 + 10*tau**3
    W = np.array(WAYPOINTS); P = W[seg] + (W[seg+1] - W[seg])*s[:, None]
    P[:, 2] -= H_P*np.sin(np.pi*tau)**(4 if REF_C2 else 2)
    q = ik_vec(P); qd = np.gradient(q, dt, axis=0, edge_order=2); qdd = np.gradient(qd, dt, axis=0, edge_order=2)
    return dict(t=t, P=P, q=q, qd=qd, qdd=qdd, N=n)

REFS = {sc: adept_reference(sc) for sc in SCALES}
for sc, R in REFS.items():
    jump = np.abs(np.diff(R["qdd"], axis=0)).max()
    print(f"scale {sc}: {R['t'][-1]:.1f} s | peak |qd| {np.abs(R['qd']).max():.2f} rad/s | peak |qdd| {np.abs(R['qdd']).max():.1f} rad/s^2 "
          f"| largest qdd change between samples {jump:.2f} rad/s^2")

# %% [markdown]
# ## LTV model, gains, reference features and tube tightening (off line, once per model × task)
#
# * `ltv_along_reference`, `mpc_gains`: as in NB01–NB02.
# * `ref_features`: `φ_k = [1, ‖q̈_r‖, ‖q̇_r‖², max over the period of ‖q̈_r(t) − q̈_r(t_k)‖, (m/M_BASE)·‖q̈_r‖]`, known off line for every task.
# * `sigma_track`: normalizers along the task, constant (`const`) or `σ_{k,i} = max(β_iᵀφ_k, floor_i)` (`ref`).
# * `unit_tightening`: support functions of `R_{k,j} = ⊕_{i<j} Φ(k+j, k+i+1) W_{k+i}` for `q = 1`, with the time-varying box `W_k = {|w_i| ≤ σ_{k,i}}`.
# * Only the control periods that lie fully inside the reference (`idx_k + STEP ≤ N − 1`) enter the score; NB01 also scored the last, partly simulated period.

# %%
def jac_fd(fun, x0, h):
    f0 = fun(x0); J = np.zeros((len(f0), len(x0)))
    for j in range(len(x0)):
        e = np.zeros(len(x0)); e[j] = h
        J[:, j] = (fun(x0 + e) - fun(x0 - e)) / (2*h)
    return J

NX = 9
def ltv_along_reference(F, R, m):
    idx = np.arange(0, R["N"], STEP); K = len(idx)
    U = np.zeros((K, 3)); A = np.zeros((K, NX, NX)); B = np.zeros((K, NX, 3)); minlam = np.inf
    for k, i in enumerate(idx):
        q, qd, qdd = R["q"][i], R["qd"][i], R["qdd"][i]
        U[k] = F(q, qd, qdd, m)
        Fq  = jac_fd(lambda v: F(v, qd, qdd, m), q, 1e-6)
        Fqd = jac_fd(lambda v: F(q, v, qdd, m), qd, 1e-6)
        Fa  = jac_fd(lambda v: F(q, qd, v, m), qdd, 1e-4)
        minlam = min(minlam, np.linalg.eigvals(0.5*(Fa + Fa.T)).real.min())
        Mi = np.linalg.inv(Fa)
        Ac = np.zeros((6, 6)); Bc = np.zeros((6, 3))
        Ac[0:3, 3:6] = np.eye(3); Ac[3:6, 0:3] = -Mi @ Fq; Ac[3:6, 3:6] = -Mi @ Fqd; Bc[3:6] = Mi
        Md = expm(np.block([[Ac, Bc], [np.zeros((3, 9))]]) * T_C)
        A[k, :6, :6], B[k, :6] = Md[:6, :6], Md[:6, 6:]
        A[k, 6:9, 0:3] = T_C*np.eye(3); A[k, 6:9, 6:9] = np.eye(3)
    return dict(idx=idx, U=U, A=A, B=B, minlam=minlam)

def mpc_gains(L):
    K = len(L["idx"]); qd = [Q_POS]*3 + [Q_VEL]*3 + [Q_INT]*3
    P = solve_discrete_are(L["A"][-1], L["B"][-1], np.diag(qd), R_TAU*np.eye(3))
    Wx = np.diag(np.tile(np.sqrt(qd), N_H)); Wx[-NX:, -NX:] = np.linalg.cholesky(P + 1e-12*np.eye(NX)).T
    W2 = Wx.T @ Wx; Kg = np.zeros((K, 3, NX))
    for k in range(K):
        ks = np.minimum(np.arange(k, k + N_H), K - 1); A, B = L["A"][ks], L["B"][ks]
        Phi = np.zeros((N_H*NX, NX)); G = np.zeros((N_H*NX, 3*N_H)); Ak = np.eye(NX)
        for j in range(N_H):
            Ak = A[j] @ Ak; Phi[j*NX:(j+1)*NX] = Ak; G[j*NX:(j+1)*NX, 3*j:3*j+3] = B[j]
            if j > 0: G[j*NX:(j+1)*NX, :3*j] = A[j] @ G[(j-1)*NX:j*NX, :3*j]
        GtW = G.T @ W2
        Kg[k] = -cho_solve(cho_factor(GtW @ G + R_TAU*np.eye(3*N_H)), GtW @ Phi)[:3]
    return Kg

FEATURES = ["1", "|qdd_r|", "|qd_r|^2", "qdd_r jump in period", "payload x |qdd_r|"]
def ref_features(R, idx, m):
    offs = np.arange(STEP + 1)
    Q = R["qdd"][np.minimum(idx[:, None] + offs, R["N"] - 1)]
    jump = np.linalg.norm(Q - Q[:, :1], axis=2).max(axis=1)
    a = np.linalg.norm(R["qdd"][idx], axis=1); v2 = np.sum(R["qd"][idx]**2, axis=1)
    return np.column_stack([np.ones(len(idx)), a, v2, jump, (m / M_BASE)*a])

def valid_steps(R, idx):
    return int(np.sum(idx + STEP <= R["N"] - 1))

def sigma_track(spec, Feat):
    if spec["kind"] == "const": return np.tile(spec["sigma"], (len(Feat), 1))
    return np.maximum(Feat @ spec["beta"], spec["floor"])

TOP_STEPS = 64                     # largest one-step scores kept per episode (for the union-bound baseline)

def sigma_for(spec, P_, sc, m):
    """Normalizers along a task: `const`/`ref` as in NB02; `torque`: sigma = |B_k| tau_sigma_k for a torque spread tau_sigma (K x 3)."""
    if spec["kind"] in ("const", "ref"): return sigma_track(spec, P_["Feat"])
    if spec["kind"] == "state": return np.maximum(spec["ssig"](REFS[sc], P_["L"]["idx"], m), 1e-12)
    ts = spec["tsig"](REFS[sc], P_["L"]["idx"], m)
    return np.maximum(np.einsum("kij,kj->ki", np.abs(P_["L"]["B"][:, :6, :]), ts), 1e-12)

EQ = np.hstack([np.eye(3), np.zeros((3, 6))])
J_ALL = N_H + N_TAIL

def unit_tightening(Phi, Kg, S9):
    """T[k, j, r] = h_{R_{k,j}}(a_r) for q = 1 with time-varying W_k = {|w_l| <= S9[k, l]}; rows 0..2 input, 3..5 joint error."""
    K = len(Phi); T = np.zeros((K, J_ALL, 6), np.float32); kk = np.arange(K)
    for j in range(1, J_ALL):
        a = np.concatenate([Kg[np.minimum(kk + j, K - 1)], np.broadcast_to(EQ, (K, 3, NX))], axis=1)
        v = a.copy(); acc = np.zeros((K, 6))
        for i in range(j - 1, -1, -1):           # v = a Phi(k+j, k+i+1); add h of W_{k+i}
            ii = np.minimum(kk + i, K - 1)
            acc += np.einsum("kri,ki->kr", np.abs(v), S9[ii])
            v = v @ Phi[ii]
        T[:, j] = acc
    return T

# %% [markdown]
# ## Tube MPC with restricted constraints, with the cause of every fallback
#
# As in NB01. When `c = 0` violates a tightened constraint, the controller first checks the tightened sets: `x_empty` if the tube on the joint error is wider than `ε_q` at some prediction step (no `c` can help), `u_empty` if the torque tube is wider than `τ_max`. Otherwise the QP is solved; `qp_infeasible` and `solver` count the remaining failures. In every case the fallback is `c = 0`.

# %%
class TubeMPC:
    def __init__(self, P_, q):
        self.P, self.q = P_, q
        L = P_["L"]; self.K = len(L["idx"]); self.U = L["U"]; self.B = L["B"]; self.Phi = P_["Phi"]; self.Kg = P_["Kg"]
        self.n_qp = 0; self.fb = dict(x_empty=0, u_empty=0, qp_infeasible=0, solver=0)
    @property
    def n_fallback(self): return sum(self.fb.values())
    def bounds(self, k):
        ks = np.minimum(np.arange(k, k + J_ALL), self.K - 1)
        tight = np.zeros((J_ALL, 6)) if self.q == 0 else self.q * self.P["T"][k].astype(float)
        ur = self.U[ks]
        lo_u = -TAU_MAX - ur + tight[:, :3]; hi_u = TAU_MAX - ur - tight[:, :3]
        hi_x = EPS_Q - tight[:, 3:]
        valid = np.arange(J_ALL) + k < self.K
        return ks, lo_u, hi_u, hi_x, valid
    def control(self, k, x):
        ks, lo_u, hi_u, hi_x, valid = self.bounds(k)
        z = x.copy(); ok = True
        for j in range(J_ALL):
            if not valid[j]: break
            u = self.Kg[ks[j]] @ z; dq = z[:3]
            if np.any(u < lo_u[j]) or np.any(u > hi_u[j]) or np.any(np.abs(dq) > hi_x[j]): ok = False; break
            z = self.Phi[ks[j]] @ z
        if ok: return self.Kg[k] @ x, 0
        if np.any(hi_x[valid] < 0): self.fb["x_empty"] += 1; return self.Kg[k] @ x, 2
        if np.any(lo_u[valid] > hi_u[valid]): self.fb["u_empty"] += 1; return self.Kg[k] @ x, 2
        return self._qp(k, x, ks, lo_u, hi_u, hi_x, valid)
    def _qp(self, k, x, ks, lo_u, hi_u, hi_x, valid):
        self.n_qp += 1; nc = 3*N_H
        z0 = x.copy(); G = np.zeros((NX, nc)); rows = []; lo = []; hi = []
        for j in range(J_ALL):
            if not valid[j]: break
            Kj = self.Kg[ks[j]]
            Au = Kj @ G
            if j < N_H: Au[:, 3*j:3*j+3] += np.eye(3)
            bu = Kj @ z0
            rows.append(Au); lo.append(lo_u[j] - bu); hi.append(hi_u[j] - bu)
            rows.append(EQ @ G); bx = EQ @ z0; lo.append(-hi_x[j] - bx); hi.append(hi_x[j] - bx)
            z0 = self.Phi[ks[j]] @ z0; G = self.Phi[ks[j]] @ G
            if j < N_H: G[:, 3*j:3*j+3] += self.B[ks[j]]
        Acon = sparse.csc_matrix(np.vstack(rows)); lo = np.concatenate(lo); hi = np.concatenate(hi)
        prob = osqp.OSQP(); prob.setup(sparse.eye(nc, format="csc")*2.0, np.zeros(nc), Acon, lo, hi,
                                       verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=4000, polish=True)
        res = prob.solve(); st = str(res.info.status)
        if st not in ("solved", "solved inaccurate"):
            self.fb["qp_infeasible" if "infeasible" in st else "solver"] += 1; return self.Kg[k] @ x, 2
        return self.Kg[k] @ x + res.x[:3], 1

# %% [markdown]
# ## Episodes and the task worker
#
# An episode is drawn as in NB02 (same seed, ids and admissible task list of the hybrid model), so every method runs the same tasks, initial errors and noise. `task_worker` runs a list of episodes of one task in a worker process; it keeps the off-line data of the task between calls and recomputes the tightening only when the normalizers change. It also checks the task off line at each tube size it is given: `x_empty_task` (the joint-error tube exceeds `ε_q` somewhere, so the tightened problem is empty at those steps), `uref_out_task` (the reference torque plus the torque tube exceeds `τ_max` somewhere) and the torque budget lost to tightening.

# %%
def all_tasks():
    return [(sc, m) for sc in SCALES for m in PAYLOADS]

def draw_episode(eid, tasks):
    rng = np.random.default_rng([SEED, eid])
    sc, m = tasks[rng.integers(len(tasks))]
    x0 = rng.uniform(-X0_BOX, X0_BOX, 3)
    K = len(REFS[sc]["q"][::STEP]); noise = rng.normal(0.0, SIGMA_TAU, (K, 3))
    return dict(eid=eid, sc=sc, m=m, dq0=x0, noise=noise)

def run_episode(P_, ep, q_tube, keep_trace=False):
    sc, m = ep["sc"], ep["m"]; R = REFS[sc]; L = P_["L"]
    ctrl = TubeMPC(P_, q_tube); K = len(L["idx"]); Kv = P_["Kv"]
    q = R["q"][0] + ep["dq0"]; qd = R["qd"][0].copy(); zeta = np.zeros(3)
    X = np.zeros((K + 1, NX)); Uapp = np.zeros((K, 3)); modes = np.zeros(K, int); viol_tau = 0
    Pe = np.zeros((R["N"], 3))
    for k in range(K):
        i = L["idx"][k]
        x = np.concatenate([q - R["q"][i], qd - R["qd"][i], zeta]); X[k] = x
        du, modes[k] = ctrl.control(k, x)
        tau_cmd = L["U"][k] + du
        if np.any(np.abs(tau_cmd) > TAU_MAX + 1e-9): viol_tau += 1
        tau = np.clip(tau_cmd, -TAU_MAX, TAU_MAX); Uapp[k] = tau - L["U"][k]
        tau_pl = tau + ep["noise"][k]
        zeta = zeta + T_C*x[:3]
        for s_ in range(STEP):
            ii = i + s_
            if ii >= R["N"]: break
            p = fk(q); Pe[ii] = p
            qd = qd + plant_qdd(q, qd, tau_pl, p, m, SEVERITY)*dt; q = q + qd*dt
        if not np.all(np.isfinite(q)) or np.abs(q).max() > 3:
            return dict(eid=ep["eid"], sc=sc, m=m, q_tube=q_tube, unstable=True), None
    iN = min(L["idx"][-1] + STEP, R["N"] - 1)
    X[K] = np.concatenate([q - R["q"][iN], qd - R["qd"][iN], zeta])
    W = (X[1:] - np.einsum("kij,kj->ki", L["A"], X[:-1]) - np.einsum("kij,kj->ki", L["B"], Uapp))[:Kv]
    e_cart = 1000*np.linalg.norm(R["P"][:len(Pe)] - Pe, axis=1)
    res = dict(eid=ep["eid"], sc=sc, m=m, q_tube=q_tube, unstable=False, max_abs_w_zeta=np.abs(W[:, 6:]).max(),
               max_dq=np.abs(X[:-1, :3]).max(), viol_x=int((np.abs(X[:-1, :3]) > EPS_Q).any(axis=1).sum()),
               viol_tau=viol_tau, n_qp=ctrl.n_qp, n_fallback=ctrl.n_fallback, K=K,
               **{f"fb_{c}": v for c, v in ctrl.fb.items()},
               rmse_um=1000*np.sqrt(np.mean(e_cart**2)), max_err_um=1000*e_cart.max())
    extra = dict(W=W[:, :6])
    if keep_trace: extra.update(X=X, modes=modes)
    return res, extra

def _worker_cache():
    mod = sys.modules.get("_cmpc_nb04_cache")
    if mod is None:
        mod = types.ModuleType("_cmpc_nb04_cache"); mod.C = {}; sys.modules["_cmpc_nb04_cache"] = mod
    return mod.C

def prepare_task(label, sc, m):
    t0 = time.time(); R = REFS[sc]; L = ltv_along_reference(FMODELS[label], R, m)
    Kg = mpc_gains(L); Phi = L["A"] + np.einsum("kij,kjl->kil", L["B"], Kg)
    return dict(L=L, Kg=Kg, Phi=Phi, Feat=ref_features(R, L["idx"], m), Kv=valid_steps(R, L["idx"]),
                t_prep=time.time() - t0, T=None, S=None, sig_id=None)

def task_check(P_, q):
    T = P_["T"]; U = P_["L"]["U"]; K = len(U)
    kj = np.arange(K)[:, None] + np.arange(J_ALL)[None, :]; valid = kj < K; ks = np.minimum(kj, K - 1)
    tu = q*T[:, :, :3].astype(float); tx = q*T[:, :, 3:].astype(float)
    x_bad = ((EPS_Q - tx).min(axis=2) < 0) & valid
    u_bad = ((TAU_MAX - np.abs(U[ks]) - tu).min(axis=2) < 0) & valid
    return dict(x_empty_task=bool(x_bad.any()), x_empty_frac_k=float(x_bad.any(axis=1).mean()),
                uref_out_task=bool(u_bad.any()), uref_out_frac_k=float(u_bad.any(axis=1).mean()),
                budget_lost_max_Nm=float(tu.max()), budget_lost_med_Nm=float(np.median(tu.max(axis=(1, 2)))),
                tube_pos_j1_med_urad=float(1e6*np.median(tx[:, 1].max(axis=1))),
                tube_pos_j1_max_urad=float(1e6*tx[:, 1].max()), tube_pos_jmax_mrad=float(1e3*tx.max()))

def task_worker(label, sc, m, jobs, spec, tasks, trace_ids=()):
    """Run the episodes `jobs` = [(eid, q), ...] of task (sc, m) with normalizers `spec` (None: q = 0, no score)."""
    C = _worker_cache(); key = (label, sc, m)
    if key not in C: C[key] = prepare_task(label, sc, m)
    P_ = C[key]
    if spec is not None and P_["sig_id"] != spec["id"]:
        S = sigma_for(spec, P_, sc, m)
        P_["T"] = unit_tightening(P_["Phi"], P_["Kg"], np.hstack([S, np.zeros((len(S), 3))])); P_["S"] = S; P_["sig_id"] = spec["id"]
    info = dict(model=label, sc=sc, m=m, t_prep_s=P_["t_prep"], min_eig_Fqdd=P_["L"]["minlam"], max_abs_uref=float(np.abs(P_["L"]["U"]).max()))
    checks = {q: task_check(P_, q) for q in sorted({q for _, q in jobs})} if spec is not None else {}
    out_ = []
    for eid, q in jobs:
        res, extra = run_episode(P_, draw_episode(eid, tasks), q, keep_trace=eid in trace_ids)
        if extra is not None:
            if spec is not None:
                st_ = (np.abs(extra["W"]) / P_["S"][:len(extra["W"])]).max(axis=1)
                res["score"] = float(st_.max()); res["_top"] = np.sort(st_)[-TOP_STEPS:]
                if q > 0: res["step_coverage"] = float(np.mean(st_ <= q))     # fraction of steps inside the tube
                if eid in trace_ids: extra["tube1"] = q*P_["T"][:, 1, 3:].max(axis=1).astype(float); extra["S"] = P_["S"]
            else:
                extra["Feat"] = P_["Feat"][:len(extra["W"])]
            if spec is not None and eid not in trace_ids: extra = None
        out_.append((res, extra))
    return info, checks, out_

try:
    from joblib import Parallel, delayed
except ImportError:
    if N_JOBS > 1: print("joblib not found: running serially"); N_JOBS = 1

def run_jobs(label, eids, qfun, spec, tasks, trace_ids=()):
    """Group episodes by task, run them (in parallel if N_JOBS > 1) and return results in the order of `eids`."""
    task_of = {e: draw_episode(e, tasks) for e in eids}
    groups = {}
    for e in eids: groups.setdefault((task_of[e]["sc"], task_of[e]["m"]), []).append((e, qfun(task_of[e])))
    calls = sorted(groups.items(), key=lambda kv: -len(kv[1])*len(REFS[kv[0][0]]["q"]))
    args = [(label, sc, m, jobs, spec, tasks, tuple(trace_ids)) for (sc, m), jobs in calls]
    if N_JOBS > 1:
        outs = Parallel(n_jobs=N_JOBS, backend="loky")(delayed(task_worker)(*a) for a in args)
    else:
        outs = [task_worker(*a) for a in args]
    by_eid, infos, checks = {}, [], []
    for (info, chk, rs), a in zip(outs, args):
        infos.append(info)
        for q, c in chk.items(): checks.append(dict(model=label, sc=a[1], m=a[2], q=q, **c))
        for res, extra in rs: by_eid[res["eid"]] = (res, extra)
    return [by_eid[e] for e in eids], infos, checks

# %% [markdown]
# ## Normalizers from the pilot episodes (`q = 0`)
#
# `const`: per-component standard deviation of `w` over all valid pilot steps (NB01). `ref`: for each component `i`, `β_i ≥ 0` by non-negative least squares of the envelope of `|w_{k,i}|` (running maximum over `ENV_WIN` steps) on `φ_k` (columns scaled for conditioning), with a floor of 20% of the mean envelope. Fitting the envelope instead of `|w|` makes `σ_k` follow the size of the peaks, which is what sets the score.

# %%
ENV_WIN = N_H                      # window (control steps) of the |w| envelope fitted by the `ref` normalizers

def fit_normalizers(pilot):
    Wall = np.vstack([ex["W"] for r, ex in pilot if ex is not None]); Fall = np.vstack([ex["Feat"] for r, ex in pilot if ex is not None])
    Eall = np.vstack([maximum_filter1d(np.abs(ex["W"]), size=ENV_WIN, axis=0, mode="nearest") for r, ex in pilot if ex is not None])
    sig_c = Wall.std(axis=0); sig_c[sig_c < 1e-12] = 1e-12
    cs = np.abs(Fall).max(axis=0); cs[cs == 0] = 1.0; Fs = Fall / cs
    beta = np.zeros((Fall.shape[1], 6)); floor = np.zeros(6); r2 = np.zeros(6)
    for i in range(6):
        y = Eall[:, i]; b, _ = nnls(Fs, y); beta[:, i] = b / cs
        floor[i] = 0.2*y.mean(); yh = np.maximum(Fall @ beta[:, i], floor[i])
        r2[i] = 1 - np.sum((y - yh)**2) / np.sum((y - y.mean())**2)
    return sig_c, beta, floor, r2

# %% [markdown]
# ## Prediction models and uncertainty models of the baselines
#
# * `HYBRID-35`: the hybrid surrogate of notebook 18 (seed 0), as in NB02.
# * `Oracle`: the exact inverse dynamics of the plant, `F(q, q̇, q̈, m) = γ(m)·Y(q, q̈)θ0 + τ_u(q, q̇)` (lower bound).
# * `ENS`: mean of the hybrid surrogates of seeds `ENS_SEEDS`; the spread of their torques along the reference, mapped through `|B_k|`, sizes its tube.
# * GP (built after the pilot, below): one Gaussian process per component of the one-step mismatch `w`, regressed on the reference state `(q_r, q̇_r, q̈_r, m)` with the pilot data of the nominal MPC, in the spirit of cautious MPC; the tube is `|μ| + z·s` with the posterior mean `μ` and standard deviation `s`.

# %%
def F_true(q, qd, qdd, m):
    return gamma_payload(m)*(regressor(q, qdd, fk(q)) @ THETA0) + tau_unmodeled(q, qd, SEVERITY)

def hybrid_path(seed):
    return os.path.join(WEIGHTS_DIR, f"ff18_{BASE[1]}_{BASE[2]}traj_seed{seed}_e150.npz")

F_BASE = load_model(BASE[1], BASE[2], 0)
ENS_F = [load_model(BASE[1], BASE[2], s) for s in ENS_SEEDS if os.path.exists(hybrid_path(s))]
if len(ENS_F) < len(ENS_SEEDS): print(f"WARNING: only {len(ENS_F)} ensemble members found in {WEIGHTS_DIR}")
def F_ens(q, qd, qdd, m): return np.mean([f(q, qd, qdd, m) for f in ENS_F], axis=0)
FMODELS.update({BASE[0]: F_BASE, "Oracle": F_true, "ENS": F_ens})

def tsig_ens(R, idx, m):
    T = np.array([[f(R["q"][i], R["qd"][i], R["qdd"][i], m) for i in idx] for f in ENS_F])
    return np.sqrt(T.std(axis=0)**2 + SIGMA_TAU**2)

class GPJoint:
    """Scalar GP regression, ARD squared-exponential kernel, ML-II hyperparameters (inputs and targets standardized)."""
    def __init__(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12; self.X = (X - self.mu)/self.sd
        self.ym, self.ys = y.mean(), y.std() + 1e-300; self.y = (y - self.ym)/self.ys
        th0 = np.concatenate([np.zeros(X.shape[1]), [0.0], [np.log(0.1)]])
        r_ = minimize(self.nlml, th0, method="L-BFGS-B", bounds=[(-3, 4)]*X.shape[1] + [(-5, 3), (-8, 1)])
        self.set(r_.x)
    def K(self, A, B, ell, sf):
        d = (A[:, None, :] - B[None, :, :])/ell; return sf**2*np.exp(-0.5*np.sum(d**2, axis=2))
    def nlml(self, th):
        ell, sf, sn = np.exp(th[:-2]), np.exp(th[-2]), np.exp(th[-1])
        Kx = self.K(self.X, self.X, ell, sf) + (sn**2 + 1e-8)*np.eye(len(self.X))
        try: L = np.linalg.cholesky(Kx)
        except np.linalg.LinAlgError: return 1e10
        a = cho_solve((L, True), self.y)
        return 0.5*self.y @ a + np.log(np.diag(L)).sum()
    def set(self, th):
        self.th = th; self.ell, self.sf, self.sn = np.exp(th[:-2]), np.exp(th[-2]), np.exp(th[-1])
        Kx = self.K(self.X, self.X, self.ell, self.sf) + (self.sn**2 + 1e-8)*np.eye(len(self.X))
        self.L = np.linalg.cholesky(Kx); self.a = cho_solve((self.L, True), self.y)
    def predict(self, Xs):
        """Posterior mean and standard deviation of a new observation (latent variance plus noise)."""
        ks = self.K((np.atleast_2d(Xs) - self.mu)/self.sd, self.X, self.ell, self.sf)
        v = np.linalg.solve(self.L, ks.T); var = np.maximum(self.sf**2 - np.sum(v**2, axis=0), 0.0) + self.sn**2
        return self.ym + self.ys*(ks @ self.a), self.ys*np.sqrt(var)

def ref_state(R, idx, m):
    return np.column_stack([R["q"][idx], R["qd"][idx], R["qdd"][idx], np.full(len(idx), float(m))])

GPS = []                           # filled after the pilot
def ssig_gp(R, idx, m):
    X = ref_state(R, idx, m); S = np.zeros((len(idx), 6))
    for i, g in enumerate(GPS):
        mu, s = g.predict(X); S[:, i] = s + np.abs(mu)/Z_GAUSS
    return S

Z_GAUSS = float(norm.ppf(1 - DELTA/2))
print(f"Gaussian quantile z_(1-delta/2) = {Z_GAUSS:.3f} | ensemble members: {len(ENS_F)}")

# %% [markdown]
# ## Run all methods on the same episodes
#
# Pilot (`q = 0`, hybrid model) → reference-dependent normalizers, as in NB02. Then each method in turn; all of them are tested on the same `N_TEST` test episodes (ids of NB02). A checkpoint is written after every method.

# %%
def conformal_quantile(s, delta):
    n = len(s); r = math.ceil((n + 1)*(1 - delta))
    return np.inf if r > n else float(np.sort(s)[r - 1])

def pilot_quantile(s, delta):
    return float(np.quantile(s, 1 - delta, method="higher"))

LAB = BASE[0]
STATE = dict(DONE=set(), SUMMARY=[], EPISODES=[], CHECKS=[], ROUNDS=[], COVD=[], INFO={})
CKPT = out("checkpoint.pkl")
if RESUME_DIR and os.path.exists(CKPT):
    STATE = pickle.load(open(CKPT, "rb")); print("resuming", RESUME_DIR, "| done:", sorted(STATE["DONE"]))
def save_checkpoint():
    tmp = CKPT + ".tmp"
    with open(tmp, "wb") as f_: pickle.dump(STATE, f_)
    os.replace(tmp, CKPT)

t_all = time.time()
grid = [(sc, m) for sc in SCALES for m in PAYLOADS]
infos = (Parallel(n_jobs=N_JOBS, backend="loky")(delayed(task_worker)(LAB, sc, m, [], None, grid) for sc, m in grid) if N_JOBS > 1
         else [task_worker(LAB, sc, m, [], None, grid) for sc, m in grid])
umax = {(i["sc"], i["m"]): i["max_abs_uref"] for i, _, _ in infos}
TASKS = [tk for tk in grid if umax[tk] <= TAU_MAX - TORQUE_MARGIN]
print(f"{LAB}: {len(TASKS)} admissible tasks of {len(grid)} | prep {(time.time()-t_all)/60:.1f} min")

# pilot at q = 0 and reference-dependent normalizers (as NB02)
pilot_ids = list(range(N_PILOT))
pilot, _, _ = run_jobs(LAB, pilot_ids, lambda ep: 0.0, None, TASKS)
sig_c, beta, floor, r2 = fit_normalizers(pilot)
SPEC_REF = dict(id=f"{LAB}|ref", kind="ref", beta=beta, floor=floor)
print(f"pilot {len(pilot)} episodes | R^2 of the envelope fit {np.round(r2, 2)}")
# worst case: the largest |w_i| over all valid steps of the pilot episodes, as a constant box, with q = 1
Wp = [ex["W"] for r, ex in pilot if ex is not None]; W_MAX = np.abs(np.vstack(Wp)).max(axis=0)
# GP of each component of w on the reference state, trained on N_GP random pilot steps
pilot_eps = {e: draw_episode(e, TASKS) for e in pilot_ids}
XP = np.vstack([ref_state(REFS[pilot_eps[r["eid"]]["sc"]], np.arange(0, REFS[pilot_eps[r["eid"]]["sc"]]["N"], STEP)[:len(ex["W"])],
                          pilot_eps[r["eid"]]["m"]) for r, ex in pilot if ex is not None])
WP = np.vstack(Wp); rng_gp = np.random.default_rng(SEED + 202); sel = rng_gp.choice(len(XP), min(N_GP, len(XP)), replace=False)
t0 = time.time(); GPS[:] = [GPJoint(XP[sel], WP[sel, i]) for i in range(6)]
hold = rng_gp.choice(np.setdiff1d(np.arange(len(XP)), sel), min(5000, len(XP) - len(sel)), replace=False)
cov_hold = [np.mean(np.abs(WP[hold, i] - g.predict(XP[hold])[0]) <= Z_GAUSS*g.predict(XP[hold])[1]) for i, g in enumerate(GPS)]
print(f"GP of w fitted in {time.time()-t0:.1f} s | per-step coverage of mu +- z s on held-out pilot steps {np.round(cov_hold, 3)}")
SPECS = {"worst_case": dict(id="wc", kind="const", sigma=W_MAX), "gp": dict(id="gp", kind="state", ssig=ssig_gp),
         "ensemble": dict(id="ens", kind="torque", tsig=tsig_ens)}

cal_ids0 = [10**6 + e for e in range(N_CAL)]
test_ids = [2*10**6 + e for e in range(N_TEST)]

def summarize(method, label, q_desc, test, chk, **kw):
    te = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r, _ in test]); ok = te[~te.unstable].copy()
    row = dict(method=method, model=label, q=q_desc, n_test=len(te), unstable=int(te.unstable.sum()),
               episodes_with_x_violation=float((ok.viol_x > 0).mean()), episodes_with_tau_violation=float((ok.viol_tau > 0).mean()),
               episodes_with_fallback=float((ok.n_fallback > 0).mean()),
               fallback_step_fraction=float(ok.n_fallback.sum()/ok.K.sum()), qp_fraction=float(ok.n_qp.sum()/ok.K.sum()),
               fb_x_empty=float((ok.fb_x_empty > 0).mean()), fb_u_empty=float((ok.fb_u_empty > 0).mean()),
               fb_qp_infeasible=float((ok.fb_qp_infeasible > 0).mean()), fb_solver=float((ok.fb_solver > 0).mean()),
               rmse_um_mean=float(ok.rmse_um.mean()), max_err_um_mean=float(ok.max_err_um.mean()), max_dq_mrad=float(1e3*ok.max_dq.max()))
    if "score" in ok and isinstance(q_desc, float) and q_desc > 0:
        cov = ok.score <= q_desc; row.update(coverage_test=float(cov.mean()), coverage_min_speed=float(cov.groupby(ok.sc).mean().min()),
                                             step_coverage=float(ok.step_coverage.mean()))
        ok["covered"] = cov
    if chk:
        c = pd.DataFrame(chk).drop_duplicates(subset=["sc", "m", "q"]); cm = {(r.sc, r.m): r for r in c.itertuples()}
        tc = pd.DataFrame([cm[(r.sc, r.m)]._asdict() for r in ok.itertuples()])
        row.update(tube_pos_j1_med_urad=float(tc.tube_pos_j1_med_urad.median()), tube_pos_jmax_mrad=float(tc.tube_pos_jmax_mrad.median()),
                   budget_lost_med_Nm=float(tc.budget_lost_med_Nm.median()), test_tasks_x_empty=float(tc.x_empty_task.mean()),
                   test_tasks_uref_out=float(tc.uref_out_task.mean()))
        for r in c.to_dict("records"): STATE["CHECKS"].append(dict(r, method=method))
    row.update(kw); row["minutes"] = (time.time() - t_m)/60
    for r in ok.to_dict("records"): STATE["EPISODES"].append(dict(r, method=method, set="test"))
    STATE["SUMMARY"].append(row)
    print(f"[{method}] q {q_desc if isinstance(q_desc, str) else round(q_desc, 4)} | coverage {row.get('coverage_test', np.nan):.3f} (steps {row.get('step_coverage', np.nan):.4f}) "
          f"| tube j=1 {row.get('tube_pos_j1_med_urad', np.nan):.1f} urad, tail {row.get('tube_pos_jmax_mrad', np.nan):.3f} mrad "
          f"| fallback {row['episodes_with_fallback']:.2f} of episodes, {row['fallback_step_fraction']:.3f} of steps "
          f"| viol_x {row['episodes_with_x_violation']:.3f} | RMSE {row['rmse_um_mean']:.1f} um | {row['minutes']:.1f} min")
    return row

def cov_by_delta(method, rule):
    """rule(delta) -> q at that level; coverage of the test scores (episodes run with the delta = DELTA tube)."""
    sc_ = np.array([r["score"] for r in STATE["EPISODES"] if r["method"] == method and "score" in r])
    for dl in DELTAS_CHECK:
        STATE["COVD"].append(dict(method=method, delta=dl, q=rule(dl), coverage=float(np.mean(sc_ <= rule(dl)))))

for method in METHODS:
    if method in STATE["DONE"]: print(f"[{method}] done (checkpoint)"); continue
    t_m = time.time()
    if method == "nominal":
        test, _, _ = run_jobs(LAB, test_ids, lambda ep: 0.0, SPEC_REF, TASKS); summarize(method, LAB, "0", test, [])
    elif method == "oracle":
        test, _, _ = run_jobs("Oracle", test_ids, lambda ep: 0.0, None, TASKS); summarize(method, "Oracle", "0", test, [])
    elif method == "proposed":
        s0 = []
        for r, ex in pilot:
            if ex is None: continue
            ep = pilot_eps[r["eid"]]; R = REFS[ep["sc"]]; idx = np.arange(0, R["N"], STEP)
            s0.append(float(np.max(np.abs(ex["W"]) / sigma_track(SPEC_REF, ref_features(R, idx, ep["m"]))[:len(ex["W"])])))
        q_p = pilot_quantile(np.array(s0), DELTA)
        for it in range(PILOT_ITERS):
            rs, _, _ = run_jobs(LAB, pilot_ids, lambda ep: q_p, SPEC_REF, TASKS)
            q_p = pilot_quantile(np.array([r["score"] for r, _ in rs if not r["unstable"]]), DELTA)
        q_t = INFLATE*q_p; cert = False
        for rnd in range(MAX_ROUNDS):
            ids = [10**6 + 10**5*rnd + e for e in range(N_CAL)]
            cal, _, _ = run_jobs(LAB, ids, lambda ep: q_t, SPEC_REF, TASKS)
            sc_cal = np.array([r["score"] if not r["unstable"] else np.inf for r, _ in cal]); q_h = conformal_quantile(sc_cal, DELTA)
            cert = q_h <= q_t
            STATE["ROUNDS"].append(dict(round=rnd + 1, q_tilde=q_t, q_hat=q_h, certified=cert))
            print(f"  round {rnd + 1}: q_tilde {q_t:.3f}, q_hat {q_h:.3f}, certified {cert}")
            if cert or not np.isfinite(q_h): break
            q_t = INFLATE*q_h
        STATE["INFO"]["proposed"] = dict(q_tilde=q_t, cal_scores=sc_cal, rounds=rnd + 1, certified=cert)
        test, _, chk = run_jobs(LAB, test_ids, lambda ep: q_t, SPEC_REF, TASKS)
        summarize(method, LAB, q_t, test, chk, certified=cert, rounds=rnd + 1, q_hat=q_h)
        pool = np.concatenate([sc_cal, [r["score"] for r, _ in test if not r["unstable"]]]); rng_s = np.random.default_rng(SEED + 7)
        for dl in DELTAS_CHECK:
            cv = []
            for _ in range(N_SPLITS):
                p_ = rng_s.permutation(len(pool)); cv.append(np.mean(pool[p_[N_CAL:]] <= conformal_quantile(pool[p_[:N_CAL]], dl)))
            STATE["COVD"].append(dict(method=method, delta=dl, q=np.nan, coverage=float(np.mean(cv)),
                                      p05=float(np.percentile(cv, 5)), p95=float(np.percentile(cv, 95))))
    elif method == "sample_max":
        q_s = float(np.max(STATE["INFO"]["proposed"]["cal_scores"]))
        test, _, chk = run_jobs(LAB, test_ids, lambda ep: q_s, SPEC_REF, TASKS); summarize(method, LAB, q_s, test, chk)
        cov_by_delta(method, lambda dl: q_s)
    elif method in ("ol_traj", "ol_union"):
        if "ol" not in STATE["INFO"]:
            ol, _, _ = run_jobs(LAB, cal_ids0, lambda ep: 0.0, SPEC_REF, TASKS)
            s_tr = np.array([r["score"] for r, _ in ol if not r["unstable"]])
            tops = np.sort(np.concatenate([r["_top"] for r, _ in ol if not r["unstable"]]))[::-1]
            Kv = np.array([r["K"] for r, _ in ol if not r["unstable"]]); STATE["INFO"]["ol"] = dict(s_tr=s_tr, tops=tops, N=int(Kv.sum()), Kbar=float(Kv.mean()))
        I = STATE["INFO"]["ol"]
        def q_union(dl):
            r_ = math.ceil((I["N"] + 1)*(1 - dl/I["Kbar"])); j = I["N"] - r_ + 1
            return float(I["tops"][j - 1]) if 1 <= j <= len(I["tops"]) else np.inf
        rule = (lambda dl: conformal_quantile(I["s_tr"], dl)) if method == "ol_traj" else q_union
        q_o = rule(DELTA)
        test, _, chk = run_jobs(LAB, test_ids, lambda ep: q_o, SPEC_REF, TASKS); summarize(method, LAB, q_o, test, chk)
        cov_by_delta(method, rule)
    else:
        label = {"worst_case": LAB, "gp": LAB, "ensemble": "ENS"}[method]; q_m = 1.0 if method == "worst_case" else Z_GAUSS
        test, _, chk = run_jobs(label, test_ids, lambda ep: q_m, SPECS[method], TASKS); summarize(method, label, q_m, test, chk)
        cov_by_delta(method, (lambda dl: 1.0) if method == "worst_case" else (lambda dl: float(norm.ppf(1 - dl/2))))
    STATE["DONE"].add(method); save_checkpoint()
print(f"\ntotal {(time.time()-t_all)/60:.1f} min")

sum_df = pd.DataFrame(STATE["SUMMARY"]); sum_df.to_csv(out("CMPC_NB04_summary.csv"), index=False)
ep_df = pd.DataFrame(STATE["EPISODES"]); ep_df.to_csv(out("CMPC_NB04_episodes.csv"), index=False)
pd.DataFrame(STATE["CHECKS"]).to_csv(out("CMPC_NB04_task_checks.csv"), index=False)
pd.DataFrame(STATE["ROUNDS"]).to_csv(out("CMPC_NB04_rounds.csv"), index=False)
covd = pd.DataFrame(STATE["COVD"]); covd.to_csv(out("CMPC_NB04_coverage_by_delta.csv"), index=False)
pd.DataFrame(dict(component=range(6), w_max=W_MAX, gp_lengthscales=[np.round(g.ell, 3).tolist() for g in GPS],
                  gp_signal=[g.sf for g in GPS], gp_noise=[g.sn for g in GPS], gp_heldout_step_coverage=cov_hold)).to_csv(out("CMPC_NB04_baseline_params.csv"), index=False)
print(sum_df.round(4).set_index("method").T.to_string())
print(covd.pivot_table(index="method", columns="delta", values="coverage").round(3))

# %% [markdown]
# ## Figures

# %%
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False})
tub = sum_df.dropna(subset=["coverage_test"]) if "coverage_test" in sum_df else sum_df.iloc[:0]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
for r in tub.itertuples():
    ax[0].plot(r.tube_pos_j1_med_urad, r.coverage_test, "o", ms=6); ax[0].annotate(r.method, (r.tube_pos_j1_med_urad, r.coverage_test), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax[0].axhline(1 - DELTA, color="k", ls=":", lw=0.9); ax[0].set_xscale("log"); ax[0].set_xlabel("one-step joint-error tube (µrad, median)"); ax[0].set_ylabel("test coverage")
xs = np.arange(len(sum_df)); w = 0.4
ax[1].bar(xs - w/2, sum_df.episodes_with_fallback, w, label="episodes with fallback"); ax[1].bar(xs + w/2, sum_df.episodes_with_x_violation, w, label=r"episodes violating $\varepsilon_q$")
ax[1].set_xticks(xs); ax[1].set_xticklabels(sum_df.method, rotation=45, ha="right", fontsize=7); ax[1].legend(frameon=False)
fig.tight_layout(); fig.savefig(out("fig_nb04_baselines.pdf")); fig.savefig(out("fig_nb04_baselines.png"), dpi=200); plt.show()

# %% [markdown]
# ## Pack the run folder

# %%
zip_path = shutil.make_archive(RUN_DIR, "zip", root_dir=RUN_DIR)
print("results:", zip_path, f"({os.path.getsize(zip_path)/1024:.0f} kB)")
print(sorted(os.listdir(RUN_DIR)))
