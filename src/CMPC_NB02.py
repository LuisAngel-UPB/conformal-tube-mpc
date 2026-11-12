# %% [markdown]
# # CMPC-NB02 — Reference-dependent normalizers and class-conditional calibration of the tube MPC (Delta robot)
#
# **Paper:** *Conformal-calibrated tube MPC with physics-informed learned dynamics for Lagrangian robots* (JAS draft, `paper/main.tex`, Sec. V).
#
# **Why this notebook.** NB01 showed that, with constant normalizers σ and a single tube size q, the certified tube is valid but not useful: the trajectory score is set by a few isolated steps, it grows with the speed of the reference, the learned surrogates do not certify smaller tubes than the identified model, and the tightened problem is often empty. A diagnosis of the NB01 mismatch traces the large scores to two sources that do not depend on the surrogate:
# 1. **Acceleration jumps of the reference.** The vertical dip `H_P·sin²(πτ)` of the Adept path has a non-zero second derivative at the segment ends, and the segments have different durations, so the reference acceleration jumps at `t = 0.25·sc` and `t = 2.25·sc`. Across those control periods the ZOH prediction cannot follow the reference and `w` on the velocity rows is 10–15 standard deviations.
# 2. **The last control period of each episode** is only partly simulated (the reference ends inside it), and NB01 compared it with a full-period prediction. NB02 drops that step from the score.
#
# **What NB02 does (per surrogate), all on the same episodes as NB01:**
# * Variant `const`: Algorithm 1 with constant σ, as in NB01, with the last-step fix. It separates the effect of the fix from that of the new normalizers.
# * Variant `ref`: **reference-dependent normalizers** `σ_{k,i} = β_iᵀ φ_k`, where `φ_k` collects features of the reference over the control period (acceleration, squared speed, acceleration jump inside the period, payload × acceleration). `β_i ≥ 0` is fitted by non-negative least squares on `|w_{k,i}|` of the pilot episodes, so σ is fixed before calibration and exchangeability (Assumption 3) is untouched. The tube tightening uses the time-varying `W_k` in the support-function recursion.
# * Variant `ref+class`: the same normalizers plus **class-conditional (Mondrian) calibration**: one tube size per speed class (time scaling), calibrated on the episodes of that class only. Theorem 1 then holds within each class.
# * For every fallback, the **cause** is recorded: empty tightened torque set, empty tightened error set, or solver failure. Off line, every task is checked for a non-empty tightened set at the deployed tube size (a necessary condition for Assumption 4), and the torque budget lost to tightening is reported.
#
# **Outputs.** Every run writes its CSV files and figures into `OUT_DIR/CMPC_NB02_<tag>/` and packs that folder into a zip.
#
# **Interrupted runs.** After each model and variant the run saves `checkpoint.pkl` in its folder. If the run stops (power cut, crash), set `RESUME_DIR` to that folder (for example `"results/CMPC_NB02_FULL_20261016_190000"`) and run the notebook again from the top: finished models and variants are loaded and skipped; only the one in progress is repeated. Keep the other settings unchanged.
#
# **Modes.** `QUICK = True` is a smoke test (one model, two speeds, two payloads, few episodes). `QUICK = False` is the long run for the paper.
#
# **Parallel run.** Episodes are grouped by task and run in `N_JOBS` worker processes with joblib (`pip install joblib` if missing). Results do not depend on `N_JOBS`; `N_JOBS = 1` runs serially without joblib. Each worker keeps the off-line data of the tasks it has seen, about 0.5–0.7 GB per worker in the full run; lower `N_JOBS` if memory is short.
#
# **Requirements.** numpy, scipy, pandas, matplotlib, osqp, joblib. The surrogate weights `ff18_{KIND}_{SIZE}traj_seed{SEED}_e150.npz` of notebook 18 must be in `WEIGHTS_DIR`.

# %%
import os, sys, types, time, json, shutil, datetime, math
import numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import expm, solve_discrete_are, cho_factor, cho_solve
from scipy.optimize import nnls
from scipy.ndimage import maximum_filter1d
from scipy import sparse
from scipy.stats import beta as beta_dist
import osqp

QUICK = True                       # True: smoke test (minutes); False: full run for the paper (hours)
WEIGHTS_DIR = "."                  # folder with ff18_*_e150.npz (notebook 18)
OUT_DIR = "results"
RESUME_DIR = None                  # to continue an interrupted run: its folder, e.g. "results/CMPC_NB02_FULL_20261016_190000"
N_JOBS = max(1, (os.cpu_count() or 2) - 1)   # worker processes (1 = serial)

# ---- plant / controller (identical to NB01)
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

# ---- task distribution D (identical to NB01)
REF_C2 = True                      # True: vertical dip sin^4 (continuous acceleration); False: the NB01 reference (sin^2, acceleration jumps)
# With the sin^4 dip the peak joint speed at time scaling 2.0 (3.6 rad/s) is that of NB01 at 1.6, and at 1.6 no task is
# admissible, so the fastest class is 2.0.
SCALES = (2.0, 2.4, 3.2, 4.8, 6.4) if REF_C2 else (1.6, 2.4, 3.2, 4.8, 6.4)
PAYLOADS = (0, 1, 2, 3, 4, 5, 6)
X0_BOX = 2.0e-4
SIGMA_TAU = 0.005

# ---- models
MODELS = [("Nominal", "NOMINAL", 0, 0), ("HYBRID-35", "HYBRID", 35, 0), ("PINN-175", "PINN", 175, 0), ("MLP-35", "MLP", 35, 0)]

# ---- Algorithm 1 and variants
VARIANTS = ("const", "ref", "ref+class")
DELTA = 0.10
N_PILOT, N_CAL, N_TEST = 50, 200, 400
PILOT_ITERS, INFLATE = 1, 1.05     # NB01: the pilot fixed point was reached at the first iterate for every model
DELTAS_CHECK = (0.05, 0.10, 0.20)
N_SPLITS = 500
SEED = 2026                        # same seed as NB01: for a given model, episode e has the same task, initial error and noise

if QUICK:
    SCALES, PAYLOADS = (SCALES[0], 3.2), (0, 6)
    MODELS = [("HYBRID-35", "HYBRID", 35, 0)]
    N_PILOT, N_CAL, N_TEST, N_SPLITS = 16, 24, 24, 100

STEP = int(round(T_C / dt))
TAG = ("QUICK_" if QUICK else "FULL_") + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESUME_DIR or os.path.join(OUT_DIR, f"CMPC_NB02_{TAG}"); os.makedirs(RUN_DIR, exist_ok=True)
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

FMODELS = {lab: load_model(kind, size, seed) for lab, kind, size, seed in MODELS}
print("models:", list(FMODELS))

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
# * `ltv_along_reference`, `mpc_gains`: as in NB01.
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
                                       verbose=False, eps_abs=1e-9, eps_rel=1e-9, max_iter=20000, polish=True)
        res = prob.solve(); st = str(res.info.status)
        if st not in ("solved", "solved inaccurate"):
            self.fb["qp_infeasible" if "infeasible" in st else "solver"] += 1; return self.Kg[k] @ x, 2
        return self.Kg[k] @ x + res.x[:3], 1

# %% [markdown]
# ## Episodes and the task worker
#
# An episode is drawn exactly as in NB01 (same seed, same ids and the same admissible task list, hence the same task, initial error and noise as NB01 for the same model). `task_worker` runs a list of episodes of one task in a worker process; it keeps the off-line data of the task between calls and recomputes the tightening only when the normalizers change. It also checks the task off line at each tube size it is given: `x_empty_task` (the joint-error tube exceeds `ε_q` somewhere, so the tightened problem is empty at those steps), `uref_out_task` (the reference torque plus the torque tube exceeds `τ_max` somewhere) and the torque budget lost to tightening.

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
    mod = sys.modules.get("_cmpc_nb02_cache")
    if mod is None:
        mod = types.ModuleType("_cmpc_nb02_cache"); mod.C = {}; sys.modules["_cmpc_nb02_cache"] = mod
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
        S = sigma_track(spec, P_["Feat"])
        P_["T"] = unit_tightening(P_["Phi"], P_["Kg"], np.hstack([S, np.zeros((len(S), 3))])); P_["S"] = S; P_["sig_id"] = spec["id"]
    info = dict(model=label, sc=sc, m=m, t_prep_s=P_["t_prep"], min_eig_Fqdd=P_["L"]["minlam"], max_abs_uref=float(np.abs(P_["L"]["U"]).max()))
    checks = {q: task_check(P_, q) for q in sorted({q for _, q in jobs})} if spec is not None else {}
    out_ = []
    for eid, q in jobs:
        res, extra = run_episode(P_, draw_episode(eid, tasks), q, keep_trace=eid in trace_ids)
        if extra is not None:
            if spec is not None:
                res["score"] = float(np.max(np.abs(extra["W"]) / P_["S"][:len(extra["W"])]))
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
# ## Algorithm 1 for each model and variant
#
# Groups: one group for `const` and `ref`; one group per time scaling for `ref+class`. In each group: pilot quantile at `q = 0`, `PILOT_ITERS` fixed-point iterations under `C_q`, inflation, then calibration on fresh episodes (`q̂_g`, certified if `q̂_g ≤ q̃_g`) and test episodes, all under the deployed tube sizes. Episode ids are those of NB01.

# %%
def conformal_quantile(s, delta):
    n = len(s); r = math.ceil((n + 1)*(1 - delta))
    return np.inf if r > n else float(np.sort(s)[r - 1])

def pilot_quantile(s, delta):
    return float(np.quantile(s, 1 - delta, method="higher"))

def group_of(variant, ep):
    return f"sc={ep['sc']:g}" if variant.endswith("+class") else "all"

ALL_EP, SUMMARY, CLASS_ROWS, TASK_ROWS, CHECK_ROWS, SIGMA_ROWS, PILOT_LOG, TRACES = [], [], [], [], [], [], [], {}
# ---- checkpoint after every (model, variant): an interrupted run continues with RESUME_DIR set to its folder
import pickle
CKPT = out("checkpoint.pkl"); DONE = set()
_STATE = ("ALL_EP", "SUMMARY", "CLASS_ROWS", "TASK_ROWS", "CHECK_ROWS", "SIGMA_ROWS", "PILOT_LOG", "TRACES", "DONE")
if RESUME_DIR and os.path.exists(CKPT):
    _st = pickle.load(open(CKPT, "rb"))
    ALL_EP, SUMMARY, CLASS_ROWS, TASK_ROWS, CHECK_ROWS, SIGMA_ROWS, PILOT_LOG, TRACES, DONE = (_st[k] for k in _STATE)
    print(f"resuming {RESUME_DIR}: already done {sorted(d for d in DONE if d[1] != 'pilot')}")
def save_checkpoint():
    g_ = globals(); tmp = CKPT + ".tmp"
    with open(tmp, "wb") as f_: pickle.dump({k: g_[k] for k in _STATE}, f_)
    os.replace(tmp, CKPT)

t_all = time.time()
for label, kind, size, seed in MODELS:
    t_m = time.time()
    if all((label, v) in DONE for v in VARIANTS):
        print(f"\n=== {label}: already done (checkpoint), skipped"); continue
    # ---- off-line preparation of every grid task (in the workers) and admissible tasks, in the grid order of NB01
    grid = all_tasks(); args = [(label, sc, m, [], None, grid) for sc, m in grid]
    infos = (Parallel(n_jobs=N_JOBS, backend="loky")(delayed(task_worker)(*a) for a in args) if N_JOBS > 1
             else [task_worker(*a) for a in args])
    infos = [i for i, _, _ in infos]
    if (label, "pilot") not in DONE: TASK_ROWS += infos
    umax = {(i["sc"], i["m"]): i["max_abs_uref"] for i in infos}
    tasks = [tk for tk in grid if umax[tk] <= TAU_MAX - TORQUE_MARGIN]
    print(f"\n=== {label}: {len(tasks)} admissible tasks of {len(grid)} (excluded: {[tk for tk in grid if tk not in tasks]}) "
          f"| min eig F_qdd {min(i['min_eig_Fqdd'] for i in infos):.3f} | prep {(time.time()-t_m)/60:.1f} min")
    # ---- pilot at q = 0 (shared by all variants)
    pilot_ids = list(range(N_PILOT))
    pilot, _, _ = run_jobs(label, pilot_ids, lambda ep: 0.0, None, tasks)
    first_pilot = (label, "pilot") not in DONE; DONE.add((label, "pilot"))
    sig_c, beta, floor, r2 = fit_normalizers(pilot)
    for i in range(6):
        if first_pilot: SIGMA_ROWS.append(dict(model=label, component=i, sigma_const=sig_c[i], floor=floor[i], r2_abs_w=r2[i],
                               **{f"beta[{f}]": beta[j, i] for j, f in enumerate(FEATURES)}))
    for r, ex in (pilot if first_pilot else []): ALL_EP.append(dict(r, model=label, variant="-", set="pilot_q0", group="-"))
    print(f"pilot {len(pilot)} episodes at q=0 | R^2 of the |w| envelope fit per component {np.round(r2, 2)} | {(time.time()-t_m)/60:.1f} min")
    specs = {"const": dict(id=f"{label}|const", kind="const", sigma=sig_c),
             "ref": dict(id=f"{label}|ref", kind="ref", beta=beta, floor=floor)}
    pilot_eps = {e: draw_episode(e, tasks) for e in pilot_ids}
    for variant in VARIANTS:
        if (label, variant) in DONE: continue
        t_v = time.time(); spec = specs["const" if variant.startswith("const") else "ref"]
        # pilot scores at q = 0 from the stored mismatch
        s0 = {}
        for r, ex in pilot:
            if ex is None: continue
            ep = pilot_eps[r["eid"]]; R = REFS[ep["sc"]]; idx = np.arange(0, R["N"], STEP)
            S = sigma_track(spec, ref_features(R, idx, ep["m"]))[:len(ex["W"])]
            s0[r["eid"]] = float(np.max(np.abs(ex["W"]) / S))
        groups = sorted({group_of(variant, ep) for ep in pilot_eps.values()}, key=str)
        def gq(vals): return pilot_quantile(np.array(vals), DELTA) if len(vals) else pilot_quantile(np.array(list(s0.values())), DELTA)
        q_g = {g: gq([s for e, s in s0.items() if group_of(variant, pilot_eps[e]) == g]) for g in groups}
        for g in groups: PILOT_LOG.append(dict(model=label, variant=variant, group=g, iter=0, q=q_g[g]))
        for it in range(1, PILOT_ITERS + 1):
            rs, _, _ = run_jobs(label, pilot_ids, lambda ep: q_g[group_of(variant, ep)], spec, tasks)
            q_g = {g: gq([r["score"] for r, _ in rs if not r["unstable"] and group_of(variant, pilot_eps[r["eid"]]) == g]) for g in groups}
            for g in groups: PILOT_LOG.append(dict(model=label, variant=variant, group=g, iter=it, q=q_g[g]))
        q_tilde = {g: INFLATE*q_g[g] for g in groups}
        qfun = lambda ep: q_tilde.get(group_of(variant, ep), max(q_tilde.values()))
        # ---- calibration and test
        cal_ids = [10**6 + e for e in range(N_CAL)]; test_ids = [2*10**6 + e for e in range(N_TEST)]
        cal, _, chk_c = run_jobs(label, cal_ids, qfun, spec, tasks)
        test, _, chk_t = run_jobs(label, test_ids, qfun, spec, tasks, trace_ids=test_ids[:3])
        for e, (r, ex) in zip(test_ids[:3], test[:3]):
            if ex is not None: TRACES[(label, variant, e)] = dict(r, **ex)
        chk = pd.DataFrame(chk_c + chk_t).drop_duplicates(subset=["sc", "m", "q"])
        for _, c in chk.iterrows(): CHECK_ROWS.append(dict(c, variant=variant))
        chk_map = {(c.sc, c.m): c for _, c in chk.iterrows()}
        q_hat, cert = {}, {}
        for g in groups:
            sg = np.array([r["score"] if not r["unstable"] else np.inf for r, _ in cal if group_of(variant, r) == g])
            q_hat[g] = conformal_quantile(sg, DELTA) if len(sg) else np.inf; cert[g] = q_hat[g] <= q_tilde[g]
        for st, rs in (("calibration", cal), ("test", test)):
            for r, _ in rs:
                g = group_of(variant, r); d = dict(r, model=label, variant=variant, set=st, group=g)
                if not r["unstable"]: d["covered"] = r["score"] <= q_tilde[g]
                c = chk_map[(r["sc"], r["m"])]; d.update(x_empty_task=c.x_empty_task, uref_out_task=c.uref_out_task)
                ALL_EP.append(d)
        te = pd.DataFrame([r for r, _ in test]); ok = te[~te.unstable].copy()
        ok["g"] = [group_of(variant, r) for _, r in ok.iterrows()]; ok["cov"] = [s <= q_tilde[g] for s, g in zip(ok.score, ok.g)]
        for g in groups:
            og = ok[ok.g == g]
            CLASS_ROWS.append(dict(model=label, variant=variant, group=g, q_tilde=q_tilde[g], q_hat=q_hat[g], certified=cert[g],
                                   n_cal=int(sum(group_of(variant, r) == g for r, _ in cal)), n_test=len(og),
                                   coverage_test=float(og["cov"].mean()) if len(og) else np.nan))
        bysc = ok.groupby("sc")["cov"].mean()
        te_chk = pd.DataFrame([chk_map[(r.sc, r.m)] for _, r in ok.iterrows()])
        SUMMARY.append(dict(model=label, variant=variant, delta=DELTA, n_groups=len(groups),
                            q_tilde=";".join(f"{q_tilde[g]:.3f}" for g in groups), q_hat=";".join(f"{q_hat[g]:.3f}" for g in groups),
                            certified=all(cert.values()), coverage_test=float(ok["cov"].mean()),
                            coverage_min_speed=float(bysc.min()), coverage_max_speed=float(bysc.max()),
                            unstable=int(te.unstable.sum()),
                            episodes_with_x_violation=float((ok.viol_x > 0).mean()), episodes_with_tau_violation=float((ok.viol_tau > 0).mean()),
                            episodes_with_fallback=float((ok.n_fallback > 0).mean()),
                            fb_x_empty=float((ok.fb_x_empty > 0).mean()), fb_u_empty=float((ok.fb_u_empty > 0).mean()),
                            fb_qp_infeasible=float((ok.fb_qp_infeasible > 0).mean()), fb_solver=float((ok.fb_solver > 0).mean()),
                            test_tasks_x_empty=float(te_chk.x_empty_task.mean()), test_tasks_uref_out=float(te_chk.uref_out_task.mean()),
                            qp_fraction=float(ok.n_qp.sum() / ok.K.sum()), fallback_step_fraction=float(ok.n_fallback.sum() / ok.K.sum()),
                            x_empty_steps_med=float(te_chk.x_empty_frac_k.median()),
                            budget_lost_med_Nm=float(te_chk.budget_lost_med_Nm.median()),
                            tube_pos_j1_med_urad=float(te_chk.tube_pos_j1_med_urad.median()),
                            tube_pos_jmax_mrad=float(te_chk.tube_pos_jmax_mrad.median()),
                            rmse_um_mean=float(ok.rmse_um.mean()), max_err_um_mean=float(ok.max_err_um.mean()),
                            max_dq_mrad=float(1e3*ok.max_dq.max()), max_abs_w_zeta=float(ok.max_abs_w_zeta.max()),
                            minutes=(time.time() - t_v)/60))
        S_ = SUMMARY[-1]
        print(f"[{variant}] q_tilde {S_['q_tilde']} | certified {S_['certified']} | coverage {S_['coverage_test']:.3f} "
              f"(by speed {S_['coverage_min_speed']:.2f}-{S_['coverage_max_speed']:.2f}) | fallback {S_['episodes_with_fallback']:.2f} of episodes, {S_['fallback_step_fraction']:.3f} of steps "
              f"(x_empty {S_['fb_x_empty']:.2f}, u_empty {S_['fb_u_empty']:.2f}, qp {S_['fb_qp_infeasible']:.2f}, solver {S_['fb_solver']:.2f}) "
              f"| tube j=1 {S_['tube_pos_j1_med_urad']:.1f} urad, max {S_['tube_pos_jmax_mrad']:.3f} mrad | budget {S_['budget_lost_med_Nm']:.3f} N m "
              f"| viol_x {S_['episodes_with_x_violation']:.3f} | {S_['minutes']:.1f} min")
        DONE.add((label, variant)); save_checkpoint()
print(f"\ntotal {(time.time()-t_all)/60:.1f} min")

ep_df = pd.DataFrame(ALL_EP); ep_df.to_csv(out("CMPC_NB02_episodes.csv"), index=False)
sum_df = pd.DataFrame(SUMMARY); sum_df.to_csv(out("CMPC_NB02_summary.csv"), index=False)
cls_df = pd.DataFrame(CLASS_ROWS); cls_df.to_csv(out("CMPC_NB02_groups.csv"), index=False)
pd.DataFrame(CHECK_ROWS).to_csv(out("CMPC_NB02_task_checks.csv"), index=False)
pd.DataFrame(TASK_ROWS).drop_duplicates(subset=["model", "sc", "m"]).to_csv(out("CMPC_NB02_tasks.csv"), index=False)
pd.DataFrame(SIGMA_ROWS).to_csv(out("CMPC_NB02_normalizers.csv"), index=False)
pd.DataFrame(PILOT_LOG).to_csv(out("CMPC_NB02_pilot_iterations.csv"), index=False)
print(sum_df.round(4).T.to_string())

# %% [markdown]
# ## E1 — Coverage over random calibration/test splits, within each group
#
# For each model, variant and group, the calibration and test scores of that group are pooled; for `N_SPLITS` random splits with `n` = number of calibration episodes of the group, `q̂` is computed at each level δ and its coverage measured on the rest, against the Beta law of Corollary 1.

# %%
rng = np.random.default_rng(SEED + 7); rows = []
sel = ep_df[ep_df.set.isin(["calibration", "test"]) & ep_df.score.notna()]
for (label, variant, g), d in sel.groupby(["model", "variant", "group"]):
    pool = d.score.to_numpy(); n = int((d.set == "calibration").sum())
    if n < 2 or len(pool) - n < 2: continue
    for dl in DELTAS_CHECK:
        l = math.floor((n + 1)*dl)
        for sp in range(N_SPLITS):
            perm = rng.permutation(len(pool)); c, t = pool[perm[:n]], pool[perm[n:]]
            qh = conformal_quantile(c, dl)
            rows.append(dict(model=label, variant=variant, group=g, delta=dl, split=sp, n=n, coverage=float(np.mean(t <= qh)),
                             beta_lo=beta_dist.ppf(0.05, n + 1 - l, l) if l > 0 else np.nan,
                             beta_hi=beta_dist.ppf(0.95, n + 1 - l, l) if l > 0 else np.nan))
split_df = pd.DataFrame(rows); split_df.to_csv(out("CMPC_NB02_split_coverage.csv"), index=False)
cov_tab = split_df.groupby(["model", "variant", "group", "delta"]).agg(
    n=("n", "first"), mean_cov=("coverage", "mean"), p05=("coverage", lambda v: np.percentile(v, 5)),
    p95=("coverage", lambda v: np.percentile(v, 95)), beta_p05=("beta_lo", "first"), beta_p95=("beta_hi", "first")).reset_index()
cov_tab["target"] = 1 - cov_tab["delta"]; cov_tab.to_csv(out("CMPC_NB02_split_coverage_summary.csv"), index=False)
print(cov_tab.round(4).to_string(index=False))

# %% [markdown]
# ## Figures

# %%
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False})
COL = {"Nominal": "#2a78d6", "MLP-35": "#eb6834", "HYBRID-35": "#1baf7a", "PINN-175": "#eda100"}
MK = {"Nominal": "o", "MLP-35": "s", "HYBRID-35": "D", "PINN-175": "^"}
labels = [m[0] for m in MODELS]
d_cs = ep_df[ep_df.set.isin(["calibration", "test"]) & ep_df.score.notna()].copy()
qt_map = {(r.model, r.variant, r.group): r.q_tilde for r in cls_df.itertuples()}
d_cs["r"] = [s / qt_map[(m_, v, g)] for s, m_, v, g in zip(d_cs.score, d_cs.model, d_cs.variant, d_cs.group)]

# (a) score / tube size by speed, one panel per variant
fig, axes = plt.subplots(1, len(VARIANTS), figsize=(3.3*len(VARIANTS), 2.8), sharey=True, squeeze=False)
for ax, variant in zip(axes[0], VARIANTS):
    for i, lab in enumerate(labels):
        for j, sc in enumerate(SCALES):
            y = d_cs[(d_cs.model == lab) & (d_cs.variant == variant) & (d_cs.sc == sc)].r.to_numpy()
            if not len(y): continue
            x = j + (i - (len(labels) - 1)/2)*0.15; lo, md, hi = np.quantile(y, [0.05, 0.5, 0.95])
            ax.vlines(x, lo, hi, color=COL[lab], lw=1.4); ax.plot(x, md, MK[lab], color=COL[lab], ms=4.5, mec="white", mew=0.6, label=lab if j == 0 else None)
    ax.axhline(1, color="k", lw=0.9); ax.set_yscale("log"); ax.set_title(variant)
    ax.set_xticks(range(len(SCALES))); ax.set_xticklabels([f"{s:g}" for s in SCALES]); ax.set_xlabel("time scaling (smaller = faster)")
axes[0][0].set_ylabel(r"$s/\tilde q_g$"); axes[0][0].legend(frameon=False)
fig.tight_layout(); fig.savefig(out("fig_nb02_score_speed.pdf")); fig.savefig(out("fig_nb02_score_speed.png"), dpi=200); plt.show()

# (b) test coverage by speed class, one panel per variant
te_df = ep_df[(ep_df.set == "test") & ep_df.score.notna()]
fig, axes = plt.subplots(1, len(VARIANTS), figsize=(3.3*len(VARIANTS), 2.6), sharey=True, squeeze=False)
for ax, variant in zip(axes[0], VARIANTS):
    for i, lab in enumerate(labels):
        c = te_df[(te_df.model == lab) & (te_df.variant == variant)].groupby("sc")["covered"].apply(lambda v: v.astype(float).mean())
        ax.plot([SCALES.index(s) + (i - (len(labels) - 1)/2)*0.15 for s in c.index], c.values, MK[lab], color=COL[lab], ms=5, label=lab)
    ax.axhline(1 - DELTA, color="k", ls=":", lw=0.9); ax.set_title(variant); ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(range(len(SCALES))); ax.set_xticklabels([f"{s:g}" for s in SCALES]); ax.set_xlabel("time scaling")
axes[0][0].set_ylabel("test coverage"); axes[0][0].legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(out("fig_nb02_class_coverage.pdf")); fig.savefig(out("fig_nb02_class_coverage.png"), dpi=200); plt.show()

# (c) position tube and fallback rate by variant
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6))
w = 0.8/len(labels)
for i, lab in enumerate(labels):
    s_ = sum_df[sum_df.model == lab].set_index("variant").reindex(list(VARIANTS))
    xs = np.arange(len(VARIANTS)) + (i - (len(labels) - 1)/2)*w
    ax[0].bar(xs, s_.tube_pos_j1_med_urad, w*0.9, color=COL[lab], label=lab)
    ax[1].bar(xs, s_.episodes_with_fallback, w*0.9, color=COL[lab])
for a_, t_ in zip(ax, ["one-step joint-error tube (µrad, median)", "test episodes with a fallback"]):
    a_.set_xticks(range(len(VARIANTS))); a_.set_xticklabels(VARIANTS); a_.set_title(t_, fontsize=9)
ax[0].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(out("fig_nb02_tube_fallback.pdf")); fig.savefig(out("fig_nb02_tube_fallback.png"), dpi=200); plt.show()

# (d) one test episode under the `ref` variant: joint error and the time-varying one-step tube
keys = [k for k in TRACES if k[1] == "ref"] or list(TRACES)
if keys:
    tr = TRACES[keys[0]]; tt = np.arange(len(tr["modes"]))*T_C
    fig, ax = plt.subplots(2, 1, figsize=(3.4, 3.2), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    ax[0].plot(tt, 1e3*np.abs(tr["X"][:-1, :3]).max(axis=1), lw=0.8, label=r"$\max_i|\delta q_i|$")
    ax[0].plot(tt, 1e3*tr["tube1"], lw=0.8, label="tube, $j=1$")
    ax[0].axhline(1e3*EPS_Q, color="k", lw=0.8, label=r"$\varepsilon_q$"); ax[0].set_yscale("log")
    ax[0].set_ylabel("mrad"); ax[0].legend(frameon=False, fontsize=7); ax[0].set_title(f"{keys[0][0]}, {keys[0][1]}, sc {tr['sc']}, {tr['m']} kg", fontsize=8)
    ax[1].plot(tt, tr["modes"], lw=0.6); ax[1].set_yticks([0, 1, 2]); ax[1].set_yticklabels(["K x", "QP", "fallback"], fontsize=7)
    ax[1].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(out("fig_nb02_episode.pdf")); fig.savefig(out("fig_nb02_episode.png"), dpi=200); plt.show()

# %% [markdown]
# ## Pack the run folder

# %%
zip_path = shutil.make_archive(RUN_DIR, "zip", root_dir=RUN_DIR)
print("results:", zip_path, f"({os.path.getsize(zip_path)/1024:.0f} kB)")
print(sorted(os.listdir(RUN_DIR)))
