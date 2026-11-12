# %% [markdown]
# # CMPC-NB05 — Distribution shift, coverage versus the number of calibration episodes, and computation (Delta robot)
#
# **Paper:** *Conformal-calibrated tube MPC with physics-informed learned dynamics for Lagrangian robots* (JAS draft, `paper/main.tex`, Sec. V; experiments E1, E5 and E8 of Sec. IV-D).
#
# **What this notebook does**, with the hybrid surrogate (35 trajectories, seed 0), reference-dependent normalizers and the controller of NB02/NB04:
# 1. **Calibration on D** (Adept family, severity 1, payload 0–6 kg), exactly as the `proposed` method of NB04 (same pilot, ids and rounds).
# 2. **E1 — coverage versus n.** A pool of in-distribution episodes run under the certified tube; for `n ∈ N_LIST`, random splits give the coverage of `q̂` against the Beta band of Corollary 1.
# 3. **E5 — shift.** The tube calibrated on D is deployed on tasks outside D: two new reference families (four loops of a circle and three of a figure eight in a horizontal plane, with peak joint speeds close to those of D), stronger or weaker unmodeled dynamics (severity 0.5, 1.5, 2) and payloads outside the training range (7 and 8 kg). For each set: coverage, violations, fallbacks and the ratio of the set's own conformal quantile to the deployed tube.
# 4. **E5 — per-family calibration.** Algorithm 1 run separately on the circle and on the figure eight (Remark 3); coverage on the same shifted test episodes.
# 5. **E8 — computation.** Per-cycle time of the controller (check of `c = 0` plus the QP when needed) on a few episodes run serially, and the off-line preparation time per task.
#
# **Outputs.** CSV files and figures in `OUT_DIR/CMPC_NB05_<tag>/`, packed into a zip at the end of every run.
#
# **Interrupted runs.** After each step the run saves `checkpoint.pkl` in its folder. If the run stops, set `RESUME_DIR` to that folder (for example `"results/CMPC_NB05_FULL_20261020_090000"`) and run the notebook again from the top: finished steps are loaded and skipped. Keep the other settings unchanged.
#
# **Modes.** `QUICK = True` is a smoke test. `QUICK = False` is the long run for the paper.
#
# **Requirements.** numpy, scipy, pandas, matplotlib, osqp, joblib; `ff18_HYBRID_35traj_seed0_e150.npz` in `WEIGHTS_DIR`.

# %%
import os, sys, types, time, json, shutil, datetime, math, pickle
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
WEIGHTS_DIR = "."                  # folder with ff18_HYBRID_35traj_seed0_e150.npz (notebook 18)
OUT_DIR = "results"
RESUME_DIR = None                  # to continue an interrupted run: its folder, e.g. "results/CMPC_NB05_FULL_20261020_090000"
N_JOBS = max(1, (os.cpu_count() or 2) - 1)   # worker processes (1 = serial)

# ---- plant / controller (identical to NB02 and NB04)
SEVERITY = 1.0
FS = 2000; dt = 1.0 / FS
T_C = 0.002
N_H = 25
TAU_MAX = 5.0
Q_POS, Q_VEL, Q_INT, R_TAU = 1e4, 1.0, 1e5, 1e-2

# ---- tube MPC
EPS_Q = 1.0e-3
N_TAIL = N_H
TORQUE_MARGIN = 0.5

# ---- calibration distribution D (identical to NB02) and shifted test sets
REF_C2 = True
SCALES = (2.0, 2.4, 3.2, 4.8, 6.4)
PAYLOADS = (0, 1, 2, 3, 4, 5, 6)
X0_BOX = 2.0e-4
SIGMA_TAU = 0.005
FAMILIES = ("adept", "circle", "eight")
SEVERITIES_SHIFT = (0.5, 1.5, 2.0)
PAYLOADS_OUT = (7, 8)
MODEL = ("HYBRID-35", "HYBRID", 35, 0)

# ---- Algorithm 1 and experiment sizes
DELTA = 0.10
N_PILOT, N_CAL, N_TEST = 50, 200, 400
PILOT_ITERS, INFLATE, MAX_ROUNDS = 1, 1.05, 3
N_POOL = 1000                      # extra in-distribution episodes for E1 (pool = calibration + test + these)
N_LIST = (50, 100, 200, 500)       # calibration sizes of E1
DELTAS_CHECK = (0.05, 0.10, 0.20)
N_SPLITS = 500
N_SHIFT = 200                      # test episodes per shifted set
N_TIMING = 10                      # episodes run serially for the timing of E8
SEED = 2026                        # same seed and ids as NB02/NB04

if QUICK:
    SCALES, PAYLOADS = (2.0, 3.2), (0, 6)
    N_PILOT, N_CAL, N_TEST, N_POOL, N_LIST, N_SPLITS, N_SHIFT, N_TIMING = 16, 24, 24, 40, (10, 20, 40), 100, 12, 2

STEP = int(round(T_C / dt))
TAG = ("QUICK_" if QUICK else "FULL_") + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESUME_DIR or os.path.join(OUT_DIR, f"CMPC_NB05_{TAG}"); os.makedirs(RUN_DIR, exist_ok=True)
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

# ---- other reference families (shift study): four loops of a circle (radius 0.10 m) or three of a figure eight (half-width 0.12 m) in the plane z = 0.52 m,
# traversed with a quintic (rest-to-rest) phase over the same duration as the Adept cycle, 2.5*scale s.
# A task key `sc` encodes the family: Adept `sc`, circle `100 + sc`, figure eight `200 + sc`.
FAM_CODE = {"adept": 0, "circle": 100, "eight": 200}
def family_of(sc): return {0: "adept", 1: "circle", 2: "eight"}[int(sc // 100)]
def loop_reference(kind, scale, loops=None, c=np.array([0.0, 0.0, 0.52]), r=0.10, a=0.12):
    loops = loops or (4 if kind == "circle" else 3)
    T = 2.5*scale; n = int(round(T*FS)) + 1; t = np.arange(n)*dt; tau = t/T
    ph = 2*np.pi*loops*(6*tau**5 - 15*tau**4 + 10*tau**3)
    P = np.tile(c, (n, 1))
    if kind == "circle": P[:, 0] += r*(np.cos(ph) - 1); P[:, 1] += r*np.sin(ph)
    else: P[:, 0] += a*np.sin(ph); P[:, 1] += a*np.sin(ph)*np.cos(ph)
    q = ik_vec(P); qd = np.gradient(q, dt, axis=0, edge_order=2); qdd = np.gradient(qd, dt, axis=0, edge_order=2)
    return dict(t=t, P=P, q=q, qd=qd, qdd=qdd, N=n)

REFS = {}
for fam in FAMILIES:
    for sc in SCALES:
        REFS[FAM_CODE[fam] + sc] = adept_reference(sc) if fam == "adept" else loop_reference(fam, sc)
for sc, R in REFS.items():
    jump = np.abs(np.diff(R["qdd"], axis=0)).max()
    print(f"{family_of(sc)} scale {sc % 100:g}: {R['t'][-1]:.1f} s | peak |qd| {np.abs(R['qd']).max():.2f} rad/s | peak |qdd| {np.abs(R['qdd']).max():.1f} rad/s^2 "
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
# An episode is drawn as in NB02 from a list of tasks `(sc, m)` or `(sc, m, severity)`; with the Adept task list of NB02 it is the same episode as in NB02 and NB04. `task_worker` runs a list of episodes of one task in a worker process; it keeps the off-line data of the task between calls and recomputes the tightening only when the normalizers change. It also checks the task off line at each tube size it is given: `x_empty_task` (the joint-error tube exceeds `ε_q` somewhere, so the tightened problem is empty at those steps), `uref_out_task` (the reference torque plus the torque tube exceeds `τ_max` somewhere) and the torque budget lost to tightening.

# %%
def all_tasks():
    return [(sc, m) for sc in SCALES for m in PAYLOADS]

def draw_episode(eid, tasks):
    rng = np.random.default_rng([SEED, eid])
    tk = tasks[rng.integers(len(tasks))]; sc, m = tk[0], tk[1]; sev = tk[2] if len(tk) > 2 else SEVERITY
    x0 = rng.uniform(-X0_BOX, X0_BOX, 3)
    K = len(REFS[sc]["q"][::STEP]); noise = rng.normal(0.0, SIGMA_TAU, (K, 3))
    return dict(eid=eid, sc=sc, m=m, sev=sev, dq0=x0, noise=noise)

def run_episode(P_, ep, q_tube, keep_trace=False):
    sc, m = ep["sc"], ep["m"]; R = REFS[sc]; L = P_["L"]
    ctrl = TubeMPC(P_, q_tube); K = len(L["idx"]); Kv = P_["Kv"]
    q = R["q"][0] + ep["dq0"]; qd = R["qd"][0].copy(); zeta = np.zeros(3)
    X = np.zeros((K + 1, NX)); Uapp = np.zeros((K, 3)); modes = np.zeros(K, int); viol_tau = 0
    Pe = np.zeros((R["N"], 3)); t_ctrl = np.zeros(K)
    for k in range(K):
        i = L["idx"][k]
        x = np.concatenate([q - R["q"][i], qd - R["qd"][i], zeta]); X[k] = x
        t_c = time.perf_counter(); du, modes[k] = ctrl.control(k, x); t_ctrl[k] = time.perf_counter() - t_c
        tau_cmd = L["U"][k] + du
        if np.any(np.abs(tau_cmd) > TAU_MAX + 1e-9): viol_tau += 1
        tau = np.clip(tau_cmd, -TAU_MAX, TAU_MAX); Uapp[k] = tau - L["U"][k]
        tau_pl = tau + ep["noise"][k]
        zeta = zeta + T_C*x[:3]
        for s_ in range(STEP):
            ii = i + s_
            if ii >= R["N"]: break
            p = fk(q); Pe[ii] = p
            qd = qd + plant_qdd(q, qd, tau_pl, p, m, ep["sev"])*dt; q = q + qd*dt
        if not np.all(np.isfinite(q)) or np.abs(q).max() > 3:
            return dict(eid=ep["eid"], sc=sc, m=m, sev=ep["sev"], q_tube=q_tube, unstable=True), None
    iN = min(L["idx"][-1] + STEP, R["N"] - 1)
    X[K] = np.concatenate([q - R["q"][iN], qd - R["qd"][iN], zeta])
    W = (X[1:] - np.einsum("kij,kj->ki", L["A"], X[:-1]) - np.einsum("kij,kj->ki", L["B"], Uapp))[:Kv]
    e_cart = 1000*np.linalg.norm(R["P"][:len(Pe)] - Pe, axis=1)
    res = dict(eid=ep["eid"], sc=sc, m=m, sev=ep["sev"], q_tube=q_tube, unstable=False,
               t_ctrl_mean_us=1e6*float(t_ctrl.mean()), t_ctrl_p99_us=1e6*float(np.percentile(t_ctrl, 99)), t_ctrl_max_us=1e6*float(t_ctrl.max()), max_abs_w_zeta=np.abs(W[:, 6:]).max(),
               max_dq=np.abs(X[:-1, :3]).max(), viol_x=int((np.abs(X[:-1, :3]) > EPS_Q).any(axis=1).sum()),
               viol_tau=viol_tau, n_qp=ctrl.n_qp, n_fallback=ctrl.n_fallback, K=K,
               **{f"fb_{c}": v for c, v in ctrl.fb.items()},
               rmse_um=1000*np.sqrt(np.mean(e_cart**2)), max_err_um=1000*e_cart.max())
    extra = dict(W=W[:, :6])
    if keep_trace: extra.update(X=X, modes=modes)
    return res, extra

def _worker_cache():
    mod = sys.modules.get("_cmpc_nb05_cache")
    if mod is None:
        mod = types.ModuleType("_cmpc_nb05_cache"); mod.C = {}; sys.modules["_cmpc_nb05_cache"] = mod
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
# ## Calibration on D, coverage versus n, shift and computation
#
# Every step writes a checkpoint. Episode ids: pilot `0..`, calibration `10^6 + 10^5·round + e`, in-distribution test `2·10^6 + e` (as NB02/NB04), pool `3·10^6 + e`, shifted test `4·10^6 + e`, per-family pilot and calibration `5·10^6 + e` and `6·10^6 + 10^5·round + e`.

# %%
FMODELS[MODEL[0]] = load_model(MODEL[1], MODEL[2], MODEL[3]); LAB = MODEL[0]

def conformal_quantile(s, delta):
    n = len(s); r = math.ceil((n + 1)*(1 - delta))
    return np.inf if r > n else float(np.sort(s)[r - 1])

def pilot_quantile(s, delta):
    return float(np.quantile(s, 1 - delta, method="higher"))

STATE = dict(DONE=set(), ROWS=[], EPISODES=[], ROUNDS=[], E1=[], INFO={})
CKPT = out("checkpoint.pkl")
if RESUME_DIR and os.path.exists(CKPT):
    STATE = pickle.load(open(CKPT, "rb")); print("resuming", RESUME_DIR, "| done:", sorted(STATE["DONE"]))
def save_checkpoint():
    tmp = CKPT + ".tmp"
    with open(tmp, "wb") as f_: pickle.dump(STATE, f_)
    os.replace(tmp, CKPT)

t_all = time.time()
# ---- off-line preparation of every candidate task and admissible task lists
cand = sorted({(FAM_CODE[f] + sc, m) for f in FAMILIES for sc in SCALES for m in PAYLOADS} | {(sc, m) for sc in SCALES for m in PAYLOADS_OUT})
infos = (Parallel(n_jobs=N_JOBS, backend="loky")(delayed(task_worker)(LAB, sc, m, [], None, cand) for sc, m in cand) if N_JOBS > 1
         else [task_worker(LAB, sc, m, [], None, cand) for sc, m in cand])
TASK_INFO = pd.DataFrame([i for i, _, _ in infos]); TASK_INFO["family"] = [family_of(s) for s in TASK_INFO.sc]
ADM = {(r.sc, r.m) for r in TASK_INFO.itertuples() if r.max_abs_uref <= TAU_MAX - TORQUE_MARGIN}
TASKS_D = [(sc, m) for sc in SCALES for m in PAYLOADS if (sc, m) in ADM]          # grid order of NB02
SETS = {f: [(FAM_CODE[f] + sc, m) for sc in SCALES for m in PAYLOADS if (FAM_CODE[f] + sc, m) in ADM] for f in FAMILIES if f != "adept"}
SETS.update({f"severity {s:g}": [(sc, m, s) for sc, m in TASKS_D] for s in SEVERITIES_SHIFT})
SETS[f"payload {'-'.join(str(m) for m in PAYLOADS_OUT)} kg"] = [(sc, m) for sc in SCALES for m in PAYLOADS_OUT if (sc, m) in ADM]
SETS = {k: v for k, v in SETS.items() if len(v)}
print(f"D: {len(TASKS_D)} admissible tasks | shifted sets: " + ", ".join(f"{k} ({len(v)} tasks)" for k, v in SETS.items())
      + f" | prep {(time.time()-t_all)/60:.1f} min")

# ---- pilot at q = 0 and reference-dependent normalizers, fitted on D only (as NB02/NB04)
pilot_ids = list(range(N_PILOT))
pilot, _, _ = run_jobs(LAB, pilot_ids, lambda ep: 0.0, None, TASKS_D)
sig_c, beta, floor, r2 = fit_normalizers(pilot)
SPEC = dict(id=f"{LAB}|ref", kind="ref", beta=beta, floor=floor)
print(f"pilot {len(pilot)} episodes | R^2 of the envelope fit {np.round(r2, 2)}")

def ep_rows(rs, name, q):
    rows = []
    for r, _ in rs:
        d = {k: v for k, v in r.items() if not k.startswith("_")}; d.update(set=name, q_deployed=q, family=family_of(r["sc"]))
        if not r["unstable"]: d["covered"] = r["score"] <= q
        rows.append(d)
    STATE["EPISODES"].extend(rows); return pd.DataFrame(rows)

def set_row(name, df, q, **kw):
    ok = df[~df.unstable]
    row = dict(set=name, n=len(df), q_deployed=q, unstable=int(df.unstable.sum()), coverage=float(ok.covered.mean()),
               coverage_min_class=float(ok.groupby("sc").covered.mean().min()),
               own_quantile=conformal_quantile(ok.score.to_numpy(), DELTA) if len(ok) >= 10 else np.nan,
               episodes_with_x_violation=float((ok.viol_x > 0).mean()), episodes_with_fallback=float((ok.n_fallback > 0).mean()),
               fallback_step_fraction=float(ok.n_fallback.sum()/ok.K.sum()), rmse_um_mean=float(ok.rmse_um.mean()),
               step_coverage=float(ok.step_coverage.mean()) if "step_coverage" in ok else np.nan, **kw)
    row["own_quantile_over_deployed"] = row["own_quantile"]/q
    STATE["ROWS"].append(row)
    print(f"[{name}] q {q:.3f} | coverage {row['coverage']:.3f} (worst class {row['coverage_min_class']:.2f}) | own quantile / deployed "
          f"{row['own_quantile_over_deployed']:.2f} | fallback {row['episodes_with_fallback']:.2f} | viol_x {row['episodes_with_x_violation']:.3f} "
          f"| RMSE {row['rmse_um_mean']:.1f} um")
    return row

def algorithm1(tasks, s0, cal_base, tag):
    """Pilot iteration from the q = 0 scores s0, then calibration rounds on fresh episodes; returns (q_tilde, q_hat, certified, cal scores)."""
    q_p = pilot_quantile(np.array(s0), DELTA)
    for it in range(PILOT_ITERS):
        rs, _, _ = run_jobs(LAB, pilot_like_ids[tag], lambda ep: q_p, SPEC, tasks)
        q_p = pilot_quantile(np.array([r["score"] for r, _ in rs if not r["unstable"]]), DELTA)
    q_t = INFLATE*q_p
    for rnd in range(MAX_ROUNDS):
        ids = [cal_base + 10**5*rnd + e for e in range(N_CAL)]
        cal, _, _ = run_jobs(LAB, ids, lambda ep: q_t, SPEC, tasks)
        sc_cal = np.array([r["score"] if not r["unstable"] else np.inf for r, _ in cal]); q_h = conformal_quantile(sc_cal, DELTA)
        cert = q_h <= q_t; STATE["ROUNDS"].append(dict(calibration=tag, round=rnd + 1, q_tilde=q_t, q_hat=q_h, certified=cert))
        print(f"  [{tag}] round {rnd + 1}: q_tilde {q_t:.3f}, q_hat {q_h:.3f}, certified {cert}")
        if cert or not np.isfinite(q_h): break
        q_t = INFLATE*q_h
    return q_t, q_h, cert, sc_cal

def scores_q0(rs, tasks_eps):
    s0 = []
    for r, ex in rs:
        if ex is None: continue
        ep = tasks_eps[r["eid"]]; R = REFS[ep["sc"]]; idx = np.arange(0, R["N"], STEP)
        s0.append(float(np.max(np.abs(ex["W"]) / sigma_track(SPEC, ref_features(R, idx, ep["m"]))[:len(ex["W"])])))
    return s0

pilot_like_ids = {"D": pilot_ids}
fam_pilot_ids = [5*10**6 + e for e in range(N_PILOT)]
for f in SETS:
    if f in ("circle", "eight"): pilot_like_ids[f] = fam_pilot_ids

steps = ["calib_D", "test_D", "pool", *[f"shift|{k}" for k in SETS], *[f"famcal|{f}" for f in SETS if f in ("circle", "eight")], "timing"]
for step in steps:
    if step in STATE["DONE"]: print(f"[{step}] done (checkpoint)"); continue
    t_s = time.time()
    if step == "calib_D":
        s0 = scores_q0(pilot, {e: draw_episode(e, TASKS_D) for e in pilot_ids})
        q_t, q_h, cert, sc_cal = algorithm1(TASKS_D, s0, 10**6, "D")
        STATE["INFO"]["D"] = dict(q_tilde=q_t, q_hat=q_h, certified=cert, cal_scores=sc_cal)
    elif step == "test_D":
        q_t = STATE["INFO"]["D"]["q_tilde"]
        rs, _, _ = run_jobs(LAB, [2*10**6 + e for e in range(N_TEST)], lambda ep: q_t, SPEC, TASKS_D)
        set_row("D (in distribution)", ep_rows(rs, "D", q_t), q_t, certified=STATE["INFO"]["D"]["certified"])
    elif step == "pool":
        q_t = STATE["INFO"]["D"]["q_tilde"]
        rs, _, _ = run_jobs(LAB, [3*10**6 + e for e in range(N_POOL)], lambda ep: q_t, SPEC, TASKS_D); ep_rows(rs, "pool", q_t)
        E = pd.DataFrame(STATE["EPISODES"]); pool = np.concatenate([STATE["INFO"]["D"]["cal_scores"],
                                                                     E[E.set.isin(["D", "pool"]) & ~E.unstable].score.to_numpy()])
        rng_s = np.random.default_rng(SEED + 7)
        for n in N_LIST:
            if n >= len(pool) - 10: continue
            for dl in DELTAS_CHECK:
                l = math.floor((n + 1)*dl); cv = []
                for _ in range(N_SPLITS):
                    p_ = rng_s.permutation(len(pool)); cv.append(np.mean(pool[p_[n:]] <= conformal_quantile(pool[p_[:n]], dl)))
                STATE["E1"].append(dict(n=n, delta=dl, target=1 - dl, mean_cov=float(np.mean(cv)), p05=float(np.percentile(cv, 5)),
                                        p95=float(np.percentile(cv, 95)), beta_p05=beta_dist.ppf(0.05, n + 1 - l, l) if l > 0 else np.nan,
                                        beta_p95=beta_dist.ppf(0.95, n + 1 - l, l) if l > 0 else np.nan, pool=len(pool)))
        print(pd.DataFrame(STATE["E1"]).round(3).to_string(index=False))
    elif step.startswith("shift|"):
        k = step.split("|", 1)[1]; q_t = STATE["INFO"]["D"]["q_tilde"]
        rs, _, _ = run_jobs(LAB, [4*10**6 + e for e in range(N_SHIFT)], lambda ep: q_t, SPEC, SETS[k])
        set_row(f"{k} (tube of D)", ep_rows(rs, k, q_t), q_t)
    elif step.startswith("famcal|"):
        f = step.split("|", 1)[1]
        pr, _, _ = run_jobs(LAB, fam_pilot_ids, lambda ep: 0.0, None, SETS[f])
        q_f, q_hf, cert_f, _ = algorithm1(SETS[f], scores_q0(pr, {e: draw_episode(e, SETS[f]) for e in fam_pilot_ids}), 6*10**6, f)
        rs, _, _ = run_jobs(LAB, [4*10**6 + e for e in range(N_SHIFT)], lambda ep: q_f, SPEC, SETS[f])
        set_row(f"{f} (calibrated on {f})", ep_rows(rs, f"{f}|own", q_f), q_f, certified=cert_f)
    elif step == "timing":
        q_t = STATE["INFO"]["D"]["q_tilde"]; ids = [2*10**6 + e for e in range(N_TIMING)]
        task_of = {e: draw_episode(e, TASKS_D) for e in ids}; rows = []
        for e in ids:                                  # serial, one episode at a time, in this process
            _, _, rs = task_worker(LAB, task_of[e]["sc"], task_of[e]["m"], [(e, q_t)], SPEC, TASKS_D)
            r = rs[0][0]; rows.append(dict(eid=e, sc=r["sc"], m=r["m"], K=r["K"], n_qp=r["n_qp"], t_ctrl_mean_us=r["t_ctrl_mean_us"],
                                           t_ctrl_p99_us=r["t_ctrl_p99_us"], t_ctrl_max_us=r["t_ctrl_max_us"]))
        STATE["INFO"]["timing"] = rows
        tdf = pd.DataFrame(rows); print(tdf.round(1).to_string(index=False))
        print(f"control period {1e6*T_C:.0f} us | mean {tdf.t_ctrl_mean_us.mean():.0f} us | p99 {tdf.t_ctrl_p99_us.max():.0f} us | max {tdf.t_ctrl_max_us.max():.0f} us"
              f" | off-line preparation per task: median {TASK_INFO.t_prep_s.median():.1f} s")
    STATE["DONE"].add(step); save_checkpoint(); print(f"  ({(time.time()-t_s)/60:.1f} min)")
print(f"\ntotal {(time.time()-t_all)/60:.1f} min")

rows_df = pd.DataFrame(STATE["ROWS"]); rows_df.to_csv(out("CMPC_NB05_sets.csv"), index=False)
pd.DataFrame(STATE["EPISODES"]).to_csv(out("CMPC_NB05_episodes.csv"), index=False)
pd.DataFrame(STATE["ROUNDS"]).to_csv(out("CMPC_NB05_rounds.csv"), index=False)
e1_df = pd.DataFrame(STATE["E1"]); e1_df.to_csv(out("CMPC_NB05_coverage_vs_n.csv"), index=False)
pd.DataFrame(STATE["INFO"].get("timing", [])).to_csv(out("CMPC_NB05_timing.csv"), index=False)
TASK_INFO.to_csv(out("CMPC_NB05_tasks.csv"), index=False)
print(rows_df.round(3).to_string(index=False))

# %% [markdown]
# ## Figures

# %%
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
for dl, c in zip(DELTAS_CHECK, ["#2a78d6", "#1baf7a", "#eda100"]):
    d = e1_df[e1_df.delta == dl].sort_values("n")
    if not len(d): continue
    ax[0].fill_between(d.n, d.beta_p05, d.beta_p95, color=c, alpha=0.15, lw=0)
    ax[0].errorbar(d.n, d.mean_cov, yerr=[d.mean_cov - d.p05, d.p95 - d.mean_cov], fmt="o-", color=c, ms=4, capsize=2, label=f"1-δ = {1-dl:.2f}")
    ax[0].axhline(1 - dl, color=c, ls=":", lw=0.8)
ax[0].set_xscale("log"); ax[0].set_xlabel("calibration episodes n"); ax[0].set_ylabel("coverage"); ax[0].legend(frameon=False)
y = np.arange(len(rows_df))
ax[1].barh(y, rows_df.coverage, color=["#1baf7a" if "calibrated on" in s or s.startswith("D") else "#eb6834" for s in rows_df.set])
ax[1].axvline(1 - DELTA, color="k", ls=":", lw=0.9); ax[1].set_yticks(y); ax[1].set_yticklabels(rows_df.set, fontsize=7); ax[1].set_xlabel("test coverage")
fig.tight_layout(); fig.savefig(out("fig_nb05_coverage_n_shift.pdf")); fig.savefig(out("fig_nb05_coverage_n_shift.png"), dpi=200); plt.show()

# %% [markdown]
# ## Pack the run folder

# %%
zip_path = shutil.make_archive(RUN_DIR, "zip", root_dir=RUN_DIR)
print("results:", zip_path, f"({os.path.getsize(zip_path)/1024:.0f} kB)")
print(sorted(os.listdir(RUN_DIR)))
