# %% [markdown]
# # CMPC-NB03 — Generality: conformal-calibrated tube MPC on a planar two-link arm
#
# **Paper:** *Conformal-calibrated tube MPC with physics-informed learned dynamics for Lagrangian robots* (JAS draft, `paper/main.tex`, Sec. IV-B and V, "Generality").
#
# **Why this notebook.** The Delta model of NB01–NB02 has a constant diagonal inertia. This notebook repeats the key experiments (E1–E3) on a serial arm whose inertia matrix depends on the configuration and couples the joints, to show that the method only needs the Lagrangian structure.
#
# **What it does.**
# 1. Builds a two-link arm moving in the vertical plane (gravity), with a 6-parameter linear-in-parameters model, a payload at the tip, an identification error in the nominal parameters and unmodeled friction, velocity coupling and a configuration-dependent torque of the same type as for the Delta (8% and 14% of the RMS torque of joints 1 and 2 on the fastest reference).
# 2. Trains the three surrogates of the paper on random-waypoint trajectories, in plain numpy (no deep-learning library needed): black-box **MLP**, **HYBRID** (identified model plus a learned residual) and **PINN** (the network outputs state-dependent parameters of the regressor, `τ = Y(q, q̇, q̈) θ(x)`). Weights are saved in the run folder.
# 3. Runs Algorithm 1 exactly as NB02 (same tube MPC, same normalizer variants, same fallback diagnostics, same E1 split test) on a pick-and-place cycle in the vertical plane, with random time scaling and payload.
# 4. E3: relates the certified tube to the torque residual of each surrogate along the references.
#
# **Outputs.** CSV files, figures and surrogate weights in `OUT_DIR/CMPC_NB03_<tag>/`, packed into a zip.
#
# **Interrupted runs.** After each model and variant the run saves `checkpoint.pkl` in its folder. If the run stops (power cut, crash), set `RESUME_DIR` to that folder (for example `"results/CMPC_NB03_FULL_20261016_190000"`) and run the notebook again from the top: finished models and variants are loaded and skipped; only the one in progress is repeated. Keep the other settings unchanged.
#
# **Modes.** `QUICK = True` is a smoke test (short training, one surrogate, two speeds, two payloads, few episodes). `QUICK = False` is the long run.
#
# **Requirements.** numpy, scipy, pandas, matplotlib, osqp, joblib (optional, for `N_JOBS > 1`).

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
OUT_DIR = "results"
RESUME_DIR = None                  # to continue an interrupted run: its folder, e.g. "results/CMPC_NB03_FULL_20261016_190000"
N_JOBS = max(1, (os.cpu_count() or 2) - 1)

# ---- plant / controller
NJ = 2                             # joints
SEVERITY = 1.0
FS = 2000; dt = 1.0 / FS
T_C = 0.002
N_H = 25
TAU_MAX = np.array([40.0, 20.0])   # [N m] per joint
Q_POS, Q_VEL, Q_INT, R_TAU = 1e4, 1.0, 1e5, 2e-4   # R of the Delta scaled by (5 N m / 40 N m)^2: same closed-loop bandwidth

# ---- tube MPC
EPS_Q = 1.0e-3                     # [rad] joint tracking constraint
N_TAIL = N_H
TORQUE_MARGIN = 0.1*TAU_MAX        # tasks whose reference torque exceeds TAU_MAX - margin are excluded from D

# ---- task distribution D
SCALES = (1.0, 1.25, 1.6, 2.2, 3.0)    # time scaling of the pick-and-place cycle (1.0 = fastest)
PAYLOADS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
X0_BOX = 2.0e-4
SIGMA_TAU = 0.05                   # [N m] actuator torque noise per control period (about 1/400 of the torque range, as for the Delta)

# ---- surrogates (trained here)
TRAIN_SIZES = {"MLP": 35, "HYBRID": 35, "PINN": 35}
EPOCHS, HIDDEN, BATCH, LR = 150, 64, 256, 2e-3
MODELS = [("Nominal", "NOMINAL", 0, 0), ("MLP-35", "MLP", 35, 0), ("HYBRID-35", "HYBRID", 35, 0), ("PINN-35", "PINN", 35, 0)]

# ---- Algorithm 1 and variants (as NB02)
VARIANTS = ("const", "ref")
DELTA = 0.10
N_PILOT, N_CAL, N_TEST = 50, 200, 400
PILOT_ITERS, INFLATE = 1, 1.05
DELTAS_CHECK = (0.05, 0.10, 0.20)
N_SPLITS = 500
SEED = 2027

if QUICK:
    SCALES, PAYLOADS = (1.0, 1.6), (0.0, 3.0)
    MODELS = [("Nominal", "NOMINAL", 0, 0), ("HYBRID-35", "HYBRID", 35, 0)]
    EPOCHS = 15
    N_PILOT, N_CAL, N_TEST, N_SPLITS = 16, 24, 24, 100

STEP = int(round(T_C / dt))
TAG = ("QUICK_" if QUICK else "FULL_") + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESUME_DIR or os.path.join(OUT_DIR, f"CMPC_NB03_{TAG}"); os.makedirs(RUN_DIR, exist_ok=True)
def out(name): return os.path.join(RUN_DIR, name)
CONFIG = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in dict(globals()).items()
          if k.isupper() and isinstance(v, (int, float, str, tuple, list, bool, dict, np.ndarray))}
json.dump(CONFIG, open(out("config.json"), "w"), indent=1, default=str)
print("run folder:", RUN_DIR, "| workers:", N_JOBS)

# %% [markdown]
# ## Two-link arm: model, unmodeled dynamics and plant
#
# Joint angles `q1` (from the horizontal) and `q2` (elbow), links of 0.40 and 0.30 m and 3 and 2 kg, rotor inertias, payload `m` at the tip. The rigid-body model is linear in 6 base parameters, `τ = Y(q, q̇, q̈) θ(m)` with `θ(m) = θ_0 + m θ_L`:
# `M11 = θ1 + 2θ2 cos q2`, `M12 = θ3 + θ2 cos q2`, `M22 = θ3 + θ6`, gravity `θ4 cos q1 + θ5 cos(q1+q2)` and `θ5 cos(q1+q2)`.
# The **identified model** uses `θ_0` with errors of a few percent (as an identification would leave) and the exact payload term. The plant adds `τ_u`: viscous and Coulomb friction, coupling between the joint speeds and a configuration-dependent torque, at severity `s = 1`.

# %%
L1, L2, M1, M2, G0 = 0.40, 0.30, 3.0, 2.0, 9.81
LC1, LC2 = L1/2, L2/2; I1, I2 = M1*L1**2/12, M2*L2**2/12; JM1, JM2 = 0.05, 0.02
M_BASE = M1 + M2                                     # mass used to normalize the payload in the reference features
THETA0 = np.array([I1 + I2 + M1*LC1**2 + M2*(L1**2 + LC2**2) + JM1, M2*L1*LC2, I2 + M2*LC2**2,
                   (M1*LC1 + M2*L1)*G0, M2*LC2*G0, JM2])
THETA_L = np.array([L1**2 + L2**2, L1*L2, L2**2, L1*G0, L2*G0, 0.0])
THETA_ID = THETA0*np.array([1.04, 0.95, 1.05, 0.97, 1.03, 1.10])
UNMODELED = dict(b_v=np.array([0.30, 0.10]), f_c=np.array([0.40, 0.12]), eps=0.02, c_cor=np.array([0.10, 0.03]),
                 k_pos=np.array([0.30, 0.10]), w_pos=3.0)

def theta(m, base=THETA0): return base + float(m)*THETA_L

def regressor(q, qd, qdd):
    c, s = np.cos(q[1]), np.sin(q[1]); c1, c12 = np.cos(q[0]), np.cos(q[0] + q[1])
    return np.array([[qdd[0], 2*c*qdd[0] + c*qdd[1] - s*(2*qd[0]*qd[1] + qd[1]**2), qdd[1], c1, c12, 0.0],
                     [0.0, c*qdd[0] + s*qd[0]**2, qdd[0] + qdd[1], 0.0, c12, qdd[1]]])

def regressor_batch(Q, Qd, Qdd):
    """Y for N samples: (N, 2, 6)."""
    c, s = np.cos(Q[:, 1]), np.sin(Q[:, 1]); c1, c12 = np.cos(Q[:, 0]), np.cos(Q[:, 0] + Q[:, 1]); z = np.zeros(len(Q))
    r1 = np.stack([Qdd[:, 0], 2*c*Qdd[:, 0] + c*Qdd[:, 1] - s*(2*Qd[:, 0]*Qd[:, 1] + Qd[:, 1]**2), Qdd[:, 1], c1, c12, z], 1)
    r2 = np.stack([z, c*Qdd[:, 0] + s*Qd[:, 0]**2, Qdd[:, 0] + Qdd[:, 1], z, c12, Qdd[:, 1]], 1)
    return np.stack([r1, r2], 1)

def tau_unmodeled(q, qd, sev):
    u = UNMODELED
    return sev*(u["b_v"]*qd + u["f_c"]*np.tanh(qd/u["eps"])
                + u["c_cor"]*(qd**2*np.cos(q) + qd[::-1]*qd)
                + u["k_pos"]*np.sin(u["w_pos"]*q))

def mass_matrix(q, th):
    c = np.cos(q[1])
    return np.array([[th[0] + 2*th[1]*c, th[2] + th[1]*c], [th[2] + th[1]*c, th[2] + th[5]]])

def plant_qdd(q, qd, tau, p, m, sev):
    th = theta(m)
    return np.linalg.solve(mass_matrix(q, th), tau - regressor(q, qd, np.zeros(2)) @ th - tau_unmodeled(q, qd, sev))

def F_true(q, qd, qdd, m):
    return regressor(q, qd, qdd) @ theta(m) + tau_unmodeled(q, qd, SEVERITY)

def F_nominal(q, qd, qdd, m):
    return regressor(q, qd, qdd) @ theta(m, THETA_ID)

def fk(q):
    return np.array([L1*np.cos(q[0]) + L2*np.cos(q[0] + q[1]), L1*np.sin(q[0]) + L2*np.sin(q[0] + q[1])])

def ik_vec(P):
    x, y = P[:, 0], P[:, 1]
    c2 = np.clip((x**2 + y**2 - L1**2 - L2**2) / (2*L1*L2), -1, 1); q2 = np.arccos(c2)
    q1 = np.arctan2(y, x) - np.arctan2(L2*np.sin(q2), L1 + L2*np.cos(q2))
    return np.stack([q1, q2], 1)

# %% [markdown]
# ## References: pick-and-place cycle in the vertical plane
#
# Quintic rest-to-rest segments in Cartesian space through six waypoints (continuous acceleration), mapped to the joints by inverse kinematics (elbow up), at time scaling `sc`.

# %%
WAYPOINTS = [np.array(p) for p in ([0.40, 0.00], [0.40, 0.20], [0.10, 0.45], [-0.20, 0.35], [-0.20, 0.10], [0.10, 0.45], [0.40, 0.00])]
BASE_T = np.concatenate([[0.0], np.cumsum([0.35, 0.55, 0.45, 0.35, 0.55, 0.55])])

def arm_reference(scale):
    nt = BASE_T*scale; n = int(round(nt[-1]*FS)) + 1; t = np.arange(n)*dt
    seg = np.clip(np.searchsorted(nt[1:], t, side="right"), 0, len(WAYPOINTS) - 2)
    tau = np.clip((t - nt[seg]) / (nt[seg+1] - nt[seg]), 0, 1); s = 6*tau**5 - 15*tau**4 + 10*tau**3
    W = np.array(WAYPOINTS); P = W[seg] + (W[seg+1] - W[seg])*s[:, None]
    q = ik_vec(P); qd = np.gradient(q, dt, axis=0, edge_order=2); qdd = np.gradient(qd, dt, axis=0, edge_order=2)
    return dict(t=t, P=P, q=q, qd=qd, qdd=qdd, N=n)

REFS = {sc: arm_reference(sc) for sc in SCALES}
for sc, R in REFS.items():
    print(f"scale {sc}: {R['t'][-1]:.2f} s | peak |qd| {np.abs(R['qd']).max():.2f} rad/s | peak |qdd| {np.abs(R['qdd']).max():.1f} rad/s^2 "
          f"| q1 in [{R['q'][:,0].min():.2f}, {R['q'][:,0].max():.2f}], q2 in [{R['q'][:,1].min():.2f}, {R['q'][:,1].max():.2f}] rad")

# %% [markdown]
# ## Training data and surrogates (numpy)
#
# Training trajectories: quintic rest-to-rest joint segments between random configurations (over a box that contains the task references with a margin) with random durations and a random payload per trajectory, sampled at 100 Hz; targets are the plant inverse dynamics plus measurement noise. All three surrogates are two-hidden-layer SiLU networks with input `[q, q̇, q̈, m, p(q)]`, trained by Adam on the mean squared torque error (normalized):
# * MLP: `τ = net(x)`;
# * HYBRID: `τ = F_nominal(ξ) + net(x)` (learned residual of the identified model);
# * PINN: `τ = Y(q, q̇, q̈) (θ_ID(m) + diag(|θ_ID(0)|) net(x))`, i.e. state-dependent parameters of the regressor, initialized at the identified model.

# %%
Q_BOX = None
def training_set(n_traj, seed, fs_s=100):
    rng = np.random.default_rng([seed, 99, n_traj]); X, Y = [], []
    qa = np.vstack([R["q"] for R in REFS.values()]) if Q_BOX is None else Q_BOX
    lo, hi = qa.min(0) - 0.25, qa.max(0) + 0.25
    for _ in range(n_traj):
        m = rng.uniform(0, max(PAYLOADS)); wp = rng.uniform(lo, hi, (7, NJ)); durs = rng.uniform(0.25, 0.8, 6)
        nt = np.concatenate([[0], np.cumsum(durs)]); t = np.arange(0, nt[-1], 1/fs_s)
        seg = np.clip(np.searchsorted(nt[1:], t, side="right"), 0, 5); tau = (t - nt[seg]) / durs[seg]
        s = 6*tau**5 - 15*tau**4 + 10*tau**3; sd = (30*tau**4 - 60*tau**3 + 30*tau**2) / durs[seg]
        sdd = (120*tau**3 - 180*tau**2 + 60*tau) / durs[seg]**2; dq = wp[seg + 1] - wp[seg]
        q = wp[seg] + dq*s[:, None]; qd = dq*sd[:, None]; qdd = dq*sdd[:, None]
        for i in range(len(t)):
            X.append(np.concatenate([q[i], qd[i], qdd[i], [m], fk(q[i])])); Y.append(F_true(q[i], qd[i], qdd[i], m))
    X, Y = np.array(X), np.array(Y)
    return X, Y + rng.normal(0, 0.02, Y.shape)

def silu(z): return z / (1.0 + np.exp(-z))
def dsilu(z):
    s = 1.0 / (1.0 + np.exp(-z)); return s + z*s*(1 - s)

class ArmNet:
    """Two-hidden-layer SiLU network; kind in {'MLP', 'HYBRID', 'PINN'}."""
    def __init__(self, kind, n_in, rng):
        self.kind = kind; n_out = 6 if kind == "PINN" else NJ; h = HIDDEN
        self.W = [rng.normal(0, 1/np.sqrt(n_in), (h, n_in)), np.zeros(h), rng.normal(0, 1/np.sqrt(h), (h, h)), np.zeros(h),
                  rng.normal(0, 0.1/np.sqrt(h), (n_out, h)), np.zeros(n_out)]
    def forward(self, Xs):
        W1, b1, W2, b2, W3, b3 = self.W
        z1 = Xs @ W1.T + b1; h1 = silu(z1); z2 = h1 @ W2.T + b2; h2 = silu(z2)
        return h2 @ W3.T + b3, (Xs, z1, h1, z2, h2)
    def backward(self, g, cache):
        Xs, z1, h1, z2, h2 = cache; W1, b1, W2, b2, W3, b3 = self.W
        gW3 = g.T @ h2; gb3 = g.sum(0); g2 = (g @ W3)*dsilu(z2)
        gW2 = g2.T @ h1; gb2 = g2.sum(0); g1 = (g2 @ W2)*dsilu(z1)
        return [g1.T @ Xs, g1.sum(0), gW2, gb2, gW3, gb3]
    # physics part
    def tau(self, X, out):
        q, qd, qdd, m = X[:, :NJ], X[:, NJ:2*NJ], X[:, 2*NJ:3*NJ], X[:, 3*NJ]
        if self.kind == "MLP": return out*self.y_std + self.y_mean, None
        Yr = regressor_batch(q, qd, qdd)
        th_id = THETA_ID[None, :] + m[:, None]*THETA_L[None, :]
        if self.kind == "HYBRID": return np.einsum("nij,nj->ni", Yr, th_id) + out*self.r_std, None
        return np.einsum("nij,nj->ni", Yr, th_id + out*self.th_scale), Yr

def train_surrogate(kind, n_traj, seed):
    X, Y = training_set(n_traj, seed); rng = np.random.default_rng([seed, 7])
    net = ArmNet(kind, X.shape[1], rng)
    net.x_mean, net.x_std = X.mean(0), X.std(0) + 1e-9; net.y_mean, net.y_std = Y.mean(0), Y.std(0)
    Fn = np.array([F_nominal(x[:NJ], x[NJ:2*NJ], x[2*NJ:3*NJ], x[3*NJ]) for x in X])
    net.r_std = (Y - Fn).std(0); net.th_scale = np.abs(THETA_ID)
    Xs = (X - net.x_mean) / net.x_std; n = len(X)
    mt = [np.zeros_like(w) for w in net.W]; vt = [np.zeros_like(w) for w in net.W]; it = 0; hist = []
    for ep in range(EPOCHS):
        perm = rng.permutation(n); lr = LR*0.5*(1 + np.cos(np.pi*ep/EPOCHS)); tot = 0.0
        for b0 in range(0, n, BATCH):
            ib = perm[b0:b0 + BATCH]; out_, cache = net.forward(Xs[ib])
            tau, Yr = net.tau(X[ib], out_); e = (tau - Y[ib]) / net.y_std; tot += np.sum(e**2)
            gt = 2*e / net.y_std / len(ib)                               # dL/dtau
            if kind == "MLP": g = gt*net.y_std
            elif kind == "HYBRID": g = gt*net.r_std
            else: g = np.einsum("nij,ni->nj", Yr, gt)*net.th_scale
            grads = net.backward(g, cache); it += 1
            for i_, (w, gw) in enumerate(zip(net.W, grads)):
                mt[i_] = 0.9*mt[i_] + 0.1*gw; vt[i_] = 0.999*vt[i_] + 0.001*gw**2
                w -= lr*(mt[i_]/(1 - 0.9**it)) / (np.sqrt(vt[i_]/(1 - 0.999**it)) + 1e-8)
        hist.append(tot / n)
    return net, hist

class ArmSurrogate:
    def __init__(self, net): self.net = net
    def __call__(self, q, qd, qdd, m):
        x = np.concatenate([q, qd, qdd, [m], fk(q)])[None, :]
        out_, _ = self.net.forward((x - self.net.x_mean) / self.net.x_std)
        return self.net.tau(x, out_)[0][0]

t0 = time.time(); FMODELS, TRAIN_LOG = {}, []
Xv, Yv = training_set(20, 12345)                                   # held-out random trajectories
for label, kind, size, seed in MODELS:
    if kind == "NOMINAL":
        FMODELS[label] = F_nominal; net = None
    else:
        net, hist = train_surrogate(kind, size, seed); FMODELS[label] = ArmSurrogate(net)
        np.savez(out(f"arm_{kind}_{size}traj_seed{seed}.npz"), *net.W, x_mean=net.x_mean, x_std=net.x_std,
                 y_mean=net.y_mean, y_std=net.y_std, r_std=net.r_std, th_scale=net.th_scale)
    F = FMODELS[label]
    Pv = np.array([F(x[:NJ], x[NJ:2*NJ], x[2*NJ:3*NJ], x[3*NJ]) for x in Xv])
    TRAIN_LOG.append(dict(model=label, kind=kind, n_traj=size, rmse_heldout_Nm=float(np.sqrt(np.mean((Pv - Yv)**2))),
                          final_loss=(hist[-1] if net is not None else np.nan)))
    print(f"{label}: held-out torque RMSE {TRAIN_LOG[-1]['rmse_heldout_Nm']:.3f} N m | {(time.time()-t0)/60:.1f} min")
pd.DataFrame(TRAIN_LOG).to_csv(out("CMPC_NB03_training.csv"), index=False)

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

NX = 3*NJ
def ltv_along_reference(F, R, m):
    idx = np.arange(0, R["N"], STEP); K = len(idx)
    U = np.zeros((K, NJ)); A = np.zeros((K, NX, NX)); B = np.zeros((K, NX, NJ)); minlam = np.inf
    for k, i in enumerate(idx):
        q, qd, qdd = R["q"][i], R["qd"][i], R["qdd"][i]
        U[k] = F(q, qd, qdd, m)
        Fq  = jac_fd(lambda v: F(v, qd, qdd, m), q, 1e-6)
        Fqd = jac_fd(lambda v: F(q, v, qdd, m), qd, 1e-6)
        Fa  = jac_fd(lambda v: F(q, qd, v, m), qdd, 1e-4)
        minlam = min(minlam, np.linalg.eigvals(0.5*(Fa + Fa.T)).real.min())
        Mi = np.linalg.inv(Fa)
        n2 = 2*NJ; Ac = np.zeros((n2, n2)); Bc = np.zeros((n2, NJ))
        Ac[:NJ, NJ:n2] = np.eye(NJ); Ac[NJ:n2, :NJ] = -Mi @ Fq; Ac[NJ:n2, NJ:n2] = -Mi @ Fqd; Bc[NJ:n2] = Mi
        Md = expm(np.block([[Ac, Bc], [np.zeros((NJ, 3*NJ))]]) * T_C)
        A[k, :n2, :n2], B[k, :n2] = Md[:n2, :n2], Md[:n2, n2:]
        A[k, n2:, :NJ] = T_C*np.eye(NJ); A[k, n2:, n2:] = np.eye(NJ)
    return dict(idx=idx, U=U, A=A, B=B, minlam=minlam)

def mpc_gains(L):
    K = len(L["idx"]); qd = [Q_POS]*NJ + [Q_VEL]*NJ + [Q_INT]*NJ
    P = solve_discrete_are(L["A"][-1], L["B"][-1], np.diag(qd), R_TAU*np.eye(NJ))
    Wx = np.diag(np.tile(np.sqrt(qd), N_H)); Wx[-NX:, -NX:] = np.linalg.cholesky(P + 1e-12*np.eye(NX)).T
    W2 = Wx.T @ Wx; Kg = np.zeros((K, NJ, NX))
    for k in range(K):
        ks = np.minimum(np.arange(k, k + N_H), K - 1); A, B = L["A"][ks], L["B"][ks]
        Phi = np.zeros((N_H*NX, NX)); G = np.zeros((N_H*NX, NJ*N_H)); Ak = np.eye(NX)
        for j in range(N_H):
            Ak = A[j] @ Ak; Phi[j*NX:(j+1)*NX] = Ak; G[j*NX:(j+1)*NX, NJ*j:NJ*j+NJ] = B[j]
            if j > 0: G[j*NX:(j+1)*NX, :NJ*j] = A[j] @ G[(j-1)*NX:j*NX, :NJ*j]
        GtW = G.T @ W2
        Kg[k] = -cho_solve(cho_factor(GtW @ G + R_TAU*np.eye(NJ*N_H)), GtW @ Phi)[:NJ]
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

EQ = np.hstack([np.eye(NJ), np.zeros((NJ, 2*NJ))])
J_ALL = N_H + N_TAIL

def unit_tightening(Phi, Kg, S9):
    """T[k, j, r] = h_{R_{k,j}}(a_r) for q = 1 with time-varying W_k = {|w_l| <= S9[k, l]}; rows 0..NJ-1 input, NJ..2NJ-1 joint error."""
    K = len(Phi); T = np.zeros((K, J_ALL, 2*NJ), np.float32); kk = np.arange(K)
    for j in range(1, J_ALL):
        a = np.concatenate([Kg[np.minimum(kk + j, K - 1)], np.broadcast_to(EQ, (K, NJ, NX))], axis=1)
        v = a.copy(); acc = np.zeros((K, 2*NJ))
        for i in range(j - 1, -1, -1):           # v = a Phi(k+j, k+i+1); add h of W_{k+i}
            ii = np.minimum(kk + i, K - 1)
            acc += np.einsum("kri,ki->kr", np.abs(v), S9[ii])
            v = v @ Phi[ii]
        T[:, j] = acc
    return T

# %% [markdown]
# ## Tube MPC with restricted constraints, with the cause of every fallback
#
# As in NB01. When `c = 0` violates a tightened constraint, the controller first checks the tightened sets: `x_empty` if the tube on the joint error is wider than `ε_q` at some prediction step (no `c` can help), `u_empty` if the torque tube is wider than `τ_max`, `x_now` if the current error already exceeds the tightened bound. Otherwise the QP is solved; `qp_infeasible` and `solver` count the remaining failures. In every case the fallback is `c = 0`.

# %%
class TubeMPC:
    def __init__(self, P_, q):
        self.P, self.q = P_, q
        L = P_["L"]; self.K = len(L["idx"]); self.U = L["U"]; self.B = L["B"]; self.Phi = P_["Phi"]; self.Kg = P_["Kg"]
        self.n_qp = 0; self.fb = dict(x_empty=0, u_empty=0, x_now=0, qp_infeasible=0, solver=0)
    @property
    def n_fallback(self): return sum(self.fb.values())
    def bounds(self, k):
        ks = np.minimum(np.arange(k, k + J_ALL), self.K - 1)
        tight = np.zeros((J_ALL, 2*NJ)) if self.q == 0 else self.q * self.P["T"][k].astype(float)
        ur = self.U[ks]
        lo_u = -TAU_MAX - ur + tight[:, :NJ]; hi_u = TAU_MAX - ur - tight[:, :NJ]
        hi_x = EPS_Q - tight[:, NJ:]
        valid = np.arange(J_ALL) + k < self.K
        return ks, lo_u, hi_u, hi_x, valid
    def control(self, k, x):
        ks, lo_u, hi_u, hi_x, valid = self.bounds(k)
        z = x.copy(); ok = True
        for j in range(J_ALL):
            if not valid[j]: break
            u = self.Kg[ks[j]] @ z; dq = z[:NJ]
            if np.any(u < lo_u[j]) or np.any(u > hi_u[j]) or np.any(np.abs(dq) > hi_x[j]): ok = False; break
            z = self.Phi[ks[j]] @ z
        if ok: return self.Kg[k] @ x, 0
        if np.any(hi_x[valid] < 0): self.fb["x_empty"] += 1; return self.Kg[k] @ x, 2
        if np.any(np.abs(x[:NJ]) > hi_x[0]): self.fb["x_now"] += 1; return self.Kg[k] @ x, 2    # current error already outside
        if np.any(lo_u[valid] > hi_u[valid]): self.fb["u_empty"] += 1; return self.Kg[k] @ x, 2
        return self._qp(k, x, ks, lo_u, hi_u, hi_x, valid)
    def _qp(self, k, x, ks, lo_u, hi_u, hi_x, valid):
        self.n_qp += 1; nc = NJ*N_H
        z0 = x.copy(); G = np.zeros((NX, nc)); rows = []; lo = []; hi = []
        for j in range(J_ALL):
            if not valid[j]: break
            Kj = self.Kg[ks[j]]
            Au = Kj @ G
            if j < N_H: Au[:, NJ*j:NJ*j+NJ] += np.eye(NJ)
            bu = Kj @ z0
            rows.append(Au); lo.append(lo_u[j] - bu); hi.append(hi_u[j] - bu)
            rows.append(EQ @ G); bx = EQ @ z0; lo.append(-hi_x[j] - bx); hi.append(hi_x[j] - bx)
            z0 = self.Phi[ks[j]] @ z0; G = self.Phi[ks[j]] @ G
            if j < N_H: G[:, NJ*j:NJ*j+NJ] += self.B[ks[j]]
        Acon = sparse.csc_matrix(np.vstack(rows)); lo = np.concatenate(lo); hi = np.concatenate(hi)
        prob = osqp.OSQP(); prob.setup(sparse.eye(nc, format="csc")*2.0, np.zeros(nc), Acon, lo, hi,
                                       verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=4000, polish=True)
        res = prob.solve(); st = str(res.info.status)
        if st not in ("solved", "solved inaccurate"):
            self.fb["qp_infeasible" if "infeasible" in st else "solver"] += 1; return self.Kg[k] @ x, 2
        return self.Kg[k] @ x + res.x[:NJ], 1

# %% [markdown]
# ## Episodes and the task worker
#
# An episode is drawn from its own seed, so the same id gives the same task, initial error and noise for every variant of a model. `task_worker` runs a list of episodes of one task in a worker process; it keeps the off-line data of the task between calls and recomputes the tightening only when the normalizers change. It also checks the task off line at each tube size it is given: `x_empty_task` (the joint-error tube exceeds `ε_q` somewhere, so the tightened problem is empty at those steps), `uref_out_task` (the reference torque plus the torque tube exceeds `τ_max` somewhere) and the torque budget lost to tightening.

# %%
def all_tasks():
    return [(sc, m) for sc in SCALES for m in PAYLOADS]

def draw_episode(eid, tasks):
    rng = np.random.default_rng([SEED, eid])
    sc, m = tasks[rng.integers(len(tasks))]
    x0 = rng.uniform(-X0_BOX, X0_BOX, NJ)
    K = len(REFS[sc]["q"][::STEP]); noise = rng.normal(0.0, SIGMA_TAU, (K, NJ))
    return dict(eid=eid, sc=sc, m=m, dq0=x0, noise=noise)

def run_episode(P_, ep, q_tube, keep_trace=False):
    sc, m = ep["sc"], ep["m"]; R = REFS[sc]; L = P_["L"]
    ctrl = TubeMPC(P_, q_tube); K = len(L["idx"]); Kv = P_["Kv"]
    q = R["q"][0] + ep["dq0"]; qd = R["qd"][0].copy(); zeta = np.zeros(NJ)
    X = np.zeros((K + 1, NX)); Uapp = np.zeros((K, NJ)); modes = np.zeros(K, int); viol_tau = 0
    Pe = np.zeros((R["N"], R["P"].shape[1]))
    for k in range(K):
        i = L["idx"][k]
        x = np.concatenate([q - R["q"][i], qd - R["qd"][i], zeta]); X[k] = x
        du, modes[k] = ctrl.control(k, x)
        tau_cmd = L["U"][k] + du
        if np.any(np.abs(tau_cmd) > TAU_MAX + 1e-9): viol_tau += 1
        tau = np.clip(tau_cmd, -TAU_MAX, TAU_MAX); Uapp[k] = tau - L["U"][k]
        tau_pl = tau + ep["noise"][k]
        zeta = zeta + T_C*x[:NJ]
        for s_ in range(STEP):
            ii = i + s_
            if ii >= R["N"]: break
            p = fk(q); Pe[ii] = p
            qd = qd + plant_qdd(q, qd, tau_pl, p, m, SEVERITY)*dt; q = q + qd*dt
        if not np.all(np.isfinite(q)) or np.abs(q - R["q"][min(i + STEP, R["N"] - 1)]).max() > 0.5:
            return dict(eid=ep["eid"], sc=sc, m=m, q_tube=q_tube, unstable=True), None
    iN = min(L["idx"][-1] + STEP, R["N"] - 1)
    X[K] = np.concatenate([q - R["q"][iN], qd - R["qd"][iN], zeta])
    W = (X[1:] - np.einsum("kij,kj->ki", L["A"], X[:-1]) - np.einsum("kij,kj->ki", L["B"], Uapp))[:Kv]
    e_cart = 1000*np.linalg.norm(R["P"][:len(Pe)] - Pe, axis=1)
    res = dict(eid=ep["eid"], sc=sc, m=m, q_tube=q_tube, unstable=False, max_abs_w_zeta=np.abs(W[:, 2*NJ:]).max(),
               max_dq=np.abs(X[:-1, :NJ]).max(), viol_x=int((np.abs(X[:-1, :NJ]) > EPS_Q).any(axis=1).sum()),
               viol_tau=viol_tau, n_qp=ctrl.n_qp, n_fallback=ctrl.n_fallback, K=K,
               **{f"fb_{c}": v for c, v in ctrl.fb.items()},
               rmse_um=1000*np.sqrt(np.mean(e_cart**2)), max_err_um=1000*e_cart.max())
    extra = dict(W=W[:, :2*NJ])
    if keep_trace: extra.update(X=X, modes=modes)
    return res, extra

def _worker_cache():
    mod = sys.modules.get("_cmpc_nb03_cache")
    if mod is None:
        mod = types.ModuleType("_cmpc_nb03_cache"); mod.C = {}; sys.modules["_cmpc_nb03_cache"] = mod
    return mod.C

def prepare_task(label, sc, m):
    t0 = time.time(); R = REFS[sc]; L = ltv_along_reference(FMODELS[label], R, m)
    Kg = mpc_gains(L); Phi = L["A"] + np.einsum("kij,kjl->kil", L["B"], Kg)
    res = np.array([F_true(R["q"][i], R["qd"][i], R["qdd"][i], m) for i in L["idx"]]) - L["U"]
    return dict(L=L, Kg=Kg, Phi=Phi, Feat=ref_features(R, L["idx"], m), Kv=valid_steps(R, L["idx"]),
                t_prep=time.time() - t0, T=None, S=None, sig_id=None, res_rms=float(np.sqrt(np.mean(res**2))))

def task_check(P_, q):
    T = P_["T"]; U = P_["L"]["U"]; K = len(U)
    kj = np.arange(K)[:, None] + np.arange(J_ALL)[None, :]; valid = kj < K; ks = np.minimum(kj, K - 1)
    tu = q*T[:, :, :NJ].astype(float); tx = q*T[:, :, NJ:].astype(float)
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
        P_["T"] = unit_tightening(P_["Phi"], P_["Kg"], np.hstack([S, np.zeros((len(S), NJ))])); P_["S"] = S; P_["sig_id"] = spec["id"]
    info = dict(model=label, sc=sc, m=m, t_prep_s=P_["t_prep"], min_eig_Fqdd=P_["L"]["minlam"], max_abs_uref=float(np.abs(P_["L"]["U"]).max()), u_ratio=float((np.abs(P_["L"]["U"]) / (TAU_MAX - TORQUE_MARGIN)).max()),
                res_rms_ref_Nm=P_["res_rms"])
    checks = {q: task_check(P_, q) for q in sorted({q for _, q in jobs})} if spec is not None else {}
    out_ = []
    for eid, q in jobs:
        res, extra = run_episode(P_, draw_episode(eid, tasks), q, keep_trace=eid in trace_ids)
        if extra is not None:
            if spec is not None:
                res["score"] = float(np.max(np.abs(extra["W"]) / P_["S"][:len(extra["W"])]))
                if eid in trace_ids: extra["tube1"] = q*P_["T"][:, 1, NJ:].max(axis=1).astype(float); extra["S"] = P_["S"]
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
    beta = np.zeros((Fall.shape[1], 2*NJ)); floor = np.zeros(2*NJ); r2 = np.zeros(2*NJ)
    for i in range(2*NJ):
        y = Eall[:, i]; b, _ = nnls(Fs, y); beta[:, i] = b / cs
        floor[i] = 0.2*y.mean(); yh = np.maximum(Fall @ beta[:, i], floor[i])
        r2[i] = 1 - np.sum((y - yh)**2) / np.sum((y - y.mean())**2)
    return sig_c, beta, floor, r2

# %% [markdown]
# ## Algorithm 1 for each model and variant
#
# Groups: one group for `const` and `ref`; one group per time scaling for `ref+class`. In each group: pilot quantile at `q = 0`, `PILOT_ITERS` fixed-point iterations under `C_q`, inflation, then calibration on fresh episodes (`q̂_g`, certified if `q̂_g ≤ q̃_g`) and test episodes, all under the deployed tube sizes. 

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
    # ---- off-line preparation of every grid task (in the workers) and admissible tasks, in grid order
    grid = all_tasks(); args = [(label, sc, m, [], None, grid) for sc, m in grid]
    infos = (Parallel(n_jobs=N_JOBS, backend="loky")(delayed(task_worker)(*a) for a in args) if N_JOBS > 1
             else [task_worker(*a) for a in args])
    infos = [i for i, _, _ in infos]
    if (label, "pilot") not in DONE: TASK_ROWS += infos
    umax = {(i["sc"], i["m"]): i["u_ratio"] for i in infos}
    tasks = [tk for tk in grid if umax[tk] <= 1.0]
    print(f"\n=== {label}: {len(tasks)} admissible tasks of {len(grid)} (excluded: {[tk for tk in grid if tk not in tasks]}) "
          f"| min eig F_qdd {min(i['min_eig_Fqdd'] for i in infos):.3f} | prep {(time.time()-t_m)/60:.1f} min")
    # ---- pilot at q = 0 (shared by all variants)
    pilot_ids = list(range(N_PILOT))
    pilot, _, _ = run_jobs(label, pilot_ids, lambda ep: 0.0, None, tasks)
    first_pilot = (label, "pilot") not in DONE; DONE.add((label, "pilot"))
    sig_c, beta, floor, r2 = fit_normalizers(pilot)
    for i in range(2*NJ):
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
                            fb_x_empty=float((ok.fb_x_empty > 0).mean()), fb_u_empty=float((ok.fb_u_empty > 0).mean()), fb_x_now=float((ok.fb_x_now > 0).mean()),
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

ep_df = pd.DataFrame(ALL_EP); ep_df.to_csv(out("CMPC_NB03_episodes.csv"), index=False)
sum_df = pd.DataFrame(SUMMARY); sum_df.to_csv(out("CMPC_NB03_summary.csv"), index=False)
cls_df = pd.DataFrame(CLASS_ROWS); cls_df.to_csv(out("CMPC_NB03_groups.csv"), index=False)
pd.DataFrame(CHECK_ROWS).to_csv(out("CMPC_NB03_task_checks.csv"), index=False)
pd.DataFrame(TASK_ROWS).drop_duplicates(subset=["model", "sc", "m"]).to_csv(out("CMPC_NB03_tasks.csv"), index=False)
pd.DataFrame(SIGMA_ROWS).to_csv(out("CMPC_NB03_normalizers.csv"), index=False)
pd.DataFrame(PILOT_LOG).to_csv(out("CMPC_NB03_pilot_iterations.csv"), index=False)
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
split_df = pd.DataFrame(rows); split_df.to_csv(out("CMPC_NB03_split_coverage.csv"), index=False)
cov_tab = split_df.groupby(["model", "variant", "group", "delta"]).agg(
    n=("n", "first"), mean_cov=("coverage", "mean"), p05=("coverage", lambda v: np.percentile(v, 5)),
    p95=("coverage", lambda v: np.percentile(v, 95)), beta_p05=("beta_lo", "first"), beta_p95=("beta_hi", "first")).reset_index()
cov_tab["target"] = 1 - cov_tab["delta"]; cov_tab.to_csv(out("CMPC_NB03_split_coverage_summary.csv"), index=False)
print(cov_tab.round(4).to_string(index=False))

# %% [markdown]
# ## Figures

# %%
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False})
PAL, MKS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"], ["o", "s", "D", "^", "v"]
COL = {m[0]: PAL[i % 5] for i, m in enumerate(MODELS)}; MK = {m[0]: MKS[i % 5] for i, m in enumerate(MODELS)}
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
fig.tight_layout(); fig.savefig(out("fig_nb03_score_speed.pdf")); fig.savefig(out("fig_nb03_score_speed.png"), dpi=200); plt.show()

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
fig.tight_layout(); fig.savefig(out("fig_nb03_class_coverage.pdf")); fig.savefig(out("fig_nb03_class_coverage.png"), dpi=200); plt.show()

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
fig.tight_layout(); fig.savefig(out("fig_nb03_tube_fallback.pdf")); fig.savefig(out("fig_nb03_tube_fallback.png"), dpi=200); plt.show()

# (d) one test episode under the `ref` variant: joint error and the time-varying one-step tube
keys = [k for k in TRACES if k[1] == "ref"] or list(TRACES)
if keys:
    tr = TRACES[keys[0]]; tt = np.arange(len(tr["modes"]))*T_C
    fig, ax = plt.subplots(2, 1, figsize=(3.4, 3.2), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    ax[0].plot(tt, 1e3*np.abs(tr["X"][:-1, :NJ]).max(axis=1), lw=0.8, label=r"$\max_i|\delta q_i|$")
    ax[0].plot(tt, 1e3*tr["tube1"], lw=0.8, label="tube, $j=1$")
    ax[0].axhline(1e3*EPS_Q, color="k", lw=0.8, label=r"$\varepsilon_q$"); ax[0].set_yscale("log")
    ax[0].set_ylabel("mrad"); ax[0].legend(frameon=False, fontsize=7); ax[0].set_title(f"{keys[0][0]}, {keys[0][1]}, sc {tr['sc']}, {tr['m']} kg", fontsize=8)
    ax[1].plot(tt, tr["modes"], lw=0.6); ax[1].set_yticks([0, 1, 2]); ax[1].set_yticklabels(["K x", "QP", "fallback"], fontsize=7)
    ax[1].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(out("fig_nb03_episode.pdf")); fig.savefig(out("fig_nb03_episode.png"), dpi=200); plt.show()

# %% [markdown]
# ## E3 — Certified tube versus the torque residual of each surrogate along the references
#
# For each surrogate: RMS of `F_true − F̂` along the admissible references (off line, no controller), against the certified one-step joint-error tube and the torque budget lost to tightening (Proposition 1 predicts that the tube scales with the residual divided by the inertia bound `m_0`).

# %%
tk_df = pd.DataFrame(TASK_ROWS).drop_duplicates(subset=["model", "sc", "m"])
tk_df = tk_df[tk_df.u_ratio <= 1.0]
e3 = tk_df.groupby("model").agg(res_rms_ref_Nm=("res_rms_ref_Nm", "median"), m0=("min_eig_Fqdd", "min")).reset_index()
e3 = e3.merge(sum_df[["model", "variant", "q_tilde", "tube_pos_j1_med_urad", "budget_lost_med_Nm", "coverage_test", "certified"]], on="model")
e3.to_csv(out("CMPC_NB03_E3.csv"), index=False); print(e3.round(4).to_string(index=False))
fig, ax = plt.subplots(1, len(VARIANTS), figsize=(3.3*len(VARIANTS), 2.6), squeeze=False)
for a_, variant in zip(ax[0], VARIANTS):
    d = e3[e3.variant == variant]
    for _, r in d.iterrows():
        a_.plot(r.res_rms_ref_Nm, r.tube_pos_j1_med_urad, MK[r.model], color=COL[r.model], ms=7, label=r.model)
    a_.set_xlabel("torque residual along the references (N m, RMS)"); a_.set_title(variant)
ax[0][0].set_ylabel("one-step joint-error tube (µrad)"); ax[0][0].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(out("fig_nb03_E3.pdf")); fig.savefig(out("fig_nb03_E3.png"), dpi=200); plt.show()

# %% [markdown]
# ## Pack the run folder

# %%
zip_path = shutil.make_archive(RUN_DIR, "zip", root_dir=RUN_DIR)
print("results:", zip_path, f"({os.path.getsize(zip_path)/1024:.0f} kB)")
print(sorted(os.listdir(RUN_DIR)))
