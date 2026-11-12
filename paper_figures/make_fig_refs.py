"""Reference trajectories of the two case studies: python3 make_fig_refs.py <output folder>"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = sys.argv[1]
plt.rcParams.update({"font.size":9, "axes.labelsize":9.5, "legend.fontsize":8, "xtick.labelsize":8.5, "ytick.labelsize":8.5,
                     "font.family":"serif", "mathtext.fontset":"cm", "axes.spines.top":False, "axes.spines.right":False,
                     "axes.edgecolor":"#555", "axes.linewidth":0.6})
q5 = lambda t: 6*t**5 - 15*t**4 + 10*t**3
# ---- Delta: Adept cycle (sin^4 dip), circle (4 loops) and figure eight (3 loops), as in NB02 and NB05
W = np.array([[0, 0, .5], [.1, .1, .55], [.1, -.1, .55], [-.1, -.1, .55], [-.1, .1, .55], [.1, .1, .55], [0, 0, .5]])
tau = np.linspace(0, 1, 400); A = []
for i in range(6):
    P = W[i] + (W[i+1] - W[i])*q5(tau)[:, None]; P[:, 2] -= 0.05*np.sin(np.pi*tau)**4; A.append(P)
A = np.vstack(A); ph = np.linspace(0, 2*np.pi, 400)
C = np.column_stack([0.10*(np.cos(ph) - 1), 0.10*np.sin(ph)]); E = np.column_stack([0.12*np.sin(ph), 0.12*np.sin(ph)*np.cos(ph)])
# ---- arm: six waypoints in the vertical plane
WA = np.array([[.4, 0], [.4, .2], [.1, .45], [-.2, .35], [-.2, .1], [.1, .45], [.4, 0]])
AR = np.vstack([WA[i] + (WA[i+1] - WA[i])*q5(tau)[:, None] for i in range(6)])
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.35), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
ax[0].plot(1e3*A[:, 0], 1e3*A[:, 1], c="#2a78d6", lw=1.4, label="Adept cycle")
ax[0].plot(1e3*C[:, 0], 1e3*C[:, 1], c="#eb6834", lw=1.1, ls="--", label="circle (E5)")
ax[0].plot(1e3*E[:, 0], 1e3*E[:, 1], c="#1baf7a", lw=1.1, ls=":", label="figure eight (E5)")
ax[0].plot(1e3*W[1:5, 0], 1e3*W[1:5, 1], "o", c="#2a78d6", ms=4)
ax[0].set_aspect("equal"); ax[0].set_xlabel("$x$ (mm)"); ax[0].set_ylabel("$y$ (mm)"); ax[0].set_title("(a) Delta, top view", fontsize=9, loc="left")
ax[0].set_xlim(-230, 140); ax[0].set_ylim(-250, 130); ax[0].legend(frameon=False, loc="lower right", ncol=1, fontsize=7.5, borderaxespad=0.0, handlelength=1.6)
s = np.linspace(0, 6, len(A)); ax[1].plot(s, 1e3*A[:, 2], c="#2a78d6", lw=1.4)
for k in range(7): ax[1].axvline(k, c="#ddd", lw=0.6, zorder=0)
ax[1].set_xlabel("segment"); ax[1].set_ylabel("$z$ (mm, downward)"); ax[1].set_title("(b) Delta, Adept height", fontsize=9, loc="left")
x_, z_ = WA[0]; c2 = (x_**2 + z_**2 - 0.4**2 - 0.3**2)/(2*0.4*0.3); q2 = -np.arccos(c2); q1 = np.arctan2(z_, x_) - np.arctan2(0.3*np.sin(q2), 0.4 + 0.3*np.cos(q2)); j = np.array([0.4*np.cos(q1), 0.4*np.sin(q1)]); e = j + 0.3*np.array([np.cos(q1+q2), np.sin(q1+q2)])
ax[2].plot([0, j[0], e[0]], [0, j[1], e[1]], "-o", c="#999", lw=2.5, ms=4, zorder=1)
ax[2].plot(AR[:, 0], AR[:, 1], c="#eda100", lw=1.4, zorder=2); ax[2].plot(WA[:-1, 0], WA[:-1, 1], "o", c="#eda100", ms=4, zorder=3)
for i, p in enumerate(WA[:-1]):
    if i != 5: ax[2].annotate("2, 5" if i == 2 else str(i), p, xytext=(4, 2), textcoords="offset points", fontsize=7.5)
ax[2].set_aspect("equal"); ax[2].set_xlabel("$x$ (m)"); ax[2].set_ylabel("$z$ (m)"); ax[2].set_title("(c) two-link arm", fontsize=9, loc="left")
ax[2].set_xlim(-0.3, 0.55); ax[2].set_ylim(-0.08, 0.55)
fig.tight_layout(pad=0.3, w_pad=0.8); [fig.savefig(f"{OUT}/fig_refs.{x}", dpi=300) for x in ("pdf", "png")]
print("ok")
