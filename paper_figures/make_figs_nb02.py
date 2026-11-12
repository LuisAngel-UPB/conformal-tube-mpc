"""Paper figures from the NB02 FULL run: python3 make_figs_nb02.py <unzipped NB02 run folder> <output folder>"""
import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D, OUT = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.size":9, "axes.labelsize":9.5, "legend.fontsize":8.5, "xtick.labelsize":8.5, "ytick.labelsize":8.5,
                     "font.family":"serif", "mathtext.fontset":"cm", "axes.spines.top":False, "axes.spines.right":False,
                     "axes.edgecolor":"#555", "axes.linewidth":0.6})
MOD = [("Nominal","Nominal","#2a78d6","o"),("MLP-35","MLP (35)","#eb6834","s"),("HYBRID-35","Hybrid (35)","#1baf7a","D"),("PINN-175","PINN (175)","#eda100","^")]

# ---- E1: split coverage, reference-dependent normalizers
cs = pd.read_csv(f"{D}/CMPC_NB02_split_coverage_summary.csv"); cs = cs[(cs.variant=="ref")&(cs.group=="all")]
fig, ax = plt.subplots(figsize=(3.45, 2.7))
for t in sorted(cs.target.unique()):
    r = cs[cs.target==t].iloc[0]
    ax.add_patch(plt.Rectangle((t-0.022, r.beta_p05), 0.044, r.beta_p95-r.beta_p05, color="#d9d9d6", lw=0, zorder=0))
ax.plot([0.76,0.99],[0.76,0.99], ls=":", c="#777", lw=0.8, zorder=1)
for i,(m,lab,c,mk) in enumerate(MOD):
    d = cs[cs.model==m].sort_values("target"); x = d.target + (i-1.5)*0.008
    ax.errorbar(x, d.mean_cov, yerr=[d.mean_cov-d.p05, d.p95-d.mean_cov], fmt=mk, ms=5, c=c, mec="white", mew=0.6,
                elinewidth=1.3, capsize=2, label=lab, zorder=3)
ax.set_xticks([0.8,0.9,0.95]); ax.set_xlim(0.76,0.99); ax.set_ylim(0.72,1.0)
ax.set_xlabel(r"target coverage $1-\delta$"); ax.set_ylabel("empirical coverage")
ax.grid(axis="y", c="#eee", lw=0.5); ax.set_axisbelow(True)
h,l = ax.get_legend_handles_labels(); h.append(plt.Rectangle((0,0),1,1,color="#d9d9d6")); l.append("Beta 5–95% band")
ax.legend(h,l, loc="upper left", frameon=False, ncol=1, handletextpad=0.4)
fig.tight_layout(pad=0.3); [fig.savefig(f"{OUT}/fig_coverage.{e}", dpi=300) for e in ("pdf","png")]

# ---- normalized score by speed: constant vs reference-dependent normalizers
e = pd.read_csv(f"{D}/CMPC_NB02_episodes.csv"); s = pd.read_csv(f"{D}/CMPC_NB02_summary.csv")
e = e[e.set.isin(["calibration","test"]) & e.variant.isin(["const","ref"])].copy()
qt = {(r.model, r.variant): float(r.q_tilde) for r in s[s.variant.isin(["const","ref"])].itertuples()}
e["r"] = e.score / [qt[(m, v)] for m, v in zip(e.model, e.variant)]
sc = sorted(e.sc.unique())
fig, axs = plt.subplots(2, 1, figsize=(3.45, 4.6), sharex=True)
for ax, v, title in zip(axs, ["const","ref"], ["(a) constant normalizers", "(b) reference-dependent normalizers"]):
    for i,(m,lab,c,mk) in enumerate(MOD):
        for j,x0 in enumerate(sc):
            y = e[(e.model==m)&(e.variant==v)&(e.sc==x0)].r.values
            if len(y) == 0: continue
            x = j + (i-1.5)*0.18
            lo,med,hi = np.quantile(y,[0.05,0.5,0.95])
            ax.vlines(x, lo, hi, color=c, lw=1.5); ax.plot(x, med, mk, ms=5, c=c, mec="white", mew=0.6, label=lab if j==1 else None)
    ax.axhline(1, c="#222", lw=0.9); ax.text(0.5, 1.08, r"$s=\tilde q$", ha="center", va="bottom", fontsize=8.5)
    ax.set_yscale("log"); ax.set_ylim(0.07, 4); ax.set_yticks([0.1,0.2,0.5,1,2]); ax.set_yticklabels(["0.1","0.2","0.5","1","2"]); ax.minorticks_off()
    ax.set_ylabel(r"$s/\tilde q$"); ax.set_title(title, fontsize=9, loc="left", pad=3)
    ax.grid(axis="y", c="#eee", lw=0.5); ax.set_axisbelow(True)
axs[0].legend(loc="lower left", frameon=False, ncol=2, handletextpad=0.3, columnspacing=0.8)
axs[1].set_xticks(range(len(sc))); axs[1].set_xticklabels([f"{v:g}" for v in sc])
axs[1].set_xlabel("time scaling (2.0 = fastest)")
fig.tight_layout(pad=0.3, h_pad=0.8); [fig.savefig(f"{OUT}/fig_score_speed.{e_}", dpi=300) for e_ in ("pdf","png")]
print("ok")
