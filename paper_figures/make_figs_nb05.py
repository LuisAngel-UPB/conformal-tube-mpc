"""Paper figure from the NB05 FULL run: python3 make_figs_nb05.py <unzipped NB05 run folder> <output folder>"""
import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D, OUT = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.size":9, "axes.labelsize":9.5, "legend.fontsize":8.5, "xtick.labelsize":8.5, "ytick.labelsize":8.5,
                     "font.family":"serif", "mathtext.fontset":"cm", "axes.spines.top":False, "axes.spines.right":False,
                     "axes.edgecolor":"#555", "axes.linewidth":0.6})
cv = pd.read_csv(f"{D}/CMPC_NB05_coverage_vs_n.csv"); st = pd.read_csv(f"{D}/CMPC_NB05_sets.csv")
fig, (a, b) = plt.subplots(2, 1, figsize=(3.45, 5.0), gridspec_kw={"height_ratios":[1, 1.15], "hspace":0.42})
ns = sorted(cv.n.unique()); xi = {n:i for i, n in enumerate(ns)}
for d, c, off in [(0.05, "#2a78d6", -0.18), (0.10, "#1baf7a", 0.0), (0.20, "#eb6834", 0.18)]:
    s = cv[np.isclose(cv.delta, d)].sort_values("n"); x = np.array([xi[n] for n in s.n]) + off
    a.bar(x, s.beta_p95 - s.beta_p05, bottom=s.beta_p05, width=0.15, color=c, alpha=0.18, lw=0)
    a.errorbar(x, s.mean_cov, yerr=[s.mean_cov - s.p05, s.p95 - s.mean_cov], fmt="o", ms=4.5, c=c, capsize=2, lw=1, label=f"$\\delta={d:.2f}$")
    a.plot([-0.5, len(ns) - 0.5], [1 - d] * 2, c=c, lw=0.6, ls="--")
a.set_xticks(range(len(ns))); a.set_xticklabels([str(n) for n in ns]); a.set_xlim(-0.5, len(ns) - 0.5)
a.set_xlabel("calibration episodes $n$"); a.set_ylabel("coverage"); a.set_ylim(0.68, 1.0)
a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=3, handletextpad=0.2, columnspacing=0.8, fontsize=8.5)
a.text(-0.3, 1.13, "(a)", transform=a.transAxes, fontsize=10, fontweight="bold")
a.grid(axis="y", c="#eee", lw=0.5); a.set_axisbelow(True)
ROWS = [("D (in distribution)", "Adept ($\\mathcal{D}$)", "#555"), ("circle (tube of D)", "circle", "#eb6834"), ("eight (tube of D)", "figure eight", "#eb6834"),
        ("severity 0.5 (tube of D)", "severity 0.5", "#eb6834"), ("severity 1.5 (tube of D)", "severity 1.5", "#eb6834"), ("severity 2 (tube of D)", "severity 2", "#eb6834"),
        ("payload 7-8 kg (tube of D)", "payload 7–8 kg", "#eb6834"), ("circle (calibrated on circle)", "circle, own tube", "#1baf7a"), ("eight (calibrated on eight)", "eight, own tube", "#1baf7a")]
sv = st.set_index("set"); y = np.arange(len(ROWS))[::-1]
for yy, (k, lab, c) in zip(y, ROWS):
    v = sv.coverage[k]; b.barh(yy, v, color=c, height=0.62)
    b.text(v + 0.02, yy, f"{v:.2f}", va="center", ha="left", fontsize=8.5, bbox=dict(fc="white", ec="none", pad=0.5))
b.axvline(0.9, c="black", lw=0.8, ls="--", zorder=0); b.set_yticks(y); b.set_yticklabels([r[1] for r in ROWS]); b.set_xlim(0, 1.13); b.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]); b.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "", "1"])
b.set_xlabel(r"coverage of $\mathcal{E}$ ($\delta=0.10$, target 0.90)")
b.text(-0.3, 1.04, "(b)", transform=b.transAxes, fontsize=10, fontweight="bold")
b.grid(axis="x", c="#eee", lw=0.5); b.set_axisbelow(True)
fig.subplots_adjust(left=0.33, right=0.97, top=0.93, bottom=0.08)
[fig.savefig(f"{OUT}/fig_nb05.{x}", dpi=300) for x in ("pdf", "png")]
print("ok")
