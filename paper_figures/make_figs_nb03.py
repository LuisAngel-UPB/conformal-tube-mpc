"""Paper figure from the NB03 FULL run (two-link arm): python3 make_figs_nb03.py <unzipped NB03 run folder> <output folder>"""
import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D, OUT = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.size":9, "axes.labelsize":9.5, "legend.fontsize":8.5, "xtick.labelsize":8.5, "ytick.labelsize":8.5,
                     "font.family":"serif", "mathtext.fontset":"cm", "axes.spines.top":False, "axes.spines.right":False,
                     "axes.edgecolor":"#555", "axes.linewidth":0.6})
MOD = [("Nominal","Nominal","#2a78d6","o"),("MLP-35","MLP (35)","#eb6834","s"),("HYBRID-35","Hybrid (35)","#1baf7a","D"),("PINN-35","PINN (35)","#eda100","^")]
e3 = pd.read_csv(f"{D}/CMPC_NB03_E3.csv"); e3["x"] = e3.res_rms_ref_Nm / e3.m0
fig, ax = plt.subplots(figsize=(3.45, 2.6))
for m, lab, c, mk in MOD:
    d = e3[e3.model == m].set_index("variant")
    ax.plot([d.x["const"]]*2, [d.tube_pos_j1_med_urad["const"], d.tube_pos_j1_med_urad["ref"]], c=c, lw=0.8, alpha=0.6)
    ax.plot(d.x["const"], d.tube_pos_j1_med_urad["const"], mk, ms=6.5, mfc="white", mec=c, mew=1.4)
    ax.plot(d.x["ref"], d.tube_pos_j1_med_urad["ref"], mk, ms=6.5, c=c, mec="white", mew=0.6, label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"residual / inertia bound $r_{\mathrm{RMS}}/m_0$ (rad/s$^2$)"); ax.set_ylabel(r"one-step tube ($\mu$rad)")
ax.set_yticks([10, 20, 50, 100]); ax.set_yticklabels(["10","20","50","100"]); ax.set_ylim(10, 160)
ax.set_xticks([2, 5, 10, 20, 50, 100]); ax.set_xticklabels(["2","5","10","20","50","100"]); ax.set_xlim(1.5, 200); ax.minorticks_off()
ax.grid(c="#eee", lw=0.5); ax.set_axisbelow(True)
h, l = ax.get_legend_handles_labels()
h += [plt.Line2D([], [], ls="", marker="o", mfc="white", mec="#555", ms=6), plt.Line2D([], [], ls="", marker="o", c="#555", ms=6)]
l += ["constant $\\sigma$", "reference-dep. $\\sigma$"]
ax.legend(h, l, loc="lower right", frameon=False, ncol=1, handletextpad=0.3, fontsize=8)
fig.tight_layout(pad=0.3); [fig.savefig(f"{OUT}/fig_arm_tube.{x}", dpi=300) for x in ("pdf","png")]
print("ok")
