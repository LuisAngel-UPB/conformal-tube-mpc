# Conformal-calibrated tube MPC with physics-informed learned dynamics

Code, trained weights and raw results for the paper

> L. Angel, "Conformal-Calibrated Tube MPC with Physics-Informed Learned Dynamics for Lagrangian Robots, with Probabilistic Tracking Guarantees," submitted to *IEEE/CAA Journal of Automatica Sinica*.

Everything is simulation: a Delta parallel robot (main case study) and a planar two-link arm (second case study). All parameters are listed in Table I of the paper.

## Contents

| Folder | What it holds |
|---|---|
| `notebooks/` | Jupyter notebooks NB02–NB05 (the runs behind the paper) |
| `src/` | The same notebooks as plain Python (`py2nb.py` converts them back to `.ipynb`) |
| `weights/` | Trained surrogates: Delta robot (`ff18_*.npz`) and two-link arm (`arm_*.npz`) |
| `results/` | CSV outputs and `config.json` of the full runs used in the paper |
| `paper_figures/` | Scripts that draw the paper figures from `results/` |
| `certificate/` | Recomputes the stability certificate of Assumption 2 |

## Notebook → paper

| Notebook | Experiment | Paper |
|---|---|---|
| `CMPC_NB02` | Closed-loop calibration on the Delta robot, four surrogates, constant vs reference-dependent normalizers (E1–E3) | Sec. V-A, Table II, Figs. 4–5 |
| `CMPC_NB04` | Baselines: sample maximum, open-loop conformal, worst-case box, GP, ensemble, oracle (E4) | Sec. V-B, Tables III–IV |
| `CMPC_NB03` | Planar two-link arm (E6); trains its own surrogates | Sec. V-C, Table V, Fig. 6 |
| `CMPC_NB05` | Coverage vs. number of calibration episodes (E1), distribution shift (E5), computation time (E7) | Sec. V-D, Fig. 7 |
| `paper_figures/make_fig_refs.py` | Reference trajectories | Fig. 3 |

## Running

```
pip install -r requirements.txt
```

1. Copy the weights into the notebook folder, or set `WEIGHTS_DIR` at the top of each notebook to `../weights`.
2. Open a notebook and run all cells. `QUICK = True` (the default) is a smoke test of a few minutes. `QUICK = False` reproduces the paper; it takes hours.
3. Each run writes its CSVs and figures to `results/CMPC_NBxx_<tag>/` and zips them. If a run is interrupted, set `RESUME_DIR` to that folder and run again; finished parts are loaded from `checkpoint.pkl`.
4. Paper figures: `python paper_figures/make_figs_nb02.py results/nb02full <out>` (likewise for NB03 and NB05).

`N_JOBS` sets the number of worker processes. The full runs used 15 workers on a laptop with an AMD Ryzen 7 PRO 5850U (8 cores, 16 threads) and 16 GB of RAM, under Windows 10 with Python 3.9.

All random draws are seeded (`SEED = 2026`), so the episodes are the same across notebooks and runs.

## Weights

* `ff18_HYBRID_35traj_seed{0,1,2}_e150.npz`: hybrid surrogate (identified rigid-body model + learned residual), 35 training trajectories, three seeds (seeds 1 and 2 are only used by the ensemble baseline of NB04).
* `ff18_MLP_35traj_seed0_e150.npz`: black-box network, 35 trajectories.
* `ff18_PINN_175traj_seed0_e150.npz`: physics-informed network that outputs the 14 dynamic parameters, 175 trajectories.
* `arm_{MLP,HYBRID,PINN}_35traj_seed0.npz`: surrogates of the two-link arm, as trained by NB03.

All networks have two hidden layers of 64 SiLU units. The `.npz` files hold the layer weights and the input/output normalization.

## License

MIT (see `LICENSE`).

## Stability certificate (Assumption 2)

`certificate/` recomputes the window-contraction constants `c`, `M`, `ρ` of Assumption 2 for every surrogate and task of both robots: run `python delta_phi.py` and `python arm_phi.py` inside that folder, then
`python analyze.py delta_phi.pkl 1e4,1e4,1e4,1,1,1,1e5,1e5,1e5 1e-2 10,25,50` and
`python analyze.py arm_phi.pkl 1e4,1e4,1,1,1e5,1e5 2e-4 10,25,50`.
Result: `c = 1`, `M = 25`, `ρ ≤ 0.891` (Delta) and `ρ ≤ 0.899` (arm).
