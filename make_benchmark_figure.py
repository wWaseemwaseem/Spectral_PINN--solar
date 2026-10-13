"""Accuracy-versus-cost figure for the PINN benchmark."""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = Path(__file__).parent
b = pd.DataFrame([json.load(open(f)) for f in glob.glob(str(R/"benchmark_results/*.json"))])
LAB = {"spectral65": "Spectral-PINN, 65 nodes (present)", "spectral_noreg65": "Ablation B, 65 nodes",
       "spectral": "Spectral-PINN, 129 nodes", "spectral_noreg": "Ablation B: no spectral penalty, 129 nodes",
       "autograd_cgl": "Ablation A: autograd on CGL", "vanilla": "Vanilla PINN", "sa": "SA-PINN",
       "rba": "RBA-PINN", "pirate": "PirateNet", "cpikan": "cPIKAN", "config": "ConFIG"}
STY = {"spectral65": ("#D55E00", "*", 17), "spectral_noreg65": ("#E69F00", "P", 10),
       "spectral": ("#F4A582", "*", 12), "spectral_noreg": ("#FDDBC7", "P", 9), "autograd_cgl": ("#CC79A7", "X", 9),
       "vanilla": ("#000000", "o", 8), "sa": ("#0072B2", "s", 8), "rba": ("#56B4E9", "D", 8),
       "pirate": ("#009E73", "^", 9), "cpikan": ("#7F7F7F", "v", 9), "config": ("#882255", "h", 9)}
plt.rcParams.update({"font.family": "serif", "font.size": 13, "mathtext.fontset": "dejavuserif"})
fig, axs = plt.subplots(1, 3, figsize=(15.5, 5.4))
cols = [("RelL2_theta", r"relative $L_2$ error of $\theta$"), ("RelErr_theta0", r"relative error of $\theta(0)$"),
        ("OffGridResidualRMS", "off-grid RMS residual")]
for ax, (c, lab), let in zip(axs, cols, "abc"):
    for m, (col, mk, ms) in STY.items():
        g = b[b.method == m]
        if g.empty: continue
        ax.scatter(g.t_train, g[c], color=col, marker=mk, s=ms*4, alpha=.35, edgecolor="none")
        ax.scatter(g.t_train.mean(), np.exp(np.log(g[c]).mean()), color=col, marker=mk, s=ms*14,
                   edgecolor="k", linewidth=.6, label=LAB[m], zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, which="both", alpha=.25, ls="--")
    ax.set_xlabel("training wall-clock time (s, one CPU core)"); ax.set_ylabel(lab); ax.set_title(f"({let})")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
fig.tight_layout(rect=(0, 0.06, 1, 1))
(R/"figures").mkdir(exist_ok=True)
fig.savefig(R/"figures/figure_benchmark.pdf", bbox_inches="tight"); fig.savefig(R/"figures/figure_benchmark.png", dpi=300, bbox_inches="tight")
