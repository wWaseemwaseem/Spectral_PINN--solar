"""Robustness of the Spectral-PINN to the number of CGL nodes and the truncation eta_inf
(baseline parameter set, seed 42, same optimisation protocol as the main runs)."""
import argparse, json, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd, torch
import spectral_pinn_paper_code as S

def run(n, eta_max, out):
    S.ETA_MAX = eta_max
    args = SimpleNamespace(test_points=400, spectral_points=n, modes=8, width=80, depth=5,
                           learning_rate=5e-4, iterations=10000, max_grad_norm=10.0, log_every=250,
                           skip_lbfgs=False, lbfgs_restarts=2, lbfgs_iterations=3000, lbfgs_log_every=50, seed=42)
    x = np.linspace(0, eta_max, 400)[:, None]
    r = S.train("base", 0.0, S.BASE_PARAMETERS.copy(), args, x, out, torch.device("cpu"), torch.float64)
    ref = r["reference"]; p = r["prediction"]; row = dict(N_nodes=n, eta_inf=eta_max)
    for name, pred, true in (("F", p[:, 0], ref[0]), ("Fprime", r["fp_prediction"], ref[1]),
                             ("theta", p[:, 1], ref[3]), ("phi", p[:, 2], ref[5])):
        e = pred - true; row[f"RelL2_{name}"] = np.linalg.norm(e) / np.linalg.norm(true); row[f"MaxAbs_{name}"] = np.abs(e).max()
    row.update({f"pinn_{k}": v for k, v in r["wall"].items()}); row.update({f"bvp_{k}": v for k, v in r["wall_ref"].items()})
    row.update({k: v for k, v in r["timing"].items() if k != "final_losses"})
    return row

if __name__ == "__main__":
    torch.set_num_threads(1); out = Path("robustness_results"); out.mkdir(exist_ok=True); rows = []
    for n, e in ((33, 20.), (65, 20.), (129, 20.), (65, 16.), (65, 30.)):
        print("== N", n, "eta_inf", e, flush=True); rows.append(run(n, e, out))
        pd.DataFrame(rows).to_csv(out / "robustness_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string())
