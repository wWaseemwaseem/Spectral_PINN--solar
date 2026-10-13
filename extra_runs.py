"""Extra runs after switching to 65 CGL nodes: benchmark rows for the 65-node Spectral-PINN
(same Adam 5000 + L-BFGS 1500 protocol) the truncation-length runs are now in robustness_study.py."""
import json, time, sys
from pathlib import Path
import pandas as pd, torch
torch.set_num_threads(1)
import spectral_pinn_paper_code as S
import benchmark_pinn_methods as B
what = sys.argv[1]
if what == "bench":
    t0 = time.perf_counter(); ref = S.numerical_solution(B.P, 400); t_bvp = time.perf_counter() - t0
    for m in ("spectral65", "spectral_noreg65"):
        for s in (42, 43, 44):
            meth = "spectral65" if m == "spectral65" else "spectral_noreg"
            model, ta, tl = B.train(meth, s, 65, 5000, 1500, 1)
            res = dict(method=m, seed=s, adam_iters=5000, lbfgs_iters=1500, n_nodes=65, t_adam=ta, t_lbfgs=tl,
                       t_train=ta+tl, t_bvp=t_bvp, n_params=sum(p.numel() for p in model.parameters()), **B.score(model, ref))
            Path(f"benchmark_results/{m}_seed{s}.json").write_text(json.dumps(res, indent=1)); print(m, s, res["RelL2_theta"], flush=True)
else:
    raise SystemExit("usage: python extra_runs.py bench")
