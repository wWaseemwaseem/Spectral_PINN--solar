"""Two-worker job queue for the benchmark (Adam 5000 + L-BFGS 1500, 3 seeds), then parametric and robustness runs."""
import os, sys, json, time, subprocess
from pathlib import Path
import torch
torch.set_num_threads(1)
import spectral_pinn_paper_code as S
import benchmark_pinn_methods as B
OUT = Path("benchmark_results"); OUT.mkdir(exist_ok=True); LOCK = Path("locks"); LOCK.mkdir(exist_ok=True)
ADAM, LB = 5000, 1500
jobs = [(m, s) for m in ["pirate", "cpikan", "config", "sa", "rba", "vanilla", "autograd_cgl", "spectral", "spectral_noreg"] for s in (42, 43, 44)]
def claim(name):
    try: os.close(os.open(LOCK/f"{name}.lock", os.O_CREAT | os.O_EXCL)); return True
    except FileExistsError: return False
t0 = time.perf_counter(); ref = S.numerical_solution(B.P, 400); t_bvp = time.perf_counter() - t0
for m, s in jobs:
    if not claim(f"{m}_{s}"): continue
    print("==", m, s, flush=True)
    model, ta, tl = B.train(m, s, 129, ADAM, LB, 1)
    res = dict(method=m, seed=s, adam_iters=ADAM, lbfgs_iters=LB, t_adam=ta, t_lbfgs=tl, t_train=ta+tl, t_bvp=t_bvp,
               n_params=sum(p.numel() for p in model.parameters()), **B.score(model, ref))
    (OUT/f"{m}_seed{s}.json").write_text(json.dumps(res, indent=1)); print(res, flush=True)
for e in ("parametric", "robustness"):
    if not claim(e): continue
    cmd = [sys.executable, "parametric_spectral_pinn.py"] if e == "parametric" else [sys.executable, "robustness_study.py"]
    subprocess.run(cmd, stdout=open(f"{e}.log", "w"), stderr=subprocess.STDOUT)
