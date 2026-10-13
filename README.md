# Spectral-PINN for the MHD hybrid-nanofluid boundary layer of a flat-plate solar collector

This repository holds the code and data for the revised manuscript *"Thermal Performance Enhancement of a Solar Collector using MHD Hybrid Nanofluid: A Physics-Guided Spectral Learning Approach"* (Results in Engineering, RINENG-D-26-19132).

Every table and figure in the paper can be regenerated from the scripts below. Results were produced on one core of an Intel Xeon CPU @ 2.8 GHz with Python 3.13, PyTorch 2.14, SciPy 1.18 and NumPy 2.5, all in float64.

## Files

| Script | Purpose | Paper |
|---|---|---|
| `spectral_pinn_paper_code.py` | Spectral-PINN, 18 parameter cases (six studies), SciPy `solve_bvp` reference, metrics, timings, profile/error/loss figures | Sec. 3, 4.1–4.8, Tables 4–5, Figs. 3–9 |
| `paper_analysis_figures.py` | wall indicators, normalised sensitivities, dashboard and heat-map | Sec. 4.9, Figs. 10–11 |
| `benchmark_pinn_methods.py` | Vanilla PINN, SA-PINN, RBA-PINN, PirateNet, cPIKAN, ConFIG and two ablations | Sec. 3.11, 4.10, Table 6, Fig. 12 |
| `run_queue.py` | runs the benchmark (3 seeds, Adam 5000 + L-BFGS 1500), then the parametric and robustness studies | — |
| `extra_runs.py` | benchmark rows for the 65-node Spectral-PINN | Table 6 |
| `parametric_spectral_pinn.py` | parametric Spectral-PINN over (M, Ec, Qh), generalisation and many-query cost | Sec. 4.11, Table 8 |
| `robustness_study.py` | sensitivity to the number of CGL nodes and to the truncation length | Sec. 4.11, Table 7 |
| `thermal_performance.py` | heat-transfer enhancement and Hottel–Whillier–Bliss collector efficiency | Sec. 4.12, Table 9, Fig. 13 |
| `make_tables.py`, `make_benchmark_figure.py` | LaTeX tables and benchmark figure from the result files | — |

## Reproduce

```bash
pip install torch numpy scipy pandas matplotlib
python spectral_pinn_paper_code.py --threads 1 --cpu --output all_case_results   # about 35 min for all 18 cases on one CPU core
python paper_analysis_figures.py --results all_case_results
python run_queue.py & python run_queue.py &                                        # two workers
python extra_runs.py bench
python thermal_performance.py
python make_tables.py && python make_benchmark_figure.py
```

To regenerate the figures from the saved CSV files without retraining, run `python spectral_pinn_paper_code.py --plots-only`.

## Data

- `all_case_results/<study>/`: profiles (Spectral-PINN and reference), error metrics, timings and wall values, training and L-BFGS histories, spectral residuals, trained models (`.pt`)
- `all_case_results/all_case_error_metrics.csv`, `all_case_timing_wall.csv`: summaries over all 18 cases
- `benchmark_results/*.json`: one file per method and seed
- `parametric_results/`, `robustness_results/`, `thermal_performance/`: the remaining studies
- `results_129_nodes/`: the 21 cases at the original resolution of 129 CGL nodes (discussed in Sec. 4.1)

The main runs use 65 CGL nodes (the default `--spectral-points 65`) on the truncated domain `ETA_MAX = 20`.

## Pressure-gradient parameter

The pressure-gradient parameter is fixed at `LAMBDA = 0`. With the outer condition F'(eta_inf) = 0 the momentum equation reduces in the far field to (1 + 1/beta) F''' = lambda, so a non-zero lambda has no decaying far-field state: it produces reverse flow (F' < 0) whose magnitude grows with the truncation length. With lambda = 0 the momentum equation has the exact solution F' = exp(-s*eta), s = sqrt((1 + M)/(1 + 1/beta)), which the reference solver and the Spectral-PINN reproduce. The earlier version (lambda = 0.1, eta_inf = 8, seven studies including a lambda sweep) is kept in `old_lambda0.1_eta8_version/` for comparison only.

On Windows with an Anaconda-based environment, set `KMP_DUPLICATE_LIB_OK=TRUE` if the OpenMP runtime clash (OMP Error #15) appears.

## Note on the reference solver

The reference solutions come from SciPy's `solve_bvp`, which uses the same fourth-order collocation algorithm (Kierzenka & Shampine, 2001) as MATLAB's `bvp4c`. The tolerance is 1e-7 with up to 30,000 mesh nodes.
