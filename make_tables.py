"""Build LaTeX tables (in red, for the revision) from the CSV/JSON results."""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).parent
SYM = {"M": "M", "beta": r"\beta", "Ec": "Ec", "Kr": "Kr", "Qh": "Q_h", "Sc": "Sc"}
BASE = {"M": .5, "beta": .1, "Ec": 4., "Kr": 5., "Qh": .3, "Sc": 3.}

def e(x, d=2):
    if not np.isfinite(x): return "--"
    m, ex = f"{x:.{d-1}e}".split("e"); return rf"${m}\times10^{{{int(ex)}}}$"

def unique_cases():
    m = pd.read_csv(R/"all_case_results/all_case_error_metrics.csv")
    t = pd.read_csv(R/"all_case_results/all_case_timing_wall.csv")
    rows = []; seen_base = False
    for (s, v), g in m.groupby(["study", "value"], sort=False):
        is_base = abs(BASE[s]-v) < 1e-12
        if is_base and seen_base: continue
        if is_base: seen_base = True; label = "Baseline"
        else: label = rf"${SYM[s]}={v:g}$"
        tt = t[(t.study == s) & (t.value == v)].iloc[0]
        r = dict(label=label, study=s, value=v, base=is_base)
        for var in ("F", "Fprime", "theta", "phi"):
            gg = g[g.variable == var].iloc[0]
            r[f"L2_{var}"] = gg.RelativeL2; r[f"Max_{var}"] = gg.MaxAbsError; r[f"R2_{var}"] = gg.R2
            r[f"MAE_{var}"] = gg.MAE; r[f"RMSE_{var}"] = gg.RMSE
        for k in ("Fpp0", "theta0", "mphip0"):
            r[f"p_{k}"] = tt[f"pinn_{k}"]; r[f"b_{k}"] = tt[f"bvp_{k}"]
            r[f"err_{k}"] = abs(tt[f"pinn_{k}"]-tt[f"bvp_{k}"])/abs(tt[f"bvp_{k}"])
        for k in ("t_adam", "t_lbfgs", "t_train", "t_inference", "t_bvp"): r[k] = tt[k]
        rows.append(r)
    order = ["Baseline"]; df = pd.DataFrame(rows)
    base = df[df.base]; rest = df[~df.base]
    return pd.concat([base, rest], ignore_index=True)

def table_accuracy(df):
    L = [r"\begin{table}[htbp]", r"\centering", r"\begin{revised}", r"\footnotesize",
         r"\caption{\rev{Accuracy and cost of the Spectral-PINN for the 13 distinct parameter cases (the baseline case is shared by all six studies). $\varepsilon_{L_2}$: relative $L_2$ error; $\varepsilon_\infty$: maximum absolute error; MAE and RMSE of $\theta$; $1-R^2$: largest value over the four fields ($R^2$: coefficient of determination); $t_{\mathrm{train}}$: Adam + L-BFGS wall-clock time; $t_{\mathrm{BVP}}$: time of one reference solution (single CPU core).}}",
         r"\label{tab:accuracy}", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lccccccccc}", r"\toprule",
         r"Case & $\varepsilon_{L_2}(F)$ & $\varepsilon_{L_2}(F')$ & $\varepsilon_{L_2}(\theta)$ & $\varepsilon_{L_2}(\phi)$ & $\varepsilon_\infty(\theta)$ & MAE$(\theta)$ & RMSE$(\theta)$ & $1-R^2$ & $t_{\mathrm{train}}$/$t_{\mathrm{BVP}}$ (s)\\", r"\midrule"]
    for _, r in df.iterrows():
        r2 = min(r[f"R2_{v}"] for v in ("F", "Fprime", "theta", "phi"))
        L.append(f"{r.label} & {e(r.L2_F)} & {e(r.L2_Fprime)} & {e(r.L2_theta)} & {e(r.L2_phi)} & {e(r.Max_theta)} & {e(r.MAE_theta)} & {e(r.RMSE_theta)} & {e(1-r2,1)} & {r.t_train:.0f} / {r.t_bvp:.2f}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{revised}", r"\end{table}"]
    return "\n".join(L)

def table_wall(df):
    L = [r"\begin{table}[htbp]", r"\centering", r"\begin{revised}", r"\footnotesize",
         r"\caption{\rev{Wall quantities predicted by the Spectral-PINN and by the reference solver. $F''(0)$, $\theta(0)$ and $-\phi'(0)$ determine the skin-friction coefficient, the Nusselt number and the Sherwood number through Eqs.~\eqref{eq:reduced_skin_friction}--\eqref{eq:reduced_sherwood}; ``rel.\ err.'' is the relative deviation of the Spectral-PINN value.}}",
         r"\label{tab:wall}", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lccccccccc}", r"\toprule",
         r" & \multicolumn{3}{c}{$F''(0)$} & \multicolumn{3}{c}{$\theta(0)$} & \multicolumn{3}{c}{$-\phi'(0)$}\\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
         r"Case & PINN & BVP & rel.\ err. & PINN & BVP & rel.\ err. & PINN & BVP & rel.\ err.\\", r"\midrule"]
    for _, r in df.iterrows():
        L.append(f"{r.label} & {r.p_Fpp0:.4f} & {r.b_Fpp0:.4f} & {e(r.err_Fpp0,2)} & {r.p_theta0:.3f} & {r.b_theta0:.3f} & {e(r.err_theta0,2)} & {r.p_mphip0:.4f} & {r.b_mphip0:.4f} & {e(r.err_mphip0,2)}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{revised}", r"\end{table}"]
    return "\n".join(L)

NAMES = {"spectral": "Spectral-PINN (present)", "spectral_noreg": "Ablation B: no spectral penalty",
         "autograd_cgl": "Ablation A: autograd on CGL nodes", "vanilla": r"Vanilla PINN \cite{raissi2019}",
         "sa": r"SA-PINN \cite{mcclenny2023}", "rba": r"RBA-PINN \cite{anagnostopoulos2024}",
         "pirate": r"PirateNet \cite{wang2024pirate}", "cpikan": r"cPIKAN \cite{shukla2024}",
         "config": r"ConFIG \cite{liu2025config}", "spectral_noreg65": r"Ablation B, 65 nodes",
         "spectral65": r"\textbf{Spectral-PINN, 65 nodes (present)}"}
ORDER = ["vanilla", "sa", "rba", "pirate", "cpikan", "config", "autograd_cgl", "spectral_noreg", "spectral", "spectral_noreg65", "spectral65"]

def bench_df():
    rows = [json.load(open(f)) for f in glob.glob(str(R/"benchmark_results/*.json"))]
    return pd.DataFrame(rows)

def ms(x, d=2):
    return rf"{e(np.mean(x),d)}\,$\pm$\,{e(np.std(x),1)}" if len(x) > 1 else e(np.mean(x), d)

def gm(x):
    return float(np.exp(np.mean(np.log(np.asarray(x)))))

def table_benchmark(b):
    L = [r"\begin{table}[htbp]", r"\centering", r"\begin{revised}", r"\scriptsize",
         r"\caption{\rev{Benchmark at the baseline parameter set (identical Adam schedule and L-BFGS refinement for all methods, three random seeds; 129 collocation points for all methods except the last block, which uses the 65 CGL nodes of the main study). Errors are geometric means over the seeds, with the range (min--max) in brackets for $\theta$. ``Off-grid res.'' is the RMS residual of Eqs.~\eqref{eq:reduced_momentum}--\eqref{eq:reduced_concentration} at 1000 points that are not collocation nodes. Times are single-core wall-clock means; $N_p$ is the number of trainable parameters. The reference \texttt{solve\_bvp} solution takes @@TBVP@@\,s.}}",
         r"\label{tab:benchmark}", r"\setlength{\tabcolsep}{2.5pt}",
         r"\begin{tabular}{lrccccccc}", r"\toprule",
         r"Method & $N_p$ & $\varepsilon_{L_2}(F')$ & $\varepsilon_{L_2}(\theta)$ [min--max] & $\varepsilon_{L_2}(\phi)$ & rel.\ err.\ $\theta(0)$ & Off-grid res. & $t_{\mathrm{train}}$ (s) & $t_{\mathrm{inf}}$ (ms)\\", r"\midrule"]
    for m in ORDER:
        g = b[b.method == m]
        if g.empty: continue
        rng = f"[{e(g.RelL2_theta.min(),2)}--{e(g.RelL2_theta.max(),2)}]"
        name = NAMES[m] if m != "spectral" else r"\textbf{Spectral-PINN, 129 nodes (present)}"
        if m in ("autograd_cgl", "spectral_noreg65"): L.append(r"\midrule")
        L.append(f"{name} & {int(g.n_params.iloc[0])} & {e(gm(g.RelL2_Fprime))} & {e(gm(g.RelL2_theta))} {rng} & {e(gm(g.RelL2_phi))} & {e(gm(g.RelErr_theta0))} & {e(gm(g.OffGridResidualRMS))} & {g.t_train.mean():.0f} & {1e3*g.t_inference.mean():.1f}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{revised}", r"\end{table}"]
    return "\n".join(L).replace("@@TBVP@@", f"{b.t_bvp.mean():.2f}")

def table_robust():
    d = pd.read_csv(R/"robustness_results/robustness_summary.csv")
    b = bench_df(); s = b[b.method == "spectral65"]
    L = [r"\begin{table}[htbp]", r"\centering", r"\begin{revised}", r"\footnotesize",
         r"\caption{\rev{Robustness of the Spectral-PINN at the baseline parameter set. Upper block: number of CGL nodes and truncation length $\eta_\infty$ (seed 42); the reference solution is recomputed on the same domain. The last two columns give the reference wall temperature and the relative change of the reference skin-friction parameter, which show how the physical solution itself depends on $\eta_\infty$. Lower block: spread over three random initialisations of the 65-node Spectral-PINN (benchmark protocol: Adam 5000 + L-BFGS 1500).}}",
         r"\label{tab:robust}", r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Configuration & $\varepsilon_{L_2}(F')$ & $\varepsilon_{L_2}(\theta)$ & $\varepsilon_{L_2}(\phi)$ & rel.\ err.\ $\theta(0)$ & $\theta_{\mathrm{BVP}}(0)$ & $\Delta F''_{\mathrm{BVP}}(0)$\\", r"\midrule"]
    ref = d[(d.N_nodes == 65) & (d.eta_inf == 20)].iloc[0]
    keep = [(33, 20.), (65, 20.), (129, 20.), (65, 16.), (65, 30.)]
    d = pd.concat([d[(d.N_nodes == n) & (d.eta_inf == e)] for n, e in keep if ((d.N_nodes == n) & (d.eta_inf == e)).any()])
    for _, r in d.iterrows():
        dth = abs(r.pinn_theta0-r.bvp_theta0)/abs(r.bvp_theta0)
        dF = (r.bvp_Fpp0-ref.bvp_Fpp0)/abs(ref.bvp_Fpp0)
        L.append(f"$N_s+1={int(r.N_nodes)}$, $\\eta_\\infty={r.eta_inf:g}$ & {e(r.RelL2_Fprime)} & {e(r.RelL2_theta)} & {e(r.RelL2_phi)} & {e(dth)} & {r.bvp_theta0:.3f} & {100*dF:+.2f}\\%\\\\")
    L.append(r"\midrule")
    for col, nm in (("RelL2_Fprime", "$F'$"),):
        pass
    L.append(r"Seeds 42/43/44 (mean $\pm$ std) & " + " & ".join(ms(s[c].values) for c in ("RelL2_Fprime", "RelL2_theta", "RelL2_phi", "RelErr_theta0")) + r" & -- & --\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{revised}", r"\end{table}"]
    return "\n".join(L)

def table_parametric():
    s = pd.read_csv(R/"parametric_results/parametric_summary.csv"); c = json.load(open(R/"parametric_results/parametric_cost.json"))
    order = ["manuscript grid (9 cases)", "interpolation (100 LHS points)", "extrapolation (4 cases)"]
    lab = {"manuscript grid (9 cases)": "Manuscript grid (9 cases)", "interpolation (100 LHS points)": "Unseen, inside box (100 LHS points)",
           "extrapolation (4 cases)": "Outside box (4 cases)"}
    L = [r"\begin{table}[htbp]", r"\centering", r"\begin{revised}", r"\footnotesize",
         r"\caption{\rev{Generalisation of the parametric Spectral-PINN trained once on $M\in[0.5,1.3]$, $Ec\in[2,6]$, $Q_h\in[0.1,0.5]$ (other parameters at baseline). Errors are relative to \texttt{solve\_bvp} on 400 points. Training cost: @@TT@@\,s; inference: @@TI@@\,ms per operating point (batched) versus @@TB@@\,ms per \texttt{solve\_bvp} solution, i.e.\ about @@SP@@ times cheaper per query; training is amortised after about @@BE@@ queries.}}",
         r"\label{tab:parametric}", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lccccc}", r"\toprule",
         r"Test set & mean $\varepsilon_{L_2}(F)$ & mean $\varepsilon_{L_2}(\theta)$ & max $\varepsilon_{L_2}(\theta)$ & mean $\varepsilon_{L_2}(\phi)$ & mean / max rel.\ err.\ $\theta(0)$\\", r"\midrule"]
    for k in order:
        r = s[s.set == k].iloc[0]
        L.append(f"{lab[k]} & {e(r.RelL2_F_mean)} & {e(r.RelL2_theta_mean)} & {e(r.RelL2_theta_max)} & {e(r.RelL2_phi_mean)} & {e(r.RelErr_theta0_mean)} / {e(r.RelErr_theta0_max)}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{revised}", r"\end{table}"]
    t = "\n".join(L)
    return (t.replace("@@TT@@", f"{c['t_train']:.0f}").replace("@@TI@@", f"{1e3*c['per_query_inference']:.1f}")
             .replace("@@TB@@", f"{1e3*c['per_query_bvp']:.0f}").replace("@@SP@@", f"{c['speedup_per_query']:.0f}")
             .replace("@@BE@@", f"{float(format(c['break_even_queries'], '.2g')):.0f}"))

def table_performance():
    d = pd.read_csv(R/"thermal_performance/thermal_performance_summary.csv")
    lab = ["Water", r"Al$_2$O$_3$/water ($\Phi_1=4\%$)", r"Al$_2$O$_3$--Cu/water ($\Phi_1=\Phi_2=2\%$)"]
    L = [r"\begin{table}[htbp]", r"\centering", r"\begin{revised}", r"\footnotesize",
         r"\caption{\rev{Step-by-step evaluation of the heat-transfer enhancement and collector efficiency (Newtonian limit, $Ec\to0$, $\lambda=0$, $M_f=0.5$, $\epsilon=0.1$, $\eta_\infty=8$; collector data: $I=1000$\,W\,m$^{-2}$, $\tau=0.90$, $\alpha_s=0.95$, $U_l=5$\,W\,m$^{-2}$K$^{-1}$, $\dot m/A_c=0.02$\,kg\,s$^{-1}$m$^{-2}$, $b=0.01$\,s$^{-1}$). $E$ is the increase of $h$ relative to water; $\eta_{c,\max}$ is the efficiency at $T_{\mathrm{in}}=T_{\mathrm{amb}}$.}}",
         r"\label{tab:performance}", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lccccccccc}", r"\toprule",
         r"Fluid & $k_{\mathrm{hnf}}/k_f$ & $\nu_{\mathrm{hnf}}/\nu_f$ & $\Pr$ & $Q_h$ & $\theta(0)$ & $h$ (W\,m$^{-2}$K$^{-1}$) & $E$ (\%) & $F_R$ & $\eta_{c,\max}$ (\%)\\", r"\midrule"]
    for l, (_, r) in zip(lab, d.iterrows()):
        L.append(f"{l} & {r.k_ratio:.4f} & {r.nu_ratio:.4f} & {r.Pr_eff:.3f} & {r.Qh:.4f} & {r.theta0:.4f} & {r.h_W_m2K:.1f} & {r.enhancement_pct:.2f} & {r.F_R:.4f} & {r.eta_max_pct:.2f}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{revised}", r"\end{table}"]
    return "\n".join(L)

if __name__ == "__main__":
    df = unique_cases(); out = R/"tables"; out.mkdir(exist_ok=True)
    (out/"accuracy.tex").write_text(table_accuracy(df)); (out/"wall.tex").write_text(table_wall(df))
    (out/"performance.tex").write_text(table_performance())
    if (R/"benchmark_results").exists(): (out/"benchmark.tex").write_text(table_benchmark(bench_df()))
    if (R/"robustness_results/robustness_summary.csv").exists(): (out/"robust.tex").write_text(table_robust())
    if (R/"parametric_results/parametric_summary.csv").exists(): (out/"parametric.tex").write_text(table_parametric())
    df.to_csv(out/"unique_cases.csv", index=False)
    print("unique-case stats: median L2 =", np.median(df[[f"L2_{v}" for v in ("F","Fprime","theta","phi")]].values),
          "max =", df[[f"L2_{v}" for v in ("F","Fprime","theta","phi")]].values.max())
