"""Create paper-ready secondary figures and tables from trained case-study data."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDIES = {
    "M": (0.5, 0.9, 1.3), "beta": (0.1, 0.3, 0.5),
    "Ec": (2., 4., 6.), "Kr": (3., 5., 7.),
    "Qh": (0.1, 0.3, 0.5),
    "Sc": (2., 3., 4.),
}
trapezoid = getattr(np, "trapezoid", None) or np.trapz  # NumPy < 2 compatibility
SYMBOL = {"M": "$M$", "beta": r"$\beta$", "Ec": "$Ec$", "Kr": "$Kr$",
          "Qh": "$Q_h$", "Sc": "$Sc$"}

def savefig(fig, out, stem):
    fig.savefig(out/f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out/f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

def thickness(eta, y, fraction=0.01):
    """First eta after which |y| stays below fraction of its wall magnitude."""
    threshold=fraction*max(abs(y[0]), 1e-14)
    for i in range(len(y)):
        if np.all(np.abs(y[i:]) <= threshold): return eta[i]
    return eta[-1]

def case_indicators(eta, f, fp, theta, phi, qh):
    fpp=np.gradient(fp, eta, edge_order=2)
    thetap=np.gradient(theta, eta, edge_order=2)
    phip=np.gradient(phi, eta, edge_order=2)
    return {
        "Wall shear Fpp(0)": fpp[0],
        "Wall temperature theta(0)": theta[0],
        # Eq. (39) of the revised manuscript (normalised formulation, k_hnf/k_f absorbed):
        "Nusselt Qh/theta(0)": qh/theta[0],
        "Reduced Sherwood -phi'(0)": -phip[0],
        "Momentum thickness eta_99": thickness(eta, fp),
        "Thermal thickness eta_99": thickness(eta, theta),
        "Concentration thickness eta_99": thickness(eta, phi),
        "Velocity integral": trapezoid(fp, eta),
        "Thermal integral": trapezoid(theta, eta),
        "Concentration integral": trapezoid(phi, eta),
    }

def load_indicators(root):
    rows=[]
    for study,values in STUDIES.items():
        d=pd.read_csv(root/study/f"profiles_{study}.csv"); eta=d["eta"].to_numpy()
        for value in values:
            p=f"{study}_{value:g}_"
            vals=case_indicators(eta, d[p+"F_solve_bvp"].to_numpy(),
                d[p+"Fprime_solve_bvp"].to_numpy(), d[p+"theta_solve_bvp"].to_numpy(),
                d[p+"phi_solve_bvp"].to_numpy(), value if study=="Qh" else 0.3)
            rows.append({"study":study,"parameter_value":value,**vals})
    return pd.DataFrame(rows)

def sensitivity_table(indicators):
    metrics=[c for c in indicators.columns if c not in ("study","parameter_value")]
    rows=[]
    for study,g in indicators.groupby("study",sort=False):
        g=g.sort_values("parameter_value"); x=g.parameter_value.to_numpy(); mid=len(g)//2
        for metric in metrics:
            y=g[metric].to_numpy(); denom=max(abs(y[mid]),1e-14)
            elasticity=((y[-1]-y[0])/denom)/((x[-1]-x[0])/x[mid])
            rows.append({"parameter":study,"response":metric,
                         "normalized_sensitivity":elasticity,
                         "absolute_sensitivity":abs(elasticity)})
    return pd.DataFrame(rows)

def write_latex(df,path,caption,label,float_format="%.4g"):
    text=df.to_latex(index=False,escape=False,float_format=lambda x: float_format%x,
                     caption=caption,label=label,position="htbp")
    path.write_text(text,encoding="utf-8")

def generate(root):
    plt.rcParams.update({"font.family":"serif","font.size":12,"mathtext.fontset":"dejavuserif"})
    out=root/"paper_additions"; out.mkdir(parents=True,exist_ok=True)
    ind=load_indicators(root); ind.to_csv(out/"table_1_transport_indicators.csv",index=False)
    write_latex(ind,out/"table_1_transport_indicators.tex",
                "Wall transport coefficients, boundary-layer thicknesses, and integrated field measures.",
                "tab:transport_indicators")
    sens=sensitivity_table(ind); sens.to_csv(out/"table_2_normalized_sensitivity.csv",index=False)
    ranked=sens.sort_values(["parameter","absolute_sensitivity"],ascending=[True,False])
    ranked=ranked.groupby("parameter",sort=False).head(3).reset_index(drop=True)
    ranked.to_csv(out/"table_2_top_sensitivities.csv",index=False)
    write_latex(ranked,out/"table_2_top_sensitivities.tex",
                "Three strongest normalized response sensitivities for each governing parameter.",
                "tab:top_sensitivities")

    errors=pd.read_csv(root/"all_case_error_metrics.csv")
    summary=errors.groupby(["study","variable"],sort=False).agg(
        Mean_MAE=("MAE","mean"),Worst_MAE=("MAE","max"),
        Worst_MaxAbsError=("MaxAbsError","max"),Worst_RelativeL2=("RelativeL2","max")).reset_index()
    summary.to_csv(out/"table_3_accuracy_summary.csv",index=False)
    write_latex(summary,out/"table_3_accuracy_summary.tex",
                "Aggregate Spectral-PINN accuracy over each parameter sweep.","tab:accuracy_summary")

    colors=("#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#56B4E9","#000000")
    targets={"M":"Wall shear Fpp(0)","beta":"Wall shear Fpp(0)",
             "Ec":"Wall temperature theta(0)","Kr":"Reduced Sherwood -phi'(0)",
             "Qh":"Nusselt Qh/theta(0)",
             "Sc":"Reduced Sherwood -phi'(0)"}
    PRETTY={"Wall shear Fpp(0)":r"$F''(0)$","Wall temperature theta(0)":r"$\theta(0)$",
            "Nusselt Qh/theta(0)":r"$Nu_xRe_x^{-1/2}=Q_h/\theta(0)$","Reduced Sherwood -phi'(0)":r"$-\phi'(0)$",
            "Momentum thickness eta_99":r"$\eta_{99}$ (velocity)","Thermal thickness eta_99":r"$\eta_{99}$ (thermal)",
            "Concentration thickness eta_99":r"$\eta_{99}$ (species)","Velocity integral":r"$\int F'\,d\eta$",
            "Thermal integral":r"$\int\theta\,d\eta$","Concentration integral":r"$\int\phi\,d\eta$"}
    fig,axs=plt.subplots(2,3,figsize=(13.5,7.6)); axs=axs.ravel()
    for i,(study,target) in enumerate(targets.items()):
        g=ind[ind.study==study].sort_values("parameter_value"); x=g.parameter_value; y=g[target]
        axs[i].plot(x,y,"o-",lw=2.2,ms=7,color=colors[i]); axs[i].set_xlabel(SYMBOL[study])
        axs[i].set_ylabel(PRETTY[target]); axs[i].grid(alpha=.25,ls="--")
    fig.tight_layout(); savefig(fig,out,"figure_8_parameter_response_dashboard")

    heat=sens.pivot(index="parameter",columns="response",values="normalized_sensitivity")
    order=["M","beta","Ec","Kr","Qh","Sc"]; heat=heat.reindex(order)
    fig,ax=plt.subplots(figsize=(14.8,5.8)); vmax=np.nanmax(abs(heat.to_numpy()))
    im=ax.imshow(heat,aspect="auto",cmap="RdBu_r",vmin=-vmax,vmax=vmax)
    ax.set_yticks(range(len(heat)),[SYMBOL[s] for s in heat.index])
    ax.set_xticks(range(len(heat.columns)),[PRETTY[c] for c in heat.columns],rotation=35,ha="right")
    for r in range(heat.shape[0]):
        for c in range(heat.shape[1]): ax.text(c,r,f"{heat.iloc[r,c]:.2f}",ha="center",va="center",fontsize=8)
    fig.colorbar(im,ax=ax,label="Normalized sensitivity (elasticity)")
    fig.tight_layout(); savefig(fig,out,"figure_9_sensitivity_heatmap")

    thick=["Momentum thickness eta_99","Thermal thickness eta_99","Concentration thickness eta_99"]
    fig,axs=plt.subplots(1,3,figsize=(15.2,4.8),sharey=True)
    labels=[]
    for study,values in STUDIES.items():
        for value in values: labels.append(f"{study}={value:g}")
    for ax,metric in zip(axs,thick):
        vals=ind[metric].to_numpy(); ax.barh(np.arange(len(vals)),vals,color="#0072B2",alpha=.82)
        ax.set_xlabel(PRETTY[metric]); ax.grid(axis="x",alpha=.22,ls="--")
    axs[0].set_yticks(np.arange(len(labels)),labels,fontsize=8); axs[1].tick_params(labelleft=False); axs[2].tick_params(labelleft=False)
    fig.tight_layout(); savefig(fig,out,"figure_10_boundary_layer_thickness_atlas")

    print(f"Saved paper additions to {out.resolve()}")

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--results",type=Path,default=Path(__file__).parent/"all_case_results")
    generate(p.parse_args().results)
