"""Spectral-PINN for the hybrid-nanofluid solar boundary-layer model.

The implementation follows Eqs. (23)--(25), (31)--(32), and (39)--(69) of
the accompanying manuscript.  Governing derivatives are formed with
Chebyshev--Gauss--Lobatto differentiation matrices (not autograd).  Autograd
is used only to optimize the neural-network parameters.
"""
from __future__ import annotations
import argparse, copy, json, random, time
from pathlib import Path
import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_bvp
from torch import nn
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

# LAMBDA (pressure-gradient parameter) must be zero for a quiescent outer stream:
# as F' -> 0 the momentum equation reduces to (1+1/beta) F''' = lambda, so any
# lambda != 0 has no decaying far-field state and gives reverse flow (F' < 0)
# whose size depends on ETA_MAX.  With lambda = 0 the momentum equation has the
# exact solution F' = exp(-s*eta), s = sqrt((1+M)/(1+1/beta)).
BETA, M, LAMBDA, EPSILON, EC, PR, QH, KR = .1, .5, 0., .1, 4., 1., .3, 5.
# s = 0.37 at the baseline (beta = 0.1), so eta = 20 is needed for F' < 1e-3.
ETA_MAX = 20.
BASE_PARAMETERS = dict(beta=BETA, M=M, lam=LAMBDA, epsilon=EPSILON,
                       Ec=EC, Pr=PR, Qh=QH, Kr=KR, Sc=3.)
CASE_STUDIES = {
    "M": (0.5, 0.9, 1.3),
    "beta": (0.1, 0.3, 0.5),
    "Ec": (2., 4., 6.),
    "Kr": (3., 5., 7.),
    "Qh": (0.1, 0.3, 0.5),
    "Sc": (2., 3., 4.),
}
STUDY_OUTPUT = {
    "M": ("fp", r"$F'(\eta)$", "figure_1_M_velocity"),
    "beta": ((1, 3), r"$\theta(\eta)$", "figure_2_beta_temperature"),
    "Ec": ((1, 3), r"$\theta(\eta)$", "figure_3_Ec_temperature"),
    "Kr": ((2, 5), r"$\phi(\eta)$", "figure_4_Kr_concentration"),
    "Qh": ((1, 3), r"$\theta(\eta)$", "figure_5_Qh_temperature"),
    "Sc": ((2, 5), r"$\phi(\eta)$", "figure_6_Sc_concentration"),
}
STUDY_NUMBER = {"M":1,"beta":2,"Ec":3,"Kr":4,"Qh":5,"Sc":6}
# The species layer is much thinner than the momentum layer; zoom its profile plot.
PROFILE_XLIM = {r"$\phi(\eta)$": (0., 4.)}
FIELD_OUTPUTS = {
    "velocity": ("fp", r"$F'(\eta)$"),
    "temperature": ((1,3), r"$\theta(\eta)$"),
    "concentration": ((2,5), r"$\phi(\eta)$"),
}
NAMES = ("momentum", "energy", "concentration", "boundary", "spectral")
WEIGHTS = (2., 5., 1., 50., 1e-6)

def chebyshev_grid(n, device, dtype):
    """Ascending CGL grid on [0, ETA_MAX] and its first three D matrices."""
    if n < 4:
        raise ValueError("At least four Chebyshev points are required")
    j = np.arange(n)
    xi = np.cos(np.pi * j / (n - 1))
    c = np.ones(n); c[[0, -1]] = 2; c *= (-1.) ** j
    dx = xi[:, None] - xi[None, :]
    d = (c[:, None] / c[None, :]) / (dx + np.eye(n))
    d -= np.diag(d.sum(axis=1))
    # Reverse to eta=0,...,eta_max and scale d/dxi -> d/deta.
    d = d[::-1, ::-1] * (2. / ETA_MAX)
    eta = (ETA_MAX * (1 + xi[::-1]) / 2)[:, None]
    mats = [torch.tensor(np.ascontiguousarray(a), device=device, dtype=dtype)
            for a in (d, d @ d, d @ d @ d)]
    return torch.tensor(eta.copy(), device=device, dtype=dtype), mats

def derivative(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True,
                               retain_graph=True)[0]

def numerical_solution(params, n=400):
    beta, m, lam = params["beta"], params["M"], params["lam"]
    epsilon, ec, pr = params["epsilon"], params["Ec"], params["Pr"]
    qh, kr, sc = params["Qh"], params["Kr"], params["Sc"]
    c = 1 + 1 / beta
    def ode(x, y):
        f, fp, fpp, th, thp, ph, php = y
        den = np.where(np.abs(1 + epsilon*th) < 1e-10, 1e-10, 1 + epsilon*th)
        return np.vstack((fp, fpp, (-f*fpp + fp**2 + m*fp + lam)/c,
            thp, -(ec*c*fpp**2 + epsilon*thp**2 + pr*f*thp + m*ec*fp**2)/den,
            php, kr*ph - sc*f*php))
    def bc(a, b):
        # Eq. (31) retains the temperature-dependent wall conductivity.
        return np.array([a[0], a[1]-1, -(1+epsilon*a[3])*a[4]-qh,
                         a[5]-1, b[1], b[3], b[5]])
    x = np.linspace(0, ETA_MAX, n); e = np.exp(-x); el = np.exp(-ETA_MAX)
    guess = np.vstack((x*e, (1-x)*e, (x-2)*e, qh*(e-el), -qh*e,
                       (e-el)/(1-el), -e/(1-el)))
    sol = solve_bvp(ode, bc, x, guess, tol=1e-7, max_nodes=30000)
    if not sol.success: raise RuntimeError(f"BVP failed for Sc={sc:g}: {sol.message}")
    return sol

class SpectralPINN(nn.Module):
    def __init__(self, modes, width, depth):
        super().__init__(); self.modes = modes; modules=[]
        sizes=[1]+[width]*depth+[3]
        for a,b in zip(sizes[:-2], sizes[1:-1]):
            layer=nn.Linear(a,b); nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
            modules += [layer, nn.Tanh()]
        layer=nn.Linear(sizes[-2],3); nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
        self.net=nn.Sequential(*modules, layer)
    def forward(self, x):
        # Equation (45): eta is the sole network input (scaled for conditioning).
        z=2*x/ETA_MAX-1
        return self.net(z)

def evaluate(model, x, params):
    y=model(x); f,th,ph=y[:,:1],y[:,1:2],y[:,2:3]
    fp,thp,php=derivative(f,x),derivative(th,x),derivative(ph,x)
    fpp,thpp,phpp=derivative(fp,x),derivative(thp,x),derivative(php,x)
    c=1+1/params["beta"]
    r=torch.cat((c*derivative(fpp,x)+f*fpp-fp**2-params["M"]*fp-params["lam"],
       (1+params["epsilon"]*th)*thpp+params["Ec"]*c*fpp**2
       +params["epsilon"]*thp**2+params["Pr"]*f*thp+params["M"]*params["Ec"]*fp**2,
       phpp+params["Sc"]*f*php-params["Kr"]*ph),1)
    return y,fp,thp,r

def spectral_evaluate(model, x, mats, params):
    y = model(x); d1, d2, d3 = mats
    f, th, ph = y[:, 0], y[:, 1], y[:, 2]
    fp, fpp, fppp = d1 @ f, d2 @ f, d3 @ f
    thp, thpp, php, phpp = d1 @ th, d2 @ th, d1 @ ph, d2 @ ph
    c = 1 + 1 / params["beta"]
    r = torch.stack((c*fppp + f*fpp - fp.square() - params["M"]*fp - params["lam"],
        (1+params["epsilon"]*th)*thpp + params["epsilon"]*thp.square()
        + params["Pr"]*f*thp + params["Ec"]*c*fpp.square()
        + params["M"]*params["Ec"]*fp.square(),
        phpp + params["Sc"]*f*php - params["Kr"]*ph), dim=1)
    return y, fp, thp, r

def losses(model, xs, mats, params):
    y, fp, thp, r = spectral_evaluate(model, xs, mats, params)
    # Wall is index 0; far field is index -1 (Eqs. 31--32).
    bc = torch.stack((y[0,0], fp[0]-1,
        -(1+params["epsilon"]*y[0,1])*thp[0]-params["Qh"], y[0,2]-1,
        fp[-1], y[-1,1], y[-1,2]))
    # Eq. (68): penalize the upper third of Chebyshev coefficients.
    n = len(xs); k = torch.arange(n,device=xs.device,dtype=xs.dtype)
    j = torch.arange(n,device=xs.device,dtype=xs.dtype)
    coeff = (torch.cos(torch.pi*k[:,None]*j[None,:]/(n-1)) @ y) / (n-1)
    spec = coeff[(2*n)//3:].square().mean()
    return [r[1:-1,i].square().mean() for i in range(3)] + [bc.square().mean(), spec]

def train(study, value, params, args, x_test, out, device, dtype):
    t0=time.perf_counter(); ref=numerical_solution(params,args.test_points)
    t_bvp=time.perf_counter()-t0
    xs,mats=chebyshev_grid(args.spectral_points,device,dtype)
    torch.manual_seed(args.seed)  # identical initialisation for every case, independent of run order
    model=SpectralPINN(args.modes,args.width,args.depth).to(device=device,dtype=dtype)
    opt=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(args.iterations,1))
    weights=torch.tensor(WEIGHTS,device=device,dtype=dtype); hist=[]; lbfgs_hist=[]
    best_loss=float("inf"); best_state=copy.deepcopy(model.state_dict())
    t_adam0=time.perf_counter()
    for step in range(1,args.iterations+1):
        raw=losses(model,xs,mats,params)
        opt.zero_grad(set_to_none=True)
        total_loss=sum(w*l for w,l in zip(weights,raw))
        current=total_loss.detach().item()
        if np.isfinite(current) and current < best_loss:
            best_loss=current; best_state=copy.deepcopy(model.state_dict())
        total_loss.backward()
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(
                f"Non-finite Adam gradient at {study}={value:g}, iteration={step}. "
                "Use float64 and/or reduce --learning-rate.")
        opt.step(); sched.step()
        if step==1 or step%args.log_every==0 or step==args.iterations:
            vals=[v.detach().item() for v in raw]; total=sum(w*v for w,v in zip(WEIGHTS,vals))
            hist.append([step,total,*vals]); print(f"{study}={value:g} {step:6d}/{args.iterations}: {total:.3e}")
    # A late Adam oscillation must not be handed to the deterministic refiner.
    model.load_state_dict(best_state)
    t_adam=time.perf_counter()-t_adam0; t_lbfgs=0.0
    print(f"{study}={value:g} best Adam loss restored: {best_loss:.3e}")
    if not args.skip_lbfgs:
        t_lb0=time.perf_counter(); evaluations=0
        for restart in range(1,args.lbfgs_restarts+1):
            opt2=torch.optim.LBFGS(model.parameters(),max_iter=args.lbfgs_iterations,
                max_eval=int(args.lbfgs_iterations*1.5),history_size=150,
                tolerance_grad=1e-12,tolerance_change=1e-15,line_search_fn="strong_wolfe")
            def closure():
                nonlocal evaluations
                opt2.zero_grad(set_to_none=True); ls=losses(model,xs,mats,params)
                total=sum(w*l for w,l in zip(weights,ls)); total.backward(); evaluations+=1
                if evaluations==1 or evaluations%args.lbfgs_log_every==0:
                    vals=[v.detach().item() for v in ls]
                    lbfgs_hist.append([evaluations,total.detach().item(),*vals])
                    print(f"{study}={value:g} L-BFGS eval {evaluations:6d}: {total.detach().item():.3e}")
                return total
            print(f"{study}={value:g} L-BFGS restart {restart}/{args.lbfgs_restarts}")
            opt2.step(closure)
        with torch.no_grad():
            final_ls=losses(model,xs,mats,params)
            final_total=sum(w*l for w,l in zip(weights,final_ls)).item()
        vals=[v.item() for v in final_ls]
        lbfgs_hist.append([evaluations,final_total,*vals])
        t_lbfgs=time.perf_counter()-t_lb0
        print(f"{study}={value:g} L-BFGS complete: {best_loss:.3e} -> {final_total:.3e}")
    xt=torch.tensor(x_test,device=device,dtype=dtype,requires_grad=True)
    y,fp,_,_=evaluate(model,xt,params)
    _,_,_,r=spectral_evaluate(model,xs,mats,params)
    # Inference cost: fields + F' on the 400-point test grid (mean of 50 repeats).
    t0=time.perf_counter()
    for _ in range(50):
        xq=torch.tensor(x_test,device=device,dtype=dtype,requires_grad=True)
        yq=model(xq); _=derivative(yq[:,:1],xq)
    t_inf=(time.perf_counter()-t0)/50
    with torch.no_grad():
        yn=model(xs); d1,d2,_=mats
        wall=dict(Fpp0=(d2@yn[:,0])[0].item(),theta0=yn[0,1].item(),
                  mphip0=-(d1@yn[:,2])[0].item())
    rs=ref.sol(np.array([0.0]))[:,0]
    wall_ref=dict(Fpp0=rs[2],theta0=rs[3],mphip0=-rs[6])
    with torch.no_grad():
        fl=losses(model,xs,mats,params)
    timing=dict(t_adam=t_adam,t_lbfgs=t_lbfgs,t_train=t_adam+t_lbfgs,t_inference=t_inf,
                t_bvp=t_bvp,final_losses=[v.item() for v in fl])
    torch.save({"model_state_dict":model.state_dict(),"study":study,"value":value,
                "parameters":params,"config":vars(args)},out/f"spectral_pinn_{study}_{value:g}.pt")
    return dict(prediction=y.detach().cpu().numpy(),fp_prediction=fp.detach().cpu().numpy()[:,0],
      reference=ref.sol(x_test[:,0]),residuals=r.detach().cpu().numpy(),
      residual_eta=xs.detach().cpu().numpy()[:,0],history=np.asarray(hist),
      lbfgs_history=np.asarray(lbfgs_hist),timing=timing,wall=wall,wall_ref=wall_ref)

def savefig(fig,out,stem):
    fig.savefig(out/f"{stem}.png",dpi=600,bbox_inches="tight",facecolor="white")
    fig.savefig(out/f"{stem}.pdf",bbox_inches="tight",facecolor="white"); plt.close(fig)

def profile(out,eta,res,study,component,ylabel,stem,include_loss=True):
    colors=("#0072B2","#D55E00","#009E73"); styles=("-","--","-.")
    symbol={"beta":r"\beta","lam":r"\lambda"}.get(study,study)
    fig,ax=plt.subplots(figsize=(6.6,5.0))
    for i,(value,r) in enumerate(res.items()):
        pred,ref=((r["fp_prediction"],r["reference"][1]) if component=="fp" else
                            (r["prediction"][:,component[0]],r["reference"][component[1]]))
        label=rf"${symbol}={value:g}$"
        ax.plot(eta,ref,"o",markevery=(4 if ylabel in PROFILE_XLIM else 20),ms=5.0,mfc="white",mew=1.2,color=colors[i],label=label+" BVP reference")
        ax.plot(eta,pred,styles[i],lw=2.2,color=colors[i],label=label+" Spectral-PINN")
    ax.set(xlabel=r"$\eta$",ylabel=ylabel); ax.grid(True,which="both",alpha=.22,ls="--")
    if ylabel in PROFILE_XLIM: ax.set_xlim(*PROFILE_XLIM[ylabel])
    ax.legend(ncol=1,fontsize=11,frameon=True,framealpha=.95)
    fig.tight_layout(); savefig(fig,out,stem+"_solution")

    fig,err_ax=plt.subplots(figsize=(6.6,5.0))
    for i,(value,r) in enumerate(res.items()):
        pred,ref=((r["fp_prediction"],r["reference"][1]) if component=="fp" else
                            (r["prediction"][:,component[0]],r["reference"][component[1]]))
        label=rf"${symbol}={value:g}$"
        err_ax.semilogy(eta,np.abs(pred-ref)+1e-16,styles[i],lw=2.0,color=colors[i],label=label)
    err_ax.set(xlabel=r"$\eta$",ylabel=f"Absolute error in {ylabel}")
    err_ax.grid(True,which="both",alpha=.20,ls="--"); err_ax.legend(fontsize=12,framealpha=.95)
    fig.tight_layout(); savefig(fig,out,stem+"_absolute_error")

    if include_loss:
        fig,loss_ax=plt.subplots(figsize=(6.6,5.0))
        for i,(value,r) in enumerate(res.items()):
            label=rf"${symbol}={value:g}$"
            h=r["history"]
            loss_ax.semilogy(h[:,0],h[:,1],styles[i],lw=2.0,color=colors[i],label=label+" Adam")
            lb=r.get("lbfgs_history",np.empty((0,7)))
            if len(lb):
                # Logged closure calls include rejected line-search trial points; plot the best loss so far.
                loss_ax.semilogy(h[-1,0]+lb[:,0],np.minimum.accumulate(lb[:,1]),":",lw=2.0,color=colors[i],label=label+" L-BFGS")
        loss_ax.set(xlabel="Optimization step",ylabel="Weighted training loss")
        loss_ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
        loss_ax.grid(True,which="both",alpha=.22,ls="--"); loss_ax.legend(ncol=2,fontsize=11)
        fig.tight_layout(); savefig(fig,out,f"figure_{STUDY_NUMBER[study]}_{study}_training_loss")

def load_study_results(out,study):
    """Rebuild the plotting data from saved CSVs without retraining models."""
    table=pd.read_csv(out/f"profiles_{study}.csv"); eta=table["eta"].to_numpy(); results={}
    for value in CASE_STUDIES[study]:
        prefix=f"{study}_{value:g}_"; prediction=np.zeros((len(eta),3)); reference=np.zeros((7,len(eta)))
        prediction[:,0]=table[prefix+"F_Spectral_PINN"]; reference[0]=table[prefix+"F_solve_bvp"]
        fp=table[prefix+"Fprime_Spectral_PINN"].to_numpy(); reference[1]=table[prefix+"Fprime_solve_bvp"]
        prediction[:,1]=table[prefix+"theta_Spectral_PINN"]; reference[3]=table[prefix+"theta_solve_bvp"]
        prediction[:,2]=table[prefix+"phi_Spectral_PINN"]; reference[5]=table[prefix+"phi_solve_bvp"]
        history=pd.read_csv(out/f"training_history_{study}_{value:g}.csv").to_numpy()
        lb_path=out/f"lbfgs_history_{study}_{value:g}.csv"
        lbfgs=pd.read_csv(lb_path).to_numpy() if lb_path.exists() else np.empty((0,7))
        results[value]=dict(prediction=prediction,fp_prediction=fp,reference=reference,
                            history=history,lbfgs_history=lbfgs)
    return eta,results

def regenerate_plots(root,studies):
    for study in studies:
        out=root/study; eta,res=load_study_results(out,study)
        for i,(field,(component,ylabel)) in enumerate(FIELD_OUTPUTS.items()):
            stem=f"figure_{STUDY_NUMBER[study]}_{study}_{field}"
            profile(out,eta,res,study,component,ylabel,stem,include_loss=(i==0))
        print(f"Regenerated separate solution, error, and loss plots for {study}")

def save_study_outputs(out,x,study,res):
    eta=x[:,0]; rows={"eta":eta}; metrics=[]; timing_rows=[]
    for value,r in res.items():
        p,ref,fp=r["prediction"],r["reference"],r["fp_prediction"]
        series={"F":(p[:,0],ref[0]),"Fprime":(fp,ref[1]),"theta":(p[:,1],ref[3]),"phi":(p[:,2],ref[5])}
        for name,(gs,bvp) in series.items():
            key=f"{study}_{value:g}_{name}"; rows[key+"_Spectral_PINN"]=gs; rows[key+"_solve_bvp"]=bvp; e=gs-bvp
            metrics.append(dict(study=study,value=value,variable=name,MAE=np.mean(abs(e)),RMSE=np.sqrt(np.mean(e**2)),
                                MaxAbsError=np.max(abs(e)),RelativeL2=np.linalg.norm(e)/(np.linalg.norm(bvp)+1e-14),
                                R2=1-np.sum(e**2)/(np.sum((bvp-bvp.mean())**2)+1e-30)))
        if "timing" in r:
            tm=r["timing"]
            timing_rows.append(dict(study=study,value=value,**{k:v for k,v in tm.items() if k!="final_losses"},
                **{f"loss_{n}":v for n,v in zip(NAMES,tm["final_losses"])},
                **{f"pinn_{k}":v for k,v in r["wall"].items()},**{f"bvp_{k}":v for k,v in r["wall_ref"].items()}))
        residual_rows={"eta":r["residual_eta"]}
        for j,name in enumerate(NAMES[:3]):
            residual_rows[name+"_residual"]=r["residuals"][:,j]
        pd.DataFrame(residual_rows).to_csv(out/f"spectral_residuals_{study}_{value:g}.csv",index=False)
        pd.DataFrame(r["history"],columns=("iteration","weighted_total",*NAMES)).to_csv(out/f"training_history_{study}_{value:g}.csv",index=False)
        if len(r["lbfgs_history"]):
            pd.DataFrame(r["lbfgs_history"],columns=("evaluation","weighted_total",*NAMES)).to_csv(
                out/f"lbfgs_history_{study}_{value:g}.csv",index=False)
    pd.DataFrame(rows).to_csv(out/f"profiles_{study}.csv",index=False)
    pd.DataFrame(metrics).to_csv(out/f"error_metrics_{study}.csv",index=False)
    if timing_rows: pd.DataFrame(timing_rows).to_csv(out/f"timing_wall_{study}.csv",index=False)
    for i,(field,(component,ylabel)) in enumerate(FIELD_OUTPUTS.items()):
        stem=f"figure_{STUDY_NUMBER[study]}_{study}_{field}"
        profile(out,eta,res,study,component,ylabel,stem,include_loss=(i==0))

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path(__file__).parent/"all_case_results")
    p.add_argument("--studies",nargs="+",choices=tuple(CASE_STUDIES),default=list(CASE_STUDIES),
                   help="Parameter studies to run (default: all six manuscript studies)")
    p.add_argument("--plots-only",action="store_true",
                   help="Regenerate figures from existing CSV files without retraining")
    p.add_argument("--iterations",type=int,default=10000); p.add_argument("--lbfgs-iterations",type=int,default=3000)
    p.add_argument("--lbfgs-restarts",type=int,default=2,
                   help="Independent L-BFGS restarts with refreshed curvature history")
    p.add_argument("--learning-rate",type=float,default=5e-4)
    p.add_argument("--max-grad-norm",type=float,default=10.0)
    p.add_argument("--modes",type=int,default=8,
                   help=argparse.SUPPRESS)  # retained for old command-line compatibility
    p.add_argument("--width",type=int,default=80); p.add_argument("--depth",type=int,default=5)
    p.add_argument("--spectral-points",type=int,default=65,
                   help="Chebyshev--Gauss--Lobatto nodes (Ns+1 in the paper)")
    p.add_argument("--test-points",type=int,default=400)
    p.add_argument("--log-every",type=int,default=250)
    p.add_argument("--lbfgs-log-every",type=int,default=50)
    p.add_argument("--skip-lbfgs",action="store_true"); p.add_argument("--cpu",action="store_true")
    p.add_argument("--float32",action="store_true",
                   help="Use float32 (not recommended for third-order spectral differentiation)")
    p.add_argument("--double",action="store_true",help=argparse.SUPPRESS)
    p.add_argument("--threads",type=int,default=0,help="CPU threads (0 = PyTorch default)")
    p.add_argument("--seed",type=int,default=42); return p.parse_args()

def main():
    a=parse_args()
    if a.threads>0: torch.set_num_threads(a.threads)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    device=torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    # Float64 is essential here: D_eta^3 strongly amplifies float32 roundoff.
    dtype=torch.float32 if a.float32 else torch.float64; a.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"serif","font.size":15,"axes.labelsize":18,"axes.titlesize":16,"mathtext.fontset":"dejavuserif",
                         "pdf.fonttype":42,"ps.fonttype":42})
    if a.plots_only:
        regenerate_plots(a.output,a.studies); return
    meta={**vars(a),"output":str(a.output.resolve()),"device":str(device),"dtype":str(dtype),
          "torch_version":torch.__version__,"cuda_available":torch.cuda.is_available(),
          "cuda_device":torch.cuda.get_device_name(0) if device.type=="cuda" else None}
    (a.output/"run_config.json").write_text(json.dumps(meta,indent=2,default=str),encoding="utf-8")
    print(f"PyTorch {torch.__version__} | device={device} | dtype={dtype}")
    x=np.linspace(0,ETA_MAX,a.test_points)[:,None]
    summary=[]
    for study in a.studies:
        study_out=a.output/study; study_out.mkdir(parents=True,exist_ok=True); results={}
        for value in CASE_STUDIES[study]:
            params=BASE_PARAMETERS.copy(); params[study]=value
            print(f"\nTraining Spectral-PINN for {study}={value:g}")
            results[value]=train(study,value,params,a,x,study_out,device,dtype)
        save_study_outputs(study_out,x,study,results)
        metrics=pd.read_csv(study_out/f"error_metrics_{study}.csv"); summary.append(metrics)
    if len(a.studies)==len(CASE_STUDIES):
        pd.concat(summary,ignore_index=True).to_csv(a.output/"all_case_error_metrics.csv",index=False)
        pd.concat([pd.read_csv(a.output/s/f"timing_wall_{s}.csv") for s in a.studies],
                  ignore_index=True).to_csv(a.output/"all_case_timing_wall.csv",index=False)
    print(f"\nSaved all requested studies to {a.output.resolve()}")
if __name__=="__main__": main()
