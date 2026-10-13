"""Parametric Spectral-PINN surrogate: one network for a box of operating conditions.

Input  : (eta, M, Ec, Qh)  with  M in [0.5, 1.3], Ec in [2, 6], Qh in [0.1, 0.5]
Output : F, theta, phi
All other parameters are fixed at the baseline values of the manuscript.  The
governing residuals are evaluated with the same Chebyshev-Gauss-Lobatto
differentiation matrices as the single-case Spectral-PINN; new parameter samples
are drawn at every Adam iteration.  After training, the surrogate is compared with
SciPy solve_bvp on unseen parameter combinations (interpolation) and outside the
training box (extrapolation), and the cost of many-query evaluation is measured.
"""
import argparse, copy, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from scipy.stats import qmc
import spectral_pinn_paper_code as S

BOX = {"M": (0.5, 1.3), "Ec": (2.0, 6.0), "Qh": (0.1, 0.5)}
KEYS = list(BOX)
ETA = S.ETA_MAX

class ParamNet(nn.Module):
    def __init__(self, width=96, depth=5):
        super().__init__(); s=[1+len(KEYS)]+[width]*depth; L=[]
        for a,b in zip(s[:-1],s[1:]):
            l=nn.Linear(a,b); nn.init.xavier_uniform_(l.weight); nn.init.zeros_(l.bias); L+=[l,nn.Tanh()]
        o=nn.Linear(width,3); nn.init.xavier_uniform_(o.weight); nn.init.zeros_(o.bias)
        self.net=nn.Sequential(*L,o)
    def forward(self, eta, p):        # eta (N,1), p (B,3) physical -> (B,N,3)
        lo=torch.tensor([BOX[k][0] for k in KEYS],dtype=p.dtype); hi=torch.tensor([BOX[k][1] for k in KEYS],dtype=p.dtype)
        pn=2*(p-lo)/(hi-lo)-1; B,N=p.shape[0],eta.shape[0]
        z=torch.cat([(2*eta/ETA-1).expand(B,N,1) if eta.dim()==2 else eta, pn[:,None,:].expand(B,N,len(KEYS))],-1)
        return self.net(z)

def batch_losses(model, eta, mats, p):
    d1,d2,d3=mats; y=model(eta,p); f,th,ph=y[...,0],y[...,1],y[...,2]
    mm=lambda D,v: v@D.T
    fp,fpp,fppp=mm(d1,f),mm(d2,f),mm(d3,f); thp,thpp=mm(d1,th),mm(d2,th); php,phpp=mm(d1,ph),mm(d2,ph)
    b=S.BASE_PARAMETERS; c=1+1/b["beta"]; M=p[:,0:1]; Ec=p[:,1:2]; Qh=p[:,2]
    rF=c*fppp+f*fpp-fp**2-M*fp-b["lam"]
    rT=(1+b["epsilon"]*th)*thpp+b["epsilon"]*thp**2+b["Pr"]*f*thp+Ec*c*fpp**2+M*Ec*fp**2
    rC=phpp+b["Sc"]*f*php-b["Kr"]*ph
    bc=torch.stack([f[:,0],fp[:,0]-1,-(1+b["epsilon"]*th[:,0])*thp[:,0]-Qh,ph[:,0]-1,fp[:,-1],th[:,-1],ph[:,-1]],1)
    return [rF[:,1:-1].square().mean(),rT[:,1:-1].square().mean(),rC[:,1:-1].square().mean(),bc.square().mean()]

def sample(n, rng, dtype):
    lo=np.array([BOX[k][0] for k in KEYS]); hi=np.array([BOX[k][1] for k in KEYS])
    return torch.tensor(lo+(hi-lo)*rng.random((n,len(KEYS))),dtype=dtype)

def params_dict(row):
    p=S.BASE_PARAMETERS.copy(); p.update({k:float(v) for k,v in zip(KEYS,row)}); return p

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--adam",type=int,default=20000)
    ap.add_argument("--lbfgs",type=int,default=3000); ap.add_argument("--batch",type=int,default=24)
    ap.add_argument("--n",type=int,default=65); ap.add_argument("--threads",type=int,default=1)
    ap.add_argument("--out",type=Path,default=Path("parametric_results")); ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args(); torch.set_num_threads(a.threads); a.out.mkdir(exist_ok=True)
    torch.manual_seed(a.seed); rng=np.random.default_rng(a.seed); dt=torch.float64
    eta,mats=S.chebyshev_grid(a.n,"cpu",dt); model=ParamNet().double()
    w=torch.tensor((2.,5.,1.,50.),dtype=dt)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.adam)
    hist=[]; t0=time.perf_counter()
    for it in range(1,a.adam+1):
        L=batch_losses(model,eta,mats,sample(a.batch,rng,dt)); tot=sum(wi*l for wi,l in zip(w,L))
        opt.zero_grad(); tot.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.); opt.step(); sch.step()
        if it%500==0: hist.append([it,tot.item()]); print(it,tot.item(),flush=True)
    t_adam=time.perf_counter()-t0
    # L-BFGS on a fixed scrambled-Sobol design of 64 operating points
    P_fix=torch.tensor(qmc.scale(qmc.Sobol(3,seed=a.seed).random(64),[BOX[k][0] for k in KEYS],[BOX[k][1] for k in KEYS]),dtype=dt)
    o2=torch.optim.LBFGS(model.parameters(),max_iter=a.lbfgs,max_eval=int(1.5*a.lbfgs),history_size=100,
                         tolerance_grad=1e-12,tolerance_change=1e-15,line_search_fn="strong_wolfe")
    def closure():
        o2.zero_grad(); l=sum(wi*li for wi,li in zip(w,batch_losses(model,eta,mats,P_fix))); l.backward(); return l
    t1=time.perf_counter(); o2.step(closure); t_lbfgs=time.perf_counter()-t1
    torch.save(model.state_dict(),a.out/"parametric_spectral_pinn.pt")
    pd.DataFrame(hist,columns=["iteration","weighted_loss"]).to_csv(a.out/"training_history.csv",index=False)

    # ---------------- evaluation
    x=np.linspace(0,ETA,400); xt=torch.tensor(x[:,None],dtype=dt)
    test_int=qmc.scale(qmc.LatinHypercube(3,seed=7).random(100),[BOX[k][0] for k in KEYS],[BOX[k][1] for k in KEYS])
    grid=[(m,4.,.3) for m in (.5,.9,1.3)]+[(.5,e,.3) for e in (2.,4.,6.)]+[(.5,4.,q) for q in (.1,.3,.5)]
    extra=[(1.5,4.,.3),(0.5,7.,.3),(0.5,4.,.6),(1.5,7.,.6)]
    rows=[]
    def evaluate(set_name, P):
        Pt=torch.tensor(np.asarray(P),dtype=dt)
        with torch.no_grad(): t0=time.perf_counter(); Y=model(xt,Pt).numpy(); t_inf=time.perf_counter()-t0
        t_b=0.0
        for i,row in enumerate(P):
            t0=time.perf_counter(); ref=S.numerical_solution(params_dict(row),400); t_b+=time.perf_counter()-t0
            R=ref.sol(x); r=dict(set=set_name,**{k:v for k,v in zip(KEYS,row)})
            for j,(nm,ri) in enumerate((("F",0),("theta",3),("phi",5))):
                e=Y[i,:,j]-R[ri]; r[f"RelL2_{nm}"]=np.linalg.norm(e)/np.linalg.norm(R[ri]); r[f"MaxAbs_{nm}"]=np.abs(e).max()
            r["RelErr_theta0"]=abs(Y[i,0,1]-R[3][0])/abs(R[3][0]); rows.append(r)
        return t_inf, t_b
    t_inf_int,t_bvp_int=evaluate("interpolation (100 LHS points)",test_int)
    evaluate("manuscript grid (9 cases)",grid); evaluate("extrapolation (4 cases)",extra)
    df=pd.DataFrame(rows); df.to_csv(a.out/"parametric_errors.csv",index=False)
    summ=df.groupby("set").agg(n=("RelL2_F","size"),RelL2_F_mean=("RelL2_F","mean"),RelL2_theta_mean=("RelL2_theta","mean"),
        RelL2_phi_mean=("RelL2_phi","mean"),RelL2_theta_max=("RelL2_theta","max"),MaxAbs_theta_max=("MaxAbs_theta","max"),
        RelErr_theta0_mean=("RelErr_theta0","mean"),RelErr_theta0_max=("RelErr_theta0","max")).reset_index()
    summ.to_csv(a.out/"parametric_summary.csv",index=False); print(summ.to_string())
    t_train=t_adam+t_lbfgs; per_bvp=t_bvp_int/100; per_inf=t_inf_int/100
    cost=dict(t_adam=t_adam,t_lbfgs=t_lbfgs,t_train=t_train,t_inference_100=t_inf_int,t_bvp_100=t_bvp_int,
              per_query_inference=per_inf,per_query_bvp=per_bvp,speedup_per_query=per_bvp/per_inf,
              break_even_queries=t_train/(per_bvp-per_inf))
    (a.out/"parametric_cost.json").write_text(json.dumps(cost,indent=1)); print(cost)

if __name__=="__main__": main()
