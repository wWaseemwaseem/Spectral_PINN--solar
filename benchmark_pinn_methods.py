"""Benchmark of the Spectral-PINN against classical and recent PINN variants.

All methods solve the same reduced boundary-value problem (Eqs. 23-25 with the
wall heat-flux condition, Eq. 31) at the baseline parameter set and are scored
against the SciPy ``solve_bvp`` reference on a uniform 400-point grid.

Methods
-------
spectral        Spectral-PINN of this work (CGL nodes + Chebyshev D-matrices)
spectral_noreg  ablation: Spectral-PINN without the spectral-tail penalty
autograd_cgl    ablation: same MLP and CGL nodes, derivatives by autograd
vanilla         PINN of Raissi et al. (2019), random collocation, static weights
sa              Self-adaptive PINN, McClenny & Braga-Neto (J. Comput. Phys. 2023)
rba             Residual-based attention PINN, Anagnostopoulos et al. (CMAME 2024)
pirate          PirateNet, Wang, Li, Chen & Perdikaris (JMLR 2024)
cpikan          Chebyshev physics-informed KAN, Shukla et al. (CMAME 2024)
config          ConFIG conflict-free gradient update, Liu, Chu & Thuerey (ICLR 2025)

Every method uses the same two-stage protocol (Adam, cosine-annealed learning
rate, followed by L-BFGS with strong-Wolfe line search; adaptive weights are
frozen during L-BFGS), the same number of interior collocation points and
float64 arithmetic.
"""
from __future__ import annotations
import argparse, copy, json, math, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
import spectral_pinn_paper_code as S

P = S.BASE_PARAMETERS.copy()
ETA = S.ETA_MAX
W_STATIC = (2., 5., 1., 50.)   # momentum, energy, concentration, boundary (as Spectral-PINN)

# ----------------------------------------------------------------------------- models
class MLP(nn.Module):
    def __init__(self, width=80, depth=5, n_in=1, n_out=3):
        super().__init__(); s=[n_in]+[width]*depth; L=[]
        for a,b in zip(s[:-1],s[1:]):
            l=nn.Linear(a,b); nn.init.xavier_uniform_(l.weight); nn.init.zeros_(l.bias); L+=[l,nn.Tanh()]
        o=nn.Linear(width,n_out); nn.init.xavier_uniform_(o.weight); nn.init.zeros_(o.bias)
        self.net=nn.Sequential(*L,o)
    def forward(self,x): return self.net(2*x/ETA-1)

class PirateNet(nn.Module):
    """PirateNet (Wang et al., JMLR 25 (2024) 402): random Fourier features, gated
    blocks and adaptive residual connections with alpha initialised at zero."""
    def __init__(self, width=80, blocks=3, s=1.0):
        super().__init__()
        self.register_buffer("B", torch.randn(1, width//2)*s)
        def lin(a,b):
            l=nn.Linear(a,b); nn.init.xavier_uniform_(l.weight); nn.init.zeros_(l.bias); return l
        self.U=lin(width,width); self.V=lin(width,width)
        self.blocks=nn.ModuleList([nn.ModuleList([lin(width,width) for _ in range(3)]) for _ in range(blocks)])
        self.alpha=nn.Parameter(torch.zeros(blocks)); self.out=lin(width,3)
    def embed(self,x):
        z=(2*x/ETA-1)@self.B.to(x.dtype); return torch.cat([torch.cos(z),torch.sin(z)],1)
    def features(self,x):
        h=self.embed(x); U=torch.tanh(self.U(h)); V=torch.tanh(self.V(h))
        for a,(l1,l2,l3) in zip(self.alpha,self.blocks):
            f=torch.tanh(l1(h)); z=f*U+(1-f)*V
            g=torch.tanh(l2(z)); z=g*U+(1-g)*V
            q=torch.tanh(l3(z)); h=a*q+(1-a)*h
        return h
    def forward(self,x): return self.out(self.features(x))
    def physics_init(self, x):
        """Physics-informed initialisation: least-squares fit of the output layer to
        a boundary-condition-satisfying initial guess (same guess as the BVP solver)."""
        with torch.no_grad():
            e=torch.exp(-x); el=math.exp(-ETA); q=P["Qh"]
            target=torch.cat([1-e, q*(e-el), (e-el)/(1-el)],1)
            H=torch.cat([self.features(x),torch.ones_like(x)],1)
            sol=torch.linalg.lstsq(H,target).solution
            self.out.weight.copy_(sol[:-1].T); self.out.bias.copy_(sol[-1])

class ChebyKANLayer(nn.Module):
    def __init__(self, n_in, n_out, degree):
        super().__init__(); self.d=degree
        self.c=nn.Parameter(torch.randn(n_in,n_out,degree+1)/(n_in*(degree+1)))
    def forward(self,x):
        x=torch.tanh(x); T=[torch.ones_like(x),x]
        for k in range(2,self.d+1): T.append(2*x*T[-1]-T[-2])
        T=torch.stack(T[:self.d+1],-1)                       # (N, n_in, d+1)
        return torch.einsum("nid,iod->no",T,self.c)

class CPIKAN(nn.Module):
    """Chebyshev KAN (Shukla et al., CMAME 431 (2024) 117290), widths [1,16,16,16,3], degree 5."""
    def __init__(self, width=16, depth=3, degree=5):
        super().__init__(); s=[1]+[width]*depth+[3]
        self.layers=nn.ModuleList([ChebyKANLayer(a,b,degree) for a,b in zip(s[:-1],s[1:])])
    def forward(self,x):
        h=2*x/ETA-1
        for l in self.layers: h=l(h)
        return h

# ----------------------------------------------------------------------------- residuals
def d(y,x): return torch.autograd.grad(y,x,torch.ones_like(y),create_graph=True)[0]

def autograd_fields(model,x):
    y=model(x); f,th,ph=y[:,:1],y[:,1:2],y[:,2:3]
    fp=d(f,x); fpp=d(fp,x); fppp=d(fpp,x); thp=d(th,x); thpp=d(thp,x); php=d(ph,x); phpp=d(php,x)
    return f,fp,fpp,fppp,th,thp,thpp,ph,php,phpp

def residuals(fields,p=P):
    f,fp,fpp,fppp,th,thp,thpp,ph,php,phpp=fields; c=1+1/p["beta"]
    rF=c*fppp+f*fpp-fp**2-p["M"]*fp-p["lam"]
    rT=(1+p["epsilon"]*th)*thpp+p["epsilon"]*thp**2+p["Pr"]*f*thp+p["Ec"]*c*fpp**2+p["M"]*p["Ec"]*fp**2
    rC=phpp+p["Sc"]*f*php-p["Kr"]*ph
    return torch.cat([rF,rT,rC],1)

def autograd_bc(model,dtype,p=P):
    x0=torch.zeros(1,1,dtype=dtype,requires_grad=True); x1=torch.full((1,1),ETA,dtype=dtype,requires_grad=True)
    y0=model(x0); y1=model(x1)
    fp0=d(y0[:,:1],x0); th0p=d(y0[:,1:2],x0); fp1=d(y1[:,:1],x1)
    return torch.cat([y0[:,0],fp0[:,0]-1,-(1+p["epsilon"]*y0[:,1])*th0p[:,0]-p["Qh"],y0[:,2]-1,
                      fp1[:,0],y1[:,1],y1[:,2]])

# ----------------------------------------------------------------------------- loss builders
class Problem:
    """Returns per-point residuals (N,3) and the 7 boundary residuals."""
    def __init__(self, method, model, n, dtype, seed):
        self.method=method; self.model=model; self.dtype=dtype
        if method.startswith("spectral"):
            self.x,self.mats=S.chebyshev_grid(n,"cpu",dtype)
        elif method=="autograd_cgl":
            self.x,_=S.chebyshev_grid(n,"cpu",dtype); self.x=self.x[1:-1].clone().requires_grad_(True)
        else:
            g=torch.Generator().manual_seed(seed+1000)
            self.x=(torch.rand(n-2,1,generator=g,dtype=dtype)*ETA).requires_grad_(True)
    def compute(self):
        if self.method.startswith("spectral"):
            y,fp,thp,r=S.spectral_evaluate(self.model,self.x,self.mats,P)
            bc=torch.stack((y[0,0],fp[0]-1,-(1+P["epsilon"]*y[0,1])*thp[0]-P["Qh"],y[0,2]-1,fp[-1],y[-1,1],y[-1,2]))
            return r[1:-1],bc
        return residuals(autograd_fields(self.model,self.x)), autograd_bc(self.model,self.dtype)
    def spectral_penalty(self):
        return S.losses(self.model,self.x,self.mats,P)[4]

def task_losses(prob, r, bc, w_pts=None, w_bc=None):
    if w_pts is None: L=[r[:,i].square().mean() for i in range(3)]
    else: L=[(w_pts[:,i]*r[:,i]).square().mean() for i in range(3)]
    Lb=bc.square().mean() if w_bc is None else (w_bc*bc).square().mean()
    return L+[Lb]

def config_direction(grads):
    """ConFIG update (Liu, Chu & Thuerey, ICLR 2025): conflict-free direction whose
    projection on every normalised task gradient is equal and positive."""
    G=torch.stack(grads)                                  # (m, P)
    U=G/G.norm(dim=1,keepdim=True).clamp_min(1e-30)
    gu=torch.linalg.pinv(U)@torch.ones(G.shape[0],dtype=G.dtype)
    gu=gu/gu.norm().clamp_min(1e-30)
    return (G@gu).sum()*gu

# ----------------------------------------------------------------------------- train / score
def build(method):
    if method=="pirate": return PirateNet()
    if method=="cpikan": return CPIKAN()
    return MLP()

def train(method, seed, n=129, adam_iters=10000, lbfgs_iters=3000, restarts=2, lr=5e-4):
    torch.manual_seed(seed); np.random.seed(seed); dtype=torch.float64
    model=build(method).double()
    prob=Problem(method,model,n,dtype,seed)
    # PirateNet's physics-informed initialisation (least-squares fit of the output layer to
    # initial-condition data) needs data that do not exist for this boundary-value problem; a
    # least-squares fit to a heuristic guess was found to degrade training, so the standard
    # initialisation (alpha = 0, Xavier weights) is used.
    params=list(model.parameters()); opt=torch.optim.Adam(params,lr=lr)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=adam_iters)
    nint=n-2; w_pts=None; w_bc=None
    if method=="sa":     # trainable point-wise weights, gradient ascent
        lam_r=torch.ones(nint,3,dtype=dtype,requires_grad=True); lam_b=torch.ones(7,dtype=dtype,requires_grad=True)
        opt_l=torch.optim.Adam([lam_r,lam_b],lr=5e-3,maximize=True)
    if method=="rba":
        lam_r=torch.zeros(nint,3,dtype=dtype); gamma,eta_star=0.999,0.01
    wstat=torch.tensor(W_STATIC,dtype=dtype)
    best=float("inf"); best_state=copy.deepcopy(model.state_dict())
    t0=time.perf_counter()
    for it in range(adam_iters):
        r,bc=prob.compute()
        if method=="sa":
            L=task_losses(prob,r,bc,lam_r**2,lam_b**2); total=sum(L)
            opt.zero_grad(); opt_l.zero_grad(); total.backward(); opt.step(); opt_l.step()
        elif method=="rba":
            with torch.no_grad():
                a=r.abs(); lam_r.mul_(gamma).add_(eta_star*a/a.max(0).values.clamp_min(1e-30))
            L=task_losses(prob,r,bc,1+lam_r,None); L[3]=W_STATIC[3]*L[3]; total=sum(L)
            opt.zero_grad(); total.backward(); opt.step()
        elif method=="config":
            L=task_losses(prob,r,bc); grads=[]
            for i,l in enumerate(L):
                g=torch.autograd.grad(l,params,retain_graph=i<len(L)-1,allow_unused=True)
                grads.append(torch.cat([(torch.zeros_like(p) if gi is None else gi).reshape(-1) for p,gi in zip(params,g)]))
            direction=config_direction(grads); k=0
            for p in params:
                m=p.numel(); p.grad=direction[k:k+m].view_as(p).clone(); k+=m
            opt.step(); total=sum(w*l for w,l in zip(wstat,L)).detach()
        else:
            L=task_losses(prob,r,bc); total=sum(w*l for w,l in zip(wstat,L))
            if method in ("spectral","spectral65"): total=total+1e-6*prob.spectral_penalty()
            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(params,10.0); opt.step()
        sched.step()
        v=float(sum(w*l for w,l in zip(wstat,L)).detach())
        if np.isfinite(v) and v<best: best=v; best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state); t_adam=time.perf_counter()-t0
    # ---- L-BFGS refinement (adaptive weights frozen)
    if method=="sa": fw_pts,fw_bc=(lam_r**2).detach(),(lam_b**2).detach()
    elif method=="rba": fw_pts,fw_bc=(1+lam_r).detach(),None
    else: fw_pts,fw_bc=None,None
    def objective():
        r,bc=prob.compute(); L=task_losses(prob,r,bc,fw_pts,fw_bc)
        if method in ("sa",): return sum(L)
        if method=="rba": return L[0]+L[1]+L[2]+W_STATIC[3]*L[3]
        tot=sum(w*l for w,l in zip(wstat,L))
        if method in ("spectral","spectral65"): tot=tot+1e-6*prob.spectral_penalty()
        return tot
    t1=time.perf_counter()
    for _ in range(restarts):
        o2=torch.optim.LBFGS(params,max_iter=lbfgs_iters,max_eval=int(1.5*lbfgs_iters),history_size=150,
                             tolerance_grad=1e-12,tolerance_change=1e-15,line_search_fn="strong_wolfe")
        def closure():
            o2.zero_grad(); l=objective(); l.backward(); return l
        o2.step(closure)
    t_lbfgs=time.perf_counter()-t1
    return model,t_adam,t_lbfgs

def score(model, ref, n_test=400, seed=0):
    dtype=torch.float64; eta=np.linspace(0,ETA,n_test)
    x=torch.tensor(eta[:,None],dtype=dtype,requires_grad=True)
    t0=time.perf_counter()
    for _ in range(50):
        xq=torch.tensor(eta[:,None],dtype=dtype,requires_grad=True); yq=model(xq); _=d(yq[:,:1],xq)
    t_inf=(time.perf_counter()-t0)/50
    fields=autograd_fields(model,x)
    f,fp,fpp,_,th,thp,_,ph,php,_=[a.detach().numpy()[:,0] for a in fields]
    R=ref.sol(eta); out={}
    for name,pred,true in (("F",f,R[0]),("Fprime",fp,R[1]),("theta",th,R[3]),("phi",ph,R[5])):
        e=pred-true
        out[f"RelL2_{name}"]=np.linalg.norm(e)/np.linalg.norm(true)
        out[f"MaxAbs_{name}"]=np.abs(e).max()
        out[f"R2_{name}"]=1-np.sum(e**2)/np.sum((true-true.mean())**2)
    out["RelErr_Fpp0"]=abs(fpp[0]-R[2][0])/abs(R[2][0])
    out["RelErr_theta0"]=abs(th[0]-R[3][0])/abs(R[3][0])
    out["RelErr_mphip0"]=abs(php[0]-R[6][0])/abs(R[6][0])
    # Off-collocation physics residual (generalisation of the PDE constraint).
    g=torch.Generator().manual_seed(12345)
    xr=(torch.rand(1000,1,generator=g,dtype=dtype)*ETA*0.98+0.01*ETA).requires_grad_(True)
    rr=residuals(autograd_fields(model,xr)).detach().numpy()
    out["OffGridResidualRMS"]=float(np.sqrt(np.mean(rr**2)))
    out["t_inference"]=t_inf
    return out

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--methods",nargs="+",default=["spectral","spectral_noreg","autograd_cgl","vanilla","sa","rba","pirate","cpikan","config"])
    ap.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    ap.add_argument("--adam",type=int,default=10000); ap.add_argument("--lbfgs",type=int,default=3000)
    ap.add_argument("--restarts",type=int,default=1); ap.add_argument("--n",type=int,default=129)
    ap.add_argument("--threads",type=int,default=1); ap.add_argument("--out",type=Path,default=Path("benchmark_results"))
    a=ap.parse_args(); torch.set_num_threads(a.threads); a.out.mkdir(exist_ok=True,parents=True)
    t0=time.perf_counter(); ref=S.numerical_solution(P,400); t_bvp=time.perf_counter()-t0
    for m in a.methods:
        for s in a.seeds:
            f=a.out/f"{m}_seed{s}.json"
            if f.exists(): continue
            print(f"== {m} seed {s}",flush=True)
            model,ta,tl=train(m,s,a.n,a.adam,a.lbfgs,a.restarts)
            res=dict(method=m,seed=s,t_adam=ta,t_lbfgs=tl,t_train=ta+tl,t_bvp=t_bvp,
                     n_params=sum(p.numel() for p in model.parameters()),**score(model,ref))
            f.write_text(json.dumps(res,indent=1)); print(res,flush=True)
