"""Thermal-performance assessment: heat-transfer enhancement and collector efficiency.

Procedure (documented in Section 4.9 of the revised manuscript)
-----------------------------------------------------------------
1. Effective properties of water, Al2O3/water and Al2O3-Cu/water from Eqs. (11)-(18).
2. Property-ratio form of the reduced equations (Newtonian limit, beta -> inf;
   Ec -> 0 because Ec = O(1e-6) at collector velocities):
       A1 F''' + F F'' - F'^2 - M_n F' - lam_n = 0,           A1 = nu_hnf/nu_f
       (1+eps th) th'' + eps th'^2 + Pr_n F th' = 0,          Pr_n = nu_f (rho cp)_hnf / k_hnf
       -(1+eps th(0)) th'(0) = Qh_n,                           Qh_n = k_f / k_hnf
   with M_n = M_f (sigma_hnf/sigma_f)/(rho_hnf/rho_f), lam_n = lam_f rho_f/rho_hnf.
   The temperature scale is dT_ref = q_W sqrt(nu_f/b)/k_f, common to all fluids, so
   that the same physical wall heat flux q_W is imposed on every fluid.
3. Heat-transfer coefficient h = q_W/(T_W - T_inf) = k_f sqrt(b/nu_f) / th(0), hence
   Nu_x Re_x^{-1/2} = 1/th(0) and the enhancement is E = th_bf(0)/th_nf(0) - 1.
4. Collector efficiency from the Hottel-Whillier-Bliss equation
       eta_c = F_R [ (tau alpha) - U_L (T_in - T_a)/G ],
   F' = h/(h+U_L),  F_R = (m cp/(A U_L)) [1 - exp(-A U_L F'/(m cp))].
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.integrate import solve_bvp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ETA_MAX = 8.0
# Thermophysical properties at 25 C (Oztop & Abu-Nada 2008; Devi & Devi 2016)
WATER = dict(rho=997.1, cp=4179.0, k=0.613, sigma=0.05, mu=8.91e-4)
AL2O3 = dict(rho=3970.0, cp=765.0, k=40.0, sigma=3.69e7)
CU    = dict(rho=8933.0, cp=385.0, k=401.0, sigma=5.96e7)
BASE  = dict(M=0.5, lam=0.0, epsilon=0.1)          # lam = 0: well-posed semi-infinite layer (see text)
COLLECTOR = dict(G=1000.0, tau=0.90, alpha=0.95, U_L=5.0, m_per_A=0.02, b=0.01)

def maxwell(k_bf, k_p, phi):
    return k_bf*(k_p+2*k_bf-2*phi*(k_bf-k_p))/(k_p+2*k_bf+phi*(k_bf-k_p))

def elec(s_bf, s_p, phi):
    r=s_p/s_bf; return s_bf*(1+3*phi*(r-1)/((r+2)-phi*(r-1)))

def properties(phi1, phi2):
    f=WATER
    rho=(1-phi2)*((1-phi1)*f["rho"]+phi1*AL2O3["rho"])+phi2*CU["rho"]
    rcp=(1-phi2)*((1-phi1)*f["rho"]*f["cp"]+phi1*AL2O3["rho"]*AL2O3["cp"])+phi2*CU["rho"]*CU["cp"]
    mu=f["mu"]/((1-phi1)**2.5*(1-phi2)**2.5)
    k=maxwell(maxwell(f["k"],AL2O3["k"],phi1),CU["k"],phi2)
    s=elec(elec(f["sigma"],AL2O3["sigma"],phi1),CU["sigma"],phi2)
    return dict(rho=rho, rhocp=rcp, cp=rcp/rho, mu=mu, nu=mu/rho, k=k, sigma=s)

def reduced_parameters(phi1, phi2):
    f=WATER; p=properties(phi1,phi2); nu_f=f["mu"]/f["rho"]
    return dict(A1=p["nu"]/nu_f,
                M=BASE["M"]*(p["sigma"]/f["sigma"])/(p["rho"]/f["rho"]),
                lam=BASE["lam"]*f["rho"]/p["rho"],
                Pr=nu_f*p["rhocp"]/p["k"], Qh=f["k"]/p["k"], epsilon=BASE["epsilon"],
                k_ratio=p["k"]/f["k"], cp=p["cp"], props=p)

def solve(rp, n=400):
    A1,M,lam,Pr,Qh,eps=rp["A1"],rp["M"],rp["lam"],rp["Pr"],rp["Qh"],rp["epsilon"]
    def ode(x,y):
        f,fp,fpp,th,thp=y
        return np.vstack((fp,fpp,(-f*fpp+fp**2+M*fp+lam)/A1,thp,-(eps*thp**2+Pr*f*thp)/(1+eps*th)))
    def bc(a,b): return np.array([a[0],a[1]-1,-(1+eps*a[3])*a[4]-Qh,b[1],b[3]])
    x=np.linspace(0,ETA_MAX,n); e=np.exp(-x)
    sol=solve_bvp(ode,bc,x,np.vstack((1-e,e,-e,Qh*e,-Qh*e)),tol=1e-8,max_nodes=100000)
    assert sol.success, sol.message
    return sol

def efficiency(h, cp, Tstar):
    c=COLLECTOR; FP=h/(h+c["U_L"]); mcp=c["m_per_A"]*cp
    FR=mcp/c["U_L"]*(1-np.exp(-c["U_L"]*FP/mcp))
    return FR*(c["tau"]*c["alpha"]-c["U_L"]*Tstar), FP, FR

def main(out=Path("thermal_performance")):
    out.mkdir(exist_ok=True); nu_f=WATER["mu"]/WATER["rho"]; c=COLLECTOR
    fluids={"Water (base fluid)":(0.0,0.0),
            "Al2O3/water nanofluid (phi1 = 4%)":(0.04,0.0),
            "Al2O3-Cu/water hybrid (phi1 = phi2 = 2%)":(0.02,0.02)}
    rows=[]; th_bf=None
    for name,(p1,p2) in fluids.items():
        rp=reduced_parameters(p1,p2); sol=solve(rp); th0=sol.sol(0)[3]
        if th_bf is None: th_bf=th0
        h=WATER["k"]*np.sqrt(c["b"]/nu_f)/th0
        eta0,FP,FR=efficiency(h,rp["cp"],0.0)
        rows.append(dict(fluid=name,phi1=p1,phi2=p2,k_ratio=rp["k_ratio"],nu_ratio=rp["A1"],
                         Pr_eff=rp["Pr"],M_eff=rp["M"],Qh=rp["Qh"],theta0=th0,
                         Nu_reduced=1/th0,h_W_m2K=h,enhancement_pct=100*(th_bf/th0-1),
                         F_prime=FP,F_R=FR,eta_max_pct=100*eta0,
                         Fpp0=sol.sol(0)[2]))
    df=pd.DataFrame(rows); df.to_csv(out/"thermal_performance_summary.csv",index=False)
    print(df.to_string())
    # loading sweep (equal total loading)
    sweep=[]
    for tot in np.linspace(0,0.04,9):
        for kind,(a,b) in {"mono":(tot,0),"hybrid":(tot/2,tot/2)}.items():
            rp=reduced_parameters(a,b); th0=solve(rp).sol(0)[3]
            sweep.append(dict(total_phi=tot,kind=kind,theta0=th0,enhancement_pct=100*(th_bf/th0-1)))
    sw=pd.DataFrame(sweep); sw.to_csv(out/"loading_sweep.csv",index=False)
    # efficiency curves
    Ts=np.linspace(0,0.08,81); curves={"Tstar":Ts}
    for r in rows: curves[r["fluid"]]=100*efficiency(r["h_W_m2K"],properties(r["phi1"],r["phi2"])["cp"],Ts)[0]
    pd.DataFrame(curves).to_csv(out/"efficiency_curves.csv",index=False)
    plt.rcParams.update({"font.family":"serif","font.size":12,"mathtext.fontset":"dejavuserif"})
    fig,ax=plt.subplots(1,2,figsize=(12,4.6))
    for kind,st in (("mono","o-"),("hybrid","s--")):
        g=sw[sw.kind==kind]
        ax[0].plot(100*g.total_phi,g.enhancement_pct,st,lw=2,
                   label="Al$_2$O$_3$/water" if kind=="mono" else "Al$_2$O$_3$–Cu/water (1:1)")
    ax[0].set(xlabel="Total nanoparticle volume fraction (%)",ylabel="Increase in $h$ relative to water (%)")
    ax[0].grid(alpha=.3,ls="--"); ax[0].legend(); ax[0].set_title("(a)")
    for (k,st) in zip(list(curves)[1:],("-","--","-.")):
        lab={"Water (base fluid)":"Water","Al2O3/water nanofluid (phi1 = 4%)":r"Al$_2$O$_3$/water ($\Phi_1=4\%$)",
             "Al2O3-Cu/water hybrid (phi1 = phi2 = 2%)":r"Al$_2$O$_3$–Cu/water ($\Phi_1=\Phi_2=2\%$)"}[k]
        ax[1].plot(Ts,curves[k],st,lw=2,label=lab)
    ax[1].set(xlabel=r"Reduced temperature $(T_{\mathrm{in}}-T_{\mathrm{amb}})/I$ (m$^2$K/W)",ylabel="Collector efficiency (%)")
    ax[1].grid(alpha=.3,ls="--"); ax[1].legend(fontsize=9); ax[1].set_title("(b)")
    fig.tight_layout(); fig.savefig(out/"figure_thermal_performance.pdf",bbox_inches="tight")
    fig.savefig(out/"figure_thermal_performance.png",dpi=400,bbox_inches="tight")
    (out/"assumptions.json").write_text(json.dumps(dict(water=WATER,al2o3=AL2O3,cu=CU,base=BASE,collector=COLLECTOR),indent=1))

if __name__=="__main__": main()
