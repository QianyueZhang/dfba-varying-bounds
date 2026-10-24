"""Check a substrate-supported, phenomenological biomass-transfer benchmark.

This uses the source intracellular network with a declared extracellular
conversion efficiency. It is not a strain-specific model of E. coli predation.
No old curve is fitted and no infeasible active state is continued as a result.
"""
from pathlib import Path
import argparse, hashlib, json, os, platform, sys, time
import numpy as np
import scipy
from scipy.io import loadmat
from scipy.sparse import csc_matrix
from scipy.optimize import linprog
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research_completion/scripts'))
from primal_dual import solve

SOURCE=ROOT/'data/models/biomass_transfer'
OUT=ROOT/'research_completion/runs/python_consistency/biomass_transfer'
def flat(x):return np.asarray(x.toarray() if hasattr(x,'toarray') else x,float).ravel()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Model:
    def __init__(self,mu):
        z=loadmat(SOURCE/'S1_Model.mat',simplify_cells=True)['Struct']
        w=loadmat(SOURCE/'S2_Model.mat',simplify_cells=True)['Struct']
        for key in ['A','b','c','lb','ub']:assert np.array_equal(flat(z[key]),flat(w[key])),key
        raw=loadmat(SOURCE/'e_coli_core.mat',simplify_cells=True)['e_coli_core']
        self.A=csc_matrix(z['A']);self.rawA=csc_matrix(raw['S']);self.b=flat(z['b']);self.c=flat(z['c'])
        self.lo=flat(z['lb']);self.up=flat(z['ub']);self.bio=12;self.g=27;self.name='shared_core';self.mu=mu
        assert raw['rxns'][self.g]=='EX_glc__D_e' and raw['rxns'][10]=='ATPM'
        self.warm=None;self.maxeq=0.;self.maxpd=0.;self.minslack=np.inf;self.calls=0;self.cache=None
    def bounds(self,G):
        if G<0:raise RuntimeError('Negative glucose trial state')
        l=self.lo.copy();u=self.up.copy();l[self.g]=-10*G/(1+G)
        return l,u
    def lp(self,G):
        l,u=self.bounds(G)
        a=linprog(-self.c,A_eq=self.A,b_eq=self.b,bounds=list(zip(l,u)),method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        if not a.success:raise RuntimeError(f'Active model infeasible at glucose {G}: {a.message}')
        return a
    def flux(self,G):
        if self.cache is not None and G==self.cache[0]:return self.cache[1]
        try:s=solve(self,G,mu=self.mu,tol=1e-11,warm=self.warm)
        except RuntimeError:s=solve(self,G,mu=self.mu,tol=1e-11)
        self.warm=s[:4];v=s[0];d=s[-1];self.calls+=1
        self.maxeq=max(self.maxeq,float(np.max(abs(self.rawA@v))));self.maxpd=max(self.maxpd,d['primal_dual_residual']);self.minslack=min(self.minslack,d['minimum_slack'])
        self.cache=(G,v)
        return v

def run(name,k=.1,eta=.5,mu=1e-8,form='amount',method='DOP853',rtol=1e-9,atol=1e-11):
    model=Model(mu);V0=5.;F=1.;Gin=20.;tf=20.;x0=.001
    # The first two variables are logarithmic amounts/concentrations, ensuring
    # positivity without clipping. Accumulators independently check each pool.
    y0=np.r_[np.log([x0*V0,x0*V0]) if form=='amount' else np.log([x0,x0]),Gin*V0 if form=='amount' else Gin,np.zeros(6)]
    def rhs(t,y):
        V=V0+F*t;m=np.exp(y[:2]) if form=='amount' else np.exp(y[:2])*V
        G=y[2]/V if form=='amount' else y[2];v=model.flux(G);g=v[12];vg=v[27]
        transfer=k*m[0]*m[1]/V;formation=g*m;consumption=.18*vg*m
        logs=np.array([g-transfer/m[0],g+eta*transfer/m[1]])
        dG=F*Gin+consumption.sum()
        if form=='concentration':logs-=F/V;dG=(dG-G*F)/V
        return np.r_[logs,dG,formation,consumption,transfer,(1-eta)*transfer]
    started=time.perf_counter();sol=solve_ivp(rhs,[0,tf],y0,method=method,rtol=rtol,atol=atol,max_step=.05,dense_output=True)
    if not sol.success:raise RuntimeError(sol.message)
    t=np.linspace(0,tf,401);y=sol.sol(t).T;V=V0+F*t
    amounts=np.c_[np.exp(y[:,:2]) if form=='amount' else np.exp(y[:,:2])*V[:,None],y[:,2] if form=='amount' else y[:,2]*V]
    concentrations=amounts/V[:,None];vv=[];bound_excess=0.;gaps=[]
    for j,G in enumerate(concentrations[:,2]):
        v=model.flux(float(G));vv.append(v);l,u=model.bounds(G);bound_excess=max(bound_excess,float(np.max(l-v)),float(np.max(v-u)))
        if j%20==0:gaps.append(float(-model.lp(G).fun-v[12]))
    expected=np.c_[x0*V0+y[:,3]-y[:,7],x0*V0+y[:,4]+eta*y[:,7],Gin*V0+F*Gin*t+y[:,5]+y[:,6]]
    errors=np.max(abs(amounts-expected),axis=0)
    # The transfer loss is explicitly inventoried. It cancels with donor loss
    # and recipient gain in this biomass-accounting identity.
    total_identity=amounts[:,0]+amounts[:,1]+y[:,8]-(2*x0*V0+y[:,3]+y[:,4])
    params={'volume_initial_L':V0,'feed_L_h':F,'glucose_feed_g_L':Gin,'horizon_h':tf,'initial_X1_X2_G':[x0,x0,Gin],'transfer_L_gDW_h':k,'conversion_efficiency':eta,'death_rate_h_inv':0,'glucose_vmax':10,'glucose_Ks_g_L':1,'mu':mu,'stored_bound_margin':1e-4,'glucose_lower_bound':'-10G/(1+G), no uniform glucose widening','ATPM_lower_bound':float(model.lo[10])}
    report={'name':name,'parameters':params,'integration':{'form':form,'method':method,'rtol':rtol,'atol':atol,'max_step_h':.05,'seconds':time.perf_counter()-started,'RHS_calls':sol.nfev},'terminal_X1_X2_G':concentrations[-1].tolist(),'minimum_X1_X2_G':concentrations.min(axis=0).tolist(),'max_stoichiometric_residual':model.maxeq,'max_primal_dual_residual':model.maxpd,'minimum_slack':model.minslack,'max_bound_excess':bound_excess,'max_LP_growth_gap':max(gaps),'barrier_gap_bound':190*mu,'max_pool_balance_errors_g':errors.tolist(),'max_total_biomass_accounting_error_g':float(np.max(abs(total_identity))),'unassimilated_biomass_g':float(y[-1,8]),'sources':{str(p.relative_to(ROOT)):sha(p) for p in SOURCE.glob('*') if p.suffix in ['.mat','.m']},'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()},'script_sha256':sha(__file__)}
    report['passed']=bool(sol.t[-1]==tf and concentrations.min()>=0 and model.maxeq<1e-10 and model.maxpd<1e-10 and bound_excess==0 and max(gaps)<190*mu and errors.max()<1e-5)
    folder=OUT/name;folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'trajectory.npz',time_h=t,volume_L=V,concentrations=concentrations,amounts=amounts,flux=np.array(vv),accumulators=y[:,3:])
    np.savetxt(folder/'trajectory.csv',np.c_[t,V,concentrations,np.array(vv)[:,[12,27]],y[:,8]],delimiter=',',header='time_h,volume_L,X1_gDW_L,X2_gDW_L,G_g_L,growth_h_inv,glucose_flux_mmol_gDW_h,unassimilated_gDW',comments='')
    (folder/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['name','terminal_X1_X2_G','minimum_X1_X2_G','max_pool_balance_errors_g','passed']}),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',choices=['primary','refined','reference','barrier']);a=p.parse_args()
    kw={'primary':{},'refined':dict(form='concentration',method='RK45',rtol=1e-10,atol=1e-12),'reference':dict(k=0,rtol=1e-10,atol=1e-12),'barrier':dict(mu=1e-9,rtol=1e-10,atol=1e-12)}[a.run]
    run(a.run,**kw)
