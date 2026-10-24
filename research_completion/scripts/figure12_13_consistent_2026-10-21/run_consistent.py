"""Consistent finite-barrier batch/fed-batch reactor comparison.

The reactor assumptions and processed reconstruction follow the specified
yeast illustration. Algebraic initialisation and threshold restarts enforce
the actual finite-barrier equations. The adopted comparison uses separate
concentration and amount integrations, each with direct KKT evaluation.
An optional full-coordinate differentiated implementation is retained as a
computational diagnostic; it is not required or represented as an adopted run.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
from pathlib import Path
import sys,json,time,hashlib,argparse
import numpy as np
from scipy.io import loadmat
from scipy.sparse import bmat,diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import linprog
from scipy.integrate import solve_ivp

RC=Path(__file__).resolve().parents[2]
ROOT=RC.parent
OUT=RC/'runs/figure12_13_consistent_2026-10-21'
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(RC/'scripts'))
from revision_precision_kkt import solve

SOURCE=ROOT/'data/models/yeast_fedbatch'
MU=1e-6;EPSILON=1e-3;KAPPA=.56;GIN=22.5;V0=5.;TF=14.
Y0=np.array([.1,22.5,0.])

class Model:
    name='Consistent yeast reactor illustration'
    bio=2984;g=2664;e=2707
    def __init__(self,ethanol_floor=10):
        m=loadmat(SOURCE/'SC_Model.mat',simplify_cells=True)['Struct']
        self.A=m['A'];self.b=m['b'];self.c=m['c'];self.l=m['lb'].copy();self.u=m['ub'].copy()
        self.ethanol_floor=ethanol_floor;self.warm=None;self.solves=0
    def bounds(self,G):
        l=self.l.copy();l[self.g]=-12.9*max(G,0)/(6.95+max(G,0));l[self.e]=self.ethanol_floor-EPSILON
        return l,self.u.copy()
    def lp(self,G):
        l,u=self.bounds(G)
        r=linprog(-self.c,A_eq=self.A,b_eq=self.b,bounds=np.c_[l,u],method='highs',
            options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        if not r.success:raise RuntimeError(r.message)
        return r
    def consistent(self,G):
        try:r=solve(self,G,mu=MU,tol=1e-11,warm=self.warm)
        except RuntimeError:r=solve(self,G,mu=MU,tol=1e-11)
        self.warm=r[:4];self.solves+=1
        return r

def rates(y,v,F,t):
    X,G,E=y;V=V0+F*t
    return np.array([(KAPPA*v[2984]-F/V)*X,.18*v[2664]*X+F/V*(GIN-G),.046*v[2707]*X-F/V*E])

def run(mode,method,rtol,max_step):
    F=0. if mode=='batch' else .5
    model=Model();n=len(model.c);clock=time.perf_counter();t0=0.;y=Y0.copy();parts=[];events=[];rhs_calls=0;diagnostics=[]
    for floor,threshold in [(10.,7.5),(5.,2.),(0.,0.)]:
        if t0>=TF-1e-12:break
        model.ethanol_floor=floor;model.warm=None
        v,lam,zl,zu,init=model.consistent(y[1]);init.update(time_h=t0,ethanol_floor=floor);diagnostics.append(init)
        if method=='differentiated':
            z0=np.r_[y,v,lam]
            def rhs(t,z):
                nonlocal rhs_calls
                rhs_calls+=1;yy=z[:3];vv=z[3:3+n];dy=rates(yy,vv,F,t);l,u=model.bounds(yy[1]);sl=vv-l;su=u-vv
                if rhs_calls%200==0:print(mode,method,'progress',t,'rhs',rhs_calls,'seconds',time.perf_counter()-clock,flush=True)
                al=MU/sl**2;au=MU/su**2;D=al+au
                dl=np.zeros(n);dl[model.g]=-12.9*6.95/(6.95+max(yy[1],0))**2*dy[1]
                K=bmat([[diags(D),model.A.T],[model.A,None]],format='csc')
                dvdl=spsolve(K,np.r_[al*dl,np.zeros(len(model.b))])
                return np.r_[dy,dvdl]
            def event(t,z):return z[1]-threshold
            atol=np.r_[np.ones(3)*1e-10,np.ones(n+len(model.b))*1e-9]
            solver='RK45'
        elif method=='concentration':
            z0=y.copy()
            def rhs(t,z):
                nonlocal rhs_calls
                rhs_calls+=1;vv,*_=model.consistent(z[1])
                if rhs_calls%200==0:print(mode,method,'progress',t,'rhs',rhs_calls,'seconds',time.perf_counter()-clock,flush=True)
                return rates(z,vv,F,t)
            def event(t,z):return z[1]-threshold
            atol=1e-9;solver='RK45'
        else:
            z0=y*(V0+F*t0)
            def rhs(t,z):
                nonlocal rhs_calls
                rhs_calls+=1;G=z[1]/(V0+F*t);vv,*_=model.consistent(G)
                if rhs_calls%200==0:print(mode,method,'progress',t,'rhs',rhs_calls,'seconds',time.perf_counter()-clock,flush=True)
                return np.array([KAPPA*vv[model.bio]*z[0],F*GIN+.18*vv[model.g]*z[0],.046*vv[model.e]*z[0]])
            def event(t,z):return z[1]/(V0+F*t)-threshold
            atol=1e-11;solver='DOP853'
        event.direction=-1;event.terminal=True
        sol=solve_ivp(rhs,(t0,TF),z0,method=solver,rtol=rtol,atol=atol,max_step=max_step,dense_output=True,events=event)
        if not sol.success:raise RuntimeError(sol.message)
        end=float(sol.t[-1]);sample=np.unique(np.r_[np.linspace(t0,end,max(2,int(np.ceil((end-t0)/.05))+1)),sol.t_events[0]])
        zz=sol.sol(sample)
        yy=zz[:3] if method in ['differentiated','concentration'] else zz/(V0+F*sample)[None,:]
        parts.append(dict(t=sample,y=yy,solution=sol,floor=floor))
        y=yy[:,-1];t0=end
        if sol.t_events[0].size:events.append(dict(time_h=end,glucose_g_L=float(y[1]),previous_floor=floor,next_floor=5 if floor==10 else 0))
        print(mode,method,'segment',floor,'end',end,'state',y.tolist(),'calls',rhs_calls,'seconds',time.perf_counter()-clock,flush=True)
    if t0<TF-1e-9:raise RuntimeError('Glucose exhaustion before the specified endpoint; requires an explicitly defined depletion mode.')
    t=np.concatenate([p['t'] if i==0 else p['t'][1:]for i,p in enumerate(parts)])
    states=np.concatenate([p['y'] if i==0 else p['y'][:,1:]for i,p in enumerate(parts)],axis=1)
    V=V0+F*t
    np.savetxt(OUT/f'{mode}_{method}_trajectory.csv',np.c_[t,V,states.T,(states*V).T],delimiter=',',
        header='time_h,volume_L,biomass_g_L,glucose_g_L,ethanol_g_L,biomass_g,glucose_g,ethanol_g',comments='')
    checks=[]
    for p in parts:
        model.ethanol_floor=p['floor'];model.warm=None
        for tt in np.linspace(p['t'][0],p['t'][-1],5):
            z=p['solution'].sol(tt);yy=z[:3] if method in ['differentiated','concentration'] else z/(V0+F*tt)
            vc,_,_,_,info=model.consistent(yy[1]);lp=model.lp(yy[1]);l,u=model.bounds(yy[1])
            row=dict(time_h=float(tt),ethanol_floor=p['floor'],**info,LP_biomass=float(-lp.fun),LP_gap=float(-lp.fun-vc[model.bio]),nominal_gap_bound=2*n*MU)
            if method=='differentiated':
                vi=z[3:3+n];li=z[3+n:]
                row.update(integrated_equality_residual=float(max(abs(model.A@vi-model.b))),
                    integrated_stationarity_residual=float(max(abs(-model.c+model.A.T@li-MU/(vi-l)+MU/(u-vi)))),
                    integrated_minimum_slack=float(min(min(vi-l),min(u-vi))),
                    integrated_vs_direct_flux_max=float(max(abs(vi-vc))),
                    integrated_vs_direct_biomass=float(vi[model.bio]-vc[model.bio]),
                    integrated_vs_direct_glucose=float(vi[model.g]-vc[model.g]),
                    integrated_vs_direct_ethanol=float(vi[model.e]-vc[model.e]))
            checks.append(row)
    report=dict(mode=mode,method=method,solver=solver,relative_tolerance=rtol,absolute_tolerance=atol.tolist() if isinstance(atol,np.ndarray)else atol,max_step_h=max_step,
        initial=Y0.tolist(),terminal=states[:,-1].tolist(),terminal_volume=float(V[-1]),terminal_amount=(states[:,-1]*V[-1]).tolist(),
        events=events,initialisation=diagnostics,sampled_checks=checks,rhs_calls=rhs_calls,KKT_solves=model.solves,wall_seconds=time.perf_counter()-clock,
        parameters=dict(mu=MU,epsilon=EPSILON,growth_scaling=KAPPA,feed_L_h=F,inlet_glucose_g_L=GIN,initial_volume_L=V0,horizon_h=TF,
            glucose_qmax=12.9,glucose_Ks=6.95,glucose_lower='-q(G)',glucose_upper=1+EPSILON,ethanol_lower='h(G)-epsilon',
            ethanol_policy='10 for G>7.5; 5 for2<G<=7.5;0 otherwise',other_bounds='processed source expanded finite bounds; infinities capped at1000'),
        source=dict(path=str(SOURCE/'SC_Model.mat'),sha256=hashlib.sha256((SOURCE/'SC_Model.mat').read_bytes()).hexdigest()))
    (OUT/f'{mode}_{method}_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['batch','fed_batch'],required=True);p.add_argument('--method',choices=['differentiated','concentration','direct'],required=True);p.add_argument('--rtol',type=float,default=1e-7);p.add_argument('--max-step',type=float,default=.1);a=p.parse_args()
    run(a.mode,a.method,a.rtol,a.max_step)
