"""Recipient-dependent pyruvate diversion with consistent finite-barrier solves.

No source files or manuscript files are modified. The imposed secretion is a
specific-flux lower bound. Dynamic tracked-pool bounds are exact physical
capacities; saved epsilon-expanded static bounds are retained and documented.
Integration terminates before loss of positive-growth feasibility. There is no
starvation extension, clipped accepted flux or automatic parameter retuning.
"""
from pathlib import Path
import argparse, hashlib, json, platform, sys, time
import numpy as np
import scipy
from scipy.io import loadmat
from scipy.integrate import solve_ivp
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'research_completion/scripts'))
from primal_dual import solve as solve_pd

SOURCES = {
    'fedbatch': ROOT / 'data/models/resource_diversion_fedbatch',
    'batch': ROOT / 'data/models/resource_diversion_batch',
}

def flat(x):
    return np.asarray(x.toarray() if hasattr(x, 'toarray') else x, float).ravel()

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def parameters(case, kp=.1, horizon=None):
    return dict(case=case, initial_volume_L=5., initial_biomass_gDW_L=.001 if case=='fedbatch' else 1.,
                initial_glucose_g_L=20., initial_pyruvate_g_L=0., feed_L_h=.1 if case=='fedbatch' else 0.,
                feed_glucose_g_L=20., death_h_inv=0. if case=='fedbatch' else .01,
                horizon_h=horizon if horizon is not None else (20. if case=='fedbatch' else 5.),
                Kp=kp, Kp_units='mmol L^3 g_glucose^-1 gDW^-3 h^-1',
                glucose_vmax_mmol_gDW_h=10., glucose_Ks_g_L=1.,
                recipient_pyruvate_vmax_mmol_gDW_h=10., recipient_pyruvate_Ks_g_L=5.,
                glucose_MW_g_mmol=.18, pyruvate_MW_g_mmol=.088)

class Model:
    def __init__(self, case, species, kp=.1, mu=1e-8, tol=1e-11):
        self.name=species;self.bio=12;self.g=27;self.py=37;self.kp=kp;self.mu=mu;self.tol=tol
        p=SOURCES[case]
        s=loadmat(p/f'{species}_Model.mat',simplify_cells=True)['Struct']
        raw=loadmat(p/'e_coli_core.mat',simplify_cells=True)['e_coli_core']
        self.A=csc_matrix(s['A']);self.rawA=csc_matrix(raw['S']);self.b=flat(s['b']);self.c=flat(s['c'])
        self.lo=flat(s['lb']);self.up=flat(s['ub']);self.warm=None;self.cache=None
        assert raw['rxns'][self.py]=='EX_pyr_e' and raw['rxns'][self.g]=='EX_glc__D_e'
        assert raw['rxns'][self.bio]=='BIOMASS_Ecoli_core_w_GAM'
        assert self.A.shape==(67,95)
        self.calls=0;self.lp_calls=0;self.retries=0;self.maxeq=0.;self.maxpd=0.;self.minslack=np.inf

    def bounds(self, state):
        x1,x2,G,P=map(float,state)
        l=self.lo.copy();u=self.up.copy()
        # Exact changing capacities prohibit uptake of absent tracked solute.
        l[self.g]=-10*max(G,0)/(1+max(G,0))
        if self.name=='S1':
            l[self.py]=self.kp*max(G,0)*max(x1,0)*max(x2,0)
        else:
            l[self.py]=-10*max(P,0)/(5+max(P,0))
        return l,u

    def lp(self,state):
        key=tuple(map(float,state))
        if self.cache is not None and self.cache[0]==key:return self.cache[1]
        l,u=self.bounds(state)
        r=linprog(-self.c,A_eq=self.A,b_eq=self.b,bounds=np.c_[l,u],method='highs',
                  options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        self.lp_calls+=1;self.cache=(key,r)
        return r

    def capacity(self,state):
        r=self.lp(state)
        return float(-r.fun) if r.success else -1.

    def flux(self,state,allvars=False):
        self.calls+=1
        try:s=solve_pd(self,state,mu=self.mu,tol=self.tol,warm=self.warm)
        except RuntimeError:
            self.retries+=1;s=solve_pd(self,state,mu=self.mu,tol=self.tol,warm=None)
        self.warm=s[:4];d=s[-1]
        self.maxeq=max(self.maxeq,float(abs(self.rawA@s[0]).max()))
        self.maxpd=max(self.maxpd,d['primal_dual_residual']);self.minslack=min(self.minslack,d['minimum_slack'])
        return s if allvars else s[0]

    def metrics(self):
        return dict(KKT_calls=self.calls,LP_calls=self.lp_calls,cold_retries=self.retries,
                    maximum_full_balance_residual=self.maxeq,maximum_primal_dual_residual=self.maxpd,
                    minimum_slack=self.minslack)

def integrate(case, kp=.1, horizon=None, mu=1e-8, rtol=1e-8, atol=1e-10, max_step=.02,
              method='DOP853', guard=1e-4):
    p=parameters(case,kp,horizon);mods=[Model(case,s,kp,mu)for s in ['S1','S2']]
    V=p['initial_volume_L'];x=p['initial_biomass_gDW_L'];F=p['feed_L_h'];Gin=p['feed_glucose_g_L'];kd=p['death_h_inv']
    # Physical amount states followed by independent rate integrals.
    y0=np.r_[V,V*x,V*x,V*p['initial_glucose_g_L'],0.,np.zeros(9)]
    trial_guard_count=0
    def env(y):return y[1:5]/y[0]
    def rhs(t,y):
        nonlocal trial_guard_count
        state=env(y)
        if min(m.capacity(state)for m in mods)<guard*.25:
            # A finite extension only for rejected RK trial points and root
            # bracketing; the terminal event occurs at the larger guard.
            trial_guard_count+=1
            return np.r_[F,np.zeros(13)]
        v=[m.flux(state)for m in mods]
        growth=np.array([v[i][12]*y[1+i]for i in range(2)])
        loss=kd*y[1:3]
        gg=np.array([.18*v[i][27]*y[1+i]for i in range(2)])
        pp=np.array([.088*v[i][37]*y[1+i]for i in range(2)])
        return np.r_[F,growth-loss,gg.sum()+F*Gin,pp.sum(),growth,loss,gg,pp,F*Gin]
    events=[]
    for m in mods:
        def margin(t,y,m=m):return m.capacity(env(y))-guard
        margin.terminal=True;margin.direction=-1;events.append(margin)
    started=time.perf_counter()
    sol=solve_ivp(rhs,(0,p['horizon_h']),y0,method=method,rtol=rtol,atol=atol,max_step=max_step,
                  events=events,dense_output=True)
    elapsed=time.perf_counter()-started
    if not sol.success:raise RuntimeError(sol.message)
    times=np.linspace(0,float(sol.t[-1]),501);Y=sol.sol(times).T
    flux=[];dual=[];zl=[];zu=[];lower=[];upper=[];lp_growth=[]
    for y in Y:
        state=env(y);row=[m.flux(state,True)for m in mods]
        flux.append([z[0]for z in row]);dual.append([z[1]for z in row]);zl.append([z[2]for z in row]);zu.append([z[3]for z in row])
        bounds=[m.bounds(state)for m in mods];lower.append([z[0]for z in bounds]);upper.append([z[1]for z in bounds])
        lp_growth.append([m.capacity(state)for m in mods])
    eventlist=[]
    for i,e in enumerate(sol.t_events):
        if len(e):eventlist.append(dict(species=i+1,time_h=float(e[0]),maximum_growth_h_inv=mods[i].capacity(env(sol.sol(e[0])))))
    arrays=dict(time_h=times,amounts=Y,flux=np.array(flux),lambda_=np.array(dual),zl=np.array(zl),zu=np.array(zu),
                lower=np.array(lower),upper=np.array(upper),lp_growth=np.array(lp_growth))
    report=dict(parameters=p,numerical=dict(mu=mu,rtol=rtol,atol=atol,max_step=max_step,method=method,guard_growth_h_inv=guard,
                    KKT_tolerance=1e-11,elapsed_s=elapsed),
                full_requested_horizon=bool(sol.t[-1]==p['horizon_h']),terminal_time_h=float(sol.t[-1]),
                terminal_concentrations=env(Y[-1]).tolist(),events=eventlist,
                guard_trial_evaluations=trial_guard_count,metrics=[m.metrics()for m in mods],
                model_definition='Specific donor pyruvate lower bound Kp*G*X1*X2; direct finite-barrier KKT at every accepted state. Exact dynamic uptake capacities and donor secretion floor; source epsilon-expanded static bounds retained. No post-feasibility continuation.',
                environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
                source_files=[dict(path=str(q.relative_to(ROOT)),sha256=digest(q))for q in SOURCES[case].iterdir()if q.suffix in ['.m','.mat']],
                script_sha256=digest(Path(__file__)))
    return arrays,report

def save(name,arrays,report):
    d=HERE/name;d.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(d/'trajectory.npz',**arrays)
    a=arrays;y=a['amounts'];v=a['flux'];t=a['time_h']
    np.savetxt(d/'trajectory.csv',np.c_[t,y[:,:5],y[:,1:5]/y[:,:1],v[:,:,12],v[:,:,27],v[:,:,37]],delimiter=',',
               header='time_h,V_L,M1_gDW,M2_gDW,MG_g,MP_g,X1_gDW_L,X2_gDW_L,G_g_L,P_g_L,growth1_h_inv,growth2_h_inv,vG1,vG2,vP1,vP2',comments='')
    (d/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (d/'run_script.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:report[k]for k in ['parameters','full_requested_horizon','terminal_time_h','terminal_concentrations','events','metrics']},indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=list(SOURCES),required=True);ap.add_argument('--name',required=True)
    ap.add_argument('--kp',type=float,default=.1);ap.add_argument('--horizon',type=float);ap.add_argument('--mu',type=float,default=1e-8)
    ap.add_argument('--refined',action='store_true');ap.add_argument('--guard',type=float,default=1e-4)
    a=ap.parse_args();kw=dict(rtol=1e-10,atol=1e-12,max_step=.01,method='RK45')if a.refined else {}
    z,r=integrate(a.case,a.kp,a.horizon,a.mu,guard=a.guard,**kw);save(a.name,z,r)
