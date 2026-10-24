"""Direct finite-barrier pilots for independent/shared-resource models.

The specified input arrays and reactor parameters define each case. A
consistent KKT solve supplies intracellular rates. An integration event stops
before the maintenance-feasibility boundary; the calculation therefore
represents the feasible active-metabolism interval only.
"""
from pathlib import Path
from time import perf_counter
import argparse, hashlib, json, platform, sys
import numpy as np
import scipy
from scipy.io import loadmat
from scipy.integrate import solve_ivp
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'research_completion/scripts'))
from primal_dual import solve as solve_pd

BASE = ROOT/'data/models'
OUT = ROOT/'research_completion/runs/python_migration/interaction_cases'
CASES = {
    'independent_identical': {'folder':'independent_pools',
        'vmax':[10.,10.], 'initial_substrate':[10.,10.], 'feed_substrate':[10.,10.],
        'duration_h':5., 'shared':False, 'figures':[14,15,16],
        'parameter_basis':'README identical-species case and retained Figure 14 horizon; current main uses 10 h.'},
    'independent_different': {'folder':'independent_pools',
        'vmax':[5.,10.], 'initial_substrate':[10.,10.], 'feed_substrate':[10.,10.],
        'duration_h':10., 'shared':False, 'figures':[17],
        'parameter_basis':'Retained Figure 17 uptake panels and thesis text: S1 vmax=5, S2 vmax=10, 10 h. README instead assigns S2 vmax=5; current main and saved models use equal vmax. Fresh KKT initialisation uses the plotted parameter order.'},
    'shared_competition': {'folder':'shared_resource',
        'vmax':[10.,5.], 'initial_substrate':[20.], 'feed_substrate':[20.],
        'duration_h':5., 'shared':True, 'figures':[18],
        'parameter_basis':'Current main_competition.m: 5 h; README separately states 10 h.'},
}
MU=1e-6
EPS=1e-4
KKT_TOL=1e-11
GUARD_G_L=.01
MAX_STEP_H=.005

def flat(value):
    return np.asarray(value.toarray() if hasattr(value,'toarray') else value,dtype=float).ravel()

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

class Core:
    def __init__(self,folder,species,vmax):
        self.path=folder/(species+'_Model.mat')
        self.source=loadmat(self.path,simplify_cells=True)['Struct']
        raw=loadmat(folder/'e_coli_core.mat',simplify_cells=True)['e_coli_core']
        self.A=csc_matrix(self.source['A']); self.raw_A=csc_matrix(raw['S'])
        self.b=flat(self.source['b']); self.c=flat(self.source['c'])
        self.lo=flat(self.source['lb']); self.up=flat(self.source['ub'])
        self.bio=12; self.g=27; self.vmax=vmax; self.name=species
        assert raw['rxns'][self.bio]=='BIOMASS_Ecoli_core_w_GAM'
        assert raw['rxns'][self.g]=='EX_glc__D_e'
        assert raw['rxns'][10]=='ATPM'
        assert self.c[self.bio]==1 and np.count_nonzero(self.c)==1
        assert self.A.shape==(67,95)
        self.warm=None; self.calls=0; self.newton_steps=0; self.retries=0
        self.max_eq=0.; self.max_raw_eq=0.; self.max_pd=0.; self.max_explicit=0.
        self.min_slack=float('inf')
        # Find the least carbon uptake compatible with unchanged maintenance,
        # mass balances and all other stored reaction bounds.
        l=self.lo.copy(); l[self.g]=-vmax
        objective=np.zeros(95); objective[self.g]=-1
        r=linprog(objective,A_eq=self.A,b_eq=self.b,bounds=list(zip(l,self.up)),method='highs')
        if not r.success: raise RuntimeError('Initial maintenance feasibility LP failed: '+r.message)
        self.minimum_uptake=float(-r.x[self.g])
        self.critical_G=self.minimum_uptake/(vmax-self.minimum_uptake)
        self.minimum_uptake_lp_residual=float(np.max(np.abs(self.A@r.x-self.b)))

    def bounds(self,G):
        if G<=0: raise RuntimeError('Nonpositive substrate; no clipping or depletion extension is applied.')
        l=self.lo.copy(); u=self.up.copy()
        l[self.g]=max(-self.vmax*G/(1+G),-self.vmax-EPS)
        l=np.minimum(l,u-2*EPS)  # exact bound construction used by the source RHS
        return l,u

    def lp(self,G):
        l,u=self.bounds(G)
        r=linprog(-self.c,A_eq=self.A,b_eq=self.b,bounds=list(zip(l,u)),method='highs')
        if not r.success: raise RuntimeError(f'LP infeasible for {self.name} at G={G}: {r.message}')
        return r

    def flux(self,G):
        self.calls+=1
        try: sol=solve_pd(self,G,mu=MU,tol=KKT_TOL,warm=self.warm)
        except RuntimeError:
            if self.warm is None: raise
            self.retries+=1; sol=solve_pd(self,G,mu=MU,tol=KKT_TOL,warm=None)
        self.warm=sol[:4]; v=sol[0]; d=sol[-1]
        self.newton_steps+=d['newton_steps']
        self.max_eq=max(self.max_eq,d['equality_residual'])
        self.max_raw_eq=max(self.max_raw_eq,float(np.max(np.abs(self.raw_A@v))))
        self.max_pd=max(self.max_pd,d['primal_dual_residual'])
        self.max_explicit=max(self.max_explicit,d['stationarity_residual'])
        self.min_slack=min(self.min_slack,d['minimum_slack'])
        return v

    def metrics(self):
        return {'KKT_calls':self.calls,'Newton_steps':self.newton_steps,'cold_retries':self.retries,
            'max_processed_equality_residual':self.max_eq,'max_full_model_equality_residual':self.max_raw_eq,
            'max_primal_dual_residual':self.max_pd,'max_reciprocal_slack_stationarity_residual':self.max_explicit,
            'minimum_slack':self.min_slack}

    def preflight(self,G0):
        initial=flat(self.source['flux']); l,u=self.bounds(G0)
        clipped=np.maximum(l+EPS,np.minimum(u-EPS,initial))
        optimum=self.lp(G0)
        scans=[]
        for G in sorted({G0,1.,.1,.01,self.critical_G, self.critical_G+GUARD_G_L},reverse=True):
            l,u=self.bounds(G)
            r=linprog(-self.c,A_eq=self.A,b_eq=self.b,bounds=list(zip(l,u)),method='highs')
            scans.append({'G_g_L':G,'feasible':bool(r.success),'LP_status':r.status})
        return {'name':self.name,'vmax':self.vmax,'Ks_g_L':1.,'source_path':str(self.path.relative_to(ROOT)),
            'source_sha256':digest(self.path),'source_initial_uptake':float(self.source['uptk']),
            'minimum_required_glucose_uptake':self.minimum_uptake,
            'critical_glucose_g_L':self.critical_G,
            'event_glucose_g_L':self.critical_G+GUARD_G_L,
            'minimum_uptake_LP_equality_residual':self.minimum_uptake_lp_residual,
            'ATPM_lower_bound':float(self.lo[10]),
            'saved_initial_equality_residual':float(np.max(np.abs(self.A@initial-self.b))),
            'source_clipped_initial_equality_residual':float(np.max(np.abs(self.A@clipped-self.b))),
            'source_clipped_initial_biomass':float(clipped[self.bio]),
            'initial_exact_LP_biomass':float(optimum.x[self.bio]),
            'feasibility_scan':scans}

def make_models(case):
    folder=BASE/case['folder']
    return [Core(folder,'S'+str(i+1),case['vmax'][i]) for i in range(2)]

def integrate(case,method,rtol,atol):
    models=make_models(case); initial=np.r_[5.,1.,1.,case['initial_substrate']]
    def rhs(t,y):
        V=y[0]; X=y[1:3]; G=y[3:]
        flux=[m.flux(G[0 if case['shared'] else i]) for i,m in enumerate(models)]
        growth=np.array([v[12] for v in flux]); uptake=np.array([v[27] for v in flux])
        dX=growth*X-.1/V*X
        metabolic=np.array([.18*np.dot(uptake,X)]) if case['shared'] else .18*uptake*X
        dG=.1/V*(np.array(case['feed_substrate'])-G)+metabolic
        return np.r_[.1,dX,dG]
    def guard(t,y):
        return min(y[3+(0 if case['shared'] else i)]-m.critical_G-GUARD_G_L for i,m in enumerate(models))
    guard.terminal=True; guard.direction=-1
    start=perf_counter()
    sol=solve_ivp(rhs,(0,case['duration_h']),initial,method=method,rtol=rtol,atol=atol,
                  max_step=MAX_STEP_H,events=guard,dense_output=True)
    elapsed=perf_counter()-start
    rec={'method':method,'rtol':rtol,'atol':atol,'max_step_h':MAX_STEP_H,
        'success':bool(sol.success),'status':int(sol.status),'message':sol.message,
        'terminal_time_h':float(sol.t[-1]),'terminal_state':sol.y[:,-1].tolist(),
        'requested_horizon_h':case['duration_h'],'complete_horizon':bool(sol.t[-1]>=case['duration_h']-1e-10),
        'RHS_evaluations':sol.nfev,'accepted_nodes':len(sol.t),'seconds':elapsed,
        'event_times':[a.tolist() for a in sol.t_events],
        'models':[m.metrics() for m in models]}
    return sol,rec

def run(key,horizon=None):
    case=CASES[key].copy()
    if horizon is not None:case['duration_h']=horizon
    dest=(OUT/f'validated_{horizon:g}h'/key) if horizon is not None else OUT/key
    dest.mkdir(parents=True,exist_ok=True)
    models=make_models(case)
    preflight=[m.preflight(case['initial_substrate'][0 if case['shared'] else i]) for i,m in enumerate(models)]
    (dest/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
    print(key,'preflight critical G:',[m.critical_G for m in models],flush=True)
    primary,r1=integrate(case,'RK45',1e-7,1e-9)
    refined,r2=integrate(case,'DOP853',1e-9,1e-11)
    grid=np.linspace(0,min(primary.t[-1],refined.t[-1]),201)
    y=refined.sol(grid); yp=primary.sol(grid); delta=np.max(np.abs(y-yp),axis=1)
    samples=make_models(case); vv=np.array([[m.flux(y[3+(0 if case['shared'] else i),j])
        for i,m in enumerate(samples)] for j in range(len(grid))])
    lpchecks=[]
    for j in [0,50,100,150,200]:
        for i,m in enumerate(samples):
            r=m.lp(y[3+(0 if case['shared'] else i),j])
            lpchecks.append({'time_h':float(grid[j]),'species':m.name,
                'biomass_gap_to_LP':float(-r.fun-vv[j,i,12]),'nominal_barrier_gap_bound':190*MU})
    columns=['time_h','V_L','X1_gDW_L','X2_gDW_L']+(['G_g_L'] if case['shared'] else ['G1_g_L','G2_g_L'])
    columns+=['v_bio1_h_inv','v_bio2_h_inv','v_glucose1_mmol_gDW_h','v_glucose2_mmol_gDW_h']
    values=np.c_[grid,y.T,vv[:,:,12],vv[:,:,27]]
    np.savetxt(dest/'trajectory.csv',values,delimiter=',',header=','.join(columns),comments='')
    np.savez_compressed(dest/'trajectory.npz',time=grid,state=y,flux=vv,primary_state=yp)
    folder=BASE/case['folder']; sources=[p for p in folder.iterdir() if p.suffix in ['.m','.mat','.txt']]
    constraint_checks=[]
    for i,m in enumerate(samples):
        excess=[]
        for j in range(len(grid)):
            l,u=m.bounds(y[3+(0 if case['shared'] else i),j])
            excess.append(max(float(np.max(l-vv[j,i])),float(np.max(vv[j,i]-u))))
        constraint_checks.append({'species':m.name,'max_flux_bound_excess':max(excess)})
    verified=bool(np.max(delta)<1e-5 and all(x['success'] for x in [r1,r2])
        and all(m.max_raw_eq<1e-9 and m.max_pd<1e-10 and m.min_slack>0 for m in samples)
        and all(x['max_flux_bound_excess']<=1e-10 for x in constraint_checks)
        and all(-1e-9<=x['biomass_gap_to_LP']<x['nominal_barrier_gap_bound'] for x in lpchecks))
    report={'case':key,'case_definition':case,'physical_parameters':{'V0_L':5.,'X0_gDW_L':[1.,1.],
        'feed_L_h':.1,'MW_glucose_g_mmol':.18,'biomass_rate_multiplier':1.,'Ks_g_L':1.},
        'numerical_parameters':{'mu':MU,'epsilon_on_stored_bounds':EPS,'dynamic_glucose_lower_bound':'-vmax*G/(1+G), exactly as source RHS; no additional epsilon expansion',
            'KKT_tolerance':KKT_TOL,'event_margin_above_LP_feasibility_boundary_g_L':GUARD_G_L},
        'sources':[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in sources],
        'script_sha256':digest(Path(__file__)),
        'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()},
        'preflight':preflight,'integrations':[r1,r2],
        'validation':{'max_refinement_difference_by_state':delta.tolist(),'threshold_g_L':1e-5,
            'refinement_passed':bool(np.max(delta)<1e-5),'sampled_models':[m.metrics() for m in samples],
            'LP_gap_checks':lpchecks,'analytic_volume_max_error':float(np.max(np.abs(y[0]-(5+.1*grid)))),
            'minimum_concentration':float(y[1:].min()),'flux_bounds':constraint_checks,
            'verified':verified},
        'adoption':False,
        'classification':'Validated feasible-prefix pilot only; the maintenance constraints prevent completion of the specified full horizon.' if not r2['complete_horizon'] else 'Validated finite-barrier calculation for the explicitly specified horizon, with integration refinement and intracellular feasibility checks.',
        'cross_language_equivalence_established':False,
        'limitations':'The retained PNGs do not provide complete numerical reference arrays. The original code clips a differentiated flux state; this pilot uses consistent algebraic finite-barrier solves and stops before the unchanged ATP-maintenance constraints lose feasibility. No model repair, parameter retuning, growth arrest or artificial continuation is applied.'}
    (dest/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    render(key,case,grid,y,dest)
    print(json.dumps({'case':key,'terminal_time':r2['terminal_time_h'],'complete_horizon':r2['complete_horizon'],'verified':verified,'max_refinement_difference':float(delta.max()),'sampled_KKT':[m.metrics() for m in samples]}),flush=True)
    if not verified:raise RuntimeError('Numerical validation did not pass; results cannot be adopted.')

def render(key,case,t,y,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(8,7),sharex=True,constrained_layout=True)
    for i,style in enumerate(['-','--']): axes[0].plot(t,y[i+1],style,label='Species '+str(i+1))
    for i in range(len(case['initial_substrate'])): axes[1].plot(t,y[i+3],label='Shared glucose' if case['shared'] else 'Resource '+str(i+1))
    axes[0].set_ylabel('Biomass (gDW L$^{-1}$)');axes[1].set_ylabel('Glucose (g L$^{-1}$)');axes[1].set_xlabel('Time (h)')
    for ax in axes:
        ax.legend();ax.grid(alpha=.2);ax.axvline(t[-1],color='grey',ls=':',label='Feasibility guard')
    status='Full selected horizon' if t[-1]>=case['duration_h']-1e-10 else 'Verified feasible prefix; specified horizon incomplete'
    fig.suptitle(key.replace('_',' ')+' — '+status,fontsize=11)
    fig.savefig(dest/'pilot.png',dpi=180);fig.savefig(dest/'pilot.svg');plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('cases',nargs='*',choices=list(CASES));parser.add_argument('--horizon',type=float);args=parser.parse_args()
    for key in args.cases or ['independent_identical','shared_competition']: run(key,args.horizon)
