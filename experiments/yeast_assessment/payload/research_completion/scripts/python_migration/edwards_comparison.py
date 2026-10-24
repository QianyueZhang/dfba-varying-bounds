"""Matched Edwards comparison: glucose capacity versus prescribed glucose.

Both cases use the same positive-uptake stoichiometry, oxygen capacity, physical
bounds, biomass objective, reactor balances and solver tolerances. Only the
glucose constraint changes. Fixed glucose is eliminated exactly; all remaining
fluxes satisfy a consistent finite-barrier KKT system at every RHS evaluation.
No input file, previous figure or manuscript is modified.
"""
from pathlib import Path
from time import perf_counter
import hashlib,json,platform,sys
import numpy as np
import scipy
from scipy.io import loadmat
from scipy.optimize import linprog
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix,hstack
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from primal_dual import solve as solve_pd
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'research_completion/runs/python_migration/edwards_matched_comparison'
SOURCE=ROOT/'new_model copy/已解决03_single_edwards_simu/03_01_single_edwards_simu/Edwards_Model.mat'
MU=1e-6
TOL=1e-11

class Model:
    def __init__(self,kind):
        source=loadmat(SOURCE,simplify_cells=True)['Struct']
        self.full_A=csc_matrix(source['A']);self.full_c=np.asarray(source['c'],float)
        assert self.full_A.shape==(8,13)
        self.kind=kind;self.name='Edwards '+kind
        self.free=np.arange(13) if kind=='capacity' else np.arange(1,13)
        self.A=self.full_A[:,self.free];self.c=self.full_c[self.free]
        self.bio=int(np.flatnonzero(self.free==8)[0]);self.g=0
        self.b=np.zeros(8);self.warm=None;self.calls=0;self.iterations=0
        self.max_equality=0.;self.max_pd=0.;self.max_stationarity=0.
        self.min_slack=float('inf');self.cold_retries=0
    def bounds(self,S):
        if S<=0:raise RuntimeError('Nonpositive substrate reached; no hidden clipping or kinetic modification is used.')
        q=12*S/(2+S);lo=np.zeros(13);up=np.full(13,100.);up[0]=q;up[9]=15.
        self.b=np.zeros(8) if self.kind=='capacity' else -self.full_A[:,0].toarray().ravel()*q
        return lo[self.free],up[self.free]
    def lp(self,S):
        l,u=self.bounds(S);r=linprog(-self.c,A_eq=self.A,b_eq=self.b,bounds=list(zip(l,u)),method='highs')
        if not r.success:raise RuntimeError(f'LP infeasible: {self.name}, S={S}, {r.message}')
        return r
    def strict_margin(self,S):
        l,u=self.bounds(S);n=len(l)
        eq=np.column_stack([self.A.toarray(),np.zeros(self.A.shape[0])])
        aub=np.r_[np.column_stack([np.eye(n),np.ones(n)]),np.column_stack([-np.eye(n),np.ones(n)])]
        r=linprog(np.r_[np.zeros(n),-1.],A_ub=aub,b_ub=np.r_[u,-l],A_eq=eq,b_eq=self.b,bounds=[(None,None)]*n+[(0,None)],method='highs')
        return {'substrate_g_L':float(S),'success':bool(r.success),'maximum_common_margin':float(r.x[-1]) if r.success else None,'message':r.message}
    def flux(self,S,cold=False):
        self.calls+=1
        try:s=solve_pd(self,S,mu=MU,tol=TOL,warm=None if cold else self.warm)
        except RuntimeError:
            if cold or self.warm is None:raise
            self.cold_retries+=1;s=solve_pd(self,S,mu=MU,tol=TOL,warm=None)
        self.warm=s[:4];x,lam,zl,zu,d=s;v=np.zeros(13);v[self.free]=x
        if self.kind=='prescribed':v[0]=12*S/(2+S)
        eq=float(np.max(np.abs(self.full_A@v)))
        self.iterations+=d['newton_steps'];self.max_equality=max(self.max_equality,eq)
        self.max_pd=max(self.max_pd,d['primal_dual_residual']);self.max_stationarity=max(self.max_stationarity,d['stationarity_residual'])
        self.min_slack=min(self.min_slack,d['minimum_slack'])
        return v,d
    def rhs(self,t,y):
        X,G,D,E=y;v,_=self.flux(G)
        return np.array([.023*v[8]*X,-.18*v[0]*X,.091*v[11]*X,.046*v[12]*X])
    def metrics(self):
        return {'KKT_solves':self.calls,'Newton_iterations':self.iterations,'cold_retries':self.cold_retries,
                'maximum_equality_residual':self.max_equality,'maximum_primal_dual_residual':self.max_pd,
                'maximum_explicit_stationarity_residual':self.max_stationarity,'minimum_free_flux_slack':self.min_slack}

def integrate(kind,method,rtol,atol):
    m=Model(kind);start=perf_counter()
    # A common 0.05 h step cap prevents high-order trial stages from crossing
    # the nonnegative-substrate domain; no state or flux clipping is applied.
    sol=solve_ivp(m.rhs,(0,7),[1.,10.,0.,0.],method=method,rtol=rtol,atol=atol,max_step=.05,dense_output=True)
    elapsed=perf_counter()-start
    if not sol.success:raise RuntimeError(sol.message)
    rec={'kind':kind,'method':method,'rtol':rtol,'atol':atol,'maximum_step_h':.05,'success':bool(sol.success),'RHS_evaluations':sol.nfev,
         'accepted_time_nodes':len(sol.t),'elapsed_s':elapsed,'terminal_state_X_G_D_E_g_L':sol.y[:,-1].tolist(),**m.metrics()}
    return sol,rec

def render_results(trajectories,grid):
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
    for kind,(y,v,q) in trajectories.items():
        fig,ax=plt.subplots(3,1,figsize=(8.5,10),sharex=True,constrained_layout=True)
        for i,label in [(0,'Biomass'),(2,'Product D'),(3,'Product E')]:ax[0].plot(grid,y[i],lw=1.8,label=label)
        ax2=ax[0].twinx();ax2.plot(grid,y[1],color='#c26920',ls='--',lw=1.8,label='Substrate A');ax2.spines['right'].set_visible(True)
        ax[0].set_ylabel('Biomass and products (g L$^{-1}$)');ax2.set_ylabel('Substrate A (g L$^{-1}$)',color='#c26920')
        ax[0].set_ylim(0,4.2);ax2.set_ylim(0,10.5)
        handles,labels=ax[0].get_legend_handles_labels();hh,ll=ax2.get_legend_handles_labels();ax[0].legend(handles+hh,labels+ll,loc='upper left',ncol=2,fontsize=9)
        if kind=='capacity':
            ax[1].fill_between(grid,0,q,color='#2674a5',alpha=.12,label='Imposed scalar interval');ax[1].plot(grid,q,'--',color='#c26920',label='Glucose capacity')
            ax[1].plot(grid,v[:,0],color='#176599',label='Selected glucose flux',lw=1.8)
        else:
            ax[1].plot(grid,q,'--',color='#c26920',lw=2.7,label='Prescribed q(G)');ax[1].plot(grid,v[:,0],color='#176599',lw=1.2,label='Glucose flux (exact equality)')
        ax[2].fill_between(grid,0,15,color='#2674a5',alpha=.12,label='Imposed scalar interval');ax[2].axhline(15,ls='--',color='#c26920',label='Oxygen capacity');ax[2].plot(grid,v[:,9],color='#176599',lw=1.8,label='Selected oxygen flux')
        for i,aa in enumerate(ax):
            aa.text(-.09,1.02,'('+chr(97+i)+')',transform=aa.transAxes,weight='bold',fontsize=13)
            if i:aa.set_ylabel('Flux (mmol gDW$^{-1}$ h$^{-1}$)');aa.legend(fontsize=9,loc='best',ncol=2);aa.set_ylim(bottom=0)
        ax[-1].set_xlabel('Time (h)');ax[-1].set_xlim(0,7)
        fig.savefig(OUT/f'edwards_{kind}.png',dpi=240);fig.savefig(OUT/f'edwards_{kind}.svg');plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    preflight={kind:[Model(kind).strict_margin(S) for S in [10,5,1,.1,.01,1e-6]] for kind in ['capacity','prescribed']}
    (OUT/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
    for scans in preflight.values():
        if not all(r['success'] and r['maximum_common_margin']>0 for r in scans):raise RuntimeError('Strict feasibility screening failed; see preflight.json.')
    records=[];trajectories={};grid=np.linspace(0,7,351);diagnostics={}
    for kind in ['capacity','prescribed']:
        print('Integrating',kind,flush=True)
        primary,r1=integrate(kind,'RK45',1e-8,1e-10);refined,r2=integrate(kind,'DOP853',1e-10,1e-12)
        records.extend([r1,r2]);y=refined.sol(grid);yp=primary.sol(grid)
        # Saved traces are sampled with fresh direct KKT solves; no interpolated
        # flux table and no raw differentiated flux state is plotted.
        check=Model(kind);vv=np.array([check.flux(s)[0] for s in y[1]])
        q=12*y[1]/(2+y[1]);de=np.max(np.abs(yp-y),axis=1)
        lp_gaps=[];cold_diffs=[]
        for i in [0,50,175,300,350]:
            cold=Model(kind);vc,_=cold.flux(y[1,i],cold=True);lp=cold.lp(y[1,i])
            lp_gaps.append({'time_h':float(grid[i]),'biomass_flux_gap_to_LP':float(-lp.fun-vv[i,8]),'theoretical_barrier_gap_bound':2*len(cold.free)*MU})
            cold_diffs.append(float(np.max(np.abs(vc-vv[i]))))
        glc_err=float(np.max(np.abs(vv[:,0]-q))) if kind=='prescribed' else float(np.max(vv[:,0]-q))
        diagnostics[kind]={'refinement_max_abs_X_G_D_E':de.tolist(),'refinement_threshold_g_L':2e-6,'refinement_passed':bool(np.max(de)<2e-6),
            'sampled_KKT':check.metrics(),'minimum_substrate_g_L':float(np.min(y[1])),
            'maximum_glucose_equality_error' if kind=='prescribed' else 'maximum_glucose_capacity_excess':glc_err,
            'maximum_oxygen_capacity_excess':float(np.max(vv[:,9]-15)),
            'minimum_flux':float(vv.min()),'maximum_cold_warm_flux_difference':max(cold_diffs),'LP_gap_checks':lp_gaps,
            'all_flux_constraints_passed':bool(vv.min()>0 and np.max(vv[:,9]-15)<=1e-10 and (glc_err<1e-10 if kind=='prescribed' else glc_err<=1e-10)),
            'full_equality_passed':bool(check.max_equality<TOL)}
        trajectories[kind]=(y,vv,q)
        np.savez_compressed(OUT/f'{kind}_trajectory.npz',time=grid,state=y,flux=vv,glucose_capacity=q)
        np.savetxt(OUT/f'{kind}_trajectory.csv',np.c_[grid,y.T,q,vv],delimiter=',',header='time_h,X_g_L,G_g_L,D_g_L,E_g_L,q_G,'+','.join('v'+str(j) for j in range(13)),comments='')
    report={'experiment':'Matched positive-uptake Edwards comparison: only the glucose constraint differs.',
        'source_model':str(SOURCE.relative_to(ROOT)),'source_model_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()},
        'parameters':{'initial_X_G_D_E_g_L':[1,10,0,0],'duration_h':7,'glucose_vmax':12,'glucose_Ks':2,'oxygen_capacity':15,
            'MW_X_G_D_E':[.023,.18,.091,.046],'barrier_mu':MU,'bound_relaxation_epsilon':0,'KKT_residual_tolerance':TOL,
            'internal_flux_bounds':[0,100],'oxygen_bounds':[0,15],
            'capacity_glucose':'0 < v_G < 12 G/(2+G), finite-barrier selection',
            'prescribed_glucose':'v_G = 12 G/(2+G), eliminated exactly; oxygen remains capacity bounded',
            'initialisation':'Consistent primal-dual KKT solve; no componentwise perturbed LP flux is reused.'},
        'preflight':preflight,'integrations':records,'validation':diagnostics,
        'comparison_note':'This is a matched recomputation, not an exact recreation of the selected historical PNGs or a claim of MATLAB/Python equivalence.',
        'timing_note':'Single-run elapsed times include initial LP/KKT setup and all RHS KKT solves, exclude plotting and postprocessing. They are reproducibility metadata, not a statistically supported algorithm ranking.',
        'passed':all(d['refinement_passed'] and d['all_flux_constraints_passed'] and d['full_equality_passed'] for d in diagnostics.values())}
    (OUT/'comparison_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    render_results(trajectories,grid)
    print(json.dumps({'passed':report['passed'],'integrations':records,'validation':diagnostics,'output':str(OUT)},indent=2),flush=True)
    if not report['passed']:raise SystemExit('Validation failed; figures are not approved for adoption.')

if __name__=='__main__':
    if '--render-only' in sys.argv:
        trajectories={}
        for kind in ['capacity','prescribed']:
            d=np.load(OUT/f'{kind}_trajectory.npz');grid=d['time'];trajectories[kind]=(d['state'],d['flux'],d['glucose_capacity'])
        render_results(trajectories,grid)
    else:main()
