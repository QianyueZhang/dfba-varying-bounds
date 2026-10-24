"""Pyruvate coupling with explicit batch starvation events.

Active cells satisfy the E. coli core constraints, including ATP maintenance.
Quiescent cells have zero intracellular metabolism and first-order biomass
loss. Growth and quiescence are separate states of the specified hybrid model.
"""
from pathlib import Path
import sys, json, hashlib, platform, time, argparse, os
import numpy as np
import scipy
from scipy.io import loadmat
from scipy.sparse import csc_matrix
from scipy.optimize import linprog
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research_completion/scripts'))
from primal_dual import solve as solve_pd
SOURCE=Path(os.environ.get('THESIS_INTERACTION_SOURCE',str(ROOT/'data/models/pyruvate_crossfeeding')))
OUT=ROOT/'research_completion/runs/python_consistency/interactions/pyruvate_batch'
EPS=1e-4
KD=.01
VOLUME=5.
MW_G=.18
MW_P=.088


def flat(v):
    return np.asarray(v.toarray() if hasattr(v,'toarray') else v,float).ravel()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Model:
    def __init__(self,species,mu,tolerance,uptake=True):
        self.name=species;self.bio=12;self.g=27;self.py=37;self.mu=mu;self.tol=tolerance
        source=loadmat(SOURCE/f'{species}_Model.mat',simplify_cells=True)['Struct']
        raw=loadmat(SOURCE/'e_coli_core.mat',simplify_cells=True)['e_coli_core']
        self.A=csc_matrix(source['A']);self.rawA=csc_matrix(raw['S'])
        self.b=flat(source['b']);self.c=flat(source['c'])
        self.lo=flat(source['lb']);self.up=flat(source['ub'])
        assert raw['rxns'][self.py]=='EX_pyr_e' and raw['rxns'][self.g]=='EX_glc__D_e'
        assert raw['rxns'][10]=='ATPM'
        self.level=2.;self.uptake=uptake;self.warm=None
        self.calls=0;self.retries=0;self.maxeq=0.;self.maxpd=0.;self.minslack=np.inf
        self.lp_calls=0;self.cache=None

    def bounds(self,state):
        G,P=map(float,state)
        l=self.lo.copy();u=self.up.copy()
        # Use exact capacity rather than permit uptake from an absent pool.
        l[self.g]=-10*max(G,0)/(1+max(G,0))
        if self.name=='S1':
            # Source epsilon is retained at positive secretion floors, while
            # the zero-production phase forbids reverse donor uptake.
            l[self.py]=max(0.,self.level-EPS)
        else:
            l[self.py]=-10*max(P,0)/(5+max(P,0)) if self.uptake else 0.
        return l,u

    def lp(self,state):
        key=(float(state[0]),float(state[1]),self.level,self.uptake)
        if self.cache and self.cache[0]==key:return self.cache[1]
        l,u=self.bounds(state)
        ans=linprog(-self.c,A_eq=self.A,b_eq=self.b,bounds=list(zip(l,u)),method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        self.lp_calls+=1;self.cache=(key,ans)
        return ans

    def capacity(self,state):
        lp=self.lp(state)
        return float(-lp.fun) if lp.success else -1.

    def flux(self,state):
        self.calls+=1
        try:sol=solve_pd(self,state,mu=self.mu,tol=self.tol,warm=self.warm)
        except RuntimeError:
            self.retries+=1;sol=solve_pd(self,state,mu=self.mu,tol=self.tol,warm=None)
        self.warm=sol[:4];v=sol[0];d=sol[-1]
        self.maxeq=max(self.maxeq,float(np.max(abs(self.rawA@v))),d['equality_residual'])
        self.maxpd=max(self.maxpd,d['primal_dual_residual']);self.minslack=min(self.minslack,d['minimum_slack'])
        return v

    def metrics(self):
        return {'KKT_calls':self.calls,'LP_calls':self.lp_calls,'cold_retries':self.retries,
                'max_full_stoichiometric_residual':self.maxeq,'max_primal_dual_residual':self.maxpd,
                'minimum_active_slack':self.minslack}


def integrate(mu=1e-8,rtol=1e-8,atol=1e-10,max_step=.02,entry=1e-5,uptake=True,prefix=False,method='DOP853'):
    mods=[Model(s,mu,1e-11,uptake)for s in ['S1','S2']]
    active=[True,True];level=2.
    # First five entries are amounts X1,X2,G1,G2,P. The remaining entries
    # independently accumulate metabolic formation/removal for balance checks.
    y0=np.r_[5.,5.,50.,50.,0.,np.zeros(9)]
    t0=0.;segments=[];events=[];started=time.perf_counter();guard_calls=0
    while t0<5.-1e-11:
        phase_active=active.copy();phase_level=level
        mods[0].level=phase_level
        for m in mods:m.warm=None;m.cache=None
        def intracellular(y,check_guard=True):
            vals=[]
            for i,m in enumerate(mods):
                if not phase_active[i]:vals.append(np.zeros(95));continue
                env=(y[2+i]/VOLUME,y[4]/VOLUME)
                if check_guard and m.capacity(env)<entry*.25:
                    # Defined only to keep event root bracketing finite if an
                    # RK trial crosses the active-domain boundary. Those trial
                    # states are not retained as active biochemical solutions.
                    vals.append(np.zeros(95));continue
                vals.append(m.flux(env))
            return vals
        def rhs(t,y):
            vv=intracellular(y)
            growth=np.array([v[12]*y[i] for i,v in enumerate(vv)])
            loss=KD*y[:2]
            gterms=np.array([MW_G*vv[i][27]*y[i] for i in range(2)])
            pterms=np.array([MW_P*vv[i][37]*y[i] for i in range(2)])
            return np.r_[growth-loss,gterms,pterms.sum(),growth,loss,gterms,pterms,0.]
        funcs=[];labels=[]
        if phase_level>0:
            threshold=7.5 if phase_level==2 else 2.
            def secretion(t,y):return y[2]/VOLUME-threshold
            secretion.terminal=True;secretion.direction=-1;funcs.append(secretion);labels.append(('secretion',threshold))
        for i,m in enumerate(mods):
            if phase_active[i]:
                def starve(t,y,i=i,m=m):return m.capacity((y[2+i]/VOLUME,y[4]/VOLUME))-entry
                starve.terminal=True;starve.direction=-1;funcs.append(starve);labels.append(('quiescent',i))
            elif i==1 and uptake and phase_active[0]:
                # A producer may restore an uptake opportunity. Hysteresis
                # avoids an immediate switch at the same resource threshold.
                def revive(t,y,i=i,m=m):return m.capacity((y[2+i]/VOLUME,y[4]/VOLUME))-2*entry
                revive.terminal=True;revive.direction=1;funcs.append(revive);labels.append(('reactivate',i))
        sol=solve_ivp(rhs,(t0,5.),y0,method=method,rtol=rtol,atol=atol,max_step=max_step,events=funcs or None,dense_output=True)
        if not sol.success:raise RuntimeError(sol.message)
        segments.append({'t0':t0,'tf':float(sol.t[-1]),'active':phase_active,'level':phase_level,'solution':sol})
        t0=float(sol.t[-1]);y0=sol.y[:,-1]
        if t0>=5.-1e-11:break
        hit=next(j for j,e in enumerate(sol.t_events)if len(e))
        kind,value=labels[hit]
        events.append({'time_h':t0,'event':kind,'species':value+1 if kind!='secretion'else None,
                       'source_glucose_threshold':value if kind=='secretion'else None,
                       'concentrations':(y0[:5]/VOLUME).tolist(),'active_before':phase_active})
        if kind=='secretion':level-=1.
        elif kind=='quiescent':
            active[value]=False
            if prefix:break
        else:active[value]=True
        if len(events)>50:raise RuntimeError('Excessive switching; no adoption.')
        print('event',kind,value,'t',t0,'state',y0[:5]/VOLUME,flush=True)
    return {'segments':segments,'events':events,'elapsed_s':time.perf_counter()-started,
            'models':mods,'mu':mu,'rtol':rtol,'atol':atol,'entry':entry,'uptake':uptake,'prefix':prefix,'method':method,'max_step':max_step}


def sample(run,n=501):
    times=np.linspace(0,run['segments'][-1]['tf'],n)
    yy=[];flux=[];modes=[];levels=[];maxbound=0.;gaps=[]
    for t in times:
        seg=next((s for s in run['segments']if s['t0']<=t<s['tf']),run['segments'][-1])
        y=seg['solution'].sol(t);yy.append(y)
        pair=[]
        for i,m in enumerate(run['models']):
            m.level=seg['level'];m.warm=None
            if seg['active'][i]:
                env=(y[2+i]/VOLUME,y[4]/VOLUME)
                v=m.flux(env);l,u=m.bounds(env)
                maxbound=max(maxbound,float(np.max(l-v)),float(np.max(v-u)))
                if len(yy)%50==1:gaps.append({'time_h':float(t),'species':i+1,'gap':m.capacity(env)-float(v[12]),'bound':190*run['mu']})
            else:v=np.zeros(95)
            pair.append(v)
        flux.append(pair);modes.append(seg['active']);levels.append(seg['level'])
    return times,np.asarray(yy),np.asarray(flux),np.asarray(modes),np.asarray(levels),maxbound,gaps


def save_run(name,run):
    out=OUT/name;out.mkdir(parents=True,exist_ok=True)
    t,y,v,active,level,maxbound,gaps=sample(run)
    np.savez_compressed(out/'trajectory.npz',time_h=t,amounts=y,flux=v,active=active,donor_secretion_level=level)
    header='time_h,X1_gDW_L,X2_gDW_L,G1_g_L,G2_g_L,P_g_L,vbio1_h_inv,vbio2_h_inv,vG1_mmol_gDW_h,vG2_mmol_gDW_h,vP1_mmol_gDW_h,vP2_mmol_gDW_h,active1,active2'
    np.savetxt(out/'trajectory.csv',np.c_[t,y[:,:5]/VOLUME,v[:,:,12],v[:,:,27],v[:,:,37],active],delimiter=',',header=header,comments='')
    expected=np.c_[5+y[:,5]-y[:,7],5+y[:,6]-y[:,8],50+y[:,9],50+y[:,10],y[:,11]+y[:,12]]
    balances=np.max(abs(y[:,:5]-expected),axis=0)
    report={'name':name,'full_horizon':bool(t[-1]==5),'terminal_time_h':float(t[-1]),'terminal_concentrations':(y[-1,:5]/VOLUME).tolist(),
            'minimum_concentrations':np.min(y[:,:5]/VOLUME,axis=0).tolist(),'maximum_pyruvate_g_L':float(y[:,4].max()/VOLUME),
            'events':run['events'],'numerical':{k:run[k]for k in ['mu','rtol','atol','entry','method','max_step','elapsed_s']},
            'active_metabolism':[m.metrics()for m in run['models']],'maximum_active_bound_excess':maxbound,
            'maximum_integrated_pool_balance_error_g':balances.tolist(),'LP_gap_checks':gaps,
            'biochemical_mapping':{'glucose_exchange_1based':28,'donor_and_recipient_pyruvate_exchange_1based':38,'ATPM_1based':11,'biomass_1based':13,'pyruvate_MW_g_mmol':MW_P},
            'scope':'Pyruvate-transfer simulation with explicit growth/quiescence states, separate glucose pools and a common extracellular pyruvate pool. Active and quiescent dynamics and switching thresholds are specified below.',
            'parameters':{'volume_L':5,'horizon_h':5,'initial_concentrations_X1_X2_G1_G2_P':[1,1,10,10,0],'feed_L_h':0,'death_rate_h_inv':KD,'pyruvate_recipient_uptake_enabled':run['uptake']},
            'active_model':'Stored finite bounds and ATP-maintenance lower bound8.3899 retained. Glucose and recipient pyruvate uptake lower bounds are exact negative kinetic capacities without minimum-width widening at zero pool. μ is stated above. Donor secretion lower bound changes2−epsilon→1−epsilon→0 at G1=7.5 and2g/L with event restart and fresh KKT state. The zero phase forbids the source epsilon-sized reverse donor uptake.',
            'quiescent_model':'At a maximum feasible biomass flux of entry h−1, active metabolism switches off; all95 intracellular fluxes become zero. Biomass continues to decay at0.01h−1. Recipient may reactivate if available resources restore maximum growth above twice entry. No ATPM demand is claimed for the zero-metabolism mode. This is a stated dormant-cell idealisation, not a measured viability model.',
            'source_files':[{'path':str(p),'sha256':sha(p)}for p in SOURCE.iterdir()if p.suffix in ['.m','.mat','.txt']],
            'script_sha256':sha(Path(__file__)),'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()},
            'adoption_ready':False}
    (out/'audit.json').write_text(json.dumps(report,indent=2))
    (out/'run_script.py').write_text(Path(__file__).read_text())
    print(name,report['terminal_concentrations'],'seconds',run['elapsed_s'],flush=True)
    return report,(t,y,v,active,level)


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--run',choices=['prefix','primary','refined','reference','threshold','barrier_sensitivity'],default='primary');a=pa.parse_args()
    config={'prefix':dict(mu=1e-6,entry=.001,prefix=True,max_step=.02),
            'primary':dict(mu=1e-8,entry=1e-5),
            'refined':dict(mu=1e-8,entry=1e-5,rtol=1e-10,atol=1e-12,max_step=.01,method='RK45'),
            'reference':dict(mu=1e-8,entry=1e-5,rtol=1e-10,atol=1e-12,max_step=.01,uptake=False,method='RK45'),
            'threshold':dict(mu=1e-9,entry=1e-6,rtol=1e-9,atol=1e-11,max_step=.01),
            'barrier_sensitivity':dict(mu=1e-9,entry=1e-5,rtol=1e-9,atol=1e-11,max_step=.01)}
    OUT.mkdir(parents=True,exist_ok=True)
    try:save_run(a.run,integrate(**config[a.run]))
    except Exception as e:
        (OUT/f'failure_{a.run}.json').write_text(json.dumps({'run':a.run,'parameters':config[a.run],'error':str(e),'script_sha256':sha(Path(__file__))},indent=2))
        raise


if __name__=='__main__':main()
