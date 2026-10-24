"""Finite-scenario planning and a model-relative surrogate manager.

All scenarios share a fixed causal feedback family. Scenario selection never
reads hidden plant settings. No recursive-feasibility guarantee is asserted.
"""
import time
import numpy as np
from scipy.optimize import minimize, linprog, least_squares
from model import DT, CAP, DESIGN, NUM_MARGIN, project_plan

BAND = DESIGN-NUM_MARGIN
PENALTY = 100.


def estimate_kappa(model, history, previous):
    if not history: return np.array(previous), {'n_intervals':0, 'status':'prior'}
    recent=history[-2:]; prior=np.asarray(previous)
    def residual(k):
        out=[]
        for before,feed,after in recent:
            pred=model.predict(before,[feed],k)['state']
            out.extend((pred[1:3]-after[1:3])/.01)
            out.append((pred[3]/pred[0]-after[3]/after[0])/.1)
        return np.r_[out,.05*(k-prior)]
    fit=least_squares(residual,prior,bounds=([.75,.75],[1.25,1.25]),
        max_nfev=18,ftol=1e-7,xtol=1e-7,gtol=1e-7,diff_step=2e-4)
    return fit.x, {'n_intervals':len(recent),'success':bool(fit.success),'cost':float(fit.cost),
        'nfev':int(fit.nfev),'at_bound':bool(np.any((fit.x<.7501)|(fit.x>1.2499))),
        'singular_values':np.linalg.svd(fit.jac[:-2],compute_uv=False)}


def scenario_set(kappa,stage,robust):
    n=7-stage; center=np.tile(kappa,(n,1))
    if not robust: return [('center',center)]
    radius=.15 if stage==0 else .05 if stage==1 else .025
    out=[('center',center)]
    for sign in (-1,1):
        out.append((f'current_{sign}',np.tile(np.clip(kappa+sign*radius*np.array([1.,-1.]),.65,1.35),(n,1))))
    # A union, not the Cartesian product of current and shift uncertainty.
    # Include a shift at the current decision time: it is still unobserved.
    for shock in (3,4):
        if shock>=stage:
            for sign in (-1,1):
                kk=center.copy(); kk[shock-stage:]=np.clip(kappa+sign*.16*np.array([1.,-1.]),.65,1.35)
                out.append((f'shift_{shock}_{sign}',kk))
    return out


class Evaluation:
    def __init__(self,model,state,target,scenarios,reference,basis):
        self.model=model; self.state=state; self.target=np.log(target)
        self.scenarios=scenarios; self.reference=reference; self.basis=basis
        self.cache={}; self.trace=[]; self.failures=[]
    def __call__(self,theta):
        theta=np.asarray(theta); key=theta.tobytes()
        if key not in self.cache:
            try:
                rows=[self.model.policy(self.state,theta,k,self.reference,self.basis) for _,k in self.scenarios]
            except Exception as ex:
                self.failures.append({'theta':theta.copy(),'error':repr(ex)}); raise
            z=np.array([r['z'] for r in rows]); lb=np.array([r['logB'] for r in rows])
            violation=max(0.,float(np.max(np.abs(z-self.target)))-BAND)
            q={'rows':rows,'z':z,'logB':lb,'jz':np.array([r['jz'] for r in rows]),
                'jlogB':np.array([r['jlogB'] for r in rows]),'violation':violation,
                'feasible':bool(violation<=1e-9),'merit':float(-lb.mean()+PENALTY*violation)}
            self.cache[key]=q; self.trace.append((theta.copy(),q))
        return self.cache[key]
    def best(self):
        feasible=[r for r in self.trace if r[1]['feasible']]
        return min(feasible or self.trace,key=lambda r:r[1]['merit'])


def direct(e,start,gain,maxiter):
    n=len(start)-1; q=e(start); x0=np.r_[start,q['violation']+1e-8]
    def con(x):
        z=e(x[:-1])['z']-e.target
        return np.r_[BAND+x[-1]-z,BAND+x[-1]+z]
    def jac(x):
        j=e(x[:-1])['jz']; return np.vstack([np.c_[-j,np.ones(len(j))],np.c_[j,np.ones(len(j))]])
    try:
        fit=minimize(lambda x:-e(x[:-1])['logB'].mean()+PENALTY*x[-1],x0,
            jac=lambda x:np.r_[-e(x[:-1])['jlogB'].mean(axis=0),PENALTY],
            method='SLSQP',bounds=[(0,.2)]*n+[(0,2 if gain else 0),(0,None)],
            constraints=[{'type':'ineq','fun':con,'jac':jac}],
            options={'maxiter':maxiter,'ftol':1e-7})
        e(fit.x[:-1]); status={'success':bool(fit.success),'message':str(fit.message),'nit':int(fit.nit)}
    except Exception as ex: status={'success':False,'message':repr(ex)}
    theta,q=e.best(); return theta,q,status


def proposal(e,center,radius,gain):
    q=e(center); p=len(center); J=q['jz']; z=q['z']; n=p-1
    A=np.vstack([np.c_[J,-np.ones(len(J))],np.c_[-J,-np.ones(len(J))]])
    b=np.r_[e.target+BAND-z+J@center,-e.target+BAND+z-J@center]
    step=np.r_[np.full(n,radius),min(1.,radius/.035)]
    lower=np.maximum(0,center-step); upper=np.minimum(np.r_[np.full(n,.2),2 if gain else 0],center+step)
    fit=linprog(np.r_[-q['jlogB'].mean(axis=0),PENALTY],A_ub=A,b_ub=b,
        bounds=list(zip(lower,upper))+[(0,None)],method='highs')
    if not fit.success: return center.copy(),{'success':False,'message':fit.message}
    candidate=fit.x[:-1]; dz=candidate-center
    pred_z=z+J@dz; pred_lb=q['logB']+q['jlogB']@dz
    return candidate,{'success':True,'radius':radius,'linear_slack':float(fit.x[-1]),
        'predicted_z':pred_z,'predicted_merit':float(-pred_lb.mean()+PENALTY*max(0,np.max(np.abs(pred_z-e.target))-BAND))}


def solve(model,state,kappa,target,start,stage,method):
    robust=method.startswith('scenario'); gain=method in ('scenario_feedback','scenario_managed')
    start=project_plan(start,state); n=len(start)
    # Fixed nominal reference and signed inverse sensitivity across all solves.
    nominal=model.policy(state,np.r_[start,0.],np.tile(kappa,(n,1)),np.zeros(n),np.zeros(n))
    reference=nominal['knots'][:-1,1]-nominal['knots'][:-1,2]
    sensitivity=nominal['jz'][:n]
    basis=np.sign(sensitivity)*np.minimum(2.,1./np.maximum(abs(sensitivity),.5))
    scenarios=scenario_set(kappa,stage,robust)
    e=Evaluation(model,state,target,scenarios,reference,basis)
    center=np.r_[start,0.]; e(center) # a=0 incumbent retained for both solvers
    events=[]; fallback=False
    maxiter=5 if method=='ce_limited' else 12
    if method!='scenario_managed': theta,q,status=direct(e,center,gain,maxiter)
    else:
        accepted=False; radius=.035
        for attempt in range(3):
            candidate,info=proposal(e,center,radius,gain)
            q=e(candidate); base=e(center)
            # Acceptance certifies only the declared finite-scenario design.
            gate=info['success'] and q['feasible'] and (not base['feasible'] or q['logB'].mean()>=base['logB'].mean()-.002)
            events.append(dict(info,attempt=attempt,action='accept' if gate else 'reject',
                theta=candidate,z=q['z'],violation=q['violation'],merit=q['merit']))
            if gate: theta=candidate; accepted=True; break
            center,_=e.best(); radius*=.5
        if not accepted:
            fallback=True; theta,q,status=direct(e,center,gain,maxiter)
        else: status={'success':True,'message':'finite-scenario gate accepted'}
    q=e(theta)
    trace=[{'theta':t,'z':r['z'],'logB':r['logB'],'violation':r['violation'],'merit':r['merit']} for t,r in e.trace]
    info={'status':status,'fallback':fallback,'events':events,'reference':reference,'basis':basis,
        'scenarios':[{'name':name,'kappas':k} for name,k in scenarios],'selected':q,
        'theta':theta,'trace':trace,'failures':e.failures}
    # Warm start is the center-model realized feed sequence, not raw base feeds.
    return q['rows'][0]['feeds'],info
