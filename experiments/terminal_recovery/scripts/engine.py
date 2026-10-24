"""Adaptive Dormand--Prince screening; all certificate wording remains numerical.

The RHS is the archived paired-rate model. Independent amount integration is
used for deployment verification, never to tune the finite decision library.
"""
from pathlib import Path
import sys,json
import numpy as np
from scipy.interpolate import PchipInterpolator
from numba import njit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
from prior_kernel import rhs
CAP=1.8; DT=2.; GUARD=1e-11; ACCEPT=float(np.log(1.01)); BAND=.0075
GRID=np.linspace(0,.2,21)
K=np.array([[1.,1.],[.975,1.025],[1.025,.975]])

def arguments(family,inhibition=None):
    data=json.loads((ROOT/'data/flux_table_eps1e-06.json').read_text()); curves=[]
    for sp in ('SC','KP'):
        rows=data[sp]['rows'];curves.append(PchipInterpolator([r['S'] for r in rows],[[r['growth'],r['glucose_flux']] for r in rows],axis=0,extrapolate=False))
    return (int(family=='analytic'),np.full(2,np.inf) if inhibition is None else np.array(inhibition,float),curves[0].x,curves[0].c,curves[1].x,curves[1].c)

def actions(x):
    cap=max(0.,min(.2,(CAP-x[0]-GUARD)/DT))
    return np.unique(np.minimum(GRID,cap))

def initial():return np.array([.5,np.log(.0005),np.log(.0005),10.])

@njit(cache=True)
def flow(y0,u,kappa,args,rtol=2e-11,atol=2e-13,maxstep=.05):
    y=y0.copy();t=0.;h=min(.005,maxstep);minimum=y[3]/y[0]; steps=0; rejected=0
    dummy=np.zeros(1)
    while t<2.:
        if steps+rejected>100000:raise ValueError('Adaptive integration budget exhausted')
        h=min(h,2.-t)
        if h<1e-13:raise ValueError('Adaptive integration step underflow')
        a=rhs(y,u,kappa,dummy,False,*args)
        b=rhs(y+h*(a/5),u,kappa,dummy,False,*args)
        c=rhs(y+h*(3*a/40+9*b/40),u,kappa,dummy,False,*args)
        d=rhs(y+h*(44*a/45-56*b/15+32*c/9),u,kappa,dummy,False,*args)
        e=rhs(y+h*(19372*a/6561-25360*b/2187+64448*c/6561-212*d/729),u,kappa,dummy,False,*args)
        f=rhs(y+h*(9017*a/3168-355*b/33+46732*c/5247+49*d/176-5103*e/18656),u,kappa,dummy,False,*args)
        yn=y+h*(35*a/384+500*c/1113+125*d/192-2187*e/6784+11*f/84)
        yn[0]=y0[0]+(t+h)*u
        g=rhs(yn,u,kappa,dummy,False,*args)
        err=h*((35/384-5179/57600)*a+(500/1113-7571/16695)*c+(125/192-393/640)*d+(-2187/6784+92097/339200)*e+(11/84-187/2100)*f-g/40)
        scale=atol+rtol*np.maximum(np.abs(y),np.abs(yn))
        norm=np.sqrt(np.mean((err/scale)**2))
        if not np.isfinite(norm):raise ValueError('Nonfinite integration error')
        if norm<=1.:
            y=yn;t+=h;steps+=1;minimum=min(minimum,y[3]/y[0])
            h=min(maxstep,h*(5. if norm==0. else min(5.,max(.2,.9*norm**(-.2)))))
        else:
            rejected+=1;h*=max(.1,.9*norm**(-.2))
    y[0]=y0[0]+2*u
    return y,minimum,steps,rejected

@njit(cache=True)
def batch(states,feeds,kappas,args,rtol=2e-11,atol=2e-13,maxstep=.05):
    out=np.empty_like(states);minimum=np.empty(len(feeds));steps=0;rejected=0
    for i in range(len(feeds)):
        out[i],minimum[i],s,r=flow(states[i],feeds[i],kappas[i],args,rtol,atol,maxstep);steps+=s;rejected+=r
    return out,minimum,steps,rejected

def one_step(x,args,kappas=K):
    u=actions(x);n=len(kappas)
    out,minimum,steps,rejected=batch(np.repeat(x[None,:],len(u)*n,axis=0),np.repeat(u,n),np.tile(kappas,(len(u),1)),args)
    return {'actions':u,'terminal':out.reshape(len(u),n,4),'min_S':minimum.reshape(len(u),n),'steps':steps,'rejected_steps':rejected,'transitions':len(u)*n}

def tree(x,args,kappas=K):
    first=one_step(x,args,kappas);branches=[];calls=first['transitions']
    for states in first['terminal']:
        # Volume is identical for a common first action, so the second action
        # alphabet is common across successor states. The future minimizer is
        # nevertheless a function of the observed state, not its scenario ID.
        maps=[one_step(state,args,kappas) for state in states]
        assert all(np.array_equal(m['actions'],maps[0]['actions']) for m in maps)
        calls+=sum(m['transitions'] for m in maps)
        branches.append({'actions':maps[0]['actions'],'terminal':np.stack([m['terminal'] for m in maps]),'min_S':np.stack([m['min_S'] for m in maps])})
    return {'first':first,'branches':branches,'transitions':calls}

def serialize(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)

def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,default=serialize)+'\n')
