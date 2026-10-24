"""Causal scenario rollouts with fixed reference and analytic policy tangents.

No true evaluation-process parameter is an input to a deployed policy.
Scenario feasibility is a finite-design property, not a robust certificate.
"""
from pathlib import Path
import time
import numpy as np
from scipy.integrate import solve_ivp
from base_model import Reactor as BaseReactor, initial_state, observable, DT, ROOT

CAP = 1.8
FEED_MAX = .2
DESIGN = .0075
ACCEPT = float(np.log(1.01))
NUM_MARGIN = 2e-6


def project_plan(u, state):
    u = np.clip(np.asarray(u, float), 0., FEED_MAX)
    total = max(0., (CAP - state[0] - 1e-11) / DT)
    if u.sum() > total:
        # Projection is part of candidate construction; evaluate the result.
        lo, hi = 0., float(u.max())
        for _ in range(50):
            middle = (lo + hi) / 2
            if np.maximum(u-middle, 0.).sum() > total: lo = middle
            else: hi = middle
        u = np.maximum(u-hi, 0.)
    return u


class Reactor(BaseReactor):
    def __init__(self, family='table', inhibition=None):
        super().__init__()
        self.family = family
        self.inhibition = inhibition

    def rates(self, substrate, kappa):
        physical = max(float(substrate), 0.)
        k = np.asarray(kappa)
        # A second, explicitly designed mathematical benchmark, not a culture fit.
        if self.family == 'analytic':
            ks = np.array([4., .6]); maximum = np.array([1.12, .92])
            yields = np.array([.46, .54])
            s = physical*k
            inhibition = np.asarray(self.inhibition) if self.inhibition is not None else np.full(2, np.inf)
            den = ks+s+s*s/inhibition
            mu = maximum*s/den
            derivative = maximum*(ks-s*s/inhibition)*k/den**2
            return np.c_[mu,-mu/yields/.18], np.c_[derivative,-derivative/yields/.18]
        if self.inhibition is None:
            return super().rates(substrate,kappa)
        # Out-of-affinity-family substrate inhibition, mapped jointly into
        # paired growth/uptake curves; prescribed stress, not new calibration.
        ks = np.array([6.95,.385]); ki = np.asarray(self.inhibition)
        raw = k*physical
        den = 1.+raw*raw/(ki*ks)
        eff = raw/den
        slope = k*(1.-raw*raw/(ki*ks))/den**2
        factor = min(1.,physical/1e-5)
        dfactor = 1e5 if 0.<physical<1e-5 else 0.
        rates=[]; derivatives=[]
        for i,(curve,derivative) in enumerate(self.curves):
            base=curve(eff[i]); rates.append(base*factor)
            derivatives.append(derivative(eff[i])*slope[i]*factor+base*dfactor)
        return np.array(rates),np.array(derivatives)

    def policy(self,state,theta,kappas,reference,basis,gradients=True,tight=False):
        tic=time.perf_counter(); self.calls+=1
        n=len(kappas); p=n+1; y=np.array(state,float).copy(); J=np.zeros((4,p))
        theta=np.asarray(theta); gain=theta[-1]
        feeds=[]; knots=[y.copy()]; branches=[]; min_s=float(y[3]/y[0]); kink_min=np.inf
        try:
            for j,kappa in enumerate(kappas):
                volume0=float(y[0]); volume_sensitivity=J[0].copy()
                raw=float(theta[j]); du=np.eye(p)[j].copy()
                if j>0:
                    err=y[1]-y[2]-reference[j]
                    raw-=gain*basis[j]*err
                    du-=gain*basis[j]*(J[1]-J[2])
                    du[-1]-=basis[j]*err
                cap=max(0.,min(FEED_MAX,(CAP-volume0-1e-11)/DT))
                kink_min=min(kink_min,abs(raw),abs(raw-cap))
                if raw<=0.:
                    feed=0.; du[:]=0.; branch='lower'
                elif raw>=cap:
                    feed=cap
                    if cap<FEED_MAX and cap>0.:
                        du=-volume_sensitivity/DT; branch='volume'
                    else: du[:]=0.; branch='upper'
                else: feed=raw; branch='interior'
                # This feed/tangent is held for the complete 2h interval.
                def rhs(t,yy):
                    v,l1,l2,q=yy[:4]; s=q/v; rates,dr=self.rates(s,kappa)
                    mass=np.exp([l1,l2]); f=np.r_[feed,rates[:,0],100.*feed+.18*(rates[:,1]@mass)]
                    if not gradients: return f
                    A=np.zeros((4,4)); A[1:3,0]=-dr[:,0]*s/v; A[1:3,3]=dr[:,0]/v
                    uptake=.18*(dr[:,1]@mass)
                    A[3,0]=-uptake*s/v; A[3,1:3]=.18*rates[:,1]*mass; A[3,3]=uptake/v
                    tangent=A@yy[4:].reshape(4,p)
                    tangent[0]+=du; tangent[3]+=100.*du
                    return np.r_[f,tangent.ravel()]
                initial=np.r_[y,J.ravel()] if gradients else y
                solution=solve_ivp(rhs,(0.,DT),initial,method='DOP853' if tight else 'LSODA',
                    rtol=5e-12 if tight else 2e-9,atol=5e-14 if tight else 2e-11,max_step=.05 if tight else .15)
                if not solution.success: raise RuntimeError(solution.message)
                y=solution.y[:4,-1].copy(); y[0]=volume0+DT*feed
                if gradients:
                    J=solution.y[4:,-1].reshape(4,p).copy(); J[0]=volume_sensitivity+DT*du
                min_s=min(min_s,float(np.min(solution.y[3]/solution.y[0])))
                if min_s < -2e-7: raise RuntimeError('Negative substrate')
                feeds.append(feed); knots.append(y.copy()); branches.append(branch)
            mass=np.exp(y[1:3]); biomass=mass.sum()
            return {'state':y,'z':float(y[1]-y[2]),'logB':float(np.log(biomass)),'B':float(biomass),
                'V':float(y[0]),'min_S':min_s,'feeds':np.array(feeds),'knots':np.array(knots),
                'jz':J[1]-J[2] if gradients else None,
                'jlogB':(mass@J[1:3])/biomass if gradients else None,
                'branches':branches,'minimum_kink_distance':float(kink_min)}
        except Exception:
            self.failures+=1; raise
        finally: self.seconds+=time.perf_counter()-tic
