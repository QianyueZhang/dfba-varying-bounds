"""Independent physical-amount DOP853 replay; does not import study model code.

Table values are shared scientific inputs. Interpolators, inhibition mapping,
physical-amount ODEs, policy clipping, and amount tangents are assembled here.
Volume and its tangent are analytical inside and at endpoints of each interval.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

DT = 2.0
CAP = 1.8
FEED_MAX = .2
GUARD = 1e-11
RTOL = 2e-13
ATOL = 2e-14
MAX_STEP = .025
TABLE = Path(__file__).resolve().parents[1] / 'data/flux_table_eps1e-06.json'


def from_log(state):
    v, l1, l2, q = np.asarray(state, float)
    return np.array([v, np.exp(l1), np.exp(l2), q])


def to_log(state):
    v, m1, m2, q = np.asarray(state, float)
    if min(m1,m2) <= 0: raise ValueError('Nonpositive biomass amount')
    return np.array([v, np.log(m1), np.log(m2), q])


def initial_amounts(ratio, substrate):
    return np.array([.5, .001*ratio/(1+ratio), .001/(1+ratio), .5*substrate])


class AmountReactor:
    def __init__(self, family='table', inhibition=None, table=TABLE):
        if family not in ('table','analytic'): raise ValueError('Unknown family')
        self.family=family
        self.inhibition=None if inhibition is None else np.asarray(inhibition,float)
        if self.inhibition is not None and (self.inhibition.shape!=(2,) or np.any(self.inhibition<=0)):
            raise ValueError('Inhibition must be a positive species pair')
        self.curves=[]
        if family=='table':
            data=json.loads(Path(table).read_text())
            for species in ['SC','KP']:
                rows=data[species]['rows']
                xx=np.array([r['S'] for r in rows]); yy=np.array([[r['growth'],r['glucose_flux']] for r in rows])
                self.curves.append((PchipInterpolator(xx,yy,axis=0,extrapolate=False),xx[0],xx[-1]))

    def rates(self, substrate, kappa):
        k=np.asarray(kappa,float); s=max(0.,float(substrate))
        if k.shape!=(2,) or np.any(k<=0): raise ValueError('Invalid kappa')
        if self.family=='analytic':
            ks=np.array([4.,.6]); mmax=np.array([1.12,.92]); yields=np.array([.46,.54]); x=k*s
            ki=np.full(2,np.inf) if self.inhibition is None else self.inhibition
            denominator=ks+x+x*x/ki
            mu=mmax*x/denominator
            dmu=mmax*k*(ks-x*x/ki)/denominator**2 if substrate>=0 else np.zeros(2)
            return np.c_[mu,-mu/(.18*yields)],np.c_[dmu,-dmu/(.18*yields)]
        x=k*s; dx=k.copy()
        if self.inhibition is not None:
            ks=np.array([6.95,.385]); den=1+x*x/(self.inhibition*ks)
            dx=k*(1-x*x/(self.inhibition*ks))/den**2
            x=x/den
        ramp=min(1.,s/1e-5); dramp=1e5 if 0<s<1e-5 else 0.
        values=[]; slopes=[]
        for i,(curve,lo,hi) in enumerate(self.curves):
            if not lo<=x[i]<=hi: raise ValueError('Outside archived response table')
            value=curve(x[i]); values.append(value*ramp)
            slopes.append(curve.derivative()(x[i])*dx[i]*ramp+value*dramp)
        return np.asarray(values),np.asarray(slopes)

    def interval(self, state, feed, kappa, tangent=None, dfeed=None):
        y=np.asarray(state,float); volume0=float(y[0]); amount0=y[1:].copy()
        gradients=tangent is not None
        if gradients:
            tangent=np.asarray(tangent,float); dfeed=np.asarray(dfeed,float)
            p=tangent.shape[1]; jv0=tangent[0].copy()
            extended=np.r_[amount0,tangent[1:].ravel()]
        else: extended=amount0
        def rhs(t,yy):
            v=volume0+t*feed; m=yy[:2]; q=yy[2]; s=q/v
            rates,slopes=self.rates(s,kappa)
            f=np.r_[rates[:,0]*m,100*feed+.18*rates[:,1]@m]
            if not gradients: return f
            ja=yy[3:].reshape(3,p); jv=jv0+t*dfeed
            ds=(ja[2]-s*jv)/v
            jm=rates[:,0,None]*ja[:2]+(m*slopes[:,0])[:,None]*ds
            jq=100*dfeed+.18*(rates[:,1]@ja[:2]+(slopes[:,1]@m)*ds)
            return np.r_[f,np.vstack([jm,jq]).ravel()]
        sol=solve_ivp(rhs,(0,DT),extended,method='DOP853',rtol=RTOL,atol=ATOL,max_step=MAX_STEP)
        if not sol.success: raise RuntimeError(sol.message)
        volume=volume0+DT*feed; endpoint=np.r_[volume,sol.y[:3,-1]]
        svalues=sol.y[2]/(volume0+feed*sol.t)
        out={'amount_state':endpoint,'state':to_log(endpoint),'min_S':float(svalues.min()),'max_S':float(svalues.max()),'rhs_calls':sol.nfev}
        if gradients: out['J']=np.vstack([jv0+DT*dfeed,sol.y[3:,-1].reshape(3,p)])
        return out

    def policy(self,state,theta,kappas,reference,basis,gradients=True):
        y=np.asarray(state,float).copy(); theta=np.asarray(theta,float)
        n=len(kappas); p=n+1; J=np.zeros((4,p)); feeds=[]; branches=[]; knots=[to_log(y)]
        minimum=float(y[3]/y[0]); gain=theta[-1]
        for j,k in enumerate(kappas):
            raw=theta[j]; du=np.zeros(p);du[j]=1.
            if j:
                error=np.log(y[1]/y[2])-reference[j]
                raw-=gain*basis[j]*error
                du-=gain*basis[j]*(J[1]/y[1]-J[2]/y[2]);du[-1]-=basis[j]*error
            cap=max(0.,min(FEED_MAX,(CAP-y[0]-GUARD)/DT))
            if cap<=0: feed=0.;du[:]=0.;branch='exhausted'
            elif raw<0: feed=0.;du[:]=0.;branch='lower'
            elif raw>cap:
                feed=cap
                if cap<FEED_MAX: du=-J[0]/DT;branch='volume'
                else: du[:]=0.;branch='upper'
            else: feed=float(raw);branch='interior'
            out=self.interval(y,feed,k,J if gradients else None,du if gradients else None)
            y=out['amount_state']; minimum=min(minimum,out['min_S'])
            if gradients:J=out['J']
            feeds.append(feed);branches.append(branch);knots.append(to_log(y))
        B=y[1:3].sum()
        return {'amount_state':y,'state':to_log(y),'z':float(np.log(y[1]/y[2])), 'logB':float(np.log(B)),'B':float(B),
          'V':float(y[0]),'feeds':np.array(feeds),'branches':branches,'knots':np.array(knots),'min_S':minimum,
          'jz':J[1]/y[1]-J[2]/y[2] if gradients else None,
          'jlogB':J[1:3].sum(axis=0)/B if gradients else None,'J':J if gradients else None}
