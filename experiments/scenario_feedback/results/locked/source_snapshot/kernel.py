"""Shared optional native RHS; same PCHIP polynomials and equations as model.py.

Compilation is a numerical implementation optimization, not a learned model.
The uncompiled implementation remains available for independent comparison.
"""
import numpy as np
try:
    from numba import njit
    AVAILABLE=True
except ImportError:
    AVAILABLE=False
    def njit(*args,**kwargs):
        return lambda f:f

@njit(cache=True)
def polynomial(x,c,s):
    j=min(len(x)-2,max(0,np.searchsorted(x,s,side='right')-1)); d=s-x[j]
    value=((c[0,j]*d+c[1,j])*d+c[2,j])*d+c[3,j]
    derivative=(3*c[0,j]*d+2*c[1,j])*d+c[2,j]
    return value,derivative

@njit(cache=True)
def rhs(y,feed,kappa,du,gradients,family,inhibition,x0,c0,x1,c1):
    v=y[0]; physical=max(0.,y[3]/v); substrate=y[3]/v
    rates=np.empty((2,2)); dr=np.empty((2,2))
    for i in range(2):
        k=kappa[i]; raw=k*physical
        if family==1:
            ks=4. if i==0 else .6; maximum=1.12 if i==0 else .92; yield_= .46 if i==0 else .54
            den=ks+raw+raw*raw/inhibition[i]
            mu=maximum*raw/den
            slope=maximum*(ks-raw*raw/inhibition[i])*k/(den*den)
            if substrate<0.:slope=0.
            rates[i,0]=mu;rates[i,1]=-mu/yield_/.18
            dr[i,0]=slope;dr[i,1]=-slope/yield_/.18
        else:
            eff=raw;scale=k
            if np.isfinite(inhibition[i]):
                ks=6.95 if i==0 else .385;den=1.+raw*raw/(inhibition[i]*ks)
                eff=raw/den;scale=k*(1.-raw*raw/(inhibition[i]*ks))/(den*den)
            if eff>100.+1e-10 or eff<0.:raise ValueError('Effective substrate outside archived table domain')
            if i==0:base,slope=polynomial(x0,c0,eff)
            else:base,slope=polynomial(x1,c1,eff)
            factor=min(1.,physical/1e-5);dfactor=1e5 if 0.<physical<1e-5 else 0.
            rates[i]=base*factor;dr[i]=slope*scale*factor+base*dfactor
    m0=np.exp(y[1]);m1=np.exp(y[2]);out=np.zeros_like(y)
    out[0]=feed;out[1]=rates[0,0];out[2]=rates[1,0]
    out[3]=100.*feed+.18*(rates[0,1]*m0+rates[1,1]*m1)
    if gradients:
        p=len(du); uptake=.18*(dr[0,1]*m0+dr[1,1]*m1)
        for j in range(p):
            vj=y[4+j];qj=y[4+3*p+j]; ds=(qj-substrate*vj)/v
            out[4+j]=du[j]
            out[4+p+j]=dr[0,0]*ds;out[4+2*p+j]=dr[1,0]*ds
            out[4+3*p+j]=uptake*ds+.18*(rates[0,1]*m0*y[4+p+j]+rates[1,1]*m1*y[4+2*p+j])+100.*du[j]
    return out
