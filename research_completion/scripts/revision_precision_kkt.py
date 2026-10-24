"""Primal-dual barrier verification with separately scaled stopping residuals.
Equality/dual tolerances allow floating-point accumulation in large systems;
complementarity is checked more tightly to resolve sensitive exchange fluxes.
"""
import numpy as np
from scipy.sparse import bmat,diags
from scipy.sparse.linalg import spsolve

def solve(model,state,mu=1e-6,tol=2e-12,maxiter=120,warm=None):
 l,u=model.bounds(state);A=model.A;c=model.c;b=model.b;n=len(c);eqtol=1e-10;dualtol=1e-10
 if warm is None:
  lp=model.lp(state);margin=np.minimum(.01,.1*(u-l));x=np.clip(lp.x,l+margin,u-margin);lam=-lp.eqlin.marginals;zl=np.maximum(lp.lower.marginals,1e-3);zu=np.maximum(-lp.upper.marginals,1e-3)
 else:
  x,lam,zl,zu=[a.copy() for a in warm];margin=np.minimum(1e-8,.01*(u-l));x=np.clip(x,l+margin,u-margin)
 def residual(x,lam,zl,zu):return -c+A.T@lam-zl+zu,A@x-b,(x-l)*zl-mu,(u-x)*zu-mu
 def scaled(rr):return np.r_[rr[0]/dualtol,rr[1]/eqtol,rr[2]/tol,rr[3]/tol]
 for it in range(maxiter):
  sl=x-l;su=u-x;rr=residual(x,lam,zl,zu);rd,rp,rl,ru=rr
  if np.max(np.abs(scaled(rr)))<1:break
  D=zl/sl+zu/su;K=bmat([[diags(D),A.T],[A,None]],format='csc');d=spsolve(K,np.r_[-rd-rl/sl+ru/su,-rp]);dx=d[:n];dl=d[n:];dzl=(-rl-zl*dx)/sl;dzu=(-ru+zu*dx)/su;alpha=1.
  for vals,dvals in [(sl,dx),(su,-dx),(zl,dzl),(zu,dzu)]:
   neg=dvals<0
   if np.any(neg):alpha=min(alpha,.995*np.min(-vals[neg]/dvals[neg]))
  old=np.linalg.norm(scaled(rr))
  for trial in range(35):
   new=[x+alpha*dx,lam+alpha*dl,zl+alpha*dzl,zu+alpha*dzu];val=np.linalg.norm(scaled(residual(*new)))
   if np.isfinite(val) and val<(1-1e-4*alpha)*old:break
   alpha*=.5
  else:raise RuntimeError(f'Scaled KKT line search {model.name}: residual {np.max(abs(scaled(rr)))}')
  x,lam,zl,zu=new
 else:raise RuntimeError(f'Scaled KKT iterations {model.name}')
 d={'equality_residual':float(max(abs(A@x-b))),'stationarity_residual':float(max(abs(-c+A.T@lam-mu/(x-l)+mu/(u-x)))),'primal_dual_residual':float(max(np.max(abs(r)) for r in residual(x,lam,zl,zu))),'minimum_slack':float(min(np.min(x-l),np.min(u-x))),'newton_steps':it,'growth':float(x[model.bio]),'glucose_flux':float(x[model.g])}
 return x,lam,zl,zu,d
