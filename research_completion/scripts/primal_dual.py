from flux_model import FluxModel,ROOT
import numpy as np
from scipy.sparse import bmat,diags
from scipy.sparse.linalg import spsolve
import time,json

def solve(model,S,mu=1e-6,tol=1e-8,maxiter=120,warm=None):
 l,u=model.bounds(S);A=model.A;c=model.c;b=model.b;n=len(c)
 if warm is None:
  lp=model.lp(S);margin=np.minimum(.01,.1*(u-l));x=np.clip(lp.x,l+margin,u-margin);lam=-lp.eqlin.marginals
  zl=np.maximum(lp.lower.marginals,1e-3);zu=np.maximum(-lp.upper.marginals,1e-3)
 else:
  x,lam,zl,zu=[a.copy() for a in warm];margin=np.minimum(1e-8,.01*(u-l));x=np.clip(x,l+margin,u-margin)
 def residual(x,lam,zl,zu):
  sl=x-l;su=u-x
  return -c+A.T@lam-zl+zu,A@x-b,sl*zl-mu,su*zu-mu
 for it in range(maxiter):
  sl=x-l;su=u-x;rd,rp,rl,ru=residual(x,lam,zl,zu)
  norm=max(np.max(np.abs(v)) for v in [rd,rp,rl,ru])
  if norm<tol:break
  D=zl/sl+zu/su;K=bmat([[diags(D),A.T],[A,None]],format='csc');right=np.r_[-rd-rl/sl+ru/su,-rp]
  d=spsolve(K,right);dx=d[:n];dl=d[n:];dzl=(-rl-zl*dx)/sl;dzu=(-ru+zu*dx)/su
  alpha=1.
  for vals,dvals in [(sl,dx),(su,-dx),(zl,dzl),(zu,dzu)]:
   neg=dvals<0
   if np.any(neg):alpha=min(alpha,.995*np.min(-vals[neg]/dvals[neg]))
  old=np.linalg.norm(np.r_[rd,rp,rl,ru])
  for trial in range(35):
   new=[x+alpha*dx,lam+alpha*dl,zl+alpha*dzl,zu+alpha*dzu];rr=residual(*new);val=np.linalg.norm(np.concatenate(rr))
   if np.isfinite(val) and val < (1-1e-4*alpha)*old:break
   alpha*=.5
  else:raise RuntimeError(f'PD line search: {model.name} S={S} residual={norm}')
  x,lam,zl,zu=new
 else:raise RuntimeError(f'PD iterations: {model.name} S={S} residual={norm}')
 # Explicit barrier stationarity is stricter near narrow bounds than primal-dual residuals.
 stat=np.max(np.abs(-c+A.T@lam-mu/(x-l)+mu/(u-x)))
 d={'equality_residual':float(np.max(np.abs(A@x-b))),'stationarity_residual':float(stat),'primal_dual_residual':float(norm),'minimum_slack':float(min(np.min(x-l),np.min(u-x))),'newton_steps':it,'growth':float(x[model.bio]),'glucose_flux':float(x[model.g])}
 return x,lam,zl,zu,d
if __name__=='__main__':
 for name in ['SC','KP']:
  m=FluxModel(name,1e-6);t=time.perf_counter();sol=solve(m,100,tol=1e-11);print(name,sol[-1],'time',time.perf_counter()-t,flush=True)
