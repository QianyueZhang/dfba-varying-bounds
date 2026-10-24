"""Primal-dual solver with separate feasibility and complementarity tests.

The finite-barrier problem is unchanged. Equality row scaling conditions the
linear system. Relative complementarity controls accuracy of weak flux choices;
floating-point subtraction limits are accounted for on nearly active bounds.
"""
import numpy as np
from scipy.sparse import bmat,diags
from scipy.sparse.linalg import spsolve


def solve(model,state,mu=1e-8,tol=1e-10,maxiter=150,warm=None,complementarity_rtol=2e-8):
 l,u=model.bounds(state);original_A=model.A;b0=model.b;c=model.c;n=len(c)
 scale=1/np.maximum(np.asarray(abs(original_A).max(axis=1).toarray()).ravel(),1.)
 A=diags(scale)@original_A;b=scale*b0
 if warm is None:
  lp=model.lp(state);margin=np.minimum(.01,.1*(u-l));x=np.clip(lp.x,l+margin,u-margin)
  lam=-lp.eqlin.marginals/scale;zl=np.maximum(lp.lower.marginals,1e-3);zu=np.maximum(-lp.upper.marginals,1e-3)
 else:
  x,lam,zl,zu=[a.copy() for a in warm];margin=np.minimum(1e-14,.000001*(u-l));x=np.clip(x,l+margin,u-margin)
 def residual(x,lam,zl,zu):
  return -c+A.T@lam-zl+zu,A@x-b,(x-l)*zl-mu,(u-x)*zu-mu
 def comp_floor(x,z,bound):
  return 12*np.finfo(float).eps*np.maximum(np.abs(x),np.abs(bound))*np.abs(z)
 def diagnostics(x,lam,zl,zu):
  rd,rp,rl,ru=residual(x,lam,zl,zu)
  eq=np.max(np.abs(original_A@x-b0));dual=np.max(abs(rd))
  rl_effective=np.maximum(0,np.abs(rl)-comp_floor(x,zl,l))
  ru_effective=np.maximum(0,np.abs(ru)-comp_floor(x,zu,u))
  relative=max(np.max(rl_effective),np.max(ru_effective))/mu
  raw_relative=max(np.max(abs(rl)),np.max(abs(ru)))/mu
  merit=np.linalg.norm(np.r_[rd,rp,rl/mu,ru/mu])
  return eq,dual,relative,raw_relative,merit
 for it in range(maxiter):
  sl=x-l;su=u-x;rd,rp,rl,ru=residual(x,lam,zl,zu)
  eq,dual,relative,raw_relative,merit=diagnostics(x,lam,zl,zu)
  if eq<tol and dual<tol and relative<complementarity_rtol:break
  diagonal=zl/sl+zu/su
  K=bmat([[diags(diagonal),A.T],[A,None]],format='csc')
  step=spsolve(K,np.r_[-rd-rl/sl+ru/su,-rp]);dx=step[:n];dlam=step[n:]
  dzl=(-rl-zl*dx)/sl;dzu=(-ru+zu*dx)/su
  alpha=1.
  for val,delta in [(sl,dx),(su,-dx),(zl,dzl),(zu,dzu)]:
   neg=delta<0
   if np.any(neg):alpha=min(alpha,.995*np.min(-val[neg]/delta[neg]))
  accepted=False
  for _ in range(40):
   new=[x+alpha*dx,lam+alpha*dlam,zl+alpha*dzl,zu+alpha*dzu]
   de=diagnostics(*new)
   if de[-1]<(1-1e-4*alpha)*merit or (de[0]<tol and de[1]<tol and de[2]<complementarity_rtol):
    accepted=True;break
   alpha*=.5
  if not accepted:raise RuntimeError(f'Scaled PD line search {model.name}: eq={eq}, dual={dual}, comp_rel={relative}, raw_comp_rel={raw_relative}, mu={mu}')
  x,lam,zl,zu=new
 else:raise RuntimeError(f'Scaled PD iterations {model.name}: eq={eq}, dual={dual}, comp_rel={relative}')
 diag={'equality_residual':float(eq),'stationarity_residual':float(dual),
       'primal_dual_residual':float(max(eq,dual,max(np.max(abs(r)) for r in residual(x,lam,zl,zu)[2:]))),
       'complementarity_relative_residual':float(relative),'raw_complementarity_relative_residual':float(raw_relative),
       'minimum_slack':float(min(np.min(x-l),np.min(u-x))),
       'newton_steps':it,'growth':float(x[model.bio]),'glucose_flux':float(x[model.g])}
 return x,lam,zl,zu,diag
