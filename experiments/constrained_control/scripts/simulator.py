"""Concentration balances for two yeast populations and seven feed intervals."""
from pathlib import Path
import json,hashlib,time,os
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp
PACKAGE=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('DFBA_CONTROL_OUTPUT',PACKAGE/'reference')).resolve()
THESIS=PACKAGE.parents[1]
TABLE=PACKAGE/'data/flux_table_eps1e-06.json'
class Reactor:
 def __init__(self,table=TABLE):
  self.table=Path(table);self.raw=json.loads(self.table.read_text());self.curves=[]
  for species in ['SC','KP']:
   rr=self.raw[species]['rows'];self.curves.append(PchipInterpolator([r['S'] for r in rr],[[r['growth'],r['glucose_flux']] for r in rr],axis=0))
 def simulate(self,u,initial,rtol=1e-7,atol=1e-9,horizon=14.):
  u=np.asarray(u,float);y=np.asarray(initial,float).copy();V0=float(y[0]);minimum=float(y[3]);calls=0
  for j,feed in enumerate(u):
   def rhs(t,y):
    nonlocal calls
    calls+=1;V,X,Y,S=y;Ss=max(S,0.);factor=min(1.,Ss/1e-5);a,b=[curve(Ss)*factor for curve in self.curves]
    return [feed,(a[0]-feed/V)*X,(b[0]-feed/V)*Y,feed/V*(100-S)+.18*(a[1]*X+b[1]*Y)]
   span=(j*horizon/len(u),(j+1)*horizon/len(u));sol=solve_ivp(rhs,span,y,method='DOP853',rtol=rtol,atol=atol,max_step=.25)
   if not sol.success:raise RuntimeError(sol.message)
   if np.min(sol.y[3]) < -1e-6:raise RuntimeError('Negative substrate beyond numerical tolerance')
   minimum=min(minimum,float(np.min(sol.y[3])));y=sol.y[:,-1]
  V,X,Y,S=y
  if X<=0 or Y<=0:raise RuntimeError('Nonpositive terminal biomass')
  return {'terminal':y.tolist(),'z':float(np.log(X/Y)),'biomass_amount_gDW':float(V*(X+Y)),'log_biomass_amount':float(np.log(V*(X+Y))),'minimum_glucose_g_L':minimum,'volume_analytic_L':V0+horizon/len(u)*float(np.sum(u)),'rhs_evaluations':calls}
def initial_from_condition(condition):
 ratio,S=condition;return [.5,.002*ratio/(1+ratio),.002/(1+ratio),S]
def features(u,condition):return np.r_[u,np.log(condition[0]),condition[1]]
def sample_feeds(rng,n):
 """Two thirds uniform cube rejection; one third includes lower feed totals."""
 out=[]
 while len(out)<n:
  u=rng.uniform(0,.2,7)
  if u.sum()>.65:continue
  if len(out)%3==0:u*=rng.uniform(.05,1.)
  out.append(u)
 return np.asarray(out)
