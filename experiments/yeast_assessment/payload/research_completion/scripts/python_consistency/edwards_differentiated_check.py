"""Check consistently initialised differentiated KKT against direct KKT dFBA.

The capacity case differentiates its glucose upper bound. The prescribed
case eliminates the glucose flux exactly and differentiates b=-N[:,0]q(G).
No flux is clipped. Initial states solve the finite-barrier KKT equations.
"""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research_completion/scripts/python_migration'))
from edwards_comparison import Model,MU
OUT=ROOT/'research_completion/runs/python_consistency/edwards'

class Differential:
 def __init__(self,kind):
  self.kind=kind;self.model=Model(kind);v,d=self.model.flux(10,cold=True)
  self.free=self.model.free;self.n=len(self.free);self.A=self.model.A.toarray();self.c=self.model.c;lam=self.model.warm[1]
  self.initial=np.r_[1.,10.,0.,0.,v[self.free],lam];self.calls=0;self.min_trial_slack=np.inf
 def unpack(self,y):
  X,G,D,E=y[:4];raw=y[4:4+self.n];v=np.zeros(13);v[self.free]=raw
  if self.kind=='prescribed':v[0]=12*G/(2+G)
  return v,raw,y[4+self.n:]
 def rhs(self,t,y):
  self.calls+=1;X,G,D,E=y[:4];v,free,_=self.unpack(y);l,u=self.model.bounds(G);sl=free-l;su=u-free
  self.min_trial_slack=min(self.min_trial_slack,float(min(sl.min(),su.min())))
  dy=np.array([.023*v[8]*X,-.18*v[0]*X,.091*v[11]*X,.046*v[12]*X]);dq=24/(2+G)**2*dy[1]
  du=np.zeros(self.n);db=np.zeros(8)
  if self.kind=='capacity':du[0]=dq
  else:db=-self.model.full_A[:,0].toarray().ravel()*dq
  zl=MU/sl**2;zu=MU/su**2;K=np.block([[np.diag(zl+zu),self.A.T],[self.A,np.zeros((8,8))]])
  return np.r_[dy,np.linalg.solve(K,np.r_[zu*du,db])]
 def residuals(self,y):
  v,raw,lam=self.unpack(y);l,u=self.model.bounds(y[1]);sl=raw-l;su=u-raw
  return [float(abs(self.A@raw-self.model.b).max()),float(abs(-self.c+self.A.T@lam-MU/sl+MU/su).max()),float(min(sl.min(),su.min()))]

def main():
 records=[]
 for kind in ['capacity','prescribed']:
  directpath=ROOT/f'research_completion/runs/python_migration/edwards_matched_comparison/{kind}_trajectory.npz';direct=np.load(directpath);tt=direct['time'];runs=[]
  for label,rtol,atol in [('tight',1e-11,1e-13),('refined',1e-12,1e-14)]:
   m=Differential(kind);start=time.perf_counter();s=solve_ivp(m.rhs,(0,7),m.initial,method='Radau',rtol=rtol,atol=atol,max_step=.05,dense_output=True);assert s.success,s.message
   y=s.sol(tt);res=np.array([m.residuals(z) for z in y.T]);flux=np.array([m.unpack(z)[0] for z in y.T]);gap=[]
   for ycol,v in zip(y.T,flux):r=m.model.lp(ycol[1]);gap.append(float(-r.fun-v[8]))
   gap=np.array(gap);np.savez_compressed(OUT/f'differentiated_{kind}_{label}.npz',time=tt,state=y[:4],flux=flux,equality_multiplier=y[4+m.n:],residuals=res,LP_gap=gap,glucose_capacity=12*y[1]/(2+y[1]))
   rec={'kind':kind,'run':label,'rtol':rtol,'atol':atol,'seconds':time.perf_counter()-start,'nfev':s.nfev,'calls':m.calls,'success':s.success,'initial_residuals':m.residuals(m.initial),'maximum_equality':float(res[:,0].max()),'maximum_stationarity':float(res[:,1].max()),'minimum_slack':float(res[:,2].min()),'minimum_trial_slack':m.min_trial_slack,'trial_state_note':'Implicit Newton trial states may leave the feasible interior and are rejected by adaptive integration; all sampled accepted output states have positive slack. No clipping is applied.','terminal_X_G_D_E':y[:4,-1].tolist(),'comparison_times':len(tt),'max_abs_concentration_difference_to_direct_X_G_D_E':np.max(abs(y[:4]-direct['state']),axis=1).tolist(),'max_abs_flux_difference_to_direct':float(abs(flux-direct['flux']).max()),'maximum_LP_biomass_gap':float(gap.max()),'minimum_LP_biomass_gap':float(gap.min()),'theoretical_barrier_gap_bound':2*m.n*MU,'source_direct_npz_sha256':hashlib.sha256(directpath.read_bytes()).hexdigest()};records.append(rec);runs.append(y[:4]);print(json.dumps(rec),flush=True)
   assert res[:,2].min()>0 and res[:,0].max()<1e-10
   assert np.max(abs(y[:4]-direct['state']))<2e-6
   assert gap.max()<2*m.n*MU+1e-7 and gap.min()>-1e-7
  records[-1]['refinement_concentration_max_abs_X_G_D_E']=np.max(abs(runs[0]-runs[1]),axis=1).tolist()
 (OUT/'differentiated_consistent_audit.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
