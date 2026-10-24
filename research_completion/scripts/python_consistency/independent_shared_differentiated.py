"""Differentiate the finite-barrier KKT system and verify against direct solves.

The fixed full-row-rank equality constraints are eliminated with an orthonormal
null-space basis. Weighted QR solves the differentiated stationarity equations
without forming an ill-conditioned normal matrix. No flux clipping, residual
feedback, resource arrest or biological extension is used.
"""
from pathlib import Path
import sys,argparse,json,time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import null_space,qr,solve_triangular,lstsq
sys.path.insert(0,str(Path(__file__).resolve().parent))
import independent_shared_audit as a
a.F=1.;a.MU=1e-6
OUT=a.OUT/'supported'

class Differential:
 def __init__(self,case):
  self.case=case;self.mods=a.make_models(case);self.n=len(self.mods);self.ng=len(case['initial_substrate']);self.np=1+self.n+self.ng
  self.Z=[null_space(m.A.toarray(),rcond=1e-12)for m in self.mods]
  self.v0=[m.flux(case['initial_substrate'][0 if case['shared']else i])for i,m in enumerate(self.mods)]
  self.widths=[z.shape[1]for z in self.Z];self.offsets=np.cumsum([self.np]+self.widths)
  self.q0=np.r_[a.V0,np.ones(self.n),case['initial_substrate'],np.zeros(sum(self.widths)),np.zeros(2*self.n)]
  self.calls=0;self.min_trial_slack=np.inf
 def fluxes(self,y):return [v+z@y[self.offsets[i]:self.offsets[i+1]]for i,(v,z)in enumerate(zip(self.v0,self.Z))]
 def rhs(self,t,y):
  self.calls+=1;V=y[0];X=y[1:1+self.n];G=y[1+self.n:self.np];vs=self.fluxes(y)
  growth=np.array([v[12]for v in vs])*X;exchange=a.MW*np.array([v[27]for v in vs])*X
  met=np.array([exchange.sum()])if self.case['shared']else exchange
  dG=a.F/V*(np.array(self.case['feed_substrate'])-G)+met
  deriv=[np.r_[a.F,growth-a.F/V*X,dG]]
  for i,(m,v,Z)in enumerate(zip(self.mods,vs,self.Z)):
   j=0 if self.case['shared']else i;l,u=m.bounds(G[j]);sl=v-l;su=u-v
   self.min_trial_slack=min(self.min_trial_slack,float(sl.min()),float(su.min()))
   wl=a.MU/sl**2;wu=a.MU/su**2;sqrtD=np.sqrt(wl+wu)
   Q,R=qr(sqrtD[:,None]*Z,mode='economic',check_finite=False)
   lower_derivative=-m.vmax/(1+G[j])**2*dG[j]
   forcing=Q[m.g,:]*(wl[m.g]/sqrtD[m.g]*lower_derivative)
   deriv.append(solve_triangular(R,forcing,check_finite=False))
  deriv.extend([growth*V,exchange*V])
  return np.concatenate(deriv)
 def diagnostics(self,y):
  V=y[0];G=y[1+self.n:self.np];values=[]
  for i,(m,v,Z)in enumerate(zip(self.mods,self.fluxes(y),self.Z)):
   l,u=m.bounds(G[0 if self.case['shared']else i]);sl=v-l;su=u-v
   target=m.c+a.MU/sl-a.MU/su
   lam=lstsq(m.A.T.toarray(),target,lapack_driver='gelsy')[0]
   eq=float(np.max(abs(m.raw_A@v)));stat=float(np.max(abs(m.A.T@lam-target)));projected=float(np.max(abs(Z.T@(-target))))
   values.append([eq,stat,projected,float(min(sl.min(),su.min()))])
  return np.asarray(values)

def run(key):
 case=a.CASES[key];dest=OUT/key;dest.mkdir(parents=True,exist_ok=True)
 direct=np.load(dest/'trajectory.npz');t=direct['time_h'];reference=direct['concentrations'];reference_flux=direct['flux'];records=[];all_runs=[]
 for label,rtol,atol,step in [('primary',1e-10,1e-12,.02),('refined',1e-11,1e-13,.01)]:
  model=Differential(case);start=time.perf_counter()
  sol=solve_ivp(model.rhs,(0,case['duration_h']),model.q0,method='DOP853',rtol=rtol,atol=atol,max_step=step,dense_output=True)
  if not sol.success:raise RuntimeError(sol.message)
  y=sol.sol(t).T;flux=np.asarray([model.fluxes(q)for q in y]);diagnostics=np.asarray([model.diagnostics(q)for q in y]);physical=y[:,:model.np]
  n=model.n;ng=model.ng;offset=model.offsets[-1]
  Xexpected=a.V0+y[:,offset:offset+n]
  Gexpected=a.V0*np.array(case['initial_substrate'])+t[:,None]*a.F*np.array(case['feed_substrate'])
  accum=y[:,offset+n:offset+2*n]
  Gexpected+=accum.sum(axis=1,keepdims=True)if case['shared']else accum
  Xbalance=np.max(abs(physical[:,1:1+n]*physical[:,0:1]-Xexpected),axis=0)
  Gbalance=np.max(abs(physical[:,1+n:]*physical[:,0:1]-Gexpected),axis=0)
  gaps=[]
  for j in range(0,len(t),50):
   for i,m in enumerate(model.mods):
    G=physical[j,1+n+(0 if case['shared']else i)];lp=m.lp(G)
    gaps.append({'time_h':float(t[j]),'species':i+1,'objective_gap':float(-lp.fun-flux[j,i,12]),'barrier_gap_bound':190*a.MU})
  record={'case':key,'run':label,'method':'DOP853','rtol':rtol,'atol':atol,'max_step_h':step,'elapsed_s':time.perf_counter()-start,'RHS_evaluations':sol.nfev,'accepted_nodes':len(sol.t),'full_horizon_completed':bool(sol.t[-1]==case['duration_h']),'minimum_trial_slack':model.min_trial_slack,'maximum_full_equality_residual':float(diagnostics[:,:,0].max()),'maximum_explicit_stationarity_residual':float(diagnostics[:,:,1].max()),'maximum_projected_stationarity_residual':float(diagnostics[:,:,2].max()),'minimum_sampled_slack':float(diagnostics[:,:,3].min()),'max_state_difference_to_direct_by_V_X_G':np.max(abs(physical-reference),axis=0).tolist(),'max_flux_difference_to_direct':float(np.max(abs(flux-reference_flux))),'terminal_state_V_X_G':physical[-1].tolist(),'biomass_amount_balance_max_error_gDW':Xbalance.tolist(),'glucose_amount_balance_max_error_g':Gbalance.tolist(),'LP_gap_checks':gaps,'initial_diagnostics':model.diagnostics(model.q0).tolist(),'source_direct_npz_sha256':a.sha(dest/'trajectory.npz'),'script_sha256':a.sha(Path(__file__)),'null_space_dimensions':[list(z.shape)for z in model.Z],'formulation':'v=v0+Zz, A Z=0. D=diag(mu/(v-l)^2+mu/(u-v)^2). Differentiate projected stationarity: (Z^T D Z) zdot=Z^T diag(mu/(v-l)^2) ldot; only glucose ldot is nonzero. Solve by weighted QR without normal-matrix formation. Initial v0 solves the complete finite-barrier KKT system. Fluxes and states are never clipped.'}
  record['validated']=bool(record['minimum_sampled_slack']>0 and record['maximum_full_equality_residual']<1e-9 and max(record['max_state_difference_to_direct_by_V_X_G'])<1e-5 and max(Gbalance)<1e-7 and max(Xbalance)<1e-7 and all(-1e-8<q['objective_gap']<q['barrier_gap_bound']+1e-8 for q in gaps))
  np.savez_compressed(dest/f'differentiated_{label}.npz',time_h=t,concentrations=physical,flux=flux,state=y,residuals=diagnostics)
  (dest/f'differentiated_{label}_audit.json').write_text(json.dumps(record,indent=2)+'\n')
  records.append(record);all_runs.append(physical)
  print(json.dumps({k:record[k]for k in ['case','run','elapsed_s','maximum_explicit_stationarity_residual','minimum_sampled_slack','max_state_difference_to_direct_by_V_X_G','max_flux_difference_to_direct','validated']}),flush=True)
  if not record['validated']:raise RuntimeError('Differentiated validation failed '+key+' '+label)
 validation={'case':key,'records':records,'primary_refined_state_max_difference_V_X_G':np.max(abs(all_runs[1]-all_runs[0]),axis=0).tolist(),'adoption_ready':all(q['validated']for q in records)}
 (dest/'differentiated_verification.json').write_text(json.dumps(validation,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',choices=list(a.CASES),required=True);args=p.parse_args();run(args.case)
