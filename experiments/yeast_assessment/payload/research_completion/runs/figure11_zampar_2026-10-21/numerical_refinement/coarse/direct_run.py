"""Integrate external balances with a direct finite-barrier KKT solve at every RHS.
No fitted LP response or interpolated flux is used by the ODE solver.
"""
from verify_barrier import *
class Direct(BarrierBatch):
 def __init__(self,p,eps,mu):super().__init__(p,eps,mu);self.warm=None;self.last_lp=None;self.records=[];self.start=time.perf_counter();self.steps=0
 def flux_direct(self,y):
  lp=self.lp(y);lp_lam=-lp.eqlin.marginals
  if self.warm is None:v,la,info=self.solve_barrier(y)
  else:
   guess=(lp.x+self.warm[0]-self.last_lp[0],lp_lam+self.warm[1]-self.last_lp[1]) if self.last_lp is not None else self.warm
   try:v,la,info=self.barrier(y,mu=self.mu,warm=guess,tol=2e-7,maxiter=100)
   except RuntimeError:v,la,info=self.solve_barrier(y)
  self.warm=(v,la);self.last_lp=(lp.x,lp_lam);self.steps+=info['newton_steps'];self.records.append(info);return v,info
 def rhs_direct(self,t,y):
  self.rhs_calls+=1;v,info=self.flux_direct(y)
  if self.rhs_calls%25==0:print('rhs',self.rhs_calls,'t',t,'state',y,'newton',info['newton_steps'],'elapsed',time.perf_counter()-self.start,flush=True)
  return self.state_rhs(y,v)

def run():
 p=np.array(list(json.loads((OUT/'fit_summary.json').read_text())['parameters'].values()));m=Direct(p,1e-5,1e-7);start=time.perf_counter();sol=solve_ivp(m.rhs_direct,(T0,TF),Y0,method='LSODA',rtol=2e-7,atol=2e-9,max_step=.35,dense_output=True);assert sol.success,sol.message
 np.savez_compressed(OUT/'direct_accepted_external_states.npz',time=sol.t,states=sol.y)
 print('ODE complete',sol.y[:,-1],'rhs',m.rhs_calls,'elapsed',time.perf_counter()-start,flush=True)
 tt=np.unique(np.r_[np.linspace(T0,TF,401),[float(r['time_relative_to_glucose_exhaustion_h'])for r in __import__('fit_batch').ROWS]]);yy=sol.sol(tt).T;ff=[];rr=[];m.warm=None
 for i,y in enumerate(yy):
  v,info=m.flux_direct(y);ff.append(v[[m.bio,m.g,m.e,m.o,203]]);rr.append([info['minimum_slack'],info['equality_residual'],info['stationarity_residual']])
  if i%50==0:print('output',i,'of',len(tt),'elapsed',time.perf_counter()-start,flush=True)
 ff=np.array(ff);rr=np.array(rr);lp=BatchModel().simulate(p,rtol=2e-9,atol=2e-11);lpy=lp.sol(tt).T;difference=np.max(abs(yy-lpy),axis=0)
 np.savetxt(OUT/'trajectory_direct_barrier.csv',np.c_[tt,yy,ff,rr],delimiter=',',header='t_h,X_gDW_L,G_g_L,E_g_L,growth_h-1,v_glucose,v_ethanol,v_oxygen,v_ATPase,min_slack,equality_residual,stationarity_residual',comments='')
 np.savez_compressed(OUT/'direct_accepted_external_states.npz',time=sol.t,states=sol.y)
 report={'physical_dynamic_lower_bounds':'Glucose/ethanol uptake lower bounds are exact minus-capacity, without epsilon subtraction; remaining/fixed bounds expanded','parameters':p.tolist(),'epsilon':m.epsilon,'mu':m.mu,'method':'External-state LSODA integration with a converged direct finite-barrier KKT solve at every RHS; LP solutions provide Newton predictors only, not the adopted flux; no flux interpolation','integrator':'LSODA','rtol':2e-7,'atol':2e-9,'max_step_h':.35,'no_flux_clipping':True,'continuous_extracellular_states':True,'accepted_step_count':len(sol.t),'max_equality_residual_all_solves':max(r['equality_residual']for r in m.records),'max_stationarity_residual_all_solves':max(r['stationarity_residual']for r in m.records),'minimum_slack_all_solves':min(r['minimum_slack']for r in m.records),'output_min_slack':float(rr[:,0].min()),'output_max_equality_residual':float(rr[:,1].max()),'output_max_stationarity_residual':float(rr[:,2].max()),'max_absolute_state_difference_from_unrelaxed_LP':dict(zip(['X_gDW_L','G_g_L','E_g_L'],difference.tolist())),'minimum_external_concentrations':dict(zip(['X_gDW_L','G_g_L','E_g_L'],np.min(yy,axis=0).tolist())),'elapsed_seconds':time.perf_counter()-start,'rhs_calls':m.rhs_calls,'total_KKT_solves':len(m.records),'total_Newton_steps':m.steps}
 (OUT/'direct_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
 assert report['minimum_slack_all_solves']>0 and report['max_stationarity_residual_all_solves']<2.01e-7
if __name__=='__main__':run()
