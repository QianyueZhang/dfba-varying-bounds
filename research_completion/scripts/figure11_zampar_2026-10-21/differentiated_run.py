"""Differentiate finite-barrier KKT equations over the complete batch trajectory.
Reinitialize only intracellular primal/dual variables at stated checkpoints;
extracellular balances remain continuous. No flux clipping or state reset.
"""
from verify_barrier import *
from scipy.sparse.linalg import splu

def run():
 start=time.perf_counter();p=np.array(list(json.loads((OUT/'fit_summary.json').read_text())['parameters'].values()));m=BarrierBatch(p,1e-5,1e-7);n=len(m.c);env=Y0.copy();segments=[];checks=[];all_t=[];all_y=[];bounds=[T0,-2.,-.25,.15,1.,4.,TF]
 for ta,tb in zip(bounds[:-1],bounds[1:]):
  z,info=m.initial(env);print('init',ta,info,'elapsed',time.perf_counter()-start,flush=True)
  sol=solve_ivp(m.rhs,(ta,tb),z,method='DOP853',rtol=2e-8,atol=2e-10,max_step=.12,dense_output=True)
  assert sol.success,sol.message
  for t,zz in zip(sol.t,sol.y.T):
   y=zz[:3];v=zz[3:3+n];la=zz[3+n:];l,u=m.bounds(y);sl=v-l;su=u-v
   checks.append([float(t),float(min(sl.min(),su.min())),float(abs(m.A@v-m.b).max()),float(abs(-m.c+m.A.T@la-m.mu/sl+m.mu/su).max())])
  env=sol.y[:3,-1];segments.append(sol);print('segment',ta,tb,'steps',len(sol.t),'state',env,'max residual',np.max(np.array(checks)[:,2:],axis=0),'minslack',np.min(np.array(checks)[:,1]),'calls',m.rhs_calls,'elapsed',time.perf_counter()-start,flush=True)
  np.savez_compressed(OUT/'differentiated_progress.npz',accepted_checks=np.array(checks),time=sol.t,state=sol.y[:3])
 tt=np.linspace(T0,TF,501);yy=[];ff=[];rr=[];direct=[]
 for t in tt:
  s=next((s for s in segments if s.t[0]<=t<s.t[-1]),segments[-1]);z=s.sol(t);y=z[:3];v=z[3:3+n];la=z[3+n:];l,u=m.bounds(y);sl=v-l;su=u-v
  yy.append(y);ff.append(v[[m.bio,m.g,m.e,m.o,203]]);rr.append([float(min(sl.min(),su.min())),float(abs(m.A@v-m.b).max()),float(abs(-m.c+m.A.T@la-m.mu/sl+m.mu/su).max())])
 # Pointwise algebraic comparison at every segment end; checked on identical states.
 for s in segments:
  z=s.y[:,-1];y=z[:3];v=z[3:3+n];va,la,inf=m.solve_barrier(y);direct.append({'t_h':float(s.t[-1]),'max_abs_flux_difference':float(abs(v-va).max()),'exchange_growth_differences':(v-va)[[m.bio,m.g,m.e,m.o,203]].tolist(),'direct':inf})
 yy=np.array(yy);ff=np.array(ff);rr=np.array(rr);lp=np.genfromtxt(OUT/'trajectory_lp.csv',delimiter=',',names=True);lpy=np.array([np.interp(tt,lp['t_h'],lp[k]) for k in ['X_gDW_L','G_g_L','E_g_L']]).T;difference=np.max(abs(yy-lpy),axis=0)
 np.savetxt(OUT/'trajectory_differentiated.csv',np.c_[tt,yy,ff,rr],delimiter=',',header='t_h,X_gDW_L,G_g_L,E_g_L,growth_h-1,v_glucose,v_ethanol,v_oxygen,v_ATPase,min_slack,equality_residual,stationarity_residual',comments='')
 report={'parameters':p.tolist(),'epsilon':m.epsilon,'mu':m.mu,'integrator':'DOP853','rtol':2e-8,'atol':2e-10,'max_step_h':.12,'intracellular_reinitialization_times_h':bounds[:-1],'no_flux_clipping':True,'continuous_extracellular_states':True,'accepted_step_count':len(checks),'max_accepted_equality_residual':float(np.max(np.array(checks)[:,2])),'max_accepted_stationarity_residual':float(np.max(np.array(checks)[:,3])),'minimum_accepted_slack':float(np.min(np.array(checks)[:,1])),'dense_sample_min_slack':float(rr[:,0].min()),'dense_sample_max_equality_residual':float(rr[:,1].max()),'dense_sample_max_stationarity_residual':float(rr[:,2].max()),'max_absolute_state_difference_from_unrelaxed_LP':dict(zip(['X_gDW_L','G_g_L','E_g_L'],difference.tolist())),'pointwise_direct_barrier_comparison':direct,'elapsed_seconds':time.perf_counter()-start,'rhs_calls':m.rhs_calls}
 (OUT/'differentiated_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
 assert report['minimum_accepted_slack']>0
if __name__=='__main__':run()
