"""Declared followup preserving a failed temporal forecast, without refitting."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,sys,time
import numpy as np
from scipy.integrate import RK45,solve_ivp
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'temporal'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def freeze():
 p=OUT/'failure_diagnostic_protocol.json'
 if p.exists():raise FileExistsError(p)
 dump(p,dict(declared_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),script_sha256=sha(__file__),primary_results_sha256=sha(OUT/'results.json'),primary_protocol_sha256=sha(OUT/'protocol.json'),
  status='Post-outcome diagnostic; does not replace failed primary forecast or refit parameters.',
  prefix='Repeat identical RK45 equations/tolerances with step API, retaining every accepted endpoint and dense interpolant. Record accepted time and failed trial time/state separately. No extrapolation beyond last accepted time.',
  feasibility='At failed trial state, remove fitted ATP lower demand and maximize ATP flux with every other bound unchanged. Compare available ATP to fitted demand; this concerns the queried state, not an exact event-time certificate.',
  matched='Finish originally planned no-refit capacity-versus-prescribed LP comparison using original all-phase parameters; 301 matched states and1001trajectory times.',
  sensitivity='Fine-tolerance pre-origin residual central differences in three log parameters with step.001; report SVD only, noCI. No full forecast rescue or clipping.'))
 print('Followup frozen')
def run():
 if (OUT/'failure_diagnostic.json').exists():raise FileExistsError('Retain diagnostic result')
 protocol=json.loads((OUT/'failure_diagnostic_protocol.json').read_text())
 for f,key in [('results.json','primary_results_sha256'),('protocol.json','primary_protocol_sha256')]:
  if sha(OUT/f)!=protocol[key]:raise RuntimeError('Primary evidence changed')
 if sha(__file__)!=protocol['script_sha256']:raise RuntimeError('Diagnostic source changed')
 spec=importlib.util.spec_from_file_location('temporal_driver',ROOT/'scripts/temporal_holdout.py');driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver);primary,m=driver.load()
 result=json.loads((OUT/'results.json').read_text());par=np.array(result['parameters']);model=m.BatchModel();last={};points=[np.r_[m.T0,m.Y0]];pieces=[];start=time.perf_counter()
 def rhs(t,y):
  last.update(time=float(t),state=y.tolist());return model.rhs(t,y,par)
 solver=RK45(rhs,m.T0,m.Y0,m.TF,rtol=2e-8,atol=2e-10,max_step=.25)
 failure=None
 try:
  while solver.status=='running':
   solver.step()
   if solver.status=='failed':raise RuntimeError('RK45 step failed')
   points.append(np.r_[solver.t,solver.y]);pieces.append((solver.t_old,solver.t,solver.dense_output()))
 except Exception as exc:failure=repr(exc)
 out=dict(primary_forecast_remains_failed=True,prefix_complete=solver.status=='finished',accepted_terminal_time=float(solver.t),accepted_terminal_state=solver.y.tolist(),failed_trial=last if failure else None,failure=failure,accepted_steps=len(points)-1)
 np.savetxt(OUT/'trajectory_temporal_prefix.csv',np.array(points),delimiter=',',header='time_h,X_gDW_L,G_g_L,E_g_L',comments='')
 obsrows=[]
 for row in m.ROWS:
  t=float(row['time_relative_to_glucose_exhaustion_h']);j=m.VARIABLES.index(row['variable'])
  if t>solver.t:continue
  yy=m.Y0 if t==m.T0 else next(fn(t) for a,b,fn in pieces if a-1e-12<=t<=b+1e-12)
  pred=float(yy[j]/m.OD_FACTOR if j==0 else yy[j]);obsrows.append(row|dict(prefix_prediction=pred,prefix_residual=pred-float(row['observed_value_as_published'])))
 with (OUT/'observations_temporal_prefix.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,list(obsrows[0]));w.writeheader();w.writerows(obsrows)
 out['prefix_observation_count']=len(obsrows);out['missing_full_window_observations']=len(m.ROWS)-len(obsrows)
 if failure:
  mm=m.BatchModel();yy=np.array(last['state']);l,u=mm.bounds_state(yy,par);l[203]=0.;indices=np.arange(len(l),dtype=np.int32);mm.h.changeColsBounds(len(l),indices,l,u);cost=np.zeros(len(l));cost[203]=-1.;mm.h.changeColsCost(len(l),indices,cost);mm.h.run();status=mm.h.getModelStatus()
  out['maintenance_feasibility']=dict(status=str(status),fitted_demand=float(par[3]))
  if status==m.highspy.HighsModelStatus.kOptimal:
   x=np.array(mm.h.getSolution().col_value);out['maintenance_feasibility'].update(maximum_supported_ATP=float(x[203]),shortfall=float(par[3]-x[203]),glucose_bound=float(l[mm.g]),ethanol_bound=float(l[mm.e]),equality_residual=float(np.max(abs(mm.A@x-mm.b))))
 # Local, pre-origin diagnostics do not require a successful future forecast.
 train=[r for r in m.ROWS if r['split']=='calibration' and float(r['time_relative_to_glucose_exhaustion_h'])<0];tt=np.array([float(r['time_relative_to_glucose_exhaustion_h']) for r in train]);ii=np.array([m.VARIABLES.index(r['variable']) for r in train]);yyobs=np.array([float(r['observed_value_as_published']) for r in train]);weights=1/np.array([2.,5.,2.])[ii]/np.sqrt([sum(ii==j) for j in ii])
 def sim(mod,p,end=m.TF):
  s=solve_ivp(lambda t,y:mod.rhs(t,y,p),(m.T0,end),m.Y0,rtol=2e-8,atol=2e-10,max_step=.25,dense_output=True)
  if not s.success:raise RuntimeError(s.message)
  return s
 def residual(lp):
  pp=np.array([np.exp(lp[0]),np.exp(lp[1]),10.,np.exp(lp[2])]);ss=sim(m.BatchModel(),pp,0.);values=ss.sol(tt);values[0]/=m.OD_FACTOR;return (values[ii,np.arange(len(ii))]-yyobs)*weights
 lp=np.log(par[[0,1,3]]);h=.001;jac=np.array([(residual(lp+np.eye(3)[j]*h)-residual(lp-np.eye(3)[j]*h))/(2*h) for j in range(3)]).T;u,s,vh=np.linalg.svd(jac,full_matrices=False)
 out['training_log_parameter_jacobian']=dict(singular_values=s.tolist(),condition_number=float(s[0]/s[-1]),column_norms=np.linalg.norm(jac,axis=0).tolist(),right_singular_vectors=vh.tolist())
 original=json.loads((OUT/'source_tree'/driver.SOURCE_RELS[5]).read_text())['parameters'];bp=np.array([original['qGmax_mmol_gDW_h'],original['qOmax_mmol_gDW_h'],10.,original['ATP_maintenance_mmol_gDW_h']])
 class Prescribed(m.BatchModel):
  def bounds_state(self,y,p):
   l,u=super().bounds_state(y,p);u[self.g]=l[self.g];return l,u
 cap=m.BatchModel();pre=Prescribed();cs=sim(cap,bp);ps=sim(pre,bp);times=np.linspace(m.T0,m.TF,1001);cy=cs.sol(times);py=ps.sol(times)
 np.savetxt(OUT/'trajectory_capacity_prescribed.csv',np.c_[times,cy.T,py.T],delimiter=',',header='time_h,capacity_X_gDW_L,capacity_G_g_L,capacity_E_g_L,prescribed_X_gDW_L,prescribed_G_g_L,prescribed_E_g_L',comments='')
 gaps=[];dflux=[]
 for y in cs.sol(np.linspace(m.T0,m.TF,301)).T:
  xc=cap.flux(y,bp);xp=pre.flux(y,bp);l,_=cap.bounds_state(y,bp);gaps.append(float(xc[cap.g]-l[cap.g]));dflux.append((xc-xp)[[cap.bio,cap.g,cap.e,cap.o]])
 out['matched_capacity_prescribed']=dict(complete=True,parameters=bp.tolist(),max_absolute_state_difference=np.max(abs(cy-py),axis=1).tolist(),matched_states=301,capacity_active_within_1e7=sum(abs(g)<=1e-7 for g in gaps),capacity_gap_range=[min(gaps),max(gaps)],maximum_selected_flux_difference=np.max(np.abs(dflux),axis=0).tolist(),flux_order=['growth','glucose','ethanol','oxygen'])
 allrows=[]
 for row in m.ROWS:
  t=float(row['time_relative_to_glucose_exhaustion_h']);j=m.VARIABLES.index(row['variable']);cp=float(cs.sol(t)[j]);pp=float(ps.sol(t)[j]);div=m.OD_FACTOR if j==0 else 1.;allrows.append(row|dict(capacity_prediction=cp/div,prescribed_prediction=pp/div))
 with (OUT/'observations_capacity_prescribed.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,list(allrows[0]));w.writeheader();w.writerows(allrows)
 out['protocol_sha256']=sha(OUT/'failure_diagnostic_protocol.json');out['elapsed_seconds']=time.perf_counter()-start;out['complete']=True;dump(OUT/'failure_diagnostic.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--run',action='store_true');a=p.parse_args();freeze() if a.freeze else run()
