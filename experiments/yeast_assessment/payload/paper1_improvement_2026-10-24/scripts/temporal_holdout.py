"""Retrospective temporal stress test of the unchanged Zampar batch model.

The protocol and exact source copies must be frozen before execution. No
observation, model law, parameter bound or original campaign file is edited.
"""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, os, platform, shutil, sys, time

STUDY=Path(__file__).resolve().parents[1]
OUT=STUDY/'temporal'
PROJECT=STUDY.parent
SOURCE_RELS=[
 'research_completion/scripts/figure11_zampar_2026-10-21/fit_batch.py',
 'research_completion/scripts/flux_model.py',
 '14_control_pso/09_Sc_Kp/SC_Model.mat',
 '14_control_pso/09_Sc_Kp/SCerevisiae_Data.mat',
 'research_completion/runs/figure11_source_review_2026-10-21/candidates/zampar2013/zampar2013_observations.csv',
 'research_completion/runs/figure11_zampar_2026-10-21/fit_summary.json']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def freeze():
 protocol=OUT/'protocol.json'
 if protocol.exists():raise FileExistsError('Retain existing temporal protocol')
 inputs={}
 for rel in SOURCE_RELS:
  src=PROJECT/rel;dst=OUT/'source_tree'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.exists():raise FileExistsError(dst)
  shutil.copyfile(src,dst);inputs[rel]=sha(src)
 rows=list(csv.DictReader((OUT/'source_tree'/SOURCE_RELS[4]).open()))
 assigned=[]
 for i,r in enumerate(rows):
  t=float(r['time_relative_to_glucose_exhaustion_h'])
  if t< -6.8-1e-8:continue
  first=r['replicate_label_as_published'].replace(' ','')=='Exp1'
  cohort=('exp1' if first else 'other_series')+('_pre' if t<0 else '_post')
  assigned.append(dict(source_row_index=i,source_file=r['source_file'],source_row=r['source_row'],variable=r['variable'],time=t,cohort=cohort))
 p=dict(declared_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),script_sha256=sha(__file__),sources_sha256=inputs,
  design='Retrospective temporal stress test; structural model and clock previously exposed to full data, not blind or independent-condition validation.',
  fit='Only Exp1-labelled observations with -6.8<=published glucose-exhaustion-relative time<0. The objective is integrated only through0h. No state reset at0h. All t>=0 are forecast without refit.',
  initial='Unchanged source initial conditions, including OD interpolation from-7.5/-6.5h. Both OD inputs precede the0h forecast origin, but initialization is retrospective at-6.8h.',
  parameter_order=['qGmax','qOmax','mATP'],parameter_start=[11.,3.8,5.],parameter_lower=[3.,.2,.001],parameter_upper=[30.,15.,20.],fixed_qE=10.,
  residual_scales=[2.,5.,2.],weight='Per-variable scale times square-root of that variable training count; equal mean-squared contribution. No measured standard-deviation interpretation.',
  optimizer=dict(method='scipy.optimize.least_squares',coordinates='log',diff_step=.002,xtol=2e-6,ftol=2e-6,gtol=2e-6,max_nfev=80,starts=1),
  integration=dict(fit_rtol=2e-6,fit_atol=2e-8,final_rtol=2e-8,final_atol=2e-10,max_step_h=.25),
  failure_policy='Preserve every failed fit trial, incomplete integration and post-origin forecast failure. Never evaluate dense output outside successfully integrated time.',
  conditional_sensitivity='After fit, perturb each fitted parameter independently by factors0.9/1.1, other parameters fixed; clip only to declared parameter bounds and record exact factor. Not confidence or predictive intervals.',
  local_jacobian='Central finite difference of fine-tolerance training residual in log parameters with step.001; SVD and column norms are local numerical diagnostics, not an identifiability proof or uncertainty intervals.',
  matched_uptake='Use original all-phase fitted parameters without refitting. Capacity is unchanged glucose lower bound; prescribed LP sets upper=lower for glucose only. Both unexpanded LPs use identical source equations, observations and initial conditions. Compare full trajectories and selected external fluxes at matched capacity-trajectory states.',
  assignments=assigned)
 write(protocol,p);print(json.dumps({'frozen':True,'assigned':len(assigned),'training':sum(r['cohort']=='exp1_pre' for r in assigned)}))


def load():
 p=json.loads((OUT/'protocol.json').read_text())
 if sha(__file__)!=p['script_sha256']:raise RuntimeError('Temporal source changed after freeze')
 for rel,h in p['sources_sha256'].items():
  if sha(OUT/'source_tree'/rel)!=h:raise RuntimeError('Copied scientific input changed: '+rel)
 # Local dependency fallback is runtime only; the scientific source copy is exact.
 sys.path.insert(0,str(PROJECT/'research_completion/runs/figure11_zampar_2026-10-21/_deps'))
 spec=importlib.util.spec_from_file_location('temporal_exact_batch',OUT/'source_tree'/SOURCE_RELS[0]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return p,m


def run():
 if (OUT/'results.json').exists():raise FileExistsError('Retain existing temporal outcomes')
 p,m=load()
 import numpy as np
 import scipy
 from scipy.integrate import solve_ivp
 from scipy.optimize import least_squares
 started=time.perf_counter();rows=m.ROWS;variables=m.VARIABLES
 cohorts=[('exp1' if r['split']=='calibration' else 'other_series')+('_pre' if float(r['time_relative_to_glucose_exhaustion_h'])<0 else '_post') for r in rows]
 train=np.array([i for i,c in enumerate(cohorts) if c=='exp1_pre']);times=np.array([float(r['time_relative_to_glucose_exhaustion_h']) for r in rows]);inds=np.array([variables.index(r['variable']) for r in rows]);obs=np.array([float(r['observed_value_as_published']) for r in rows])
 weights=1/np.array([2.,5.,2.])[inds[train]]/np.sqrt(np.array([sum(inds[train]==j) for j in inds[train]]))
 model=m.BatchModel();ledger=[]
 def simulate(par,end=m.TF,fine=True,mod=model):
  sol=solve_ivp(lambda t,y:mod.rhs(t,y,par),(m.T0,end),m.Y0,rtol=2e-8 if fine else 2e-6,atol=2e-10 if fine else 2e-8,dense_output=True,max_step=.25)
  if not sol.success or sol.t[-1]<end-1e-10:raise RuntimeError('Incomplete integration: '+sol.message+' at '+str(sol.t[-1]))
  return sol
 def predict(sol,indices):
  vals=sol.sol(times[indices]);vals[0]/=m.OD_FACTOR
  return vals[inds[indices],np.arange(len(indices))]
 def unpack(logp):return np.array([np.exp(logp[0]),np.exp(logp[1]),10.,np.exp(logp[2])])
 def residual(logp,record=True,fine=False):
  par=unpack(logp);ts=time.perf_counter()
  try:
   sol=simulate(par,end=0.,fine=fine);r=(predict(sol,train)-obs[train])*weights;failure=None
  except Exception as exc:
   r=np.full(len(train),10.);failure=repr(exc)
  if record:
   row=dict(evaluation=len(ledger),parameters=par.tolist(),squared_objective=float(r@r),failure=failure,seconds=time.perf_counter()-ts);ledger.append(row)
   with (OUT/'fit_evaluations.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   if len(ledger)%10==0:print(json.dumps({'evaluations':len(ledger),'objective':float(r@r),'seconds':time.perf_counter()-started}),flush=True)
  if not record and failure:raise RuntimeError(failure)
  return r
 fit=least_squares(residual,np.log(p['parameter_start']),bounds=(np.log(p['parameter_lower']),np.log(p['parameter_upper'])),diff_step=.002,xtol=2e-6,ftol=2e-6,gtol=2e-6,max_nfev=80)
 par=unpack(fit.x)
 def metrics(pred):
  result={}
  for cohort in sorted(set(cohorts)):
   result[cohort]={}
   for j,v in enumerate(variables):
    ii=np.array([i for i,c in enumerate(cohorts) if c==cohort and inds[i]==j]);r=pred[ii]-obs[ii];span=float(np.ptp(obs[ii]))
    result[cohort][v]=dict(n=len(ii),rmse=float(np.sqrt(np.mean(r*r))),mae=float(np.mean(abs(r))),mean_bias=float(np.mean(r)),observed_range=[float(min(obs[ii])),float(max(obs[ii]))],range_normalised_rmse_percent=float(100*np.sqrt(np.mean(r*r))/span) if span>0 else None)
  return result
 result=dict(complete=False,protocol_sha256=sha(OUT/'protocol.json'),parameters=par.tolist(),fit=dict(success=bool(fit.success),message=str(fit.message),nfev=int(fit.nfev),evaluations=len(ledger),failed_trials=sum(r['failure'] is not None for r in ledger),objective=float(2*fit.cost),active_mask=fit.active_mask.tolist()),assigned_counts={c:cohorts.count(c) for c in sorted(set(cohorts))})
 try:
  sol=simulate(par);pred=predict(sol,np.arange(len(rows)));result['forecast']=dict(complete=True,metrics=metrics(pred),terminal=sol.y[:,-1].tolist(),state_at_origin=sol.sol(0.).tolist())
 except Exception as exc:
  result['forecast']=dict(complete=False,failure=repr(exc));write(OUT/'results.json',result);raise
 original=json.loads((OUT/'source_tree'/SOURCE_RELS[5]).read_text());bp=original['parameters'];baseline=np.array([bp['qGmax_mmol_gDW_h'],bp['qOmax_mmol_gDW_h'],bp['qEmax_mmol_gDW_h'],bp['ATP_maintenance_mmol_gDW_h']]);bsol=simulate(baseline);bpred=predict(bsol,np.arange(len(rows)))
 result['existing_all_phase_parameters_reference']=dict(parameters=baseline.tolist(),metrics=metrics(bpred),interpretation='Reference fitted using post-origin Exp1 outcomes; not a competing temporal forecast.')
 fields=list(rows[0])+['temporal_cohort','temporal_prediction','temporal_residual','all_phase_reference_prediction','all_phase_reference_residual']
 with (OUT/'observations_predictions.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fields);w.writeheader()
  for i,r in enumerate(rows):w.writerow(r|dict(temporal_cohort=cohorts[i],temporal_prediction=pred[i],temporal_residual=pred[i]-obs[i],all_phase_reference_prediction=bpred[i],all_phase_reference_residual=bpred[i]-obs[i]))
 tt=np.linspace(m.T0,m.TF,1001);yy=sol.sol(tt);by=bsol.sol(tt)
 np.savetxt(OUT/'trajectory_temporal.csv',np.c_[tt,yy.T,yy[0]/m.OD_FACTOR,by.T,by[0]/m.OD_FACTOR],delimiter=',',header='time_h,temporal_X_gDW_L,temporal_G_g_L,temporal_E_g_L,temporal_OD600,reference_X_gDW_L,reference_G_g_L,reference_E_g_L,reference_OD600',comments='')
 result['local_sensitivity']={};h=.001;cols=[]
 for j,name in enumerate(p['parameter_order']):
  unit=np.eye(3)[j]*h
  try:
   if fit.x[j]-h<np.log(p['parameter_lower'][j]) or fit.x[j]+h>np.log(p['parameter_upper'][j]):raise RuntimeError('Central diagnostic would cross declared parameter bounds')
   cols.append((residual(fit.x+unit,record=False,fine=True)-residual(fit.x-unit,record=False,fine=True))/(2*h))
  except Exception as exc:result['local_sensitivity']['jacobian_failure']=repr(exc);break
 if len(cols)==3:
  jac=np.array(cols).T;u,s,vh=np.linalg.svd(jac,full_matrices=False)
  result['local_sensitivity']['log_parameter_jacobian']=dict(singular_values=s.tolist(),condition_number=float(s[0]/s[-1]) if s[-1]>0 else None,column_norms=np.linalg.norm(jac,axis=0).tolist(),right_singular_vectors=vh.tolist(),note='Fine-tolerance local numerical diagnostic; no confidence interval or global identifiability conclusion.')
 profiles=[]
 for j,name in enumerate(p['parameter_order']):
  for factor in [.9,1.1]:
   lp=fit.x.copy();lp[j]=np.log(np.clip(np.exp(lp[j])*factor,p['parameter_lower'][j],p['parameter_upper'][j]));pp=unpack(lp)
   row=dict(parameter=name,factor_requested=factor,factor_actual=float(np.exp(lp[j]-fit.x[j])),parameters=pp.tolist())
   try:
    rs=residual(lp,record=False,fine=True);ss=simulate(pp);pr=predict(ss,np.arange(len(rows)));row.update(complete=True,training_objective=float(rs@rs),metrics=metrics(pr))
   except Exception as exc:row.update(complete=False,failure=repr(exc))
   profiles.append(row)
 result['local_sensitivity']['conditional_parameter_perturbations']=profiles
 class Prescribed(m.BatchModel):
  def bounds_state(self,y,params):
   l,u=super().bounds_state(y,params);u[self.g]=l[self.g];return l,u
 prescribed=Prescribed()
 matched=dict(parameters=baseline.tolist(),same_parameter_no_refit=True)
 try:
  psol=simulate(baseline,mod=prescribed);py=psol.sol(tt);ppred=predict(psol,np.arange(len(rows)));matched.update(complete=True,metrics=metrics(ppred),max_absolute_state_difference=np.max(abs(py-by),axis=1).tolist(),terminal=py[:,-1].tolist())
  np.savetxt(OUT/'trajectory_capacity_prescribed.csv',np.c_[tt,by.T,py.T],delimiter=',',header='time_h,capacity_X_gDW_L,capacity_G_g_L,capacity_E_g_L,prescribed_X_gDW_L,prescribed_G_g_L,prescribed_E_g_L',comments='')
  states=bsol.sol(np.linspace(m.T0,m.TF,301));gaps=[];external=[]
  for y in states.T:
   xc=model.flux(y,baseline);xp=prescribed.flux(y,baseline);l,_=model.bounds_state(y,baseline);gaps.append(float(xc[model.g]-l[model.g]));external.append((xc-xp)[[model.bio,model.g,model.e,model.o]])
  matched['matched_state_checks']=dict(n=301,glucose_capacity_gap_max=float(max(gaps)),glucose_capacity_gap_min=float(min(gaps)),capacity_active_within_1e7=sum(abs(g)<=1e-7 for g in gaps),maximum_selected_flux_difference=np.max(np.abs(np.asarray(external)),axis=0).tolist(),flux_order=['growth','glucose','ethanol','oxygen'])
 except Exception as exc:matched.update(complete=False,failure=repr(exc))
 result['matched_capacity_prescribed']=matched
 result['numerics']=dict(highs_version=model.h.version(),screening_model_max_equality_residual=model.max_residual,screening_model_max_bound_violation=model.max_bound_violation,screening_model_lp_calls=model.calls,prescribed_max_equality_residual=prescribed.max_residual,prescribed_max_bound_violation=prescribed.max_bound_violation,prescribed_lp_calls=prescribed.calls)
 result['source_hashes_unchanged']=all(sha(OUT/'source_tree'/rel)==h for rel,h in p['sources_sha256'].items())
 result['environment']=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform())
 result['complete']=True;result['elapsed_seconds']=time.perf_counter()-started;write(OUT/'results.json',result)
 print(json.dumps({'complete':True,'elapsed_seconds':result['elapsed_seconds'],'parameters':par.tolist(),'matched_complete':matched['complete']}),flush=True)


if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--run',action='store_true');a=ap.parse_args()
 if a.freeze:freeze()
 else:run()
