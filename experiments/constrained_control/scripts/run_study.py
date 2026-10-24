"""Equal-budget constrained control with training-only surrogate transforms."""
from pathlib import Path
import json,time,hashlib,warnings,sys
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from simulator import ROOT,THESIS,TABLE,Reactor,initial_from_condition,features,sample_feeds
P=json.loads((ROOT/'protocol.json').read_text());P.update(json.loads((ROOT/'protocol_amendment_02.json').read_text())['changes']);P['reference_search']=None
D=ROOT/'data';R=ROOT/'results';reactor=Reactor();clock0=time.perf_counter();costs={'data':{},'fit':{},'test_label_seconds':0.,'common_response_table_build':'Shared infrastructure excluded equally from measured campaign cost'}
def dump(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')
def make_data(split):
 rng=np.random.default_rng(P['data_feed_seeds'][split]);xx=[];yy=[];conditions=[];controls=[];started=time.perf_counter();failures=[]
 for ci,condition in enumerate(P['conditions'][split]):
  feeds=sample_feeds(rng,P['feeds_per_condition'][split]);initial=initial_from_condition(condition)
  for u in feeds:
   try:q=reactor.simulate(u,initial)
   except Exception as ex:failures.append({'condition':ci,'control':u.tolist(),'error':str(ex)});continue
   xx.append(features(u,condition));yy.append([q['z'],q['log_biomass_amount']]);conditions.append(ci);controls.append(u)
  print('LABEL',split,ci+1,len(xx),flush=True)
 a={'X':np.asarray(xx),'Y':np.asarray(yy),'condition_index':np.asarray(conditions),'controls':np.asarray(controls)};np.savez(D/(split+'.npz'),**a);meta={'seconds':time.perf_counter()-started,'successful_evaluations':len(xx),'failures':failures};costs['data'][split]=meta;dump(D/(split+'_metadata.json'),meta)
 if failures:raise RuntimeError('Label failures recorded; stopping before model selection to avoid silent sampling changes')
 return a
from surrogate_model import Surrogate, export_models
class BudgetEnd(Exception):pass
class SimulationFailure(RuntimeError):pass
class Evaluator:
 def __init__(self,initial,target,budget):self.initial=initial;self.target=target;self.budget=budget;self.cache={};self.trace=[];self.started=time.perf_counter();self.failed=[]
 def __call__(self,u):
  u=np.asarray(u,float);key=u.tobytes()
  if key not in self.cache:
   if len(self.cache)>=self.budget:raise BudgetEnd()
   t=time.perf_counter()
   try:q=reactor.simulate(u,self.initial)
   except Exception as ex:
    failure={'control':u.tolist(),'seconds':time.perf_counter()-t,'error':str(ex),'failed':True};self.failed.append(failure);self.cache[key]=failure;raise SimulationFailure(str(ex))
   err=abs(q['z']-np.log(self.target));volume=q['terminal'][0];violation=max(0.,err-P['operating_domain']['acceptance_log_ratio_tolerance'])+max(0.,volume-1.8)
   q.update(control=u.tolist(),evaluation=len(self.cache)+1,elapsed_seconds=time.perf_counter()-self.started,simulation_seconds=time.perf_counter()-t,log_ratio_error=err,violation=violation,feasible=bool(err<=P['operating_domain']['acceptance_log_ratio_tolerance'] and volume<=1.8+1e-8));self.cache[key]=q;self.trace.append(q)
  if self.cache[key].get('failed'):raise SimulationFailure(self.cache[key]['error'])
  return self.cache[key]
 def selected(self):
  margin=[q for q in self.trace if q['log_ratio_error']<=.009+1e-7 and q['terminal'][0]<=1.8+1e-8]
  feasible=[q for q in self.trace if q['feasible']]
  if margin:return max(margin,key=lambda q:q['biomass_amount_gDW']),'optimization_margin_feasible'
  if feasible:return max(feasible,key=lambda q:q['biomass_amount_gDW']),'acceptance_only_fallback'
  return min(self.trace,key=lambda q:(q['violation'],-q['biomass_amount_gDW'])),'least_violation_fallback'
def optimize(e,start,target,subbudget=None):
 begin=len(e.cache);zt=np.log(target)
 def lookup(u):
  if subbudget is not None and len(e.cache)-begin>=subbudget and np.asarray(u,float).tobytes() not in e.cache:raise BudgetEnd()
  return e(u)
 cons=[{'type':'ineq','fun':lambda u:1.8-(.5+2*np.sum(u))},{'type':'ineq','fun':lambda u:.009-(lookup(u)['z']-zt)},{'type':'ineq','fun':lambda u:.009+(lookup(u)['z']-zt)}]
 try:
  ans=minimize(lambda u:-lookup(u)['biomass_amount_gDW']/100,start,method='SLSQP',bounds=[(0,.2)]*7,constraints=cons,options=P['slsqp']);return {'success':bool(ans.success),'message':str(ans.message),'new_evaluations':len(e.cache)-begin}
 except BudgetEnd:return {'success':False,'message':'evaluation cap','new_evaluations':len(e.cache)-begin}
 except Exception as ex:return {'success':False,'message':'recorded simulator/optimizer failure: '+repr(ex),'new_evaluations':len(e.cache)-begin}
def propose(model,condition,target,starts):
 cache={};started=time.perf_counter();zt=np.log(target);records=[];statuses=[]
 def pred(u):
  key=np.asarray(u,float).tobytes()
  if key not in cache:
   z,logB=model.predict(features(u,condition))[0];B=float(np.exp(logB));violation=max(0.,abs(z-zt)-.009)+max(0.,.5+2*np.sum(u)-1.8);q={'u':np.asarray(u).tolist(),'z':float(z),'B':B,'violation':float(violation)};cache[key]=q;records.append(q)
  return cache[key]
 cons=[{'type':'ineq','fun':lambda u:1.8-(.5+2*np.sum(u))},{'type':'ineq','fun':lambda u:.009-(pred(u)['z']-zt)},{'type':'ineq','fun':lambda u:.009+(pred(u)['z']-zt)}]
 for start in starts:
  ans=minimize(lambda u:-pred(u)['B']/100,start,method='SLSQP',bounds=[(0,.2)]*7,constraints=cons,options={'maxiter':80,'ftol':1e-9,'eps':1e-6});statuses.append({'success':bool(ans.success),'message':str(ans.message)})
 feasible=[q for q in records if q['violation']<=1e-7];best=max(feasible,key=lambda q:q['B']) if feasible else min(records,key=lambda q:(q['violation'],-q['B']))
 return np.asarray(best['u']),{'seconds':time.perf_counter()-started,'unique_surrogate_evaluations':len(cache),'selected_prediction':best,'statuses':statuses}
def main():
 if (R/'selection_lock.json').exists():raise RuntimeError('Selection lock exists: inspect partial output before rerunning; no automatic overwrite')
 train=make_data('training');val=make_data('validation');xs=StandardScaler().fit(train['X']);ys=StandardScaler().fit(train['Y']);xt=xs.transform(train['X']);yt=ys.transform(train['Y']);xv=xs.transform(val['X']);yv=ys.transform(val['Y']);models={};selection={}
 for family in ['quadratic_ridge','mlp']:
  cand=[];t0=time.perf_counter()
  settings=P['surrogates'][family]
  for setting in settings['alphas'] if family=='quadratic_ridge' else settings['initialization_seeds']:
   model=make_pipeline(PolynomialFeatures(2,include_bias=False),Ridge(alpha=setting)) if family=='quadratic_ridge' else MLPRegressor(hidden_layer_sizes=(15,),activation='logistic',solver='lbfgs',alpha=1e-4,max_iter=1500,tol=1e-7,random_state=setting)
   t=time.perf_counter()
   with warnings.catch_warnings(record=True) as ww:
    warnings.simplefilter('always');model.fit(xt,yt)
   score=float(np.mean((model.predict(xv)-yv)**2));row={'setting':setting,'validation_joint_standardized_mse':score,'seconds':time.perf_counter()-t,'warnings':[str(w.message) for w in ww]};cand.append((score,model,row));print('FIT',family,setting,score,flush=True)
  winner=min(cand,key=lambda x:x[0]);models[family]=Surrogate(winner[1],xs,ys);selection[family]={'selected':winner[2],'candidates':[c[2] for c in cand]};costs['fit'][family]=time.perf_counter()-t0
 export_models(models,R/'locked_models.json');lock={'protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),'amendment_01_sha256':hashlib.sha256((ROOT/'protocol_amendment_01.json').read_bytes()).hexdigest(),'amendment_02_sha256':hashlib.sha256((ROOT/'protocol_amendment_02.json').read_bytes()).hexdigest(),'amendment_03_sha256':hashlib.sha256((ROOT/'protocol_amendment_03.json').read_bytes()).hexdigest(),'model_sha256':hashlib.sha256((R/'locked_models.json').read_bytes()).hexdigest(),'selection':selection,'costs_at_lock':costs,'source_hashes':{f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in ['run_study.py','simulator.py','surrogate_model.py']},'locked_before_test_labels':True,'locked_before_test_optimization':True};dump(R/'selection_lock.json',lock)
 test=make_data('test');metrics={}
 for name,model in models.items():
  pred=model.predict(test['X']);res=pred-test['Y'];metrics[name]={'log_ratio_rmse':float(np.sqrt(np.mean(res[:,0]**2))),'log_biomass_amount_rmse':float(np.sqrt(np.mean(res[:,1]**2))),'biomass_amount_rmse_gDW':float(np.sqrt(np.mean((np.exp(pred[:,1])-np.exp(test['Y'][:,1]))**2)))};np.savez(D/(name+'_test_predictions.npz'),truth=test['Y'],prediction=pred,condition_index=test['condition_index'])
 dump(R/'prediction_metrics.json',metrics)
 trainconditions=np.asarray(P['conditions']['training']);cx=np.c_[np.log(trainconditions[:,0]),trainconditions[:,1]];cscale=StandardScaler().fit(cx);cx=cscale.transform(cx)
 cases=[];allrows=[];campaign0=time.perf_counter()
 for ci,condition in enumerate(P['conditions']['test']):
  for ti,target in enumerate(P['operating_domain']['target_ratios']):
   caseid=f'condition_{ci}_target_{target:g}';initial=initial_from_condition(condition);near=int(np.argmin(np.sum((cx-cscale.transform([[np.log(condition[0]),condition[1]]]))**2,axis=1)));idx=np.flatnonzero(train['condition_index']==near);ranking=sorted(idx,key=lambda k:(max(0.,abs(train['Y'][k,0]-np.log(target))-.009),-np.exp(train['Y'][k,1])))
   for ri,seed in enumerate(P['repeat_seeds']):
    starts=sample_feeds(np.random.default_rng(seed+10000*ci+100*ti),3)
    for method in P['methods']:
     if time.perf_counter()-campaign0>P['execution_time_limit_seconds']:
      dump(R/'TIME_CAP.json',{'completed_runs':len(allrows),'elapsed_seconds':time.perf_counter()-campaign0,'note':'Not convergence; report incomplete coverage'});return
     run0=time.perf_counter();e=Evaluator(initial,target,P['mechanistic_evaluation_cap']);proposal=None;execution_error=None;statuses=[]
     try:
      if method=='direct_multistart':statuses=[optimize(e,start,target,100) for start in starts]
      else:
       if method=='direct_single_start':start=starts[0]
       elif method=='database_warmstart':start=train['controls'][ranking[ri]]
       else:start,proposal=propose(models[method],condition,target,starts)
       statuses=[optimize(e,start,target)]
     except Exception as ex:execution_error=repr(ex)
     selected=None;selection_mode='no_successful_evaluation';tight={'feasible':False,'status':'not evaluated'};tight_seconds=0.
     if e.trace:
      selected,selection_mode=e.selected();tight0=time.perf_counter()
      try:
       tight=reactor.simulate(selected['control'],initial,rtol=1e-9,atol=1e-11);tighterr=abs(tight['z']-np.log(target));tight.update(log_ratio_error=tighterr,ratio_relative_error_percent=100*abs(np.exp(tight['z'])/target-1),feasible=bool(tighterr<=P['operating_domain']['acceptance_log_ratio_tolerance'] and tight['terminal'][0]<=1.8+1e-8),ratio_constraint_margin=P['operating_domain']['acceptance_log_ratio_tolerance']-tighterr,volume_constraint_margin_L=1.8-tight['terminal'][0],status='completed')
      except Exception as ex:tight={'feasible':False,'status':'failed','error':repr(ex)}
      tight_seconds=time.perf_counter()-tight0
     feasibletrace=[q for q in e.trace if q['feasible']]
     row={'case_id':caseid,'condition_id':ci,'condition':condition,'initial':initial,'target_ratio':target,'seed':seed,'method':method,'controls':selected['control'] if selected else None,'actual_unique_mechanistic_evaluations':len(e.cache),'successful_mechanistic_evaluation_count':len(e.trace),'failed_mechanistic_evaluation_count':len(e.failed),'total_mechanistic_attempts':len(e.cache),'mechanistic_seconds_sum':sum(q['simulation_seconds'] for q in e.trace)+sum(q['seconds'] for q in e.failed),'proposal':proposal,'correction_or_direct_statuses':statuses,'execution_error':execution_error,'first_feasible_ordinary_evaluation':feasibletrace[0]['evaluation'] if feasibletrace else None,'selection_mode':selection_mode,'selected_ordinary':selected,'tight_verification':tight,'tight_verification_seconds':tight_seconds,'online_total_seconds':time.perf_counter()-run0,'failed_evaluations':e.failed,'database_source_condition':near if method=='database_warmstart' else None,'trace':e.trace}
     allrows.append(row);outfile=R/'policies'/f'{caseid}_seed_{seed}_{method}.json';outfile.parent.mkdir(exist_ok=True);dump(outfile,row);dump(R/'campaign_index.json',{'completed_runs':len(allrows),'expected_runs':4*3*3*len(P['methods']),'elapsed_seconds':time.perf_counter()-campaign0,'last_file':str(outfile)})
     if ci==0 and target==1 and seed==3101 and method in ['direct_multistart','mlp'] and row['controls'] is not None:
      request={'controls':row['controls'],'initial':initial,'target_ratio':target,'condition_id':ci,'method':method,'seed':seed,'volume_cap_L':1.8,'selected_by':'prospective protocol regardless of method ranking','source_policy_file':str(outfile)};dump(R/f'direct_input_{method}.json',request)
     print('RUN',caseid,seed,method,len(e.cache),tight['feasible'],round(tight.get('biomass_amount_gDW',float('nan')),4),round(row['online_total_seconds'],3),flush=True)
 dump(R/'campaign_complete.json',{'completed_runs':len(allrows),'expected_runs':4*3*3*len(P['methods']),'campaign_seconds':time.perf_counter()-campaign0,'total_script_seconds':time.perf_counter()-clock0,'costs':costs,'selection_lock_model_sha256':lock['model_sha256']})
 # Prespecified four-corner initial-condition sensitivity, one fixed repeat per method/case.
 sensitivity=[];sens0=time.perf_counter()
 for row in allrows:
  if row['seed']!=3101:continue
  for ratio_mult in [.95,1.05]:
   for glucose_mult in [.95,1.05]:
    record={'case_id':row['case_id'],'method':row['method'],'ratio_multiplier':ratio_mult,'glucose_multiplier':glucose_mult};t0=time.perf_counter()
    if row['controls'] is None:record.update(status='no_policy',feasible=False)
    else:
     try:
      cond=[row['condition'][0]*ratio_mult,row['condition'][1]*glucose_mult];q=reactor.simulate(row['controls'],initial_from_condition(cond),rtol=1e-9,atol=1e-11);err=abs(q['z']-np.log(row['target_ratio']));record.update(status='completed',log_ratio_error=err,ratio_relative_error_percent=100*abs(np.exp(q['z'])/row['target_ratio']-1),biomass_amount_gDW=q['biomass_amount_gDW'],feasible=bool(err<=P['operating_domain']['acceptance_log_ratio_tolerance'] and q['terminal'][0]<=1.8+1e-8))
     except Exception as ex:record.update(status='failed',feasible=False,error=repr(ex))
    record['seconds']=time.perf_counter()-t0;sensitivity.append(record)
 dump(R/'initial_condition_sensitivity.json',{'seconds':time.perf_counter()-sens0,'cases':sensitivity,'scope':'Four fixed +/-5% initial-condition perturbations per seed3101 policy, not probabilities or feedback tests'})
 print('COMPLETE',len(allrows),flush=True)
if __name__=='__main__':main()
