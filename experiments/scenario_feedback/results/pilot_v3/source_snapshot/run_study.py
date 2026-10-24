"""Causal rollout, complete failure records, locked source identity, serial timing."""
import argparse, hashlib, json, platform, shutil, sys, time
from pathlib import Path
import numpy as np
import scipy
from model import Reactor,initial_state,observable,ROOT,DT,CAP,ACCEPT
from controllers import estimate_kappa,solve
import kernel
METHODS=['ce_limited','ce_direct','scenario_open','scenario_feedback','scenario_managed']

def serial(v):
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,np.generic): return v.item()
    raise TypeError(type(v).__name__)
def save(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(data,indent=2,default=serial)+'\n'); temp.replace(path)
def observe(y,noise):
    y=y.copy();y[1:3]+=noise[:2];y[3]=max(0,y[3]+noise[2]*y[0]); return y

def episode(case,method,checkpoint=None):
    model=Reactor(case.get('family','table')); plant=Reactor(case.get('family','table'),case.get('inhibition'))
    truth=initial_state(case['initial_ratio'],case['initial_glucose']); noises=np.asarray(case.get('noise',np.zeros((8,3))))
    state=observe(truth,noises[0]); history=[]; kappa=np.ones(2); previous=np.full(7,.065)
    stages=[]; states=[truth.copy()]; total=0.; estimation_seconds=0.; interrupted=False
    for stage in range(7):
        tic=time.perf_counter(); calls=model.calls
        try:
            est0=time.perf_counter()
            try: kappa,est=estimate_kappa(model,history,kappa)
            finally: estimation_seconds+=time.perf_counter()-est0
            plan,info=solve(model,state,kappa,case['target_ratio'],previous if stage==0 else previous[1:],stage,method)
            requested=float(plan[0]); feed=max(0.,min(.2,requested,(CAP-truth[0]-1e-11)/DT))
        except Exception as ex:
            # Deterministic physical backup. Does not guarantee composition.
            interrupted=True; info={'controller_exception':repr(ex)}; est={}; plan=np.zeros(7-stage); requested=0.;feed=0.
        elapsed=time.perf_counter()-tic; total+=elapsed
        if checkpoint: checkpoint({'completed_stages':stages,'pending_stage':stage,'pending_feed':feed,'controller_seconds':total,'model_calls':model.calls,'model_failures':model.failures,'estimator_seconds':estimation_seconds})
        kk=case['kappa_after'] if stage>=case.get('change_stage',99) else case['kappa_before']
        actual=plant.predict(truth,[feed],kk,tight=True,record=True)
        next_truth=actual['state']; next_truth[0]=truth[0]+DT*feed
        next_state=observe(next_truth,noises[stage+1])
        history.append((state.copy(),feed,next_state.copy()))
        record={'stage':stage,'observed_state':state,'kappa_estimate':kappa.copy(),'estimation':est,
            'decision':info,'requested_feed':requested,'feed':feed,'projection_change':feed-requested,
            'controller_seconds':elapsed,'model_calls':model.calls-calls,'true_start_state':truth,
            'true_end_state':next_truth,'true_kappa':kk,'plant_min_S':actual['min_S'],'plant_max_S':actual['max_S']}
        stages.append(record); states.append(next_truth.copy()); previous=plan; state=next_state;truth=next_truth
        if checkpoint: checkpoint({'completed_stages':stages,'controller_seconds':total,'model_calls':model.calls,'model_failures':model.failures,'estimator_seconds':estimation_seconds})
    out=observable(truth); error=abs(out['log_ratio']-np.log(case['target_ratio']))
    physical=all(0<=s['feed']<=.2 and s['true_end_state'][0]<=CAP and s['plant_min_S']>=-2e-7 for s in stages)
    out.update(log_ratio_error=float(error),terminal_feasible=bool(error<=ACCEPT and physical),
        operational_success=bool(error<=ACCEPT and physical and not interrupted),
        physical_valid=bool(physical),controller_failure=interrupted)
    return {'case':case,'method':method,'result':out,'stages':stages,'states':states,'controller_seconds':total,
        'estimator_seconds':estimation_seconds,'model_calls':model.calls,'model_failures':model.failures,'plant_seconds':plant.seconds}

def hashes():
    files=list((ROOT/'scripts').glob('*.py'))+[ROOT/'data/flux_table_eps1e-06.json']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.name in ('base_model.py','model.py','controllers.py','run_study.py','freeze_protocol.py','kernel.py') or p.suffix=='.json'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['pilot','evaluation'],default='pilot');parser.add_argument('--label',default='pilot_v1')
    parser.add_argument('--methods',nargs='+',default=METHODS);parser.add_argument('--limit',type=int);args=parser.parse_args()
    if args.mode=='pilot':
        cases=[dict(id='dev_'+name,family=family,initial_ratio=1.,initial_glucose=20.,target_ratio=target,
            kappa_before=[1.,1.],kappa_after=after,change_stage=stage) for name,family,target,after,stage in
            [('nominal','table',1.,[1.,1.],99),('shift','table',1.,[1.15,.85],3),('analytic','analytic',1.,[1.15,.85],3)]]
        if args.limit:cases=cases[:args.limit]
    else:
        protocol=json.loads((ROOT/'protocol.json').read_text()); lock=json.loads((ROOT/'LOCK.json').read_text())
        if args.limit or args.methods!=protocol['methods']:raise RuntimeError('Full locked study required')
        for p,h in lock['hashes'].items():
            if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise RuntimeError('Source changed: '+p)
        cases=protocol['cases']
    dest=ROOT/'results'/args.label;dest.mkdir(parents=True,exist_ok=True)
    compile0=time.perf_counter()
    if kernel.AVAILABLE:
        warm=Reactor(); st=initial_state(1.,20.)
        warm.predict(st,[.05],[1.,1.]);warm.policy(st,[.05,0.],[[1.,1.]],[0.],[0.])
    compilation_seconds=time.perf_counter()-compile0
    native_versions={}
    if kernel.AVAILABLE:
        import numba,llvmlite
        native_versions={'numba':numba.__version__,'llvmlite':llvmlite.__version__}
    manifest={'mode':args.mode,'hashes':hashes(),'cases':cases,'methods':args.methods,'python':sys.version,'scipy':scipy.__version__,
        'numpy':np.__version__,'platform':platform.platform(),'native_rhs':kernel.AVAILABLE,
        'native_initialization_seconds':compilation_seconds,'native_versions':native_versions,
        'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    if (dest/'execution.json').exists():
        previous=json.loads((dest/'execution.json').read_text())
        for k in ('mode','hashes','cases','methods','native_rhs','native_versions','numpy','scipy'):
            if previous[k]!=manifest[k]:raise RuntimeError('Run identity differs: '+k)
    else:
        save(dest/'execution.json',manifest); snapshot=dest/'source_snapshot';snapshot.mkdir(exist_ok=True)
        for p in manifest['hashes']:shutil.copy2(ROOT/p,snapshot/Path(p).name)
        if args.mode=='evaluation':
            for name in ('protocol.json','LOCK.json'):shutil.copy2(ROOT/name,snapshot/name)
    for i,case in enumerate(cases):
        order=args.methods[i%len(args.methods):]+args.methods[:i%len(args.methods)]
        for method in order:
            path=dest/f'{case["id"]}__{method}.json';partial=path.with_suffix('.partial.json')
            if path.exists():
                saved=json.loads(path.read_text())
                if saved.get('case')!=case or saved.get('method')!=method or saved.get('source_hashes')!=manifest['hashes']:
                    raise RuntimeError('Saved episode identity mismatch: '+str(path))
                continue
            # Never erase cost/evidence from an interrupted attempt.
            if partial.exists():partial.rename(dest/(partial.stem+'.interrupted.'+str(time.time_ns())+'.json'))
            tic=time.perf_counter()
            try:row=episode(case,method,lambda d:save(partial,d));row['status']='completed'
            except Exception as ex:
                prefix=json.loads(partial.read_text()) if partial.exists() else {}
                row={'case':case,'method':method,'status':'failed','error':repr(ex),'partial':prefix,
                    'controller_seconds':prefix.get('controller_seconds',0.),'model_calls':prefix.get('model_calls',0),
                    'model_failures':prefix.get('model_failures',0),'estimator_seconds':prefix.get('estimator_seconds',0.),
                    'result':{'terminal_feasible':False,'operational_success':False}}
            row['wall_seconds']=time.perf_counter()-tic;row['source_hashes']=manifest['hashes'];save(path,row)
            if partial.exists():partial.unlink()
            print(case['id'],method,row['result'],round(row['controller_seconds'],2),flush=True)
    inventory=[]
    for case in cases:
        for method in args.methods:
            path=dest/f'{case["id"]}__{method}.json';row=json.loads(path.read_text())
            if row.get('case')!=case or row.get('method')!=method or row.get('source_hashes')!=manifest['hashes']:
                raise RuntimeError('Final episode identity mismatch')
            inventory.append({'file':path.name,'status':row['status'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    save(dest/'complete.json',{'episodes':len(inventory),'expected':len(cases)*len(args.methods),'hashes':manifest['hashes'],
        'inventory':inventory,'interrupted_attempt_files':[p.name for p in dest.glob('*.interrupted.*.json')]})
if __name__=='__main__':main()
