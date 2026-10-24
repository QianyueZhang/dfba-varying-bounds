"""Portable runner: python scripts/continuous_run.py --workers 3."""
from pathlib import Path
import argparse,concurrent.futures,gzip,hashlib,json,os,time
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
from engine import arguments,save
from continuous_solver import solve
from continuous_audit import audit
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/continuous'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    protocol=json.loads((ROOT/'protocols/continuous_protocol.json').read_text())
    assert protocol['frozen_before_evaluation']
    for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in protocol['input_sha256'].items():assert sha(ROOT/name)==digest,name
    gate=json.loads((ROOT/'audit/grid_observation_integrator_gate.json').read_text())
    assert gate['complete'] and gate['passed']
    for name,digest in gate['source_sha256'].items():assert sha(ROOT/name)==digest,name
    return protocol


def grid_seed(root,target,mode):
    data=json.loads((ROOT/f'results/grid_observation_{root}_grid41.json').read_text())
    record=next(r for r in data['records'] if r['mode']==('full' if mode=='causal' else 'open'))
    policy=next(p for p in record['fixed_policies'] if p['target']==target)
    if mode=='open':second=[policy['second_actions'][0]['feed']]
    else:
        second=[None]*3
        for action in policy['second_actions']:
            for i in action['mask']:second[i]=action['feed']
        assert all(v is not None for v in second)
    return [policy['first_feed']]+second


def one_root(root):
    protocol=verify();args=arguments(root['family']);results=[]
    for target in protocol['targets']:
        common=None
        for mode in ['open','causal']:
            stamp=f'{root["id"]}_target_{str(target).replace(".","p")}_{mode}'
            summary_path=OUT/(stamp+'.json')
            if summary_path.exists():
                existing=json.loads(summary_path.read_text())
                assert existing['protocol_sha256']==sha(ROOT/'protocols/continuous_protocol.json')
                assert existing['complete']
                results.append(existing)
                if mode=='open':common=existing['optimization']['selected']
                continue
            tick=time.perf_counter()
            optimization=solve(root['state'],args,target,mode,grid_seed(root['id'],target,mode),common)
            if mode=='open':common=optimization['selected']
            ledger=optimization.pop('candidates')
            with gzip.open(OUT/(stamp+'_candidates.json.gz'),'wt') as stream:
                json.dump(ledger,stream,separators=(',',':'),allow_nan=False)
            deployment=audit(root,args,optimization)
            result=dict(root=root['id'],family=root['family'],target=target,mode=mode,
                optimization=optimization,deployment=deployment,
                candidate_ledger=stamp+'_candidates.json.gz',
                candidate_ledger_sha256=sha(OUT/(stamp+'_candidates.json.gz')),
                complete=bool(optimization['selected']['valid'] and deployment['complete']),
                protocol_sha256=sha(ROOT/'protocols/continuous_protocol.json'),
                elapsed_seconds=time.perf_counter()-tick)
            save(summary_path,result);results.append(result)
            print(root['id'],target,mode,'plan',optimization['selected']['max_error'],
                  'deploy',deployment['deployed_error'],'seconds',result['elapsed_seconds'],flush=True)
    return root['id'],len(results)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--root');opts=parser.parse_args();protocol=verify();OUT.mkdir(parents=True,exist_ok=True)
    roots=[r for r in json.loads((ROOT/'data/prescribed_roots.json').read_text()) if r['timing']=='uniform']
    if opts.root:roots=[r for r in roots if r['id']==opts.root]
    started=time.perf_counter()
    if opts.workers==1:
        for root in roots:one_root(root)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=opts.workers) as pool:
            for result in pool.map(one_root,roots):print('ROOT COMPLETE',result,flush=True)
    verify()
    rows=[json.loads(p.read_text()) for p in sorted(OUT.glob('*_target_*.json'))]
    summary=dict(complete=len(rows)==72 and all(r['complete'] for r in rows),problems=36,
                 policy_executions=len(rows),elapsed_wall_seconds=time.perf_counter()-started,
                 protocol_sha256=sha(ROOT/'protocols/continuous_protocol.json'),methods={})
    for mode in ['open','causal']:
        subset=[r for r in rows if r['mode']==mode]
        summary['methods'][mode]=dict(problems=len(subset),
            planned={label:sum(r['optimization']['selected'][label] for r in subset) for label in ['inner','outer','actual']},
            deployed={label:sum(r['deployment']['deployed_labels'][label] for r in subset if r['deployment']['complete']) for label in ['inner','outer','actual']},
            completed_paths=sum(sum(p['complete'] for p in r['deployment']['paths']) for r in subset),
            physical_problem_failures=sum(not r['deployment']['all_physical'] for r in subset),
            matched_numerical_problem_failures=sum(not r['deployment']['all_matched_comparisons_pass'] for r in subset),
            chained_numerical_problem_failures=sum(not r['deployment']['all_chained_comparisons_pass'] for r in subset),
            label_problem_disagreements=sum(not r['deployment']['all_chained_labels_agree'] for r in subset),
            optimizer_calls=sum(len(r['optimization']['solver_statuses']) for r in subset),
            optimizer_nonconvergence=sum(sum(not s['success'] for s in r['optimization']['solver_statuses']) for r in subset),
            optimization_transitions=sum(r['optimization']['costs']['transitions'] for r in subset),
            continuation_transitions=sum(sum(p['costs']['transitions'] for p in r['deployment']['second_policies']) for r in subset),
            optimization_seconds=sum(r['optimization']['seconds'] for r in subset),
            deployment_seconds=sum(r['deployment']['seconds'] for r in subset),
            max_matched_log_ratio_error=max((r['deployment']['max_matched_log_ratio_error'] for r in subset),default=None),
            max_chained_log_ratio_error=max((r['deployment']['max_chained_log_ratio_error'] for r in subset),default=None))
    summary['shared_grid41_construction']=[json.loads((ROOT/f'results/trees/{r["id"]}_41.json').read_text())
                                         for r in json.loads((ROOT/'data/prescribed_roots.json').read_text()) if r['timing']=='uniform']
    save(OUT/'summary.json',summary);print(json.dumps(summary['methods'],indent=2),flush=True)

if __name__=='__main__':main()
