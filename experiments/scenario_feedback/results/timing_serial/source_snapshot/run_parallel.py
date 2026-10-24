"""Locked effectiveness campaign in separate processes; isolated timing panel.

Parallel wall-time totals are diagnostic only, not comparative timing evidence.
The preselected serial panel is rerun after all parallel workers have exited.
"""
import argparse,hashlib,json,os,platform,shutil,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import scipy
import kernel
from model import ROOT,Reactor,initial_state
from run_study import hashes,save,persist_episode

def initialize(dest):
    tic=time.perf_counter();m=Reactor();s=initial_state(1.,20.)
    m.predict(s,[.05],[1.,1.]);m.policy(s,[.05,0.],[[1.,1.]],[0.],[0.])
    save(dest/f'worker_init_{os.getpid()}.json',{'pid':os.getpid(),'seconds':time.perf_counter()-tic,'native':kernel.AVAILABLE})

def task(case,index,dest,manifest):
    methods=manifest['methods'];order=methods[index%len(methods):]+methods[:index%len(methods)]
    for method in order:persist_episode(case,method,dest,manifest)
    return case['id']

def complete(dest,manifest):
    inventory=[]
    for c in manifest['cases']:
        for m in manifest['methods']:
            p=dest/f'{c["id"]}__{m}.json';d=json.loads(p.read_text())
            if d.get('case')!=c or d.get('method')!=m or d.get('source_hashes')!=manifest['hashes']:raise RuntimeError('Record identity mismatch')
            inventory.append({'file':p.name,'status':d['status'],'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    save(dest/'complete.json',{'episodes':len(inventory),'expected':len(manifest['cases'])*len(manifest['methods']),
        'inventory':inventory,'hashes':manifest['hashes'],'interrupted_attempt_files':[p.name for p in dest.glob('*.interrupted.*.json')]})

def main():
    p=argparse.ArgumentParser();p.add_argument('--timing',action='store_true');args=p.parse_args()
    protocol=json.loads((ROOT/'protocol.json').read_text());lock=json.loads((ROOT/'LOCK.json').read_text())
    for name,h in lock['hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Locked source changed: '+name)
    if not kernel.AVAILABLE:raise RuntimeError('Locked native kernel requires numba/llvmlite')
    import numba,llvmlite
    cases=protocol['cases']
    if args.timing:cases=[cases[i] for i in protocol['serial_timing_case_ordinals']]
    dest=ROOT/'results'/('timing_serial' if args.timing else 'locked');dest.mkdir(parents=True,exist_ok=True)
    manifest={'mode':'evaluation','hashes':hashes(),'cases':cases,'methods':protocol['methods'],'python':sys.version,
        'numpy':np.__version__,'scipy':scipy.__version__,'native_rhs':kernel.AVAILABLE,
        'native_versions':{'numba':numba.__version__,'llvmlite':llvmlite.__version__},'platform':platform.platform(),
        'workers':1 if args.timing else 4,'timing_inference':bool(args.timing),
        'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    if (dest/'execution.json').exists():
        old=json.loads((dest/'execution.json').read_text())
        for key in ('mode','hashes','cases','methods','native_versions','numpy','scipy','workers'):
            if old[key]!=manifest[key]:raise RuntimeError('Resume identity changed: '+key)
    else:
        save(dest/'execution.json',manifest);snap=dest/'source_snapshot';snap.mkdir(exist_ok=True)
        for name in list(manifest['hashes'])+['protocol.json','LOCK.json']:shutil.copy2(ROOT/name,snap/Path(name).name)
    if args.timing:
        initialize(dest)
        for i,c in enumerate(cases):task(c,i,dest,manifest)
    else:
        with ProcessPoolExecutor(max_workers=4,initializer=initialize,initargs=(dest,)) as pool:
            futures=[pool.submit(task,c,i,dest,manifest) for i,c in enumerate(cases)]
            for f in as_completed(futures):print('CASE_COMPLETE',f.result(),flush=True)
    complete(dest,manifest)
if __name__=='__main__':main()
