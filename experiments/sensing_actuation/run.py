"""Inspect control-study evidence or prepare a new calculation workspace."""
from pathlib import Path
import argparse, ast, hashlib, json, subprocess, sys

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def integrity():
    errors=[];manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    for row in manifest['files']:
        p=ROOT/row['path']
        if not p.is_file() or sha(p)!=row['sha256']:errors.append('File identity: '+row['path'])
        elif p.suffix=='.py':ast.parse(p.read_text())
    locked=json.loads((ROOT/'protocols/grid_observation_amendment.json').read_text())
    for name,value in locked['source_sha256'].items():
        if sha(ROOT/name)!=value:errors.append('Numerical source lock: '+name)
    return errors,len(manifest['files'])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['verify','prepare','numerical-smoke'])
    parser.add_argument('destination',nargs='?',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':
        if args.destination is None:parser.error('prepare requires a new directory')
        return subprocess.run([sys.executable,str(ROOT/'scripts/prepare_reproduction.py'),'--destination',str(args.destination.resolve())],cwd=ROOT).returncode
    out=ROOT/'local_results';out.mkdir(exist_ok=True)
    if args.command=='numerical-smoke':
        import numpy as np
        sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'audit')]
        import engine
        from independent_amount_model import AmountReactor,from_log
        records=[]
        for family in ['table','analytic']:
            initial=engine.initial();kappa=np.array([.975,1.025]);feed=.07
            screen,_,_,_=engine.flow(initial,feed,kappa,engine.arguments(family))
            ref=AmountReactor(family).interval(from_log(initial),feed,kappa)
            error=float(np.max(abs(screen-ref['state'])))
            records.append({'family':family,'maximum_state_error':error,'passed':error<=2e-7})
        report={'passed':all(r['passed'] for r in records),'checks_total':2,'checks_passed':sum(r['passed'] for r in records),'records':records,
                'scope':'Two independent model-family installation comparisons; no policy optimisation or efficacy campaign.'}
        (out/'numerical_smoke.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
        return int(not report['passed'])
    errors,count=integrity()
    integrity_passed = not errors
    p=subprocess.run([sys.executable,str(ROOT/'verify_evidence.py')],cwd=ROOT,capture_output=True,text=True)
    if p.returncode:errors.append(p.stderr or p.stdout)
    result=out/'saved_evidence_recount.json'
    evidence=json.loads(result.read_text()) if result.exists() else {'passed_recount':False,'issues':['No recount report']}
    if not evidence['passed_recount']:errors+=evidence['issues']
    report={'passed':not errors,'checks_total':2,'checks_passed':int(integrity_passed)+int(p.returncode == 0 and evidence['passed_recount']),
            'files_checked':count,'scientific_counts':evidence.get('continuous'),
            'strict_boundary_audits_passed':{'primary':False,'adaptive':False},'errors':errors,
            'scope':'Distributed file identity, frozen numerical sources and complete saved-outcome arithmetic. Strict endpoint failures remain part of the evidence. No full numerical campaign is run.'}
    (out/'package_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return int(bool(errors))

if __name__=='__main__':raise SystemExit(main())
