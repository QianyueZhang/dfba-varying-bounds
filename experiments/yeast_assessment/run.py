"""Verify supplied assessment records or prepare an isolated calculation."""
from pathlib import Path
import argparse, json, subprocess, sys

ROOT=Path(__file__).resolve().parent
PAYLOAD=ROOT/'payload'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['verify','prepare','figures'])
    parser.add_argument('destination',nargs='?',type=Path)
    parser.add_argument('--case',default='analytic',choices=['analytic','numerical-lp','numerical-tight','numerical-small-barrier','numerical-all','temporal','temporal-diagnostic','external','recalibration'])
    parser.add_argument('--source-root',type=Path)
    args=parser.parse_args()
    if args.command=='verify':
        p=subprocess.run([sys.executable,str(PAYLOAD/'verify_records.py')],cwd=PAYLOAD,capture_output=True,text=True)
        if p.returncode:print(p.stderr,file=sys.stderr)
        try:report=json.loads(p.stdout)
        except json.JSONDecodeError:report={'passed':False,'errors':[p.stdout,p.stderr]}
        dest=ROOT/'local_results/package_verification.json';dest.parent.mkdir(exist_ok=True)
        dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
        return p.returncode or int(not report.get('passed'))
    if args.command=='figures':return subprocess.run([sys.executable,str(ROOT/'render_figures.py')],cwd=ROOT).returncode
    if args.destination is None:parser.error('prepare requires a new destination directory')
    command=[sys.executable,str(PAYLOAD/'run_study.py'),'--case',args.case,'--work-dir',str(args.destination.resolve()),'--prepare-only']
    if args.source_root:command+=['--source-root',str(args.source_root.resolve())]
    return subprocess.run(command,cwd=PAYLOAD).returncode

if __name__=='__main__':raise SystemExit(main())
