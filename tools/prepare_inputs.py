"""Restore exact local model inputs by content hash; never download implicitly."""
from pathlib import Path
import argparse, hashlib, json, os, shutil

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'data/model_inputs.json'

def digest(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--search-root',type=Path,help='A local directory containing model files you are entitled to use.')
 parser.add_argument('--check',action='store_true',help='Check prepared inputs without copying.')
 args=parser.parse_args();rows=json.loads(MANIFEST.read_text())['inputs']
 pending=[r for r in rows if not (ROOT/r['path']).is_file()or digest(ROOT/r['path'])!=r['sha256']]
 if args.search_root and not args.check:
  needed={r['sha256']for r in pending};by_size={r['bytes']for r in pending};found={}
  for base,dirs,files in os.walk(args.search_root):
   dirs[:]=[d for d in dirs if d not in ['.git','.venv','__pycache__','node_modules']]
   for name in files:
    if not name.lower().endswith('.mat'):continue
    p=Path(base)/name
    try:
     if p.stat().st_size not in by_size:continue
     key=digest(p)
    except OSError:continue
    if key in needed:found.setdefault(key,p)
   if needed.issubset(found):break
  for r in pending:
   if r['sha256']not in found:continue
   target=ROOT/r['path'];target.parent.mkdir(parents=True,exist_ok=True)
   shutil.copyfile(found[r['sha256']],target)
 rows=[{'path':r['path'],'sha256':r['sha256'],'present_and_matching':(ROOT/r['path']).is_file()and digest(ROOT/r['path'])==r['sha256']}for r in rows]
 print(json.dumps({'matching':sum(r['present_and_matching']for r in rows),'required':len(rows),'missing':[r['path']for r in rows if not r['present_and_matching']]},indent=2))
 if not all(r['present_and_matching']for r in rows):raise SystemExit(1)

if __name__=='__main__':main()

