"""Short installation, frozen-data and source-parity checks for this release."""
from pathlib import Path
import argparse,ast,hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'local_verification';OUT.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'matplotlib_cache'))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
sys.path.insert(0,str(ROOT/'research_completion/scripts'))
import numpy as np
from control_experiments import Reactor

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def check(label,passed,detail=None):checks.append({'check':label,'passed':bool(passed),'detail':detail})
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--require-models',action='store_true');args=parser.parse_args()
 expected=json.loads((ROOT/'tests/expected_smoke.json').read_text())
 sources=[p for folder in ['research_completion','tools','tests']for p in (ROOT/folder).rglob('*.py')if '__pycache__'not in p.parts]
 for p in sources:ast.parse(p.read_text())
 check('Python source syntax',True,{'files':len(sources)})
 import scipy,sklearn,matplotlib,PIL,openpyxl,highspy
 versions={'numpy':np.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__,'matplotlib':matplotlib.__version__,'Pillow':PIL.__version__,'openpyxl':openpyxl.__version__,'highspy':highspy.Highs().version()}
 pinned=dict(x.split('==')for x in (ROOT/'requirements.txt').read_text().splitlines()if x.strip())
 check('Pinned numerical and plotting imports',all(versions[k]==v for k,v in pinned.items()),versions)
 for row in expected['controls']:
  reactor=Reactor(table_path=ROOT/'research_completion/runs'/row['table'])
  numeric={k:[[x['S'],x['growth'],x['glucose_flux']]for x in reactor.table[k]['rows']]for k in ['SC','KP']}
  same=hashlib.sha256(json.dumps(numeric,sort_keys=True).encode()).hexdigest()==row['numerical_table_sha256']
  actual=reactor.simulate(row['control'],1e-9,1e-11)
  check('Control replay: '+row['table']+'/'+row['name'],same and np.allclose(actual['terminal'],row['terminal'],rtol=1e-9,atol=1e-11)and np.isclose(actual['objective'],row['objective'],rtol=1e-8,atol=1e-12),{'max_terminal_difference':float(np.max(abs(np.array(actual['terminal'])-row['terminal']))),'objective_difference':actual['objective']-row['objective']})
 data=np.load(ROOT/'research_completion/runs/surrogate_dataset.npz',allow_pickle=False)
 indices=[data[k]for k in ['train','validation','test']]
 check('Frozen training split is disjoint and complete',[len(x)for x in indices]==[700,150,150]and len(np.unique(np.concatenate(indices)))==1000 and sorted(np.concatenate(indices).tolist())==list(range(1000)))
 check('Neural targets equal terminal log ratios',np.allclose(data['target_log_ratio'],np.log(data['terminal_states'][:,1]/data['terminal_states'][:,2]),rtol=1e-13,atol=1e-13))
 import csv
 rows=list(csv.DictReader((ROOT/'data/zampar2013/zampar2013_observations.csv').open()))
 eligible=[r for r in rows if float(r['time_relative_to_glucose_exhaustion_h'])>=-6.8-1e-8]
 calibration=[r for r in eligible if r['replicate_label_as_published'].replace(' ','')=='Exp1']
 check('Published observation allocation',len(rows)==266 and len(eligible)==256 and len(calibration)==45 and len(eligible)-len(calibration)==211)
 inputs=json.loads((ROOT/'data/model_inputs.json').read_text())['inputs']
 matching=[r for r in inputs if (ROOT/r['path']).exists()and digest(ROOT/r['path'])==r['sha256']]
 if args.require_models:check('All supplied model inputs match',len(matching)==len(inputs),{'matching':len(matching),'required':len(inputs)})
 if len(matching)==len(inputs):
  sys.path.insert(0,str(ROOT/'research_completion/scripts/python_migration'))
  from edwards_comparison import Model
  for row in expected['edwards']:
   m=Model(row['name']);v,d=m.flux(row['substrate'],cold=True)
   check('Edwards finite-barrier '+row['name'],np.allclose(v,row['flux'],rtol=1e-8,atol=1e-9)and d['equality_residual']<1e-9 and d['primal_dual_residual']<1e-9,{'maximum_flux_difference':float(np.max(abs(v-row['flux']))),'equality_residual':d['equality_residual']})
  from flux_model import FluxModel
  sys.path.insert(0,str(ROOT/'research_completion/runs/cstr_dynamic_range_2026-10-21'))
  import benchmark
  for row in expected['models']:
   m=FluxModel(row['name'],1e-6)if row['kind']=='yeast'else benchmark.Model(row['name'],epsilon=1e-7,light=True)
   lp=m.lp(row['input'])if row['kind']=='yeast'else m.lp((row['time_h'],benchmark.Y0))
   check('Model/LP '+row['kind']+'/'+row['name'],list(m.A.shape)==row['shape']and np.isclose(-lp.fun,row['LP_growth'],rtol=1e-9,atol=1e-10),{'shape':list(m.A.shape),'growth_difference':float(-lp.fun-row['LP_growth'])})
 else:print('Genome-scale input checks skipped: prepare exact model inputs, then use --require-models.')
 import matplotlib.pyplot as plt
 r=expected['controls'][1];tr=Reactor(table_path=ROOT/'research_completion/runs'/r['table']).simulate(r['control'],1e-9,1e-11,dense=True)
 fig,ax=plt.subplots(2,1,figsize=(6,5),sharex=True)
 ax[0].plot(tr['time'],tr['state'][:,1],label='S. cerevisiae');ax[0].plot(tr['time'],tr['state'][:,2],'--',label='K. phaffii');ax[0].set_ylabel('Biomass (gDW/L)');ax[0].legend()
 ax[1].plot(tr['time'],tr['state'][:,3]);ax[1].set(xlabel='Time (h)',ylabel='Glucose (g/L)');fig.tight_layout();fig.savefig(OUT/'control_smoke.png',dpi=160);plt.close(fig)
 from PIL import Image
 with Image.open(OUT/'control_smoke.png')as im:check('Representative plot renders',im.width==960 and im.height==800,{'size':list(im.size)})
 report={'scope':'Source syntax, installed versions, six frozen/refined table replays, dataset/observations, optional supplied-model hashes and static LP/KKT checks, one figure. No full optimisation or long-horizon study rerun.','passed':all(c['passed']for c in checks),'checks_passed':sum(c['passed']for c in checks),'checks_total':len(checks),'versions':versions,'model_inputs_supplied':len(matching)==len(inputs),'checks':checks}
 (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:report[k]for k in ['passed','checks_passed','checks_total','model_inputs_supplied']},indent=2))
 if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
