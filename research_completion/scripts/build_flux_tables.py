from flux_model import FluxModel,ROOT
from primal_dual import solve as pd_solve
import numpy as np,json,time,sys
from scipy.interpolate import PchipInterpolator

epsilon=float(sys.argv[1]) if len(sys.argv)>1 else 1e-3
mu=1e-6
out=ROOT/'research_completion/runs'/f'flux_table_eps{epsilon:g}.json'
old=json.loads(out.read_text()) if out.exists() else {}
for name in ['SC','KP']:
 model=FluxModel(name,epsilon);rows=old.get(name,{}).get('rows',[]);seen={r['S'] for r in rows};warm=None
 grid=np.unique(np.r_[0,np.geomspace(1e-5,1,25),np.linspace(1,20,25),np.linspace(20,100,17)])
 for S in grid[::-1]:
  if float(S) in seen:continue
  t=time.perf_counter()
  try:
   if epsilon<1e-4:
    x,l,zl,zu,d=pd_solve(model,float(S),mu,tol=1e-11,warm=warm);warm=(x,l,zl,zu)
   else:x,l,d=model.barrier(float(S),mu,warm);warm=(x,l)
  except RuntimeError:
   if epsilon<1e-4:x,l,zl,zu,d=pd_solve(model,float(S),mu,tol=1e-11);warm=(x,l,zl,zu)
   else:x,l,d=model.barrier(float(S),mu);warm=(x,l)
  rows.append({'S':float(S),**d,'seconds':time.perf_counter()-t})
  rows.sort(key=lambda r:r['S']);old[name]={'model':model.provenance,'mu':mu,'rows':rows}
  out.write_text(json.dumps(old,indent=2));print(name,S,d['growth'],d['glucose_flux'],d['newton_steps'],flush=True)
 # Independent midpoint checks, refine when needed to control output interpolation.
 for level in range(7):
  rows.sort(key=lambda r:r['S']);s=np.array([r['S'] for r in rows]);v=np.array([[r['growth'],r['glucose_flux']] for r in rows]);fit=PchipInterpolator(s,v,axis=0);add=[]
  # First level all midpoints; later only newly divided intervals needing recheck.
  intervals=list(zip(s[:-1],s[1:])) if level==0 else intervals
  checks=[]
  for a,b in intervals:
   S=(a+b)/2;t=time.perf_counter()
   try:
    if epsilon<1e-4:
     x,l,zl,zu,d=pd_solve(model,float(S),mu,tol=1e-11,warm=warm);warm=(x,l,zl,zu)
    else:x,l,d=model.barrier(float(S),mu,warm);warm=(x,l)
   except RuntimeError:
    if epsilon<1e-4:x,l,zl,zu,d=pd_solve(model,float(S),mu,tol=1e-11);warm=(x,l,zl,zu)
    else:x,l,d=model.barrier(float(S),mu);warm=(x,l)
   actual=np.array([d['growth'],d['glucose_flux']]);err=float(np.max(np.abs(actual-fit(S))));checks.append({'S':float(S),'max_absolute_flux_error':err})
   if err>2e-5:add.append({'S':float(S),**d,'seconds':time.perf_counter()-t})
  old[name].setdefault('interpolation_checks',[]).append({'level':level,'checks':checks,'max_absolute_error':max(c['max_absolute_flux_error'] for c in checks) if checks else 0})
  intervals=[]
  for r in add:
   at=np.searchsorted(s,r['S']);intervals.extend([(s[at-1],r['S']),(r['S'],s[at])])
  rows+=add;old[name]['rows']=sorted(rows,key=lambda r:r['S']);out.write_text(json.dumps(old,indent=2));print(name,'refine',level,'added',len(add),flush=True)
  if not add:break
print('complete',out,flush=True)
