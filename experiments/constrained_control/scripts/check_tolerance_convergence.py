from pathlib import Path
import json,sys,os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
PACKAGE=Path(__file__).resolve().parents[1];R=PACKAGE/'reference';H=Path(os.environ.get('DFBA_CONTROL_AUDIT',PACKAGE.parents[1]/'local_verification/control_balances'));H.mkdir(parents=True,exist_ok=True)
from simulator import Reactor
load=lambda p:json.loads(p.read_text());raw=load(PACKAGE/'data/flux_table_eps1e-06.json');curves=[PchipInterpolator([r['S'] for r in raw[s]['rows']],[[r['growth'],r['glucose_flux']] for r in raw[s]['rows']],axis=0) for s in ['SC','KP']]
reactor=Reactor(PACKAGE/'data/flux_table_eps1e-06.json')
def amount(p,rtol):
 u=p['controls'];v0=p['initial'][0];m=v0*np.array(p['initial'][1:]);
 for j,f in enumerate(u):
  a=2*j;b=a+2;vstart=v0+2*sum(u[:j])
  def rhs(t,m):
   S=max(0.,m[2]/(vstart+f*(t-a)));q=np.array([c(S) for c in curves])*min(1.,S/1e-5)
   return [q[0,0]*m[0],q[1,0]*m[1],f*100+.18*(q[0,1]*m[0]+q[1,1]*m[1])]
  sol=solve_ivp(rhs,(a,b),m,method='DOP853',rtol=rtol,atol=rtol/100,max_step=.125);assert sol.success;m=sol.y[:,-1]
 return float(np.log(m[0]/m[1]))
names=['condition_2_target_2_seed_3101_direct_single_start','condition_1_target_2_seed_3103_direct_multistart','condition_2_target_2_seed_3102_mlp'];result=[]
for n in names:
 p=load(R/'results/policies'/f'{n}.json');steps=[]
 for rt in [1e-9,1e-10,1e-11]:
  z=reactor.simulate(p['controls'],p['initial'],rtol=rt,atol=rt/100)['z'];za=amount(p,rt);steps.append(dict(rtol=rt,z_concentration=z,z_amount=za,difference=abs(z-za)))
 result.append(dict(policy=n,steps=steps,final_formulation_difference=steps[-1]['difference'],last_concentration_change=abs(steps[-1]['z_concentration']-steps[-2]['z_concentration']),last_amount_change=abs(steps[-1]['z_amount']-steps[-2]['z_amount'])))
(H/'tolerance_convergence.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
