"""Compare compiled and original NumPy/SciPy paths before evaluation lock."""
import json,time
import numpy as np
from model import Reactor,initial_state,ROOT
rows=[]
for family,inhib in [('table',None),('table',[240.,180.]),('analytic',None),('analytic',[240.,180.])]:
    fast=Reactor(family,inhib,True); slow=Reactor(family,inhib,False)
    for ratio,substrate,feeds in [(1.,20.,[.065]*7),(.8,18.,[0.,.16,.04,.12,.02,.08,.01])]:
        state=initial_state(ratio,substrate);ks=np.tile([.92,1.08],(7,1));ref=np.linspace(np.log(ratio),0.,7);basis=np.array([.2,.3,.5,-.4,-1.,-.6,-.5]);theta=np.r_[feeds,.8]
        a=fast.policy(state,theta,ks,ref,basis,tight=True);b=slow.policy(state,theta,ks,ref,basis,tight=True)
        errors={k:float(np.max(np.abs(np.asarray(a[k])-np.asarray(b[k])))) for k in ('state','jz','jlogB','feeds')}
        pa=fast.predict(state,feeds,[.92,1.08],gradients=True,tight=True);pb=slow.predict(state,feeds,[.92,1.08],gradients=True,tight=True)
        errors.update({'predict_'+k:float(np.max(np.abs(np.asarray(pa[k])-np.asarray(pb[k])))) for k in ('state','jz','jlogB')})
        rows.append({'family':family,'inhibition':inhib,'errors':errors,'pass':max(errors.values())<2e-6})
out={'native_available':fast.fast,'threshold_max_absolute':2e-6,'rows':rows,'passed':all(r['pass'] for r in rows)}
(ROOT/'audit/kernel_equivalence.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
assert fast.fast and out['passed']
