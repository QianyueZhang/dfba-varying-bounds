"""Focused fixed-policy audit, frozen tolerances declared before first run."""
from pathlib import Path
import hashlib,json,platform,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'core_snapshot_01/scripts'))
from model import Reactor
from independent_amount_model import AmountReactor,from_log,to_log

THRESHOLDS={
 'ordinary_state_z_logB_abs':2e-6,
 'ordinary_tangent_atol':5e-5,'ordinary_tangent_rtol':5e-5,
 'tight_state_z_logB_abs':2e-8,
 'tight_tangent_atol':1e-6,'tight_tangent_rtol':1e-6,
 'independent_fd_atol':2e-6,'independent_fd_rtol':2e-6,
 'volume_abs':2e-14,'feed_abs':2e-6,
 'finite_difference_step':2e-6,
 'note':'Finite numerical regression tolerances; not rigorous error enclosures. Exact clipping kinks use the documented inward one-sided convention.'}

def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
def close(a,b,atol,rtol=0):return bool(np.allclose(a,b,atol=atol,rtol=rtol))

def main():
 start=time.perf_counter();snapshot=HERE/'core_snapshot_01'
 hashes={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.rglob('*') if p.is_file()}
 protocol={'thresholds':THRESHOLDS,'snapshot_hashes':hashes,'python':sys.version,'platform':platform.platform(),'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 lock=HERE/'sensitivity_probe_protocol.json'
 if lock.exists():
  old=json.loads(lock.read_text())
  if old['thresholds']!=THRESHOLDS or old['snapshot_hashes']!=hashes:raise RuntimeError('Probe protocol changed')
 else:lock.write_text(json.dumps(protocol,indent=2)+'\n')
 cases=[
  dict(name='analytic_interior',family='analytic',state=[.7,.3,.25,2.8],base=[.05,.045,.04],gain=.8,basis=[0,-.1,.12],offset=[0,.035,-.025],kappas=[[1,1],[1.1,.9],[.95,1.05]],expected=['interior']*3),
  dict(name='analytic_fixed_upper_lower',family='analytic',state=[.5,.2,.15,2.5],base=[.04,.18,.02],gain=1.,basis=[0,.25,.25],offset=[0,.25,-.3],kappas=[[1,1]]*3,expected=['interior','upper','lower']),
  dict(name='analytic_volume_guard',family='analytic',state=[1.5,5.,4.,4.5],base=[.05,.12],gain=.6,basis=[0,.1],offset=[0,0],kappas=[[1,1]]*2,expected=['interior','volume']),
  dict(name='table_interior',family='table',state=[1.1,5.,4.,3.3],base=[.05,.1,.03],gain=.6,basis=[0,-.08,.1],offset=[0,.03,-.03],kappas=[[1,1],[1.1,.9],[.95,1.05]],expected=['interior']*3),
  dict(name='analytic_inhibition',family='analytic',inhibition=[30.,20.],state=[.7,.3,.25,2.8],base=[.05,.045,.04],gain=.8,basis=[0,-.1,.12],offset=[0,.035,-.025],kappas=[[1,1],[1.1,.9],[.95,1.05]],expected=['interior']*3),
  dict(name='table_inhibition',family='table',inhibition=[35.,45.],state=[1.1,5.,4.,3.3],base=[.05,.1,.03],gain=.6,basis=[0,-.08,.1],offset=[0,.03,-.03],kappas=[[1,1],[1.1,.9],[.95,1.05]],expected=['interior']*3),
 ]
 reports=[]
 for c in cases:
  independent=AmountReactor(c['family'],c.get('inhibition'));core=Reactor(c['family'],c.get('inhibition'))
  state=np.array(c['state']); n=len(c['base']); theta=np.r_[c['base'],c['gain']];basis=np.array(c['basis']);ks=np.array(c['kappas'])
  warm=independent.policy(state,np.r_[c['base'],0.],ks,np.zeros(n),np.zeros(n),False)
  reference=warm['knots'][:-1,1]-warm['knots'][:-1,2]+c['offset']
  truth=independent.policy(state,theta,ks,reference,basis)
  report={'name':c['name'],'case':c,'reference':reference,'independent_branches':truth['branches'],'checks':{},'comparisons':{}}
  report['checks']['expected_branches']=truth['branches']==c['expected']
  for tight in (False,True):
   name='tight' if tight else 'ordinary';q=core.policy(to_log(state),theta,ks,reference,basis,tight=tight)
   diffs={key:float(np.max(np.abs(np.asarray(q[key])-truth[key]))) for key in ('z','logB','jz','jlogB','feeds','V')}
   report['comparisons'][name]=diffs
   report['checks'][name+'_states']=all(diffs[k]<=THRESHOLDS[name+'_state_z_logB_abs'] for k in ('z','logB'))
   report['checks'][name+'_tangents']=all(close(q[k],truth[k],THRESHOLDS[name+'_tangent_atol'],THRESHOLDS[name+'_tangent_rtol']) for k in ('jz','jlogB'))
   report['checks'][name+'_feeds']=diffs['feeds']<=THRESHOLDS['feed_abs']
   report['checks'][name+'_volume']=diffs['V']<=THRESHOLDS['feed_abs']*2*n
   report['checks'][name+'_branches']=q['branches']==truth['branches']
  if c['family']=='analytic':
   h=THRESHOLDS['finite_difference_step'];fd=[];stable=True
   for j in range(n+1):
    direction=np.zeros(n+1);direction[j]=h
    plus=independent.policy(state,theta+direction,ks,reference,basis,False)
    minus=independent.policy(state,theta-direction,ks,reference,basis,False)
    stable &= plus['branches']==truth['branches']==minus['branches']
    fd.append([(plus[key]-minus[key])/(2*h) for key in ('z','logB')])
   fd=np.array(fd).T
   report['checks']['fd_stable_branch']=stable
   report['checks']['independent_amount_tangent_vs_fd']=all(close(truth[key],fd[i],THRESHOLDS['independent_fd_atol'],THRESHOLDS['independent_fd_rtol']) for i,key in enumerate(('jz','jlogB')))
   report['comparisons']['fd_max_abs']=float(np.max(np.abs(np.array([truth['jz'],truth['jlogB']])-fd)))
  if c['name']=='analytic_volume_guard':
   report['checks']['volume_gradient_zero']=close(truth['J'][0],np.zeros(n+1),THRESHOLDS['volume_abs'])
   report['checks']['volume_cap']=truth['V']<=1.8 and abs(truth['V']-(1.8-1e-11))<=THRESHOLDS['volume_abs']
  reports.append(report)
 guard=[]
 for volume in (1.8-1e-12,1.8,1.80001):
  state=np.array([volume,.3,.25,volume*3]);q=AmountReactor('analytic').policy(state,[.1,1.],[[1,1]],[0],[0])
  guard.append({'initial_volume':volume,'feed':q['feeds'][0],'final_volume':q['V'],'zero_feed':q['feeds'][0]==0.,'volume_unchanged':q['V']==volume,'initial_state_valid':volume<=1.8,'already_invalid_stays_invalid':q['V']>1.8 if volume>1.8 else True})
 kinks=[]
 for feed,sign in ((0.,1),(.2,-1)):
  state=np.array([.5,.2,.15,2.5]);core=Reactor('analytic');q=core.policy(to_log(state),[feed,.6],[[1,1]],[0],[0],tight=True)
  h=THRESHOLDS['finite_difference_step'];other=core.policy(to_log(state),[feed+sign*h,.6],[[1,1]],[0],[0],tight=True)
  fd=(other['z']-q['z'])/(sign*h)
  kinks.append({'feed':feed,'jz':q['jz'][0],'inward_fd':fd,'absolute_error':abs(q['jz'][0]-fd),'pass':close(q['jz'][0],fd,2e-4,2e-4),'gain_has_no_effect':q['jz'][-1]==0 and q['jlogB'][-1]==0})
 passed=all(all(r['checks'].values()) for r in reports) and all(g['zero_feed'] and g['volume_unchanged'] and g['already_invalid_stays_invalid'] for g in guard) and all(k['pass'] and k['gain_has_no_effect'] for k in kinks)
 out={'passed':passed,'reports':reports,'volume_guard':guard,'kink_convention':kinks,'elapsed_seconds':time.perf_counter()-start,'protocol_sha256':hashlib.sha256(lock.read_bytes()).hexdigest()}
 (HERE/'sensitivity_probe_results.json').write_text(json.dumps(out,indent=2,default=serial)+'\n')
 print(json.dumps({'passed':passed,'elapsed_seconds':out['elapsed_seconds'],'cases':[{'name':r['name'],'checks':r['checks'],'errors':r['comparisons']} for r in reports],'kinks':kinks},indent=2,default=serial))
if __name__=='__main__':main()
