"""Frozen-grid target sets and bounded-ratio observation policies.

This is finite numerical decision-tree algebra, not an exact ODE certificate.
The stage-two API accepts a measured log ratio only; it never receives a
disturbance index or the true biomass/substrate state.
"""
from pathlib import Path
import argparse, hashlib, itertools, json, math, sys, time
import numpy as np
import engine

ROOT = Path(__file__).resolve().parents[1]
SIZES = (41, 81, 161)
EPSILONS = (0., .001, .005, .01, .02)
PAD = 1e-7
INNER = .00748
FIXED = (.5, .75, 1., 1.25, 1.5, 2.)
RAW = np.arange(161, dtype=float) * (.2 / 160)
LOCKED = ('scripts/grid_observation.py', 'scripts/engine.py',
          'vendor/prior_kernel.py', 'data/flux_table_eps1e-06.json',
          'data/prescribed_roots.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hashes():
    return {p: sha(ROOT / p) for p in LOCKED}


def roots():
    return [r for r in json.loads((ROOT/'data/prescribed_roots.json').read_text())
            if r['timing'] == 'uniform']


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False,
                              default=engine.serialize)+'\n')


def freeze():
    p = ROOT/'protocols/grid_observation.json'
    if p.exists():
        raise FileExistsError('Frozen protocol exists; do not overwrite it')
    data = dict(
        name='Nested action resolution and bounded ratio observation',
        root_ids=[r['id'] for r in roots()],
        root_status='Prescribed states copied unchanged; no prefix regeneration',
        grids=list(SIZES), raw_action_formula='i*(0.2/160), i=0..160; strides4,2,1',
        clipping='clip to max(0,min(0.2,(1.8−V−1e−11)/2)); deduplicate',
        horizon_intervals=2, interval_hours=2, affinity=engine.K.tolist(),
        inner_band=INNER, outer_band=.0075, actual_band=math.log(1.01),
        fixed_target_ratios=list(FIXED),
        fixed_feasibility='error<=inner band, no additional feasibility tolerance',
        information='Known prescribed initial state and exact volume. Second decision receives measured z=log(M1/M2) only. No hidden scenario label or true biomass/substrate input.',
        epsilon=list(EPSILONS), compatibility_pad=PAD,
        effective_observation_radius='epsilon+1e−7; bounded support, not a probability model. Pad is a numerical guard, not a proved ODE error bound.',
        support='Closed predicted-branch intervals. Enumerate every endpoint and every intervening open cell, then every distinct nonempty compatible branch subset. Endpoint coincidences are retained.',
        set_recursion='Union over first feeds of intersection over realizable observation subsets C of union over shared second feeds of [max_(j in C,k) z_terminal−inner,min_(j in C,k) z_terminal+inner].',
        comparators=['exact full-state branch observation', 'common open two-feed plan'],
        policy='Minimax terminal log-ratio loss. Deterministic ascending feed-index ties. A common second action for every branch compatible with the measured ratio. Empty compatibility is explicitly out of support.',
        physical_checks='Finite states; V>0 and V<=1.8+2e−12; sampled min substrate concentration>=−2e−7. No continuous-path enclosure claimed.',
        convergence=dict(pairs=[[41,81],[81,161]],
            relative_symmetric_difference_limit=.05,
            endpoint_hausdorff_limit=.001,
            definition='length(A symmetric_difference B)/length(finer B); Hausdorff distance between all component endpoints',
            empty='Both empty: vacuous descriptive pass, not positive convergence evidence. One empty: fail; Hausdorff undefined.',
            singleton='If finer total length is zero, relative measure is undefined and length gate requires exact set equality; endpoint gate remains mandatory.',
            interpretation='Descriptive stability gates only, not convergence proofs, continuous-control bounds or rigorous uncertainty intervals'),
        integrator=dict(rtol=2e-13,atol=2e-15,maxstep_hours=.01),
        gate='Require audit/grid_observation_integrator_gate.json passed=true before tree outcome evaluation',
        source_sha256=hashes())
    dump(p, data)
    print(json.dumps({'frozen':str(p),'sha256':sha(p)}),flush=True)


def union(intervals):
    ans=[]
    for a,b in sorted((float(a),float(b)) for a,b in intervals if a<=b):
        if ans and a<=ans[-1][1]: ans[-1][1]=max(ans[-1][1],b)
        else: ans.append([a,b])
    return ans


def intersection(a,b):
    ans=[];i=j=0
    while i<len(a) and j<len(b):
        lo=max(a[i][0],b[j][0]);hi=min(a[i][1],b[j][1])
        if lo<=hi:ans.append([lo,hi])
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return union(ans)


def length(a): return sum(hi-lo for lo,hi in a)


def contains(a,z): return any(lo<=z<=hi for lo,hi in a)


def contained(a,b):
    return all(any(c<=lo and hi<=d for c,d in b) for lo,hi in a)


def compatible_masks(z_centres,radius):
    """All realizable nonempty masks; closed endpoints and open cells explicit."""
    z=np.asarray(z_centres,float)
    ends=np.unique(np.concatenate((z-radius,z+radius)))
    cells=[]
    for v in ends:
        mask=tuple(np.flatnonzero((z-radius<=v)&(v<=z+radius)).tolist())
        if mask: cells.append(dict(kind='singleton',lower=float(v),upper=float(v),representative=float(v),mask=list(mask)))
    for lo,hi in zip(ends[:-1],ends[1:]):
        v=float(lo+(hi-lo)/2)
        mask=tuple(np.flatnonzero((z-radius<v)&(v<z+radius)).tolist())
        if mask: cells.append(dict(kind='open',lower=float(lo),upper=float(hi),representative=v,mask=list(mask)))
    masks=sorted(set(tuple(c['mask']) for c in cells))
    return masks,cells


def masks_for(first_state,mode,epsilon=None):
    if mode=='open':return [(0,1,2)],[]
    if mode=='full':
        # Equal full states cannot be distinguished by scenario identity.
        groups={}
        for j,x in enumerate(first_state):groups.setdefault(tuple(x),[]).append(j)
        return sorted(tuple(x) for x in groups.values()),[]
    return compatible_masks(first_state[:,1]-first_state[:,2],float(epsilon)+PAD)


def alphabet(x,n=161):
    cap=max(0.,min(.2,(1.8-float(x[0])-engine.GUARD)/engine.DT))
    return np.unique(np.minimum(RAW[::160//(n-1)],cap))


def physical(states,minimum):
    return np.all(np.isfinite(states),axis=-1)&(states[...,0]>0)&(states[...,0]<=1.8+2e-12)&(minimum>=-2e-7)


def build_tree(root):
    t0=time.perf_counter();args=engine.arguments(root['family']);x=np.array(root['state'])
    u0=alphabet(x);n0=len(u0)
    first,mins,steps,rejected=engine.batch(np.repeat(x[None,:],n0*3,axis=0),np.repeat(u0,3),np.tile(engine.K,(n0,1)),args)
    first=first.reshape(n0,3,4);mins=mins.reshape(n0,3)
    u1=np.zeros((n0,161));count=np.zeros(n0,dtype=np.int64)
    terminal=np.zeros((n0,3,161,3,4));second_min=np.zeros((n0,3,161,3));calls=n0*3
    for i in range(n0):
        us=alphabet(first[i,0]);n=len(us);u1[i,:n]=us;count[i]=n
        assert all(np.array_equal(us,alphabet(s)) for s in first[i])
        states=np.repeat(first[i],n*3,axis=0)
        feeds=np.tile(np.repeat(us,3),3)
        ks=np.tile(engine.K,(3*n,1))
        out,mn,ss,rr=engine.batch(states,feeds,ks,args)
        terminal[i,:,:n,:,:]=out.reshape(3,n,3,4)
        second_min[i,:,:n,:]=mn.reshape(3,n,3)
        steps+=ss;rejected+=rr;calls+=len(feeds)
        if i%20==0:print(json.dumps({'root':root['id'],'first_actions_complete':i+1,'first_actions_total':n0}),flush=True)
    data=dict(u0=u0,first=first,first_min=mins,u1=u1,n1=count,terminal=terminal,second_min=second_min)
    cache=ROOT/'results/trees'/f"{root['id']}_161.npz";cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,**data)
    dump(cache.with_suffix('.json'),dict(root=root,raw_grid=161,transitions=calls,accepted_steps=int(steps),rejected_steps=int(rejected),seconds=time.perf_counter()-t0,source_sha256=hashes(),npz_sha256=sha(cache)))
    return data


def subset_tree(tree,n):
    wanted=alphabet(tree['first'][0,0],n)
    # First state at zero feed has the prescribed root volume.
    i0=np.array([int(np.flatnonzero(tree['u0']==u)[0]) for u in wanted])
    out=[]
    for i in i0:
        us=alphabet(tree['first'][i,0],n)
        old=tree['u1'][i,:tree['n1'][i]]
        ix=np.array([int(np.flatnonzero(old==u)[0]) for u in us])
        terminal=tree['terminal'][i][:,ix,:,:]
        valid=physical(terminal,tree['second_min'][i][:,ix,:])
        first_valid=physical(tree['first'][i],tree['first_min'][i])
        out.append(dict(u0=float(tree['u0'][i]),first=tree['first'][i],u1=us,
                        z=terminal[...,1]-terminal[...,2],
                        biomass=np.exp(terminal[...,1])+np.exp(terminal[...,2]),
                        valid=valid,first_valid=first_valid))
    return out


def target_sets(tree,mode,epsilon=None):
    every=[];byfirst=[]
    for branch in tree:
        masks,cells=masks_for(branch['first'],mode,epsilon)
        current=None
        if np.all(branch['first_valid']):
            for mask in masks:
                zz=branch['z'][list(mask),:,:]
                valid=np.all(branch['valid'][list(mask),:,:],axis=(0,2))
                lo=np.max(zz,axis=(0,2))-INNER;hi=np.min(zz,axis=(0,2))+INNER
                u=union(zip(lo[valid],hi[valid]))
                current=u if current is None else intersection(current,u)
                if not current:break
        current=[] if current is None else current
        every.extend(current);byfirst.append(current)
    intervals=union(every)
    return dict(intervals=intervals,log_length=length(intervals),
                ratio_length=sum(math.exp(b)-math.exp(a) for a,b in intervals),
                components=len(intervals),singletons=sum(a==b for a,b in intervals),
                by_first_action=byfirst)


def fixed_policy(tree,target,mode,epsilon=None):
    zstar=math.log(target);best=None
    for i,branch in enumerate(tree):
        if not np.all(branch['first_valid']):continue
        masks,cells=masks_for(branch['first'],mode,epsilon)
        choices=[];worst=0.;worstB=float('inf')
        for mask in masks:
            zz=branch['z'][list(mask),:,:]
            valid=np.all(branch['valid'][list(mask),:,:],axis=(0,2))
            errs=np.max(np.abs(zz-zstar),axis=(0,2));errs[~valid]=np.inf
            j=int(np.argmin(errs))
            if not math.isfinite(float(errs[j])):worst=float('inf');break
            biomass=float(np.min(branch['biomass'][list(mask),j,:]))
            choices.append(dict(mask=list(mask),feed=float(branch['u1'][j]),action_index=j,error=float(errs[j]),minimum_biomass=biomass))
            worst=max(worst,float(errs[j]));worstB=min(worstB,biomass)
        if math.isfinite(worst) and (best is None or worst<best['worst_error']):
            best=dict(target=target,first_feed=branch['u0'],first_action_index=i,
                      mode=mode,epsilon=epsilon,effective_radius=None if epsilon is None else epsilon+PAD,
                      predicted_z=(branch['first'][:,1]-branch['first'][:,2]).tolist(),
                      predicted_first_states=branch['first'].tolist(),
                      worst_error=worst,worst_biomass=worstB,inner=worst<=INNER,
                      outer=worst<=.0075,actual=worst<=math.log(1.01),
                      second_actions=choices,measurement_cells=cells)
    return best if best is not None else dict(target=target,mode=mode,epsilon=epsilon,unresolved=True)


def choose_second(policy,measured_z):
    """No-oracle bounded-observation API: return shared feed or explicit error."""
    if policy['mode']!='ratio':raise ValueError('Ratio-only policy required')
    z=np.asarray(policy['predicted_z']);r=policy['effective_radius']
    mask=np.flatnonzero((z-r<=measured_z)&(measured_z<=z+r)).tolist()
    if not mask:raise ValueError('Measurement outside predicted observation support')
    for c in policy['second_actions']:
        if c['mask']==mask:return c['feed']
    raise ValueError('Compatible subset absent from frozen policy')


def convergence(a,b):
    la=length(a);lb=length(b);common=length(intersection(a,b))
    sym=max(0.,la+lb-2*common)
    if not a and not b:return dict(status='both_empty_vacuous',relative_symmetric_difference=0.,endpoint_hausdorff=None,passed=True,positive_convergence_evidence=False)
    if not a or not b:return dict(status='one_empty',relative_symmetric_difference=None if lb==0 else sym/lb,endpoint_hausdorff=None,passed=False,positive_convergence_evidence=False)
    aa=np.array(a).ravel();bb=np.array(b).ravel()
    distances=np.abs(aa[:,None]-bb[None,:]);haus=float(max(np.min(distances,axis=0).max(),np.min(distances,axis=1).max()))
    rel=None if lb==0 else sym/lb
    measure_ok=(a==b) if lb==0 else rel<=.05
    return dict(status='singleton_finer' if lb==0 else 'nonempty',symmetric_difference_length=sym,
                relative_symmetric_difference=rel,endpoint_hausdorff=haus,
                passed=bool(measure_ok and haus<=.001),positive_convergence_evidence=bool(measure_ok and haus<=.001))


def run():
    protocol=ROOT/'protocols/grid_observation.json'
    locked=json.loads(protocol.read_text())
    if hashes()!=locked['source_sha256']:raise RuntimeError('Protocol source lock mismatch')
    gate_path=ROOT/'audit/grid_observation_integrator_gate.json'
    gate=json.loads(gate_path.read_text())
    if not gate.get('passed'):raise RuntimeError('Independent integrator gate did not pass')
    for name in ('scripts/engine.py','vendor/prior_kernel.py','data/flux_table_eps1e-06.json','data/prescribed_roots.json'):
        if gate.get('source_sha256',{}).get(name)!=sha(ROOT/name):
            raise RuntimeError('Independent gate source lock mismatch: '+name)
    t0=time.perf_counter();records=[];comparisons=[];checks=[]
    modes=[('full',None),('open',None)]+[('ratio',e) for e in EPSILONS]
    for root in roots():
        cache=ROOT/'results/trees'/f"{root['id']}_161.npz"
        if cache.exists():
            meta=json.loads(cache.with_suffix('.json').read_text())
            if meta['source_sha256']!=hashes() or meta['npz_sha256']!=sha(cache):raise RuntimeError('Cache lock mismatch')
            with np.load(cache) as d:tree={k:d[k] for k in d.files}
        else:tree=build_tree(root)
        local={}
        for n in SIZES:
            sub=subset_tree(tree,n)
            for mode,eps in modes:
                sets=target_sets(sub,mode,eps)
                policies=[fixed_policy(sub,t,mode,eps) for t in FIXED]
                record=dict(root=root['id'],family=root['family'],volume=root['volume'],grid=n,mode=mode,epsilon=eps,**sets,
                            fixed_policies=policies,all_path_fixed_inner=sum(p.get('inner',False) for p in policies),
                            all_path_fixed_outer=sum(p.get('outer',False) for p in policies),
                            all_path_fixed_actual=sum(p.get('actual',False) for p in policies))
                records.append(record);local[n,mode,eps]=record
            full=local[n,'full',None]['intervals'];opened=local[n,'open',None]['intervals']
            prev=full
            for eps in EPSILONS:
                obs=local[n,'ratio',eps]['intervals']
                checks.append(dict(root=root['id'],grid=n,epsilon=eps,open_contained=contained(opened,obs),observation_contained_in_full=contained(obs,full),epsilon_monotonic=contained(obs,prev)))
                prev=obs
        for mode,eps in modes:
            for coarse,fine in zip(SIZES[:-1],SIZES[1:]):
                a=local[coarse,mode,eps]['intervals'];b=local[fine,mode,eps]['intervals']
                comparisons.append(dict(root=root['id'],mode=mode,epsilon=eps,coarse=coarse,fine=fine,nested_containment=contained(a,b),**convergence(a,b)))
        dump(ROOT/'results'/f"grid_observation_{root['id']}.json",dict(root=root,records=[r for r in records if r['root']==root['id']],comparisons=[c for c in comparisons if c['root']==root['id']],complete=True))
        print(json.dumps({'root_completed':root['id'],'seconds_elapsed':time.perf_counter()-t0}),flush=True)
    passed=all(all(c[k] for k in ['open_contained','observation_contained_in_full','epsilon_monotonic']) for c in checks) and all(c['nested_containment'] for c in comparisons)
    result=dict(complete=True,source_sha256=hashes(),protocol_sha256=sha(protocol),integrator_gate_sha256=sha(gate_path),
                records=records,convergence=comparisons,algebra_checks=checks,algebra_passed=passed,
                elapsed_seconds=time.perf_counter()-t0,interpretation='No global continuous-control or biological validation claim; convergence gates are descriptive.')
    dump(ROOT/'results/grid_observation.json',result)
    print(json.dumps({'complete':True,'algebra_passed':passed,'seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.run:run()
    else:p.error('Choose --freeze or --run')
