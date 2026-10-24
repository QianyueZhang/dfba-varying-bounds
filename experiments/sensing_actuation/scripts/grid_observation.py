"""Frozen-grid target sets and bounded-ratio observation policies.

This is finite numerical decision-tree algebra, not an exact ODE certificate.
The stage-two API accepts a measured log ratio only; it never receives a
disturbance index or the true biomass/substrate state.
"""
from pathlib import Path
import argparse, hashlib, itertools, json, math, sys, time, concurrent.futures
import numpy as np
import engine

ROOT = Path(__file__).resolve().parents[1]
SIZES = (41, 81, 161)
EPSILONS = (0., .001, .005, .01, .02)
PAD = 1e-7
INNER = .00748
FIXED = (.5, .75, 1., 1.25, 1.5, 2.)
RAW = np.arange(321, dtype=float) * (.2 / 320)
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


def amend():
    p=ROOT/'protocols/grid_observation_amendment.json'
    if p.exists():raise FileExistsError('Amendment already frozen')
    original=ROOT/'protocols/grid_observation.json'
    dump(p,dict(original_protocol_sha256=sha(original),
                reason='Pre-outcome addition requested by coordinating investigator; no tree outcomes evaluated before amendment.',
                utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                primary_grids=list(SIZES),adaptive_grid=321,
                adaptive_trigger='Every root failing either descriptive gate for81→161 in ANY full/open/ratio mode, without selecting by direction or positivity of effect.',
                adaptive_budget_seconds=3600,
                budget_scope='One total additional wall-clock budget across all adaptive roots, after the complete161-phase result is saved. All triggered roots start concurrently; deadline checked between first-action subtrees. Incomplete roots and partial caches remain unresolved.',
                reuse='161-point transitions reused exactly for nested321 actions; compute missing transitions only.',
                action_formula='i*(0.2/320), i=0..320; strides8,4,2,1 for41,81,161,321.',
                workers=6,threads_per_worker=1,
                original_source_snapshot='protocols/frozen_grid_observation_v1.py',
                secondary_sensing_frontier='For each fixed root-target at finest completed grid, enumerate all7 nonempty branch subsets. A subset lacking a common feasible second action prevents recovery once closed observation intervals intersect, at effective radius span(predicted z)/2. For each first action take the smallest such critical radius, then maximize over first actions. Subtract pad for epsilon supremum; finite limiting boundary excluded. If all3 have a feasible shared action, unbounded sensing error is allowed in the finite model. Descriptive finite-model sensing budget, not calibrated assay accuracy.',
                source_sha256=hashes()))
    print(json.dumps({'amendment_frozen':str(p),'sha256':sha(p)}),flush=True)


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
    return np.unique(np.minimum(RAW[::320//(n-1)],cap))


def physical(states,minimum):
    return np.all(np.isfinite(states),axis=-1)&(states[...,0]>0)&(states[...,0]<=1.8+2e-12)&(minimum>=-2e-7)


def build_tree(root,n_grid=161,deadline=None):
    t0=time.perf_counter();args=engine.arguments(root['family']);x=np.array(root['state'])
    u0=alphabet(x,n_grid);n0=len(u0);old=None;old_index={};reused=0
    lower={81:41,161:81,321:161}.get(n_grid)
    lower_path=ROOT/'results/trees'/f"{root['id']}_{lower}.npz"
    if lower is not None and lower_path.exists():
        with np.load(lower_path) as d:old={k:d[k] for k in d.files}
        old_index={float(u):i for i,u in enumerate(old['u0'])}
    first=np.zeros((n0,3,4));mins=np.zeros((n0,3));steps=rejected=calls=0
    missing=[i for i,u in enumerate(u0) if float(u) not in old_index]
    if missing:
        ff,mm,ss,rr=engine.batch(np.repeat(x[None,:],len(missing)*3,axis=0),np.repeat(u0[missing],3),np.tile(engine.K,(len(missing),1)),args)
        first[missing]=ff.reshape(len(missing),3,4);mins[missing]=mm.reshape(len(missing),3)
        steps+=ss;rejected+=rr;calls+=len(missing)*3
    if old is not None:
        for i,u in enumerate(u0):
            if float(u) in old_index:
                oi=old_index[float(u)];first[i]=old['first'][oi];mins[i]=old['first_min'][oi];reused+=3
    u1=np.zeros((n0,n_grid));count=np.zeros(n0,dtype=np.int64)
    terminal=np.zeros((n0,3,n_grid,3,4));second_min=np.zeros((n0,3,n_grid,3))
    completed_first=0
    for i in range(n0):
        if deadline is not None and time.monotonic()>=deadline:break
        us=alphabet(first[i,0],n_grid);n=len(us);u1[i,:n]=us;count[i]=n
        assert all(np.array_equal(us,alphabet(s,n_grid)) for s in first[i])
        old_second={}
        if float(u0[i]) in old_index:
            oi=old_index[float(u0[i])];old_second={float(u):j for j,u in enumerate(old['u1'][oi,:old['n1'][oi]])}
            for j,u in enumerate(us):
                if float(u) in old_second:
                    oj=old_second[float(u)];terminal[i,:,j,:,:]=old['terminal'][oi,:,oj,:,:];second_min[i,:,j,:]=old['second_min'][oi,:,oj,:];reused+=9
        new=[j for j,u in enumerate(us) if float(u) not in old_second]
        if new:
            nn=len(new);states=np.repeat(first[i],nn*3,axis=0)
            feeds=np.tile(np.repeat(us[new],3),3);ks=np.tile(engine.K,(3*nn,1))
            out,mn,ss,rr=engine.batch(states,feeds,ks,args)
            terminal[i][:,new,:,:]=out.reshape(3,nn,3,4);second_min[i][:,new,:]=mn.reshape(3,nn,3)
            steps+=ss;rejected+=rr;calls+=len(feeds)
        completed_first=i+1
        if i%20==0:print(json.dumps({'root':root['id'],'grid':n_grid,'first_actions_complete':i+1,'first_actions_total':n0}),flush=True)
    data=dict(u0=u0,first=first,first_min=mins,u1=u1,n1=count,terminal=terminal,second_min=second_min)
    complete=completed_first==n0;suffix='' if complete else '_partial'
    cache=ROOT/'results/trees'/f"{root['id']}_{n_grid}{suffix}.npz";cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,**data)
    dump(cache.with_suffix('.json'),dict(root=root,raw_grid=n_grid,complete=complete,completed_first_actions=completed_first,total_first_actions=n0,new_transitions=calls,reused_transitions=reused,accepted_steps=int(steps),rejected_steps=int(rejected),seconds=time.perf_counter()-t0,source_sha256=hashes(),npz_sha256=sha(cache)))
    return data if complete else None


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


def sensing_frontier(tree,target):
    """Closed-support frontier from common-intersection subsets, no new ODEs.

    Nonmaximal subsets add no constraint beyond a maximal realizable subset:
    an action valid for the latter is valid for the former. Thus all seven
    subsets yield the same recovery criterion as measurement-cell enumeration.
    """
    best=None;zs=math.log(target)
    masks=[m for size in (1,2,3) for m in itertools.combinations(range(3),size)]
    for i,b in enumerate(tree):
        if not np.all(b['first_valid']):continue
        critical=math.inf;limiting=[];singleton_failed=False
        centres=b['first'][:,1]-b['first'][:,2]
        for mask in masks:
            valid=np.all(b['valid'][list(mask)],axis=(0,2))
            err=np.max(np.abs(b['z'][list(mask)]-zs),axis=(0,2));err[~valid]=np.inf
            if float(np.min(err))>INNER:
                if len(mask)==1:singleton_failed=True;break
                cut=float((max(centres[list(mask)])-min(centres[list(mask)]))/2)
                if cut<critical:critical=cut;limiting=[list(mask)]
                elif cut==critical:limiting.append(list(mask))
        if singleton_failed:continue
        if best is None or critical>best['_critical']:
            best=dict(_critical=critical,first_action_index=i,first_feed=b['u0'],limiting_subsets=limiting)
    if best is None:return dict(target=target,zero_error_ratio_recoverable=False,status='no_full_state_recovery')
    critical=best.pop('_critical')
    if math.isinf(critical):return dict(target=target,**best,status='common_open_succeeds',zero_error_ratio_recoverable=True,epsilon_supremum=None,effective_radius_supremum=None,unbounded=True,boundary_included=True)
    eps=critical-PAD
    return dict(target=target,**best,status='positive_sensing_budget' if eps>0 else 'not_recoverable_with_pad',zero_error_ratio_recoverable=eps>0,epsilon_supremum=max(0.,eps),effective_radius_supremum=critical,unbounded=False,boundary_included=False)


def check_locks():
    protocol=ROOT/'protocols/grid_observation_amendment.json'
    locked=json.loads(protocol.read_text())
    if hashes()!=locked['source_sha256']:raise RuntimeError('Protocol source lock mismatch')
    gate_path=ROOT/'audit/grid_observation_integrator_gate.json'
    gate=json.loads(gate_path.read_text())
    if not gate.get('passed'):raise RuntimeError('Independent integrator gate did not pass')
    for name in ('scripts/engine.py','vendor/prior_kernel.py','data/flux_table_eps1e-06.json','data/prescribed_roots.json'):
        if gate.get('source_sha256',{}).get(name)!=sha(ROOT/name):
            raise RuntimeError('Independent gate source lock mismatch: '+name)
    return protocol,gate_path


def load_or_build(root,n,deadline=None):
    cache=ROOT/'results/trees'/f"{root['id']}_{n}.npz"
    if cache.exists():
        meta=json.loads(cache.with_suffix('.json').read_text())
        if meta['source_sha256']!=hashes() or meta['npz_sha256']!=sha(cache):raise RuntimeError('Cache lock mismatch')
        with np.load(cache) as d:return {k:d[k] for k in d.files}
    return build_tree(root,n,deadline)


def analyse_grid(root,tree,n):
    records=[];checks=[];sub=subset_tree(tree,n)
    modes=[('full',None),('open',None)]+[('ratio',e) for e in EPSILONS]
    local={}
    for mode,eps in modes:
        sets=target_sets(sub,mode,eps);policies=[fixed_policy(sub,t,mode,eps) for t in FIXED]
        record=dict(root=root['id'],family=root['family'],volume=root['volume'],grid=n,mode=mode,epsilon=eps,**sets,
                    fixed_policies=policies,all_path_fixed_inner=sum(p.get('inner',False) for p in policies),
                    all_path_fixed_outer=sum(p.get('outer',False) for p in policies),
                    all_path_fixed_actual=sum(p.get('actual',False) for p in policies))
        records.append(record);local[mode,eps]=record
    full=local['full',None]['intervals'];opened=local['open',None]['intervals'];prev=full
    for eps in EPSILONS:
        obs=local['ratio',eps]['intervals']
        checks.append(dict(root=root['id'],grid=n,epsilon=eps,open_contained=contained(opened,obs),observation_contained_in_full=contained(obs,full),epsilon_monotonic=contained(obs,prev)))
        prev=obs
    frontiers=[sensing_frontier(sub,t) for t in FIXED]
    return dict(root=root,grid=n,records=records,algebra_checks=checks,sensing_frontiers=frontiers,complete=True)


def compare_records(records,coarse,fine):
    local={(r['mode'],r['epsilon'],r['grid']):r for r in records};out=[]
    for mode,eps in [('full',None),('open',None)]+[('ratio',e) for e in EPSILONS]:
        a=local[mode,eps,coarse]['intervals'];b=local[mode,eps,fine]['intervals']
        out.append(dict(root=records[0]['root'],mode=mode,epsilon=eps,coarse=coarse,fine=fine,nested_containment=contained(a,b),**convergence(a,b)))
    return out


def run_root(root,adaptive=False,deadline=None):
    check_locks();t0=time.perf_counter();records=[];checks=[];frontiers=[]
    sizes=(321,) if adaptive else SIZES
    for n in sizes:
        tree=load_or_build(root,n,deadline)
        if tree is None:
            failure=dict(root=root['id'],grid=n,complete=False,reason='Adaptive wall-clock budget exhausted',seconds=time.perf_counter()-t0)
            dump(ROOT/'results'/f"grid_observation_{root['id']}_adaptive_unresolved.json",failure)
            return failure
        result=analyse_grid(root,tree,n)
        dump(ROOT/'results'/f"grid_observation_{root['id']}_grid{n}.json",result)
        records.extend(result['records']);checks.extend(result['algebra_checks']);frontiers=result['sensing_frontiers']
        print(json.dumps({'root_grid_completed':root['id'],'grid':n,'seconds':time.perf_counter()-t0}),flush=True)
    comparisons=[]
    if adaptive:
        previous=json.loads((ROOT/'results'/f"grid_observation_{root['id']}_grid161.json").read_text())
        comparisons=compare_records(previous['records']+records,161,321)
    else:
        for a,b in zip(SIZES[:-1],SIZES[1:]):comparisons.extend(compare_records(records,a,b))
    result=dict(root=root,records=records,comparisons=comparisons,algebra_checks=checks,sensing_frontiers=frontiers,complete=True,seconds=time.perf_counter()-t0)
    suffix='_adaptive' if adaptive else ''
    dump(ROOT/'results'/f"grid_observation_{root['id']}{suffix}.json",result)
    return result


def run():
    protocol,gate_path=check_locks();t0=time.perf_counter();parts=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(run_root,r):r['id'] for r in roots()}
        for fut in concurrent.futures.as_completed(futures):
            parts.append(fut.result())
            print(json.dumps({'root_completed':futures[fut],'seconds_elapsed':time.perf_counter()-t0}),flush=True)
    records=[r for p in parts for r in p['records']];comparisons=[r for p in parts for r in p['comparisons']];checks=[r for p in parts for r in p['algebra_checks']]
    passed=all(all(c[k] for k in ['open_contained','observation_contained_in_full','epsilon_monotonic']) for c in checks) and all(c['nested_containment'] for c in comparisons)
    result=dict(complete=True,source_sha256=hashes(),protocol_sha256=sha(protocol),integrator_gate_sha256=sha(gate_path),
                records=records,convergence=comparisons,algebra_checks=checks,algebra_passed=passed,
                sensing_frontiers=[dict(root=p['root']['id'],grid=161,targets=p['sensing_frontiers']) for p in parts],
                elapsed_seconds=time.perf_counter()-t0,interpretation='No global continuous-control or biological validation claim; convergence gates are descriptive.')
    dump(ROOT/'results/grid_observation.json',result)
    print(json.dumps({'primary_complete':True,'algebra_passed':passed,'seconds':result['elapsed_seconds']}),flush=True)
    triggered=sorted(set(c['root'] for c in comparisons if c['fine']==161 and not c['passed']))
    start=time.monotonic();deadline=start+3600;adaptive=[]
    dump(ROOT/'results/grid_observation_adaptive_assignment.json',dict(triggered_roots=triggered,deadline_seconds=3600,primary_result_sha256=sha(ROOT/'results/grid_observation.json')))
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(run_root,r,True,deadline):r['id'] for r in roots() if r['id'] in triggered}
        for fut in concurrent.futures.as_completed(futures):adaptive.append(fut.result())
    dump(ROOT/'results/grid_observation_adaptive.json',dict(complete=all(p['complete'] for p in adaptive),triggered_roots=triggered,results=adaptive,wall_seconds=time.monotonic()-start,source_sha256=hashes()))
    print(json.dumps({'adaptive_complete':all(p['complete'] for p in adaptive),'triggered_roots':triggered,'seconds':time.monotonic()-start}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--amend',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.amend:amend()
    elif a.run:run()
    else:p.error('Choose --freeze, --amend or --run')
