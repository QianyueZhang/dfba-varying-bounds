"""Independent finite-game checks without reactor integration or outcome tuning."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,itertools,json,math,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import grid_observation as model


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new(path,data):
    with path.open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')


def masks_by_subset_coverage(centers,radius):
    """An exact mask exists iff its common support is not covered by outsiders.

    Independent of endpoint/open-cell enumeration in the implementation.
    """
    result=[]
    for size in [1,2,3]:
        for subset in itertools.combinations(range(3),size):
            lo=max(centers[i]-radius for i in subset);hi=min(centers[i]+radius for i in subset)
            if lo>hi:continue
            excluded=sorted((centers[i]-radius,centers[i]+radius) for i in range(3) if i not in subset)
            merged=[]
            for a,b in excluded:
                if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
                else:merged.append([a,b])
            if not any(a<=lo and hi<=b for a,b in merged):result.append(subset)
    return sorted(result)


def brute_game(tree,target,mode,epsilon):
    best=math.inf;z=math.log(target)
    for branch in tree:
        if not np.all(branch['first_valid']):continue
        if mode=='open':masks=[(0,1,2)]
        elif mode=='full':
            groups={}
            for j,state in enumerate(branch['first']):groups.setdefault(tuple(state),[]).append(j)
            masks=[tuple(group) for group in groups.values()]
        else:masks=masks_by_subset_coverage(branch['first'][:,1]-branch['first'][:,2],epsilon+model.PAD)
        for assignment in itertools.product(range(len(branch['u1'])),repeat=len(masks)):
            worst=0.
            for mask,action in zip(masks,assignment):
                if not all(branch['valid'][j,action,k] for j in mask for k in range(3)):
                    worst=math.inf;break
                worst=max(worst,max(abs(float(branch['z'][j,action,k])-z) for j in mask for k in range(3)))
            best=min(best,worst)
    return best


def main():
    protocol=ROOT/'protocols/audit_grid_observation_algebra.json'
    result=ROOT/'audit/grid_observation_algebra_results.json'
    sources=['scripts/grid_observation.py','scripts/engine.py','audit/grid_observation_algebra_review.py']
    if protocol.exists() or result.exists():raise FileExistsError('Algebra review already declared/run')
    lock={p:sha(ROOT/p) for p in sources}
    new(protocol,{'declared_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':lock,
        'seed':20261024,'random_games':30,'first_actions':2,'second_actions':2,'branches':3,'last_affinities':3,
        'oracle':'Exhaustive enumeration of every second-action mapping per compatible subset, and direct subset/common-support coverage for observation geometry.',
        'scope':'Synthetic finite-game identities only; no ODE integrations or reactor outcome selection.'})
    failures=[];counts={'mask_geometries':0,'finite_games':0,'set_memberships':0,'frontier_checks':0,'api_cell_checks':0}
    for centers in [[0.,0.,0.],[0.,.02,.04],[0.,0.,.02],[-.04,0.,.04],[0.,.01,.02],[1.,1.,1.1]]:
        for radius in [0.,1e-7,.005,.01,.02,.05]:
            expected=masks_by_subset_coverage(centers,radius)
            actual,cells=model.compatible_masks(centers,radius);counts['mask_geometries']+=1
            if actual!=expected:failures.append({'kind':'mask_geometry','centers':centers,'radius':radius,'expected':expected,'actual':actual})
    rng=np.random.default_rng(20261024)
    for game in range(30):
        tree=[]
        for i in range(2):
            centers=np.array([0.,.012,.024])+rng.normal(0,.002,3)
            first=np.c_[np.full(3,1.+.1*i),centers/2,-centers/2,np.full(3,10.)]
            z=rng.uniform(-.022,.022,(3,2,1))+rng.normal(0,.001,(3,2,3))
            valid=rng.random((3,2,3))>.025
            tree.append({'u0':float(i*.1),'first':first,'u1':np.array([0.,.1]),'z':z,
                         'biomass':np.full((3,2,3),2.),'valid':valid,'first_valid':np.ones(3,dtype=bool)})
        for target in [.99,1.,1.01]:
            for mode,epsilon in [('full',None),('open',None),('ratio',0.),('ratio',.005),('ratio',.02)]:
                expected=brute_game(tree,target,mode,epsilon);policy=model.fixed_policy(tree,target,mode,epsilon)
                actual=math.inf if policy.get('unresolved') else policy['worst_error'];counts['finite_games']+=1
                if not (actual==expected or (math.isfinite(actual) and math.isfinite(expected) and abs(actual-expected)<=1e-15)):
                    failures.append({'kind':'minimax','game':game,'target':target,'mode':mode,'epsilon':epsilon,'expected':None if math.isinf(expected) else expected,'actual':None if math.isinf(actual) else actual})
                sets=model.target_sets(tree,mode,epsilon)['intervals'];contains=model.contains(sets,math.log(target));counts['set_memberships']+=1
                if contains!=(expected<=model.INNER):failures.append({'kind':'target_set','game':game,'target':target,'mode':mode,'epsilon':epsilon})
                if mode=='ratio' and not policy.get('unresolved'):
                    for cell in policy['measurement_cells']:
                        feed=model.choose_second(policy,cell['representative']);counts['api_cell_checks']+=1
                        choices=[a for a in policy['second_actions'] if a['mask']==cell['mask']]
                        if len(choices)!=1 or choices[0]['feed']!=feed:failures.append({'kind':'policy_api','game':game,'target':target,'epsilon':epsilon,'cell':cell})
            frontier=model.sensing_frontier(tree,target)
            for epsilon in [0.,.0005,.001,.005,.01,.02,.1]:
                p=model.fixed_policy(tree,target,'ratio',epsilon)
                actual=False if p.get('unresolved') else p['inner']
                predicted=(frontier['status']=='common_open_succeeds' or
                           (frontier['status']=='positive_sensing_budget' and epsilon<frontier['epsilon_supremum']))
                counts['frontier_checks']+=1
                if predicted!=actual:failures.append({'kind':'sensing_frontier','game':game,'target':target,'epsilon':epsilon,'frontier':frontier,'actual':actual})
    unchanged={p:sha(ROOT/p) for p in sources}==lock
    out={'complete':True,'passed':not failures and unchanged,'source_sha256':lock,'sources_unchanged':unchanged,
         'protocol_sha256':sha(protocol),'counts':counts,'failures':failures,
         'interpretation':'Independent synthetic algebra checks; no reactor outcome validation and no floating-point boundary enclosure.'}
    new(result,out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
