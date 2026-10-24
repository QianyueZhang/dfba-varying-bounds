"""Post hoc set-distance diagnostic; never changes frozen stability gates."""
from pathlib import Path
import argparse, hashlib, json, math, time

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def member(intervals,x):return any(a<=x<=b for a,b in intervals)


def distance(x,intervals):
    if member(intervals,x):return 0.
    return min(abs(x-v) for ab in intervals for v in ab)


def directed(a,b):
    """Exact candidate formula for closed finite unions, evaluated in float64."""
    candidates=[v for ab in a for v in ab]
    for left,right in zip(b[:-1],b[1:]):
        midpoint=left[1]+(right[0]-left[1])/2
        if member(a,midpoint):candidates.append(midpoint)
    return max(distance(x,b) for x in candidates)


def hausdorff(a,b):
    if not a and not b:return dict(status='both_empty',distance=0.,positive_evidence=False)
    if not a or not b:return dict(status='one_empty',distance=None,extended_distance='infinity',positive_evidence=False)
    ab=directed(a,b);ba=directed(b,a)
    return dict(status='nonempty',distance=max(ab,ba),coarse_to_fine=ab,fine_to_coarse=ba,positive_evidence=True)


def freeze():
    path=ROOT/'protocols/diagnostic_set_hausdorff.json'
    if path.exists():raise FileExistsError('Existing diagnostic protocol must be retained')
    write(path,dict(post_hoc=True,
                    declared_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    motivation='After inspecting some81→161 outcomes, endpoint Hausdorff appeared sensitive to component mergers. This separate diagnostic measures distance between all points in the target sets.',
                    scope='Every completed primary grid pair and every completed adaptive161→321 pair, all6roots and all7information modes. No outcome-based subset selection.',
                    formula='For directed A→B evaluate distance to B at every endpoint of A and every midpoint of a gap of B that belongs to A; take the maximum. Symmetric distance is max of both directions.',
                    preservation='Do not replace frozen endpoint criterion, relative symmetric-difference criterion, primary convergence labels, or adaptive root selection.',
                    interpretation='Geometry of the computed finite target sets only. No new ODE computation, extrapolation to continuous actions or exact dynamics.',
                    script_sha256=sha(__file__)))


def run():
    protocol=ROOT/'protocols/diagnostic_set_hausdorff.json'
    p=json.loads(protocol.read_text())
    if p['script_sha256']!=sha(__file__):raise RuntimeError('Diagnostic source changed after freeze')
    # Known geometry checks include disappearance of internal component endpoints.
    tests=[([[0.,.49],[.51,1.]],[[0.,1.]],.01),
           ([[0.,0.]],[[1.,1.]],1.),([[0.,1.]],[[0.,2.]],1.),
           ([[0.,1.]],[[3.,4.]],3.),([[0.,2.]],[[.5,1.5]],.5)]
    for a,b,expected in tests:
        if abs(hausdorff(a,b)['distance']-expected)>1e-14:raise AssertionError((a,b))
    primary_path=ROOT/'results/grid_observation.json'
    primary=json.loads(primary_path.read_text());records=list(primary['records']);comparisons=list(primary['convergence'])
    inputs={str(primary_path.relative_to(ROOT)):sha(primary_path)}
    ap=ROOT/'results/grid_observation_adaptive.json'
    if ap.exists():
        inputs[str(ap.relative_to(ROOT))]=sha(ap)
        for result in json.loads(ap.read_text())['results']:
            if result['complete']:records.extend(result['records']);comparisons.extend(result['comparisons'])
    lookup={(r['root'],r['mode'],r['epsilon'],r['grid']):r for r in records}
    out=[]
    for c in comparisons:
        key=(c['root'],c['mode'],c['epsilon'])
        a=lookup[*key,c['coarse']]['intervals'];b=lookup[*key,c['fine']]['intervals']
        d=hausdorff(a,b)
        out.append(dict(root=c['root'],mode=c['mode'],epsilon=c['epsilon'],coarse=c['coarse'],fine=c['fine'],
                        original_gate_passed=c['passed'],relative_symmetric_difference=c.get('relative_symmetric_difference'),
                        endpoint_hausdorff=c['endpoint_hausdorff'],set_hausdorff=d,
                        diagnostic_set_distance_at_most_001=None if d['distance'] is None else d['distance']<=.001))
    write(ROOT/'results/diagnostic_set_hausdorff.json',dict(complete=True,post_hoc=True,protocol_sha256=sha(protocol),script_sha256=sha(__file__),input_sha256=inputs,geometry_checks_passed=len(tests),records=out))
    print(json.dumps({'complete':True,'diagnostic_records':len(out)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.run:run()
    else:p.error('Choose --freeze or --run')
