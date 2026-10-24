"""Read-only verification of all computed interval sets, with explicit topology."""
from pathlib import Path
import sys,json,hashlib,math
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from target_sets import union,intersection

def subset(a,b):
    return all(any(c<=lo and hi<=d for c,d in b) for lo,hi in a)

def difference(a,b):
    output=[]
    for lo,hi in a:
        pieces=[dict(lo=lo,hi=hi,left_closed=True,right_closed=True)]
        for c,d in b:
            nxt=[]
            for p in pieces:
                l,h=p['lo'],p['hi']
                if d<l or c>h:nxt.append(p);continue
                if c>l:nxt.append(dict(lo=l,hi=min(c,h),left_closed=p['left_closed'],right_closed=False))
                if d<h:nxt.append(dict(lo=max(d,l),hi=h,left_closed=False,right_closed=p['right_closed']))
            pieces=nxt
        output.extend(pieces)
    return output

def main():
    primary=json.loads((ROOT/'results/target_sets.json').read_text());extra=json.loads((ROOT/'results/target_set_sensitivity.json').read_text());rows=[]
    for r in primary+extra:
        a,b=r['feedback'],r['open'];assert a==union(a) and b==union(b)
        assert subset(b,a)
        diff=difference(a,b)
        assert abs(sum(p['hi']-p['lo'] for p in diff)-(r['feedback_log_length']-r['open_log_length']))<1e-12
        shifts=[]
        for t in (.5,.75,1.,1.25,1.5,2.):
            z=math.log(t);dist=min((max(lo-z,z-hi,0.) for lo,hi in a),default=None)
            shifts.append(dict(requested_ratio=t,minimum_absolute_log_target_change=dist))
        rows.append(dict(root=r['root'],grid=r['grid'],open_subset_feedback=True,feedback_minus_open=diff,target_change=shifts))
    # Nested-grid set inclusion, using exact stored interval endpoints.
    nested=[]
    for r in primary:
        if not r['root'].endswith('uniform'):continue
        small=next(x for x in extra if x['root']==r['root'] and x['grid']==11)
        large=next(x for x in extra if x['root']==r['root'] and x['grid']==41)
        for key in ('feedback','open'):
            ok1=subset(small[key],r[key]);ok2=subset(r[key],large[key]);assert ok1 and ok2
            nested.append(dict(root=r['root'],set=key,eleven_in_twentyone=ok1,twentyone_in_fortyone=ok2))
    inputs=list((ROOT/'results').glob('tree_*.npz'))+[ROOT/'results/roots.json',ROOT/'results/target_sets.json',ROOT/'results/target_set_sensitivity.json',ROOT/'EXPLORATORY_REFINEMENT_AMENDMENT.md',Path(__file__)]
    out=dict(passed=True,records=rows,nested=nested,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},interpretation='Set algebra on numerical transitions; no ODE enclosure or verification of all interval targets.')
    (ROOT/'audit/target_algebra_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print('Passed',len(rows),'direct interval containment checks and',len(nested)*2,'nested inclusions.')

if __name__=='__main__':main()
