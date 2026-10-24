"""Exploratory complete target-set algebra on numerical finite-game trees."""
import json,hashlib,time
import numpy as np
import engine as e
from policies import INNER,PHYSICAL,observable,decision
from campaign import load_tree

def union(intervals):
    ordered=sorted((float(lo),float(hi)) for lo,hi in intervals if lo<=hi)
    out=[]
    for lo,hi in ordered:
        if out and lo<=out[-1][1]:out[-1][1]=max(hi,out[-1][1])
        else:out.append([lo,hi])
    return out

def intersection(a,b):
    out=[];i=j=0
    while i<len(a) and j<len(b):
        lo=max(a[i][0],b[j][0]);hi=min(a[i][1],b[j][1])
        if lo<=hi:out.append([lo,hi])
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return union(out)

def interval(z):return [float(np.max(z)-INNER),float(np.min(z)+INNER)]

def calculate(t):
    fb=[];ol=[];per_action=[]
    for i,branch in enumerate(t['branches']):
        z,b=observable(branch['terminal']);valid=(branch['min_S'].min(axis=2)>=PHYSICAL)&(branch['terminal'][...,0].max(axis=2)<=e.CAP)
        valid &= t['first']['min_S'][i,:,None]>=PHYSICAL
        aset=None
        for k in range(len(e.K)):
            local=union([interval(z[k,j,:]) for j in range(z.shape[1]) if valid[k,j]])
            aset=local if aset is None else intersection(aset,local)
        per_action.append(aset);fb.extend(aset)
        ol.extend([interval(z[:,j,:]) for j in range(z.shape[1]) if valid[:,j].all()])
    fb=union(fb);ol=union(ol)
    common=intersection(fb,ol)
    def length(a):return float(sum(hi-lo for lo,hi in a))
    assert abs(length(common)-length(ol))<1e-12
    return dict(feedback=fb,open=ol,by_first_action=per_action,feedback_log_length=length(fb),open_log_length=length(ol),feedback_ratio_length=float(sum(np.exp(hi)-np.exp(lo) for lo,hi in fb)),open_ratio_length=float(sum(np.exp(hi)-np.exp(lo) for lo,hi in ol)))

def main():
    roots=json.loads((e.ROOT/'results/roots.json').read_text());out=[];witnesses=[]
    lock={'protocol_sha256':hashlib.sha256((e.ROOT/'EXPLORATORY_TARGET_SET_PROTOCOL.md').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()}
    e.save(e.ROOT/'audit/target_set_lock.json',lock)
    for root in roots:
        tree=load_tree(e.ROOT/'results'/f"tree_{root['id']}.npz");sets=calculate(tree);out.append(dict(root=root['id'],grid=21,**sets))
        for i,(lo,hi) in enumerate(sets['feedback']):
            target=float(np.exp((lo+hi)/2))
            choices={m:decision(np.array(root['state']),e.arguments(root['family']),target,2,m,tree) for m in ('recovery_economic','recovery_filter')}
            assert all(c['recoverable'] for c in choices.values())
            witnesses.append(dict(root=root['id'],component=i,log_interval=[lo,hi],target=target,choices=choices))
        print(root['id'],'components',len(sets['feedback']),'length',sets['feedback_log_length'],flush=True)
    e.save(e.ROOT/'results/target_sets.json',out);e.save(e.ROOT/'results/target_set_witnesses.json',witnesses)
    sensitivity=[]
    for root in roots:
        if root['timing']!='uniform':continue
        for n in (11,41):
            e.GRID=np.linspace(0,.2,n);t=e.tree(np.array(root['state']),e.arguments(root['family']));sensitivity.append(dict(root=root['id'],grid=n,**calculate(t)))
    e.GRID=np.linspace(0,.2,21);e.save(e.ROOT/'results/target_set_sensitivity.json',sensitivity)

if __name__=='__main__':main()
