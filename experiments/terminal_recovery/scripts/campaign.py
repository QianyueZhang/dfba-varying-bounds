"""Frozen root panel and exhaustive recovery maps. See PROTOCOL.json."""
import argparse,hashlib,json,time,sys
from pathlib import Path
import numpy as np
import engine as e
from policies import two_values,decision,one_values,INNER
ROOT=e.ROOT
TARGETS=[.5,.75,1.,1.25,1.5,2.]
METHODS=['nominal_replan','open_grid','open_replan','recovery_economic','recovery_filter']

def roots():
    output=[]
    for family in ('analytic','table'):
        args=e.arguments(family)
        for volume in (.9,1.3,1.7):
            total=(volume-.5)/2;remaining=total;early=[]
            for j in range(5):
                u=min(.2,remaining);early.append(u);remaining-=u
            for timing,feeds in [('early',early),('late',early[::-1]),('uniform',[total/5]*5)]:
                x=e.initial();knots=[x.copy()]
                for u in feeds:x=e.flow(x,u,e.K[0],args)[0];knots.append(x.copy())
                assert abs(x[0]-volume)<1e-13
                output.append(dict(id=f'{family}_v{volume:.1f}_{timing}',family=family,volume=volume,timing=timing,feeds=feeds,state=x,knots=knots))
    return output

def save_tree(path,t):
    fields={'actions0':t['first']['actions'],'first':t['first']['terminal'],'first_minS':t['first']['min_S']}
    for i,b in enumerate(t['branches']):
        for key in ('actions','terminal','min_S'):fields[f'{i}_{key}']=b[key]
    np.savez_compressed(path,**fields)

def load_tree(path):
    a=np.load(path);branches=[]
    for i in range(len(a['actions0'])):branches.append({k:a[f'{i}_{k}'] for k in ('actions','terminal','min_S')})
    return {'first':{'actions':a['actions0'],'terminal':a['first'],'min_S':a['first_minS']},'branches':branches}

def main():
    panel=roots();e.save(ROOT/'results/roots.json',panel);allrows=[];maps=[]
    for root in panel:
        tick=time.perf_counter();args=e.arguments(root['family']);x=np.array(root['state']);t=e.tree(x,args)
        save_tree(ROOT/'results'/f"tree_{root['id']}.npz",t)
        for target in TARGETS:
            v=two_values(t,target);out={key:v[key] for key in ('q','W_by_action','safe','J','W','open','nominal')}
            choices={method:decision(x,args,target,2,method,t) for method in METHODS}
            allrows.append(dict(root=root['id'],target=target,values=out,choices=choices))
        targets=np.exp(np.linspace(np.log(.25),np.log(4.),161));j=[];jo=[];safe=[]
        for target in targets:
            v=two_values(t,target);j.append(v['J']);jo.append(v['open']['J']);safe.append(len(v['safe']))
        maps.append(dict(root=root['id'],targets=targets,J_feedback=j,J_open=jo,safe_actions=safe,transitions=t['transitions'],seconds=time.perf_counter()-tick))
        e.save(ROOT/'results/primary_design.json',allrows);e.save(ROOT/'results/target_maps.json',maps)
        print(root['id'],f"{time.perf_counter()-tick:.2f}s",flush=True)
    # Nested grids on the six predeclared uniform-history roots; all fixed targets.
    sensitivity=[]
    for root in panel:
        if root['timing']!='uniform':continue
        for n in (11,41):
            e.GRID=np.linspace(0,.2,n);t=e.tree(np.array(root['state']),e.arguments(root['family']))
            for target in TARGETS:
                v=two_values(t,target);sensitivity.append(dict(root=root['id'],target=target,grid=n,J=v['J'],J_open=v['open']['J'],safe_count=len(v['safe']),W=v['W']))
            print('sensitivity',root['id'],n,flush=True)
    e.GRID=np.linspace(0,.2,21);e.save(ROOT/'results/grid_sensitivity.json',sensitivity)

if __name__=='__main__':main()
