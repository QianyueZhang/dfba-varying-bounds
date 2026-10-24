"""Frozen, best-found continuous minimax comparators; no global certificate.

All controls use a capacity-respecting unit-box parameterization. The causal
tree has one shared first feed and state-dependent second feeds. Deployment
recomputes the second feed using only the observed full state and a common
seed list. The true disturbance index is never passed to that controller.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar
from engine import CAP, GUARD, K, BAND, ACCEPT, flow

INNER = .00748
TOL = 1e-10
POPULATION = 24
DE_GENERATIONS = 24
LOCAL_ITERATIONS = 100
DE_SEEDS = [20261024, 20261025]


def cap(state):
    return max(0., min(.2, (CAP-float(state[0])-GUARD)/2))


def physical(state, minimum):
    return bool(np.all(np.isfinite(state)) and state[0] <= CAP+1e-13
                and state[0] > 0 and minimum >= -2e-7)


def labels(error):
    return dict(inner=bool(error <= INNER), outer=bool(error <= BAND),
                actual=bool(error <= ACCEPT))


def solve(state, args, target, mode, grid_feeds, open_incumbent=None):
    started=time.perf_counter(); state=np.asarray(state,float)
    dim=2 if mode=='open' else 4; logtarget=float(np.log(target))
    records=[]; cache={}; statuses=[]; origin='initial'
    counts=dict(evaluation_requests=0, cache_hits=0, transitions=0,
                accepted_steps=0, rejected_steps=0)
    c0=cap(state)

    def encode(feeds):
        feeds=np.asarray(feeds,float); c1=cap(state+np.array([2*feeds[0],0,0,0]))
        return np.clip(np.r_[feeds[0]/c0 if c0 else 0,
                             feeds[1:]/c1 if c1 else feeds[1:]*0],0,1)

    def decode(value):
        value=np.clip(np.asarray(value,float),0,1); u0=float(value[0]*c0)
        c1=cap(state+np.array([2*u0,0,0,0]))
        return np.r_[u0,value[1:]*c1]

    def transition(x,u,k):
        out,minimum,steps,rejected=flow(x,float(u),k,args)
        counts['transitions']+=1;counts['accepted_steps']+=int(steps)
        counts['rejected_steps']+=int(rejected)
        return out,float(minimum)

    def evaluate(value,feed_override=None):
        counts['evaluation_requests']+=1
        value=np.clip(np.asarray(value,float),0,1)
        key=(value.tobytes(),None if feed_override is None else np.asarray(feed_override,float).tobytes())
        if key in cache:
            counts['cache_hits']+=1; row=cache[key]
            if origin not in row['origins']: row['origins'].append(origin)
            return row
        feeds=decode(value) if feed_override is None else np.asarray(feed_override,float)
        row=dict(index=len(records),x=value.tolist(),feeds=feeds.tolist(),origins=[origin],
                 valid=False,error=None,causal_state_distinct=True)
        try:
            first=[];minimum=float(state[3]/state[0])
            for k in K:
                y,m=transition(state,feeds[0],k);first.append(y);minimum=min(minimum,m)
            first=np.asarray(first);terminal=[]
            # Exact identical observations cannot legitimately receive different
            # controls. Such a trial is retained but excluded from selection.
            causal_valid=True
            if mode=='causal':
                for i in range(3):
                    for j in range(i):
                        if np.array_equal(first[i],first[j]) and feeds[i+1]!=feeds[j+1]:
                            causal_valid=False
                row['causal_state_distinct']=len({x.tobytes() for x in first})==3
            for i,y in enumerate(first):
                u1=feeds[1] if mode=='open' else feeds[i+1]
                for k in K:
                    z,m=transition(y,u1,k);terminal.append(z);minimum=min(minimum,m)
            terminal=np.asarray(terminal);diff=terminal[:,1]-terminal[:,2]-logtarget
            valid=causal_valid and all(physical(y,minimum) for y in np.r_[first,terminal])
            row.update(valid=bool(valid),causal_valid=bool(causal_valid),
                       max_error=float(np.max(np.abs(diff))),differences=diff.tolist(),
                       worst_logB=float(np.min(np.logaddexp(terminal[:,1],terminal[:,2]))),
                       min_S=minimum,first_states=first.tolist(),terminal_states=terminal.tolist(),
                       first_state_min_distance=float(min(np.linalg.norm(first[i]-first[j])
                                                         for i in range(3) for j in range(i))))
            row.update(labels(row['max_error']))
        except Exception as exc: row['error']=repr(exc)
        records.append(row);cache[key]=row
        return row

    def rank(row):
        return (not row['valid'],row.get('max_error',np.inf),-row.get('worst_logB',-np.inf),row['index'])

    def objective(value):
        row=evaluate(value)
        return row['max_error'] if row['valid'] else 1e3

    origin='refined_grid41_incumbent';grid=encode(grid_feeds);evaluate(grid,grid_feeds)
    deterministic=[grid,np.full(dim,.5),np.zeros(dim),np.ones(dim)]
    if mode=='causal' and open_incumbent is not None:
        origin='embedded_best_found_common_open'
        common=encode([open_incumbent['feeds'][0]]+[open_incumbent['feeds'][1]]*3)
        evaluate(common,[open_incumbent['feeds'][0]]+[open_incumbent['feeds'][1]]*3)
        deterministic.append(common)
    for value in deterministic:
        origin='deterministic_seed';evaluate(value)
    de_best=[]
    for seed in DE_SEEDS:
        origin='differential_evolution_'+str(seed)
        rng=np.random.default_rng(seed);population=rng.random((POPULATION,dim))
        for i,value in enumerate(deterministic):population[i]=value
        tick=time.perf_counter()
        try:
            fit=differential_evolution(objective,[(0.,1.)]*dim,seed=seed,
                init=population,maxiter=DE_GENERATIONS,tol=0,atol=0,
                polish=False,updating='immediate',workers=1)
            row=evaluate(fit.x);de_best.append(np.asarray(row['x']))
            statuses.append(dict(stage='DE',seed=seed,success=bool(fit.success),
                message=str(fit.message),iterations=int(fit.nit),nfev=int(fit.nfev),
                candidate_index=row['index'],seconds=time.perf_counter()-tick))
        except Exception as exc:
            statuses.append(dict(stage='DE',seed=seed,success=False,error=repr(exc),
                                 seconds=time.perf_counter()-tick))
            de_best.append(np.asarray(min(records,key=rank)['x']))
    # Exactly five declared local starts: grid, both DE endpoints, midpoint,
    # and the current best retained candidate (may duplicate another start).
    starts=[grid]+de_best+[np.full(dim,.5),np.asarray(min(records,key=rank)['x'])]
    for number,start in enumerate(starts):
        origin='SLSQP_start_'+str(number);row=evaluate(start);tick=time.perf_counter()
        slack=row.get('max_error',1e3)
        def constraint(value):
            trial=evaluate(value[:dim])
            if not trial['valid']:return np.full(18,-1e3)
            diff=np.asarray(trial['differences'])
            return np.r_[value[-1]-diff,value[-1]+diff]
        try:
            fit=minimize(lambda v:float(v[-1]),np.r_[start,slack],method='SLSQP',
                bounds=[(0.,1.)]*dim+[(0.,None)],constraints=[dict(type='ineq',fun=constraint)],
                options=dict(maxiter=LOCAL_ITERATIONS,ftol=1e-11,eps=1e-6))
            row=evaluate(fit.x[:dim])
            statuses.append(dict(stage='SLSQP',start=number,success=bool(fit.success),
                status=int(fit.status),message=str(fit.message),iterations=int(fit.nit),
                nfev=int(fit.nfev),candidate_index=row['index'],seconds=time.perf_counter()-tick))
        except Exception as exc:
            statuses.append(dict(stage='SLSQP',start=number,success=False,error=repr(exc),
                                 seconds=time.perf_counter()-tick))
    selected=min(records,key=rank)
    counts['unique_candidates']=len(records)
    counts['invalid_candidates']=sum(not r['valid'] for r in records)
    # Preserve every evaluated control and metric, with complete states for the
    # selected plan; individual trials are replayable from root and feeds.
    ledger=[{k:v for k,v in r.items() if k not in ('first_states','terminal_states')}
            for r in records]
    return dict(mode=mode,target=target,selected=selected,solver_statuses=statuses,
                costs=counts,seconds=time.perf_counter()-started,candidates=ledger)


def causal_second(state,args,target,seed_feeds):
    """Observation-only minimax continuation; all supplied seeds are common.

    Fixed161-point scan + one bounded refinement around every discrete local
    minimum. This is an evaluated-candidate policy, not a scalar global proof.
    """
    start=time.perf_counter();state=np.asarray(state,float); upper=cap(state)
    logtarget=float(np.log(target));cache={};records=[];statuses=[]
    costs=dict(requests=0,transitions=0,accepted_steps=0,rejected_steps=0)
    def evaluate(u):
        costs['requests']+=1;u=float(np.clip(u,0,upper))
        if u in cache:return cache[u]
        row=dict(feed=u,valid=False,error=None)
        try:
            terminal=[];minimum=float(state[3]/state[0])
            for k in K:
                y,m,s,r=flow(state,u,k,args);terminal.append(y);minimum=min(minimum,float(m))
                costs['transitions']+=1;costs['accepted_steps']+=int(s);costs['rejected_steps']+=int(r)
            terminal=np.asarray(terminal);row.update(valid=all(physical(y,minimum) for y in terminal),
                max_error=float(np.max(np.abs(terminal[:,1]-terminal[:,2]-logtarget))),
                worst_logB=float(np.min(np.logaddexp(terminal[:,1],terminal[:,2]))),min_S=minimum)
        except Exception as exc: row['error']=repr(exc)
        records.append(row);cache[u]=row;return row
    def value(u):
        row=evaluate(u);return row['max_error'] if row['valid'] else 1e3
    grid=np.linspace(0,upper,161)
    errors=[value(u) for u in grid]
    for u in seed_feeds:evaluate(u)
    for i in range(1,len(grid)-1):
        if errors[i]<=errors[i-1] and errors[i]<=errors[i+1] and (errors[i]<errors[i-1] or errors[i]<errors[i+1]):
            try:
                fit=minimize_scalar(value,bounds=(grid[i-1],grid[i+1]),method='bounded',
                                    options=dict(maxiter=64,xatol=1e-11))
                evaluate(fit.x);statuses.append(dict(index=i,success=bool(fit.success),
                    message=str(fit.message),nfev=int(fit.nfev)))
            except Exception as exc:statuses.append(dict(index=i,success=False,error=repr(exc)))
    best=min(records,key=lambda r:(not r['valid'],r.get('max_error',np.inf),
                                  -r.get('worst_logB',-np.inf),r['feed']))
    return dict(feed=best['feed'],selected=best,candidates=records,solver_statuses=statuses,
                costs=costs,seconds=time.perf_counter()-start,
                information='Observed full state, target, fixed model and common seed list only')
