"""Independent physical-amount deployment of the two continuous policies."""
from pathlib import Path
import sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'audit'))
import independent_amount_model as reference
from engine import flow,K
from continuous_solver import causal_second, labels, physical

reference.RTOL=float(100*np.finfo(float).eps)
reference.ATOL=2e-15
reference.MAX_STEP=.005


def comparison(engine_state, amount_state):
    actual=np.asarray(amount_state,float);screen=reference.from_log(engine_state)
    delta=np.abs(screen-actual);allow=2e-7+2e-8*np.abs(actual[1:])
    zerr=abs(float(engine_state[1]-engine_state[2]-np.log(actual[1]/actual[2])))
    return dict(passed=bool(delta[0]<=2e-13 and np.all(delta[1:]<=allow) and zerr<=1e-7),
                component_errors=delta.tolist(),amount_allowances=allow.tolist(),
                max_amount_error_over_allowance=float(np.max(delta[1:]/allow)),log_ratio_error=zerr)


def audit(root,args,solution):
    started=time.perf_counter();model=reference.AmountReactor(root['family'])
    selected=solution['selected'];mode=solution['mode'];target=solution['target']
    initial=reference.from_log(root['state']);u0=selected['feeds'][0]
    paths=[];first=[];second_policies=[];failures=[];refcalls=0;matchedcalls=0;chaincalls=0
    rhs_calls=0
    # Every observation receives the same predeclared list, never the seed
    # indexed by the simulator's true first disturbance.
    common_seeds=list(selected['feeds'][1:])
    for i,k0 in enumerate(K):
        entry=dict(first_path_index=i,complete=False)
        try:
            observed=model.interval(initial,u0,k0);refcalls+=1;rhs_calls+=observed['rhs_calls']
            same,minimum,s,r=flow(np.asarray(root['state']),u0,k0,args);matchedcalls+=1
            first_comparison=comparison(same,observed['amount_state'])
            entry.update(complete=True,state=observed['state'].tolist(),
                         amounts=observed['amount_state'].tolist(),comparison=first_comparison,
                         min_S=observed['min_S'])
            if mode=='causal':
                policy=causal_second(observed['state'],args,target,common_seeds)
                second_policies.append(policy);u1=policy['feed']
            else:
                u1=selected['feeds'][1]
            entry['second_feed']=u1
            for j,k1 in enumerate(K):
                row=dict(path=[i,j],complete=False,feeds=[u0,u1])
                try:
                    end=model.interval(observed['amount_state'],u1,k1);refcalls+=1;rhs_calls+=end['rhs_calls']
                    matched,m,s,r=flow(observed['state'],u1,k1,args);matchedcalls+=1
                    chained,cm,s,r=flow(same,u1,k1,args);chaincalls+=1
                    z=float(np.log(end['amount_state'][1]/end['amount_state'][2]));error=abs(z-np.log(target))
                    matched_error=abs(float(matched[1]-matched[2])-np.log(target))
                    chain_error=abs(float(chained[1]-chained[2])-np.log(target))
                    good=physical(end['state'],min(observed['min_S'],end['min_S']))
                    row.update(complete=True,terminal_state=end['state'].tolist(),
                        terminal_amounts=end['amount_state'].tolist(),error=float(error),
                        worst_path_biomass=float(np.sum(end['amount_state'][1:3])),physical_valid=good,
                        min_S=min(observed['min_S'],end['min_S']),labels=labels(error),
                        matched_comparison=comparison(matched,end['amount_state']),
                        chained_comparison=comparison(chained,end['amount_state']),
                        matched_labels_agree=labels(error)==labels(matched_error),
                        chained_labels_agree=labels(error)==labels(chain_error))
                except Exception as exc:row['exception']=repr(exc);failures.append(row)
                paths.append(row)
        except Exception as exc:entry['exception']=repr(exc);failures.append(entry)
        first.append(entry)
    complete=len(paths)==9 and all(p['complete'] for p in paths)
    comparisons=[f['comparison'] for f in first if f['complete']]
    comparisons += [p['matched_comparison'] for p in paths if p['complete']]
    chained=[p['chained_comparison'] for p in paths if p['complete']]
    return dict(mode=mode,target=target,complete=complete,first=first,paths=paths,
        common_second_seed_feeds=common_seeds,second_policies=second_policies,exceptions=failures,
        planned_tree_error=selected.get('max_error'),
        deployed_error=max(p['error'] for p in paths) if complete else None,
        deployed_labels=labels(max(p['error'] for p in paths)) if complete else None,
        all_physical=complete and all(p['physical_valid'] for p in paths),
        all_matched_comparisons_pass=complete and all(c['passed'] for c in comparisons),
        all_chained_comparisons_pass=complete and all(c['passed'] for c in chained),
        all_matched_labels_agree=complete and all(p['matched_labels_agree'] for p in paths),
        all_chained_labels_agree=complete and all(p['chained_labels_agree'] for p in paths),
        max_matched_log_ratio_error=max((c['log_ratio_error'] for c in comparisons),default=None),
        max_chained_log_ratio_error=max((c['log_ratio_error'] for c in chained),default=None),
        max_matched_amount_error_over_allowance=max((c['max_amount_error_over_allowance'] for c in comparisons),default=None),
        costs=dict(independent_intervals=refcalls,independent_rhs_calls=rhs_calls,
                   matched_engine_intervals=matchedcalls,chained_engine_intervals=chaincalls),
        seconds=time.perf_counter()-started,
        interpretation='Causal deployment recomputes from the independent actual state; planned tree outcome is a distinct diagnostic.')
