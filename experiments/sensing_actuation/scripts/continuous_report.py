"""Read-only aggregate of all frozen continuous outputs; gates physical success."""
from pathlib import Path
import hashlib,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/continuous'


def main():
    rows=[json.loads(p.read_text()) for p in sorted(OUT.glob('*_target_*.json'))]
    assert len(rows)==72 and len({(r['root'],r['target'],r['mode']) for r in rows})==72
    assert all(r['complete'] for r in rows)
    report=dict(complete=True,methods={},reporting_note='Raw deployed_labels are error-only. Formal successes require complete execution, a valid selected plan, and all independent physical checks.',
                reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                protocol_sha256=hashlib.sha256((ROOT/'protocols/continuous_protocol.json').read_bytes()).hexdigest(),
                input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*_target_*.json'))})
    all_physical_differences=0
    for mode in ['open','causal']:
        subset=[r for r in rows if r['mode']==mode]
        valid=lambda r:r['complete'] and r['optimization']['selected']['valid'] and r['deployment']['all_physical']
        success={label:sum(valid(r) and r['deployment']['deployed_labels'][label] for r in subset)
                 for label in ['inner','outer','actual']}
        raw={label:sum(r['deployment']['deployed_labels'][label] for r in subset) for label in success}
        all_physical_differences+=sum(raw[k]-success[k] for k in success)
        statuses=[s for r in subset for s in r['optimization']['solver_statuses']]
        continuation=[p for r in subset for p in r['deployment']['second_policies']]
        scalarstatus=[s for p in continuation for s in p['solver_statuses']]
        inner_cases=[dict(root=r['root'],target=r['target'],error=r['deployment']['deployed_error'])
                     for r in subset if valid(r) and r['deployment']['deployed_labels']['inner']]
        amount_checks=[]
        for r in subset:
            amount_checks += [f['comparison'] for f in r['deployment']['first']]
            amount_checks += [p['matched_comparison'] for p in r['deployment']['paths']]
        report['methods'][mode]=dict(problems=36,paths=324,deployed_successes=success,
            planned_error_only_successes={label:sum(r['optimization']['selected'][label] for r in subset) for label in success},
            raw_error_only_counts=raw,inner_success_cases=inner_cases,
            all_physical=all(valid(r) for r in subset),all_matched_numeric=all(r['deployment']['all_matched_comparisons_pass'] for r in subset),
            all_chained_numeric=all(r['deployment']['all_chained_comparisons_pass'] for r in subset),
            all_matched_labels=all(r['deployment']['all_matched_labels_agree'] for r in subset),
            all_chained_labels=all(r['deployment']['all_chained_labels_agree'] for r in subset),
            max_matched_log_ratio_error=max(r['deployment']['max_matched_log_ratio_error'] for r in subset),
            max_chained_log_ratio_error=max(r['deployment']['max_chained_log_ratio_error'] for r in subset),
            max_matched_amount_error_over_allowance=max(c['max_amount_error_over_allowance'] for c in amount_checks),
            matched_interval_comparisons=len(amount_checks),same_feed_chained_comparisons=324,
            de_calls=sum(s['stage']=='DE' for s in statuses),
            de_budget_exhaustions=sum(s['stage']=='DE' and not s['success'] and 'Maximum number of iterations' in s.get('message','') for s in statuses),
            de_other_nonconvergence=sum(s['stage']=='DE' and not s['success'] and 'Maximum number of iterations' not in s.get('message','') and 'error' not in s for s in statuses),
            slsqp_calls=sum(s['stage']=='SLSQP' for s in statuses),
            slsqp_nonconvergence=sum(s['stage']=='SLSQP' and not s['success'] and 'error' not in s for s in statuses),
            optimization_exceptions=sum('error' in s for s in statuses),
            invalid_candidates=sum(r['optimization']['costs']['invalid_candidates'] for r in subset),
            unique_candidates=sum(r['optimization']['costs']['unique_candidates'] for r in subset),
            optimization_transitions=sum(r['optimization']['costs']['transitions'] for r in subset),
            optimization_seconds=sum(r['optimization']['seconds'] for r in subset),
            deployment_seconds=sum(r['deployment']['seconds'] for r in subset),
            scalar_calls=len(scalarstatus),scalar_nonconvergence=sum(not s['success'] for s in scalarstatus),
            continuation_decisions=len(continuation),continuation_transitions=sum(p['costs']['transitions'] for p in continuation),
            continuation_seconds=sum(p['seconds'] for p in continuation),
            independent_intervals=sum(r['deployment']['costs']['independent_intervals'] for r in subset),
            independent_rhs_calls=sum(r['deployment']['costs']['independent_rhs_calls'] for r in subset),
            matched_engine_intervals=sum(r['deployment']['costs']['matched_engine_intervals'] for r in subset),
            chained_engine_intervals=sum(r['deployment']['costs']['chained_engine_intervals'] for r in subset),
            maximum_absolute_planned_deployed_error_change=max(abs(r['optimization']['selected']['max_error']-r['deployment']['deployed_error']) for r in subset))
    paired=[]
    for key in sorted({(r['root'],r['target']) for r in rows}):
        common=next(r for r in rows if (r['root'],r['target'],r['mode'])==(*key,'open'))
        causal=next(r for r in rows if (r['root'],r['target'],r['mode'])==(*key,'causal'))
        a=common['deployment']['deployed_error'];b=causal['deployment']['deployed_error']
        paired.append(dict(root=key[0],target=key[1],open_error=a,causal_error=b,
                           difference_open_minus_causal=a-b,
                           comparison_scale=2e-7,causal_lower_by_scale=a-b>2e-7,open_lower_by_scale=b-a>2e-7))
    report['paired']=paired
    report['paired_counts_at_2e_7']=dict(causal_lower=sum(p['causal_lower_by_scale'] for p in paired),
        open_lower=sum(p['open_lower_by_scale'] for p in paired),
        within_scale=sum(abs(p['difference_open_minus_causal'])<=2e-7 for p in paired))
    report['physical_gating_changed_success_labels']=all_physical_differences
    metadata=[json.loads(p.read_text()) for p in sorted((ROOT/'results/trees').glob('*uniform_41.json'))]
    report['shared_grid41_prerequisite']=dict(roots=len(metadata),
        transitions=sum(m['new_transitions'] for m in metadata),seconds=sum(m['seconds'] for m in metadata),
        accounting='One shared offline tree per root, reused across both methods and allsix targets; not zero-cost online information.')
    (OUT/'formal_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['input_sha256','paired']},indent=2))

if __name__=='__main__':main()
