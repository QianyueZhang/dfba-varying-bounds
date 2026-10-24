"""Descriptive aggregation of the complete, immutable six-arm campaign."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from model import ROOT
from run_study import METHODS, save

CAMPAIGN = ROOT/'results/frozen_evaluation'
OUTDIR = ROOT/'local_results/summary'

REGIMES = ['nominal','constant_mismatch','step_mismatch','noisy_mismatch']
LABELS = {'open_loop':'Open loop','nominal_mpc':'Nominal MPC',
          'adaptive_direct_mpc':'Adaptive direct','adaptive_limited_direct':'Limited direct',
          'adaptive_ungated':'Ungated local','adaptive_verified':'Verified local'}


def stats(values):
    x = np.asarray(values, float)
    if not len(x): return {'n': 0}
    return {'n': len(x), 'mean': float(x.mean()), 'median': float(np.median(x)),
            'min': float(x.min()), 'max': float(x.max()), 'sum': float(x.sum())}


def main():
    directory = CAMPAIGN
    protocol = json.loads((ROOT/'protocol.json').read_text())
    complete = json.loads((directory/'complete.json').read_text())
    assert complete['episodes'] == complete['expected'] == 144
    rows = []
    raw = {}
    for case in protocol['cases']:
        for method in METHODS:
            path = directory/f'{case["id"]}__{method}.json'
            d = json.loads(path.read_text()); raw[case['id'], method] = d
            assert d['case'] == case and d['method'] == method
            events = [e for s in d.get('stages', []) for e in s['decision'].get('events', [])]
            result = d['result']
            last = d.get('stages', [{}])[-1]
            row = {'case': case['id'], 'cluster': case['cluster'], 'regime': case['regime'],
                   'target_ratio': case['target_ratio'], 'method': method, 'status': d['status'],
                   'feasible': bool(result['terminal_feasible']),
                   'B_gDW': result.get('biomass_amount_gDW'),
                   'ratio_error_percent': result.get('ratio_relative_error_percent'),
                   'log_ratio_error': result.get('log_ratio_error'),
                   'controller_seconds': d['controller_seconds'],
                   'estimator_seconds': d.get('estimator_seconds'),
                   'model_calls': d.get('model_calls'),
                   'fallback_stages': sum(s['decision'].get('fallback',False) for s in d.get('stages',[])),
                   'accepted_proposals': sum(e['action']=='accept' for e in events),
                   'accepted_after_correction': sum(e['action']=='accept' and e['attempt']>0 for e in events),
                   'rejected_proposals': sum(e['action']=='reject' for e in events),
                   'final_model_feasible': last.get('selected_model_feasible'),
                   'observation_clip_count': sum(s.get('next_substrate_observation_clipped',False) for s in d.get('stages',[])),
                   'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            rows.append(row)
    summary = {'scope': 'Synthetic closed-loop computational stress study; no experimental cultures',
               'condition_clusters': 6, 'cases': 24, 'episodes': 144,
               'failed_episodes': [r for r in rows if r['status']!='completed'],
               'regimes': {}, 'methods': {}, 'paired': {}}
    for regime in REGIMES:
        summary['regimes'][regime] = {}
        for method in METHODS:
            group = [r for r in rows if r['regime']==regime and r['method']==method]
            completed = [r for r in group if r['status']=='completed']
            summary['regimes'][regime][method] = {
                'n':len(group), 'feasible':sum(r['feasible'] for r in group),
                'ratio_error_percent':stats([r['ratio_error_percent'] for r in completed]),
                'controller_seconds':stats([r['controller_seconds'] for r in group]),
                'final_model_pass_actual_fail':sum(r['final_model_feasible'] is True and not r['feasible'] for r in group)}
    for method in METHODS:
        group = [r for r in rows if r['method']==method]
        decisions = [s['controller_seconds'] for d in raw.values() if d['method']==method for s in d.get('stages',[])]
        summary['methods'][method] = {
            'n':len(group), 'feasible':sum(r['feasible'] for r in group),
            'controller_seconds':stats([r['controller_seconds'] for r in group]),
            'estimator_seconds':stats([r['estimator_seconds'] for r in group if r['estimator_seconds'] is not None]),
            'decision_seconds':stats(decisions),
            'model_calls':sum(r['model_calls'] or 0 for r in group),
            'fallback_stages':sum(r['fallback_stages'] for r in group),
            'accepted_proposals':sum(r['accepted_proposals'] for r in group),
            'accepted_after_correction':sum(r['accepted_after_correction'] for r in group),
            'rejected_proposals':sum(r['rejected_proposals'] for r in group),
            'observation_clips':sum(r['observation_clip_count'] for r in group)}
    for method in ('adaptive_limited_direct','adaptive_ungated','adaptive_verified'):
        for reference in ('adaptive_direct_mpc','adaptive_limited_direct'):
            if method == reference: continue
            pairs = []
            for case in protocol['cases']:
                a,b = raw[case['id'],method],raw[case['id'],reference]
                both = a['result']['terminal_feasible'] and b['result']['terminal_feasible']
                if both:
                    difference = 100*(a['result']['biomass_amount_gDW']/b['result']['biomass_amount_gDW']-1)
                    pairs.append({'case':case['id'], 'regime':case['regime'],
                                  'biomass_difference_percent':difference,
                                  'time_ratio':a['controller_seconds']/b['controller_seconds'],
                                  'within_quality_and_faster':bool(difference>=-.5 and a['controller_seconds']<b['controller_seconds'])})
            summary['paired'][method+'__vs__'+reference] = {
                'jointly_feasible':len(pairs),
                'biomass_difference_percent':stats([p['biomass_difference_percent'] for p in pairs]),
                'time_ratio':stats([p['time_ratio'] for p in pairs]),
                'within_quality_and_faster':sum(p['within_quality_and_faster'] for p in pairs), 'pairs':pairs}
    save(OUTDIR/'summary.json', summary)
    with (OUTDIR/'episode_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['% Generated from the locked campaign; counts are cases, not reliability estimates.',
             r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'Method & Nominal & Fixed & Shift & Noisy & Total time (s)\\', r'\midrule']
    for method in METHODS:
        counts = [str(summary['regimes'][regime][method]['feasible'])+'/6' for regime in REGIMES]
        lines.append(LABELS[method]+' & '+' & '.join(counts)+f" & {summary['methods'][method]['controller_seconds']['sum']:.1f}"+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (OUTDIR/'results_table.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k in ('methods','regimes')},indent=2))


if __name__=='__main__': main()
