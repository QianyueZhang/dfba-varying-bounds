"""Two manual analytic development states; never reads evaluation results."""
from pathlib import Path
import datetime
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'audit'))
from independent_amount_model import AmountReactor, from_log
from continuous_baseline import solve_open, COMPARISON_TOLERANCE
from engine import arguments, K


def main():
    folder = ROOT / 'results' / 'development_continuous_baseline'
    folder.mkdir(parents=True, exist_ok=True)
    protocol_path = folder / 'protocol.json'
    if protocol_path.exists():
        raise RuntimeError('Development records exist; do not overwrite them')
    cases = [
        dict(id='manual_analytic_high_substrate', state=[.82, float(np.log(.02)), float(np.log(.02)), 49.2],
             target=1.8, grid_plan=[.1, .1], grid_feasible_plan=[.1, .1]),
        dict(id='manual_analytic_low_substrate', state=[1.52, float(np.log(9.)), float(np.log(8.)), 3.04],
             target=.6, grid_plan=[.04, .04], grid_feasible_plan=None),
    ]
    protocol = dict(scope='Two manual analytic development states, not evaluation or paper-effectiveness evidence',
                    declared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), cases=cases,
                    K=K.tolist(), independent_replay_tolerance=2e-6,
                    source_sha256={name: hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest()
                                   for name in ('continuous_baseline.py', 'engine.py', Path(__file__).name)})
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')
    summaries = []
    for case in cases:
        x = np.asarray(case['state'])
        original = x.copy()
        result = solve_open(x, arguments('analytic'), case['target'], case['grid_plan'], case['grid_feasible_plan'])
        assert np.array_equal(x, original), 'Input state was mutated'
        assert len(result['verification']['paths']) == 9
        assert len([s for s in result['solver_statuses'] if s['stage'] == 'minimax']) == 2
        assert len([s for s in result['solver_statuses'] if s['stage'] == 'economic']) <= 1
        assert all(v is not None and 0 <= v <= .2 for v in result['plan'])
        assert 2 * sum(result['plan']) <= result['verification']['capacity_L']
        index = result['selected_candidate_index']
        assert index is not None, 'Both manual states must return a physically valid numerical candidate'
        selected = result['candidates'][index]
        assert selected['numerical_valid'] and selected['physical_valid']
        assert any('grid_incumbent' in c['origins'] for c in result['candidates'])
        feasible = [c for c in result['candidates'] if c['feasible']]
        if feasible:
            assert result['feasible']
            assert selected['worst_logB'] == max(c['worst_logB'] for c in feasible)
        else:
            assert not result['feasible']
            valid = [c for c in result['candidates'] if c['numerical_valid'] and c['physical_valid']]
            assert selected['max_error'] == min(c['max_error'] for c in valid)
        independent = AmountReactor('analytic')
        replays = []
        for i, k0 in enumerate(K):
            first = independent.interval(from_log(x), result['plan'][0], k0)
            for j, k1 in enumerate(K):
                terminal = independent.interval(first['amount_state'], result['plan'][1], k1)
                state = terminal['amount_state']
                z = float(np.log(state[1] / state[2]))
                logB = float(np.log(state[1] + state[2]))
                saved = selected['paths'][i * 3 + j]
                replays.append(dict(path=[i, j], z=z, logB=logB,
                                    z_difference=z-saved['z'], logB_difference=logB-saved['logB'],
                                    state=state.tolist()))
        max_z = max(abs(p['z_difference']) for p in replays)
        max_logB = max(abs(p['logB_difference']) for p in replays)
        assert max_z <= protocol['independent_replay_tolerance']
        assert max_logB <= protocol['independent_replay_tolerance']
        independent_error = max(abs(p['z'] - np.log(case['target'])) for p in replays)
        independent_label = bool(independent_error <= result['verification']['inner_band'] + COMPARISON_TOLERANCE)
        result['development_independent_replay'] = dict(paths=replays, max_z_difference=max_z,
                                                       max_logB_difference=max_logB, max_error=independent_error,
                                                       inner_label=independent_label,
                                                       inner_label_agrees=independent_label == result['feasible'])
        # Strict serialization catches NaN/Infinity leaking from rejected trials.
        (folder / (case['id'] + '.json')).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        summary = dict(case=case['id'], assertions_passed=True, seconds=result['seconds'],
                       feasible=result['feasible'], plan=result['plan'], best_error=result['best_error'], worstB=result['worstB'],
                       count=result['count'], solver_statuses=result['solver_statuses'],
                       max_independent_z_difference=max_z, max_independent_logB_difference=max_logB,
                       independent_inner_label_agrees=independent_label == result['feasible'])
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)
    (folder / 'summary.json').write_text(json.dumps(dict(complete=True, cases=summaries,
                                                        scope=protocol['scope']), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
