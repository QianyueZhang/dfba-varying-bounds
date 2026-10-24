"""Frozen full-trajectory numerical decomposition; original sources are read-only."""
from pathlib import Path
import argparse, csv, datetime, hashlib, importlib.metadata, json, platform, sys, time, traceback
import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'research_completion/runs/figure11_zampar_2026-10-21'
CODE = ROOT / 'research_completion/scripts/figure11_zampar_2026-10-21'
sys.path.insert(0, str(CODE))
from fit_batch import BatchModel, Y0, T0, TF, OD_FACTOR, SOURCE
from direct_run import Direct

PROTOCOL = HERE / 'protocol.json'
PAIRS = {'tight_pair': (1e-6, 1e-8), 'fixed_expansion_small_barrier': (1e-5, 1e-8)}
EPS = {'0': 0., '1e-6': 1e-6, '1e-5': 1e-5}
STATE_NAMES = ['X_gDW_L', 'G_g_L', 'E_g_L']
VARIABLES = ['biomass_OD600', 'glucose', 'ethanol']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def observations():
    rows = [r for r in csv.DictReader(SOURCE.open())
            if float(r['time_relative_to_glucose_exhaustion_h']) >= T0 - 1e-8]
    for r in rows:
        r['split'] = ('calibration' if r['replicate_label_as_published'].replace(' ', '') == 'Exp1'
                      else 'withheld_series')
    return rows


def evaluation_times():
    return np.unique(np.r_[np.linspace(T0, TF, 401),
        [float(r['time_relative_to_glucose_exhaustion_h']) for r in observations()]])


def runtime():
    return dict(python=sys.version, platform=platform.platform(),
        packages={k: importlib.metadata.version(k) for k in ['numpy', 'scipy', 'highspy']})


def declare():
    if PROTOCOL.exists() or (HERE / 'results').exists():
        raise FileExistsError('Fresh numerical experiment required')
    paths = [Path(__file__), SOURCE, CODE / 'fit_batch.py', CODE / 'direct_run.py',
             CODE / 'verify_barrier.py', ROOT / 'research_completion/scripts/flux_model.py',
             ROOT / '14_control_pso/09_Sc_Kp/SC_Model.mat',
             ROOT / '14_control_pso/09_Sc_Kp/SCerevisiae_Data.mat',
             OLD / 'fit_summary.json', OLD / 'direct_audit.json',
             OLD / 'trajectory_direct_barrier.csv', OLD / 'final_observation_metrics.json',
             OLD / 'numerical_refinement/coarse/trajectory_direct_barrier.csv',
             OLD / 'numerical_refinement/coarse/direct_run.py']
    fitted = json.loads((OLD / 'fit_summary.json').read_text())
    p = fitted['parameters']
    protocol = dict(declared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Full-trajectory numerical decomposition with frozen fitted parameters; no new calibration or biological validation.',
        source_sha256={str(q.relative_to(ROOT)): sha(q) for q in paths},
        parameters=[p[k] for k in ['qGmax_mmol_gDW_h', 'qOmax_mmol_gDW_h',
                                   'qEmax_mmol_gDW_h', 'ATP_maintenance_mmol_gDW_h']],
        fixed_parameters=fitted['fixed_parameters'], initial_state=Y0.tolist(), interval_h=[T0, TF],
        physical_bounds='Reuse archived BatchModel.bounds_state unchanged: dynamic glucose/ethanol lower bounds exactly minus capacity; remaining/fixed bounds expanded.',
        direct_cases={k: dict(epsilon=e, mu=m) for k, (e, m) in PAIRS.items()},
        archived_direct=dict(epsilon=1e-5, mu=1e-7, source=str((OLD / 'trajectory_direct_barrier.csv').relative_to(ROOT))),
        direct_integrator=dict(method='LSODA', rtol=2e-8, atol=[2e-10, 1e-13, 2e-10], max_step_h=.25),
        direct_KKT=dict(requested_stationarity_tolerance=2e-7, same_archived_Newton_algorithm=True),
        LP=dict(epsilon=EPS, method='RK45', fine_rtol=2e-9, fine_atol=2e-11,
                coarse_rtol=2e-8, coarse_atol=2e-10, max_step_h=.25, LP_feasibility_tolerance=1e-9),
        output_times='Union of401 uniform times and every original eligible observation time; no time realignment.',
        observation_counts=dict(all=256, calibration=45, withheld=211),
        decomposition='Signed expansion=LP_epsilon-LP0; barrier_difference=finite_epsilon_mu-LP_epsilon; total=finite_epsilon_mu-LP0. Store signed terms, exact sum residual, and cancellation fraction; no root-sum-square attribution.',
        integration_error='Fresh LP fine/coarse differences for each epsilon. Existing adopted direct coarse/fine difference separately; it is not a refinement estimate for the two new pairs. Finite-minus-LP differences also include residual integration/solver errors.',
        metrics='Per original variable observation times and counts, all/calibration/withheld: signed mean, maximum absolute and RMS; OD=X/0.486. Repeated observations retained without claiming independent replicates.',
        checks=dict(complete_interval=True, finite_state=True, min_external_concentration=-1e-9,
                    minimum_slack_strictly_positive=True, equality_residual_max=1e-8,
                    stationarity_residual_max=2.01e-7,
                    growth_gap_lower=-1e-8, growth_gap_upper='2*n*mu+1e-7 at saved outputs',
                    decomposition_identity_absolute=1e-12),
        observation_scale='Compare numerical differences with archived per-variable withheld RMSE in matching units; this is descriptive, not experimental uncertainty or statistical significance.',
        failure_rule='Retain every failed/incomplete case, RHS state and exception. No parameter, bound, tolerance or threshold retuning. Do not overwrite original or new completed outputs.',
        runtime=runtime(),
        runtime_limitations='Archived adopted pair was generated under its recorded older NumPy/SciPy environment. Fresh runs use the recorded current environment; cross-runtime comparisons are numerical diagnostics, not pure analytical error bounds.')
    new_json(PROTOCOL, protocol)
    print(json.dumps({'declared': str(PROTOCOL), 'sha256': sha(PROTOCOL)}, indent=2))


def verify():
    p = json.loads(PROTOCOL.read_text())
    for name, digest in p['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Frozen input changed: ' + name)
    if p['initial_state'] != Y0.tolist() or p['interval_h'] != [T0, TF]:
        raise RuntimeError('Initial state or interval changed')
    return p


def save_curve(path, times, values, extra=None, extra_names=None):
    data = np.c_[times, values] if extra is None else np.c_[times, values, extra]
    with Path(path).open('x') as f:
        np.savetxt(f, data, delimiter=',', header=','.join(['t_h'] + STATE_NAMES + (extra_names or [])), comments='')


def run_lp():
    p = verify(); out = HERE / 'results/lp'; out.mkdir(parents=True, exist_ok=False)
    times = evaluation_times(); params = np.array(p['parameters']); records = []
    for tag, eps in EPS.items():
        for level, rtol, atol in [('fine', 2e-9, 2e-11), ('coarse', 2e-8, 2e-10)]:
            tick = time.perf_counter(); m = BatchModel(eps)
            rec = dict(epsilon=eps, level=level, complete=False)
            try:
                sol = m.simulate(params, rtol=rtol, atol=atol)
                if not sol.success:
                    raise RuntimeError(sol.message)
                states = sol.sol(times).T
                save_curve(out / f'eps_{tag}_{level}.csv', times, states)
                rec.update(complete=True, LP_calls=m.calls,
                    max_equality_residual=m.max_residual, max_bound_violation=m.max_bound_violation,
                    minimum_states=np.min(states, axis=0).tolist(),
                    checks_passed=bool(np.all(np.isfinite(states)) and np.min(states) >= -1e-9
                                       and m.max_residual <= 1e-8 and m.max_bound_violation <= 1e-8))
            except Exception as exc:
                rec.update(exception=repr(exc), traceback=traceback.format_exc(), checks_passed=False)
            rec['elapsed_seconds'] = time.perf_counter() - tick
            records.append(rec); new_json(out / f'eps_{tag}_{level}.json', rec)
            print(json.dumps(rec), flush=True)
    verify()
    new_json(out / 'summary.json', dict(complete=all(x['complete'] for x in records),
        passed=all(x['checks_passed'] for x in records), records=records,
        protocol_sha256=sha(PROTOCOL), inputs_unchanged=True, runtime=runtime()))


def run_direct(name):
    p = verify(); case = p['direct_cases'][name]
    out = HERE / 'results' / name; out.mkdir(parents=True, exist_ok=False)
    params = np.array(p['parameters']); m = Direct(params, case['epsilon'], case['mu'])
    tick = time.perf_counter(); calls = []
    result = dict(case=name, **case, complete=False, protocol_sha256=sha(PROTOCOL), runtime=runtime())
    def rhs(t, state):
        calls.append([float(t)] + np.asarray(state).tolist())
        return m.rhs_direct(t, state)
    try:
        sol = solve_ivp(rhs, (T0, TF), Y0, method='LSODA', rtol=2e-8,
                        atol=[2e-10, 1e-13, 2e-10], max_step=.25, dense_output=True)
        np.savez_compressed(out / 'accepted_states.npz', time=sol.t, states=sol.y)
        if not sol.success:
            raise RuntimeError(sol.message)
        times = evaluation_times(); states = sol.sol(times).T; rows = []; m.warm = None
        for state in states:
            v, info = m.flux_direct(state)
            gap = float(m.c @ (m.last_lp[0] - v))
            rows.append([float(v[k]) for k in [m.bio, m.g, m.e, m.o, 203]] +
                        [info[k] for k in ['minimum_slack', 'equality_residual', 'stationarity_residual']] + [gap])
        rows = np.array(rows)
        save_curve(out / 'trajectory.csv', times, states, rows,
            ['growth_h-1', 'v_glucose', 'v_ethanol', 'v_oxygen', 'v_ATPase',
             'min_slack', 'equality_residual', 'stationarity_residual', 'growth_gap_to_same_state_LP'])
        checks = dict(complete_interval=bool(sol.t[0] == T0 and sol.t[-1] == TF),
            finite_states=bool(np.all(np.isfinite(states))),
            external_states=bool(np.min(states) >= -1e-9),
            positive_slacks=min(x['minimum_slack'] for x in m.records) > 0,
            equality=max(x['equality_residual'] for x in m.records) <= 1e-8,
            stationarity=max(x['stationarity_residual'] for x in m.records) <= 2.01e-7,
            output_growth_gap=bool(np.min(rows[:, -1]) >= -1e-8 and
                                  np.max(rows[:, -1]) <= 2*len(m.c)*m.mu + 1e-7))
        result.update(complete=True, passed=all(checks.values()), checks=checks,
            minimum_states=np.min(states, axis=0).tolist(), accepted_steps=len(sol.t),
            maximum_equality=max(x['equality_residual'] for x in m.records),
            maximum_stationarity=max(x['stationarity_residual'] for x in m.records),
            minimum_slack=min(x['minimum_slack'] for x in m.records),
            output_growth_gap_range=[float(np.min(rows[:, -1])), float(np.max(rows[:, -1]))])
    except Exception as exc:
        result.update(passed=False, exception=repr(exc), traceback=traceback.format_exc())
    np.savez_compressed(out / 'rhs_evaluations.npz', values=np.array(calls))
    new_json(out / 'KKT_records.json', m.records)
    result.update(elapsed_seconds=time.perf_counter()-tick, rhs_calls=m.rhs_calls,
                  total_KKT_solves=len(m.records), total_Newton_steps=m.steps)
    try:
        verify(); result['inputs_unchanged'] = True
    except Exception as exc:
        result['inputs_unchanged'] = False; result['source_failure'] = repr(exc); result['passed'] = False
    new_json(out / 'result.json', result)
    print(json.dumps(result, indent=2), flush=True)


def sample(path, t, j):
    a = np.genfromtxt(path, delimiter=',', names=True)
    if np.min(abs(a['t_h'] - t)) > 1e-10:
        raise ValueError('Requested observation time is absent from ' + str(path))
    return float(np.interp(t, a['t_h'], a[STATE_NAMES[j]])) / (OD_FACTOR if j == 0 else 1.)


def report():
    p = verify(); out = HERE / 'results'; destination = out / 'decomposition.json'
    if destination.exists() or (out / 'observation_decomposition.csv').exists():
        raise FileExistsError('Final decomposition already exists')
    lp_summary = json.loads((out / 'lp/summary.json').read_text())
    if not lp_summary['complete']:
        raise RuntimeError('LP reference campaign incomplete; preserve failures before reporting')
    cases = [('archived_adopted', '1e-5', OLD/'trajectory_direct_barrier.csv')]
    statuses = {}
    for name, (eps, mu) in PAIRS.items():
        status = json.loads((out / name / 'result.json').read_text()); statuses[name] = status
        if status['complete']:
            cases.append((name, '1e-6' if eps == 1e-6 else '1e-5', out/name/'trajectory.csv'))
    # Load once, while requiring exact inclusion of every observation time.
    cached = {}
    def get(path, t, j):
        key = str(path)
        if key not in cached:
            cached[key] = np.genfromtxt(path, delimiter=',', names=True)
        a = cached[key]
        if np.min(abs(a['t_h'] - t)) > 1e-10:
            raise ValueError('Observation time missing from ' + key)
        return float(np.interp(t, a['t_h'], a[STATE_NAMES[j]])) / (OD_FACTOR if j == 0 else 1.)
    all_rows = []
    for name, eps, path in cases:
        for row in observations():
            t = float(row['time_relative_to_glucose_exhaustion_h']); j = VARIABLES.index(row['variable'])
            a = get(out/'lp/eps_0_fine.csv', t, j)
            b = get(out/f'lp/eps_{eps}_fine.csv', t, j)
            c = get(path, t, j)
            expansion, barrier, total = b-a, c-b, c-a
            denom = abs(expansion)+abs(barrier)
            all_rows.append(dict(case=name, variable=row['variable'], unit=row['observed_unit'],
                time_h=t, split=row['split'], series_id=row['series_id'], source_cell=row['source_value_cell'],
                observation=float(row['observed_value_as_published']), lp0=a, lp_epsilon=b, finite=c,
                expansion=expansion, barrier_difference=barrier, total=total,
                decomposition_identity_error=expansion+barrier-total,
                cancellation_fraction=0. if denom == 0 else 1.-abs(total)/denom,
                lp0_refinement=get(out/'lp/eps_0_coarse.csv', t, j)-a,
                expanded_lp_refinement=get(out/f'lp/eps_{eps}_coarse.csv', t, j)-b,
                archived_direct_refinement=get(OLD/'numerical_refinement/coarse/trajectory_direct_barrier.csv', t, j)
                                            -get(OLD/'trajectory_direct_barrier.csv', t, j)))
    fields=list(all_rows[0])
    with (out/'observation_decomposition.csv').open('x', newline='') as f:
        w=csv.DictWriter(f, fields); w.writeheader(); w.writerows(all_rows)
    data_rmse=json.loads((OLD/'final_observation_metrics.json').read_text())['withheld_series']
    summaries=[]
    components=['expansion','barrier_difference','total','lp0_refinement','expanded_lp_refinement','archived_direct_refinement']
    for name, _, _ in cases:
        for var in VARIABLES:
            for split in ['all','calibration','withheld_series']:
                group=[q for q in all_rows if q['case']==name and q['variable']==var and (split=='all' or q['split']==split)]
                metrics={}
                for component in components:
                    values=np.array([q[component] for q in group]); largest=float(np.max(abs(values)))
                    metrics[component]=dict(maximum_absolute=largest, RMS=float(np.sqrt(np.mean(values**2))),
                        mean_signed=float(np.mean(values)), maximum_over_archived_withheld_RMSE=largest/data_rmse[var]['rmse'])
                summaries.append(dict(case=name,variable=var,split=split,n=len(group),metrics=metrics,
                    maximum_cancellation_fraction=max(q['cancellation_fraction'] for q in group),
                    opposite_signed_components=sum(q['expansion']*q['barrier_difference']<0 for q in group)))
    identity=max(abs(q['decomposition_identity_error']) for q in all_rows)
    verify()
    new_json(destination,dict(complete=len(cases)==3, protocol_sha256=sha(PROTOCOL), inputs_unchanged=True,
        new_direct_statuses=statuses, LP_passed=lp_summary['passed'], summaries=summaries,
        maximum_decomposition_identity_error=identity, identity_passed=identity<=1e-12,
        archived_withheld_RMSE={k:v['rmse'] for k,v in data_rmse.items()},
        scope=p['integration_error'], runtime_limitations=p['runtime_limitations'],
        input_sha256={str(Path(q).relative_to(ROOT)):sha(q) for q in cached}))
    print(json.dumps({'complete':len(cases)==3,'observation_rows':len(all_rows),'identity_error':identity,
                      'output':str(destination)},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--declare',action='store_true')
    group.add_argument('--run-lp',action='store_true')
    group.add_argument('--run-direct',choices=list(PAIRS))
    group.add_argument('--report',action='store_true')
    args=parser.parse_args()
    if args.declare: declare()
    elif args.run_lp: run_lp()
    elif args.run_direct: run_direct(args.run_direct)
    else: report()
