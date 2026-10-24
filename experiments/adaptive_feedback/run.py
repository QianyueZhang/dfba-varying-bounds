"""Portable commands for the adaptive coculture feedback study."""
from pathlib import Path
from copy import deepcopy
import argparse
import ast
import hashlib
import json
import math
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'local_results/matplotlib_cache'))
sys.path.insert(0, str(ROOT/'scripts'))
import numpy as np
from model import Reactor, initial_state


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    def serial(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=serial, allow_nan=False)+'\n')


class StripDocs(ast.NodeTransformer):
    def generic_visit(self, node):
        super().generic_visit(node)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
        return node


def ast_digest(text):
    return hashlib.sha256(ast.dump(StripDocs().visit(ast.parse(text)), include_attributes=False).encode()).hexdigest()


def numeric_items(obj, path=''):
    if isinstance(obj, dict):
        return sum((numeric_items(v, path+'/'+str(k)) for k,v in sorted(obj.items())), [])
    if isinstance(obj, list):
        return sum((numeric_items(v, path+'/'+str(i)) for i,v in enumerate(obj)), [])
    if obj is None or isinstance(obj, (float, int, bool)):
        return [(path, type(obj).__name__, obj)]
    return []


def verify_release():
    manifest = json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    errors = []
    for entry in manifest['files']:
        path = ROOT/entry['release']
        if not path.is_file() or digest(path) != entry['release_sha256']:
            errors.append('release file differs: '+entry['release']); continue
        if 'release_numeric_payload_sha256' in entry:
            numeric = hashlib.sha256(json.dumps(numeric_items(json.loads(path.read_text())), sort_keys=True, allow_nan=False).encode()).hexdigest()
            if numeric != entry['release_numeric_payload_sha256'] or numeric != entry['original_numeric_payload_sha256']:
                errors.append('numeric payload differs: '+entry['release'])
        if entry['change'] == 'module documentation only; executable AST unchanged':
            observed = ast_digest(path.read_text())
            if observed != entry['original_computational_ast_sha256'] or observed != entry['release_computational_ast_sha256']:
                errors.append('computational AST differs: '+entry['release'])
        if 'approved_diagnostic_string_change' in entry:
            change = entry['approved_diagnostic_string_change']
            source = path.read_text()
            if source.count(change['release']) != 1:
                errors.append('diagnostic phrase count differs: '+entry['release'])
            canonical = source.replace(change['release'], change['normalisation_token'])
            node = next(n for n in ast.parse(canonical).body if isinstance(n, ast.FunctionDef) and n.name == change['function'])
            observed = hashlib.sha256(ast.dump(StripDocs().visit(node), include_attributes=False).encode()).hexdigest()
            if observed != change['normalised_function_ast_sha256']:
                errors.append('diagnostic function scientific AST differs: '+entry['release'])
        if 'unchanged_function_or_class_asts' in entry:
            tree = ast.parse(path.read_text())
            actual = {node.name: hashlib.sha256(ast.dump(StripDocs().visit(deepcopy(node)), include_attributes=False).encode()).hexdigest() for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
            for name, expected in entry['unchanged_function_or_class_asts'].items():
                if actual.get(name) != expected: errors.append('function AST differs: '+entry['release']+'/'+name)
    lock = json.loads((ROOT/'LOCK.json').read_text())
    for name, expected in lock['hashes'].items():
        if digest(ROOT/name) != expected: errors.append('release lock differs: '+name)
    if errors: raise RuntimeError('\n'.join(errors))
    return {'origin_mapped_files': len(manifest['files']), 'release_hashes_pass': True,
            'original_numeric_payloads_preserved': True, 'core_executable_asts_preserved': True}


def load_campaign(label):
    verify_release()
    if Path(label).name != label: raise ValueError('Use a results directory name, without a path')
    campaign = ROOT/'results'/label
    protocol = json.loads((ROOT/'protocol.json').read_text())
    executed = json.loads((ROOT/'reference/executed_lock.json').read_text())
    release = json.loads((ROOT/'LOCK.json').read_text())
    execution = json.loads((campaign/'execution.json').read_text())
    complete = json.loads((campaign/'complete.json').read_text())
    identity = execution['hashes']
    historical = {k:v for k,v in executed['hashes'].items() if k != 'protocol.json'}
    current = {k:v for k,v in release['hashes'].items() if k != 'protocol.json'}
    if identity == historical:
        identity_kind = 'recorded_execution'
        if label != 'frozen_evaluation': raise RuntimeError('Recorded execution must use the supplied frozen directory')
    elif identity == current:
        identity_kind = 'release_execution'
        for name in ('model.py', 'controllers.py', 'run_study.py'):
            if digest(campaign/'source_snapshot'/name) != release['hashes']['scripts/'+name]:
                raise RuntimeError('Runtime source snapshot differs: '+name)
        if digest(campaign/'source_snapshot/protocol.json') != release['hashes']['protocol.json']:
            raise RuntimeError('Runtime protocol snapshot differs')
    else:
        raise RuntimeError('Campaign matches neither recorded nor current release source identities')
    if execution['mode'] != 'evaluation' or execution['cases'] != protocol['cases'] or execution['methods'] != protocol['methods']:
        raise RuntimeError('Campaign does not match the complete protocol')
    n = len(protocol['cases'])*len(protocol['methods'])
    if complete['status'] != 'complete' or complete['episodes'] != n or complete['expected'] != n or complete['hashes'] != identity:
        raise RuntimeError('Campaign completion or source identities differ')
    expected_files = {f'{case["id"]}__{method}.json' for case in protocol['cases'] for method in protocol['methods']}
    if {p.name for p in campaign.glob('*__*.json')} != expected_files:
        raise RuntimeError('Campaign episode coverage differs')
    rows = {}
    for case in protocol['cases']:
        for method in protocol['methods']:
            path = campaign/f'{case["id"]}__{method}.json'
            row = json.loads(path.read_text())
            provenance = {'source_hashes': identity, 'mode': 'evaluation',
                          'case_sha256': hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()}
            if row['case'] != case or row['method'] != method or row['provenance'] != provenance:
                raise RuntimeError('Episode identity differs: '+path.name)
            rows[case['id'], method] = row
    return campaign, protocol, rows, identity_kind


def check(numerical=False):
    integrity = verify_release()
    campaign, protocol, rows, identity_kind = load_campaign('frozen_evaluation')
    errors = []
    counts = {method: 0 for method in protocol['methods']}
    times = {method: 0.0 for method in protocol['methods']}
    intervals = 0
    for (case_id, method), row in rows.items():
        if row['status'] != 'completed' or len(row['stages']) != 7: errors.append(case_id+'/'+method+'/completion'); continue
        intervals += len(row['stages'])
        end = row['true_states'][-1]
        z = end[1]-end[2]
        biomass = math.exp(end[1])+math.exp(end[2])
        physical = (min(s['plant_min_S'] for s in row['stages']) >= -2e-7
                    and all(-1e-10 <= u <= .2+1e-10 for u in row['controls'])
                    and all(y[0] <= 1.8+1e-8 and all(math.isfinite(x) for x in y) for y in row['true_states']))
        accepted = physical and abs(z-math.log(row['case']['target_ratio'])) <= math.log(1.01) and end[0] <= 1.8+1e-8
        if bool(accepted) != row['result']['terminal_feasible'] or not math.isclose(biomass, row['result']['biomass_amount_gDW'], rel_tol=1e-12):
            errors.append(case_id+'/'+method+'/endpoint')
        if abs(end[0]-(.5+2*math.fsum(row['controls']))) > 1e-12: errors.append(case_id+'/'+method+'/volume')
        if abs(row['controller_seconds']-math.fsum(s['controller_seconds'] for s in row['stages'])) > 1e-8: errors.append(case_id+'/'+method+'/cost')
        counts[method] += accepted
        times[method] += row['controller_seconds']
    summary = json.loads((ROOT/'results/summary.json').read_text())
    for method in protocol['methods']:
        if counts[method] != summary['methods'][method]['feasible'] or abs(times[method]-summary['methods'][method]['controller_seconds']['sum']) > 1e-8:
            errors.append(method+'/aggregate')
    strict = json.loads((ROOT/'independent_audit/frozen_evaluation_outcome_audit/audit.json').read_text())
    if strict['passed'] is not False or sum(len(e['issues']) for e in strict['episodes']) != 9:
        errors.append('strict numerical flags changed')
    for episode in strict['episodes']:
        row = rows[episode['case_id'], episode['method']]
        independent = episode['terminal']['independent_terminal_ratio_volume_feasible'] and episode['terminal']['independent_physical_valid']
        if independent != row['result']['terminal_feasible']: errors.append('independent terminal classification')
    post = json.loads((ROOT/'independent_audit/frozen_evaluation_final_interval_grid_refined/report.json').read_text())
    failed = {key for key,row in rows.items() if not row['result']['terminal_feasible']}
    if failed != {(r['case_id'],r['method']) for r in post['results']} or len(failed) != 58 or sum(r['feasible_recovery_found'] for r in post['results']) != 18:
        errors.append('retrospective coverage/counts')
    numeric_report = None
    if numerical:
        from outcome_audit import AmountReference, state_errors
        reactor = Reactor(); state = initial_state(1.1, 21.)
        controls = np.array([.04,.08,.055,.07,.04,.06,.05]); kappa = [.9,1.1]
        prediction = reactor.predict(state, controls, kappa, gradients=True, tight=True)
        finite = []
        for j in range(7):
            perturb = np.zeros(7); perturb[j] = 2e-4
            a = reactor.predict(state, controls+perturb, kappa, tight=True)
            b = reactor.predict(state, controls-perturb, kappa, tight=True)
            finite.append([(a['z']-b['z'])/4e-4, (a['logB']-b['logB'])/4e-4])
        gradient_error = float(np.max(abs(np.asarray(finite)-np.c_[prediction['jz'], prediction['jlogB']])))
        reference = AmountReference(ROOT/'data/flux_table_eps1e-06.json')
        reference_state = state.copy()
        for u in controls: reference_state = reference.segment(reference_state,u,kappa)['state']
        err = state_errors(prediction['state'], reference_state)
        numerical_pass = gradient_error < 2e-4 and err['z_absolute'] < 2e-6 and err['biomass_relative'] < 2e-6 and err['volume_absolute_L'] < 1e-10
        numeric_report = {'passed': numerical_pass, 'gradient_max_absolute_error': gradient_error, 'amount_reference_errors': err,
                          'scope': 'One 14 h feed vector, analytic sensitivities versus central finite differences and a separately coded physical-amount integrator; not a feedback-campaign rerun.'}
        if not numerical_pass: errors.append('numerical smoke check')
    report = {**integrity, 'passed': not errors, 'execution_identity': identity_kind, 'episodes': len(rows), 'intervals': intervals,
              'acceptance_counts': counts, 'controller_seconds': times, 'strict_numerical_audit_passed': False,
              'retained_strict_flags': 9, 'retrospective_failed_cases': 58, 'retrospective_feasible_alternatives': 18,
              'numerical_smoke': numeric_report, 'errors': errors}
    save(ROOT/'local_results/package_verification.json', report)
    print(json.dumps(report, indent=2))
    if errors: raise SystemExit(1)


def fresh_output(kind, label):
    if Path(label).name != label: raise ValueError('Output label cannot contain a path')
    output = ROOT/'local_results'/f'{kind}_{label}'
    output.mkdir(parents=True, exist_ok=False)
    return output


def audit(label, output_label):
    from outcome_audit import AmountReference, audit_episode, TOLERANCES
    campaign, protocol, rows, identity_kind = load_campaign(label)
    out = fresh_output('audit', output_label)
    reference = AmountReference(ROOT/'data/flux_table_eps1e-06.json'); reactor = Reactor()
    started = time.perf_counter(); episodes=[]
    for case in protocol['cases']:
        for method in protocol['methods']:
            item = audit_episode(rows[case['id'],method],case,method,reference,reactor)
            item['episode_sha256'] = digest(campaign/f'{case["id"]}__{method}.json')
            episodes.append(item)
            save(out/'audit.partial.json', {'episodes':episodes})
            print(f'{len(episodes)}/144 {case["id"]} {method}: {item["issues"] or "PASS"}', flush=True)
    report={'execution_identity':identity_kind,'tolerances':TOLERANCES,'episodes':episodes,
            'audit_wall_seconds':time.perf_counter()-started,
            'passed':all(e['passed'] for e in episodes), 'retained_flag_count':sum(len(e['issues']) for e in episodes),
            'scope':'Independent physical-amount replay and prescribed tight tail checks; all cost excluded from controller timings.'}
    save(out/'audit.json',report); (out/'audit.partial.json').unlink()
    print(json.dumps({k:v for k,v in report.items() if k!='episodes'},indent=2))
    if not report['passed']: raise SystemExit(1)


def posthoc(label, output_label):
    from posthoc_diagnostic import diagnose, episode_context
    campaign, protocol, rows, identity_kind = load_campaign(label)
    out=fresh_output('final_interval',output_label); reactor=Reactor(); results=[]
    started=time.perf_counter()
    for case in protocol['cases']:
        context=[episode_context(rows[case['id'],m]) for m in protocol['methods']]
        for method in protocol['methods']:
            row=rows[case['id'],method]
            if row['status']!='completed' or row['result']['terminal_feasible']: continue
            item=diagnose(row,case,reactor,True)
            item.update(episode_sha256=digest(campaign/f'{case["id"]}__{method}.json'),same_case_policy_context=context)
            results.append(item);save(out/f'{case["id"]}__{method}.json',item)
            print(case['id'],method,item.get('conclusion',item['status']),flush=True)
    report={'execution_identity':identity_kind,'results':results,
            'summary':{'failed_completed_episodes':len(results),'feasible_alternatives':sum(r.get('feasible_recovery_found',False) for r in results),
                       'diagnostic_model_calls':reactor.calls,'diagnostic_wall_seconds':time.perf_counter()-started},
            'scope':'Retrospective 101-point constant-feed search with local refinement from actual final-interval states. Hidden affinity is used only here; finite-search failure is not proof of unreachability.'}
    save(out/'report.json',report);print(json.dumps(report['summary'],indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['verify','simulate','summary','audit','posthoc','plots'])
    parser.add_argument('--numerical',action='store_true',help='Also run a 14 h sensitivity/amount check')
    parser.add_argument('--campaign',default='frozen_evaluation')
    parser.add_argument('--label',default='local_evaluation',help='Fresh simulation or analysis output label')
    args=parser.parse_args()
    if args.command=='verify': return check(args.numerical)
    verify_release()
    if args.command=='simulate':
        if not args.label.startswith('local_') or Path(args.label).name!=args.label:
            raise ValueError('Fresh simulation labels must begin local_ and contain no path')
        destination=ROOT/'results'/args.label
        if destination.exists():raise RuntimeError('Refusing to replace an existing run; choose a fresh --label')
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_study.py'),'--mode','evaluation','--label',args.label],check=True)
    elif args.command=='summary':
        campaign,_,_,_=load_campaign(args.campaign)
        import summarize
        summarize.CAMPAIGN=campaign;summarize.OUTDIR=fresh_output('summary',args.label);summarize.main()
    elif args.command=='audit': audit(args.campaign,args.label)
    elif args.command=='posthoc': posthoc(args.campaign,args.label)
    elif args.command=='plots': subprocess.run([sys.executable,'-B',str(ROOT/'scripts/plot_results.py')],check=True)


if __name__=='__main__': main()
