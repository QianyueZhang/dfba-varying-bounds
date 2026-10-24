"""Independent aggregation of the saved control evidence; no model execution."""
from pathlib import Path
from collections import Counter, defaultdict
import json, math, hashlib, gzip

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'local_results'
OUT.mkdir(exist_ok=True)
BANDS = {'inner': .00748, 'outer': .0075, 'actual': math.log(1.01)}
issues = []
inputs = {}

def read(path):
    path = ROOT / path
    inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())

def label(error):
    return {k: error <= value for k, value in BANDS.items()}

def require(condition, message):
    if not condition:
        issues.append(message)

records = []
for path in sorted((ROOT / 'results').glob('grid_observation_*_grid*.json')):
    d = read(path.relative_to(ROOT))
    require(d['complete'], f'Incomplete result {path.name}')
    records.extend(d['records'])

fixed = defaultdict(Counter)
rootsets = defaultdict(set)
for r in records:
    key = (r['grid'], r['mode'], r['epsilon'])
    rootsets[key].add(r['root'])
    for p in r['fixed_policies']:
        fixed[key]['problems'] += 1
        if p.get('unresolved'):
            issues.append(f'Unresolved fixed policy {key}')
            continue
        expected = label(p['worst_error'])
        require(all(p[k] == value for k,value in expected.items()), f'Wrong stored label {r["root"]}, {key}, {p["target"]}')
        for k,value in expected.items():
            fixed[key][k] += value

def stability(a,b):
    # Independently compute union lengths, overlap and endpoint Hausdorff.
    length=lambda s:sum(v-u for u,v in s)
    if not a and not b:return 'vacuous'
    if not a or not b:return 'fail'
    overlap=sum(max(0., min(y,v)-max(x,u)) for x,y in a for u,v in b)
    fine=length(b); sym=max(0., length(a)+fine-2*overlap)
    ae=[x for z in a for x in z];be=[x for z in b for x in z]
    h=max(max(min(abs(x-y) for y in be) for x in ae), max(min(abs(x-y) for y in ae) for x in be))
    return 'positive_pass' if ((a==b) if fine==0 else sym/fine<=.05) and h<=.001 else 'fail'

lookup={(r['root'],r['grid'],r['mode'],r['epsilon']):r for r in records}
stabilities={}
for coarse,fine in [(41,81),(81,161),(161,321)]:
    count=Counter()
    for r in records:
        if r['grid']!=fine:continue
        old=lookup[r['root'],coarse,r['mode'],r['epsilon']]
        count[stability(old['intervals'],r['intervals'])]+=1
    stabilities[f'{coarse}_to_{fine}']=dict(count)

deployment={}
for phase,filename in [('primary','grid_observation_deployment.json'),('adaptive','grid_observation_adaptive_deployment.json')]:
    d=read('audit/'+filename); counts=Counter(); bins=defaultdict(Counter); flips=Counter(); maxexcess=0.
    for case in d['cases']:
        cohort='fixed' if 'fixed' in case['cohort'] else 'component'
        pos='fixed' if cohort=='fixed' else ','.join(s['position'] for s in case['selection'])
        key=cohort+':'+pos
        counts['cases']+=1;counts[cohort+'_cases']+=1
        counts[cohort+'_episodes']+=len(case['episodes'])
        counts['logical_intervals']+=len(case['first_stages'])+len(case['episodes'])
        bins[key]['cases']+=1
        complete=all(e['complete'] for e in case['episodes']) and case['complete']
        physical=complete and all(e['physical_valid'] for e in case['episodes'])
        require(complete and physical, f'Incomplete or invalid deployment {phase}:{case["id"]}')
        err=max(e['error'] for e in case['episodes'])
        require(err==case['maximum_error'], f'Max error differs {phase}:{case["id"]}')
        expected={k:physical and value for k,value in label(err).items()}
        require(expected==case['all_paths_bands'],f'Deployment labels differ {phase}:{case["id"]}')
        for band,passed in expected.items():
            bins[key][band]+=passed
            pred=case['predicted_policy'][band]
            bins[key]['predicted_'+band]+=pred
            if pred != passed:
                flips[f'{cohort}:{pos}:{band}:{"positive_to_failure" if pred else "negative_to_success"}']+=1
                if pred:maxexcess=max(maxexcess,err-BANDS[band])
        require(not case['exceptions'],f'Controller exception {phase}:{case["id"]}')
    for t in d['transitions']:
        require(t['complete'] and t['comparison']['passed'],f'Failed numerical transition {phase}:{t["id"]}')
    deployment[phase]={'counts':dict(counts),'groups':dict(bins),'flips':dict(flips),'maximum_positive_to_failure_excess':maxexcess,
        'unique_transitions':len(d['transitions']), 'max_ratio_discrepancy':max(t['comparison']['log_ratio_error'] for t in d['transitions']),
        'max_amount_allowance_fraction':max(t['comparison']['max_amount_error_over_allowance'] for t in d['transitions']),
        'saved_overall_audit_passed':d['passed']}

continuous=defaultdict(Counter); paired=defaultdict(dict); maxmatched=maxchain=maxamount=0.
continuous_files=sorted((ROOT/'results/continuous').glob('*_target_*_*.json'))
for path in continuous_files:
    d=read(path.relative_to(ROOT));mode=d['mode'];c=continuous[mode];c['problems']+=1;dep=d['deployment'];opt=d['optimization']
    require(d['complete'] and dep['complete'],f'Continuous incomplete {path.name}')
    require(len(dep['paths'])==9,f'Continuous denominator {path.name}')
    c['paths']+=len(dep['paths']);c['matched_intervals']+=len(dep['first'])+len(dep['paths'])
    physical=all(p['physical_valid'] and p['complete'] for p in dep['paths'])
    require(physical and opt['selected']['valid'],f'Continuous physical invalid {path.name}')
    err=max(p['error'] for p in dep['paths']); require(err==dep['deployed_error'],f'Continuous max {path.name}')
    predicted=label(opt['selected']['max_error']);actual=label(err)
    require(actual==dep['deployed_labels'],f'Continuous labels {path.name}')
    c['planned_deployed_labels_equal']+=predicted==actual
    for band,yes in actual.items():c[band]+=physical and yes
    for f in dep['first']:
        require(f['comparison']['passed'],f'Continuous first numerical {path.name}')
        maxmatched=max(maxmatched,f['comparison']['log_ratio_error']);maxamount=max(maxamount,f['comparison']['max_amount_error_over_allowance'])
    for p in dep['paths']:
        for prefix in ['matched','chained']:
            require(p[prefix+'_comparison']['passed'] and p[prefix+'_labels_agree'],f'Continuous path numerical {path.name}')
        maxmatched=max(maxmatched,p['matched_comparison']['log_ratio_error']);maxchain=max(maxchain,p['chained_comparison']['log_ratio_error'])
        maxamount=max(maxamount,p['matched_comparison']['max_amount_error_over_allowance'])
    c['additional_transitions']+=opt['costs']['transitions']
    for s in opt['solver_statuses']:
        c[s['stage']+'_calls']+=1;c[s['stage']+'_nonconverged']+=not s['success']
    for s in dep['second_policies']:
        c['online_decisions']+=1;c['online_transitions']+=s['costs']['transitions']
        for status in s['solver_statuses']:
            c['scalar_refinements']+=1;c['scalar_nonconverged']+=not status['success']
    ledger=ROOT/'results/continuous'/d['candidate_ledger']
    inputs[str(ledger.relative_to(ROOT))]=hashlib.sha256(ledger.read_bytes()).hexdigest()
    require(inputs[str(ledger.relative_to(ROOT))]==d['candidate_ledger_sha256'],f'Candidate ledger hash {path.name}')
    with gzip.open(ledger,'rt') as f: candidates=json.load(f)
    if isinstance(candidates,dict): candidates=candidates['candidates']
    c['candidates']+=len(candidates); c['invalid_candidates']+=sum(not x['valid'] for x in candidates)
    require(len(candidates)==opt['costs']['unique_candidates'],f'Ledger count {path.name}')
    best=min(x['max_error'] for x in candidates if x['valid'])
    require(best==opt['selected']['max_error'],f'Selected error not best {path.name}')
    paired[d['root'],d['target']][mode]=err

pair_counts=Counter()
for key,value in paired.items():
    diff=value['open']-value['causal']
    pair_counts['causal_lower' if diff>2e-7 else 'open_lower' if diff< -2e-7 else 'within_scale']+=1

out={'scope':'Fresh independent aggregation of saved result/case/candidate records, not a rerun of all model transitions.',
     'fixed_grid_counts':[{'grid':k[0],'mode':k[1],'epsilon':k[2],'roots':len(rootsets[k]),**v} for k,v in fixed.items()],
     'stability':stabilities,'deployment':deployment,'continuous':dict(continuous),'continuous_paired_counts':dict(pair_counts),
     'continuous_numerical_maxima':{'matched_log_ratio':maxmatched,'chained_log_ratio':maxchain,'amount_allowance_fraction':maxamount},
     'issues':issues,'passed_recount':not issues,'input_sha256':inputs}
(OUT/'saved_evidence_recount.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['input_sha256','fixed_grid_counts']},indent=2))
