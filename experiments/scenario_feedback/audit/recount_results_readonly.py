#!/usr/bin/env python3
"""Read-only stdlib recount. Writes only a separate review output when requested."""
import argparse,json,hashlib,math,statistics,collections
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
root=Path(__file__).resolve().parents[1];src=root/'results/locked'
manifest=json.loads((src/'execution.json').read_text());complete=json.loads((src/'complete.json').read_text());summary=json.loads((root/'results/summary.json').read_text())
locked=json.loads((root/'LOCK.json').read_text());inventory={x['file']:x for x in complete['inventory']}
assert len(inventory)==440==len(complete['inventory'])
methods=manifest['methods'];cases=manifest['cases'];rows={};discrepancies=[]
status_messages={m:collections.Counter() for m in methods};update_family={m:collections.defaultdict(collections.Counter) for m in methods}
counters={m:collections.Counter() for m in methods};family={m:collections.defaultdict(collections.Counter) for m in methods};regime={m:collections.defaultdict(collections.Counter) for m in methods}
for case in cases:
 for method in methods:
  name=f'{case["id"]}__{method}.json';path=src/name;blob=path.read_bytes();r=json.loads(blob)
  assert hashlib.sha256(blob).hexdigest()==inventory[name]['sha256'],name
  assert r['case']==case and r['method']==method and r['source_hashes']==manifest['hashes'],name
  assert r['status']==inventory[name]['status']=='completed',name
  stages=r['stages'];assert len(stages)==7 and len(r['states'])==8,name
  final=r['states'][-1];target=case['target_ratio'];error=abs(final[1]-final[2]-math.log(target));mass=math.exp(final[1])+math.exp(final[2])
  physical=all(0<=s['feed']<=.2 and s['true_end_state'][0]<=1.8 and s['plant_min_S']>=-2e-7 for s in stages)
  exception=any('controller_exception' in s['decision'] for s in stages)
  passed=error<=math.log(1.01) and physical
  assert passed==r['result']['terminal_feasible'] and physical==r['result']['physical_valid'] and exception==r['result']['controller_failure'],name
  assert (passed and not exception)==r['result']['operational_success'],name
  assert abs(mass-r['result']['biomass_amount_gDW'])<1e-10*max(1,mass),name
  assert abs(error-r['result']['log_ratio_error'])<1e-12,name
  c=counters[method];c['n']+=1;c['success']+=passed;c['physical_invalid']+=not physical;c['controller_exception_episodes']+=exception;c['operational_success']+=passed and not exception;c['total_updates']+=len(stages);c['model_calls']+=r['model_calls']
  for stage in stages:
   d=stage['decision'];status=d.get('status',{});status_messages[method][(bool(status.get('success',False)),status.get('message','<missing>'))]+=1
   uf=update_family[method][case['family']];uf['updates']+=1;uf['selected_numerically_feasible']+=d.get('selected',{}).get('feasible',False);uf['solver_reported_unsuccessful']+=not status.get('success',False);uf['fallbacks']+=d.get('fallback',False)
   c['fallback_updates']+=d.get('fallback',False);c['selected_numerically_feasible_updates']+=d.get('selected',{}).get('feasible',False);c['solver_reported_unsuccessful_updates']+=not d.get('status',{}).get('success',False)
   for event in d.get('events',[]):
    c['accepted_updates']+=event['action']=='accept';c['corrected_accepts']+=event['action']=='accept' and event['attempt']>0;c['rejected_proposals']+=event['action']=='reject';c['verification_errors']+=event['action']=='verification_error'
  for group,field in [(family,'family'),(regime,'regime')]:
   group[method][case[field]]['n']+=1;group[method][case[field]]['success']+=passed
  rows[case['id'],method]={'success':passed,'mass':mass,'error':error,'calls':r['model_calls'],'family':case['family'],'regime':case['regime'],'cluster':case['cluster']}
for method in methods:
 s=summary['summary'][method];c=counters[method]
 for k in ['n','success','physical_invalid','controller_exception_episodes','operational_success','total_updates','model_calls','fallback_updates','accepted_updates','corrected_accepts','rejected_proposals']:
  if s[k]!=c[k]:discrepancies.append([method,k,s[k],c[k]])
 for groups,field in [(family,'family'),(regime,'regime')]:
  if dict(groups[method])!=s['grouped'][field]:discrepancies.append([method,field])
pairs={}
requested=[('scenario_feedback','ce_direct'),('scenario_managed','scenario_feedback'),('scenario_feedback','scenario_open'),('scenario_managed','scenario_open'),('scenario_managed','ce_direct'),('scenario_open','ce_direct'),('scenario_managed','ce_limited')]
for a,b in requested:
 joint=[];wins=[];losses=[];both_failed=[];cluster=collections.defaultdict(list);byfamily={}
 for c in cases:
  ra,rb=rows[c['id'],a],rows[c['id'],b];delta=int(ra['success'])-int(rb['success']);cluster[c['cluster']].append(delta)
  if delta==1:wins.append(c['id'])
  if delta==-1:losses.append(c['id'])
  if ra['success'] and rb['success']:joint.append((c['id'],ra['mass']/rb['mass'],ra['calls']/rb['calls'],c['family']))
  if not ra['success'] and not rb['success']:both_failed.append(c['id'])
 for f in ['table','analytic']:
  j=[x for x in joint if x[3]==f]
  byfamily[f]={'joint_n':len(j),'median_biomass_ratio':statistics.median(x[1] for x in j),'median_model_call_ratio':statistics.median(x[2] for x in j),'wins':sum(x.startswith(f+'_') for x in wins),'losses':sum(x.startswith(f+'_') for x in losses)}
 k=a+'__'+b;stat={'success_difference':len(wins)-len(losses),'wins':len(wins),'losses':len(losses),'joint_n':len(joint),'both_failed_n':len(both_failed),'median_joint_biomass_ratio':statistics.median(x[1] for x in joint),'median_joint_model_call_ratio':statistics.median(x[2] for x in joint),'joint_biomass_at_least_99_5pct':sum(x[1]>=.995 for x in joint),'joint_biomass_greater':sum(x[1]>1 for x in joint),'cluster_differences':{c:sum(v) for c,v in cluster.items()},'by_family':byfamily,'wins_ids':wins,'losses_ids':losses}
 pairs[k]=stat
 if k in summary['pairs']:
  for field in ['success_difference','wins','losses','joint_n','median_joint_biomass_ratio']:
   if abs(summary['pairs'][k][field]-stat[field])>1e-12:discrepancies.append([k,field])
out={'scope':'Read-only recount from all 440 executed episode files. No simulation; no use of parallel wall-time comparisons; source outcomes reconstructed from terminal log states and physical checks.','complete_manifest_sha256':hashlib.sha256((src/'complete.json').read_bytes()).hexdigest(),'summary_sha256':hashlib.sha256((root/'results/summary.json').read_bytes()).hexdigest(),'hash_and_identity_checked_episodes':len(rows),'summary_discrepancies':discrepancies,'update_diagnostics':{m:{'status_message_distribution':[{'success':k[0],'message':k[1],'count':v} for k,v in status_messages[m].items()],'by_family':dict(update_family[m])} for m in methods},'summary':{m:dict(counters[m],family=dict(family[m]),regime=dict(regime[m])) for m in methods},'pairs':pairs,'biological_reliability_inference':False}
assert not discrepancies,discrepancies
if args.output:args.output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({**out,'pairs':{k:{kk:vv for kk,vv in v.items() if not kk.endswith('_ids')} for k,v in pairs.items()}},indent=2))
