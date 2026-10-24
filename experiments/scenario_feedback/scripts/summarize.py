"""Summarize all assigned cases; retain failures and paired/cluster structure."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import ROOT,ACCEPT
from run_study import save
LABEL={'ce_limited':'CE direct (5)','ce_direct':'CE direct (12)','scenario_open':'Scenarios, open loop','scenario_feedback':'Scenarios, feedback','scenario_managed':'Feedback + manager'}
COLORS=['#657482','#334f6d','#b18b38','#177b89','#a44554']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--label',default='locked');args=ap.parse_args()
    src=ROOT/'results'/args.label;man=json.loads((src/'execution.json').read_text());methods=man['methods']
    rows=[json.loads((src/f'{c["id"]}__{m}.json').read_text()) for c in man['cases'] for m in methods]
    by={(r['case']['id'],r['method']):r for r in rows};summary={}
    for method in methods:
        rr=[r for r in rows if r['method']==method];stages=[s for r in rr for s in r.get('stages',[])]
        grouped={}
        for field in ('family','regime'):
            grouped[field]={g:{'success':sum(r['result']['terminal_feasible'] for r in rr if r['case'][field]==g),
                'n':sum(r['case'][field]==g for r in rr)} for g in sorted({r['case'][field] for r in rr})}
        events=[e for s in stages for e in s['decision'].get('events',[])]
        summary[method]={'n':len(rr),'success':sum(r['result']['terminal_feasible'] for r in rr),
            'operational_success':sum(r['result'].get('operational_success',False) for r in rr),
            'physical_invalid':sum(not r['result'].get('physical_valid',False) for r in rr),
            'execution_failures':sum(r['status']!='completed' for r in rr),
            'controller_exception_episodes':sum(r['result'].get('controller_failure',True) for r in rr),
            'total_seconds':sum(r['controller_seconds'] for r in rr),'estimator_seconds':sum(r.get('estimator_seconds',0) for r in rr),
            'model_calls':sum(r.get('model_calls',0) for r in rr),'median_seconds':float(np.median([r['controller_seconds'] for r in rr])),
            'median_error':float(np.median([r['result'].get('log_ratio_error',np.nan) for r in rr])),
            'fallback_updates':sum(s['decision'].get('fallback',False) for s in stages),'total_updates':len(stages),
            'accepted_updates':sum(e['action']=='accept' for e in events),'corrected_accepts':sum(e['action']=='accept' and e['attempt']>0 for e in events),
            'rejected_proposals':sum(e['action']=='reject' for e in events),'grouped':grouped}
    pairs={}
    for candidate,base in [('scenario_feedback','ce_direct'),('scenario_managed','scenario_feedback'),('scenario_managed','ce_limited'),('scenario_feedback','scenario_open')]:
        diffs=[];joint=[];cluster=defaultdict(list)
        for c in man['cases']:
            a,b=by[c['id'],candidate],by[c['id'],base];d=int(a['result']['terminal_feasible'])-int(b['result']['terminal_feasible'])
            diffs.append(d);cluster[c['cluster']].append(d)
            if a['result']['terminal_feasible'] and b['result']['terminal_feasible']:
                joint.append({'case':c['id'],'time_ratio':a['controller_seconds']/b['controller_seconds'],
                    'biomass_ratio':a['result']['biomass_amount_gDW']/b['result']['biomass_amount_gDW']})
        means=np.array([np.mean(v) for v in cluster.values()]);rng=np.random.default_rng(241024)
        strata=[np.array([np.mean(v) for k,v in cluster.items() if k.startswith(family+'_')]) for family in ('table','analytic')]
        boot=np.mean([rng.choice(v,size=(10000,len(v)),replace=True).mean(axis=1) for v in strata],axis=0)
        pairs[candidate+'__'+base]={'success_difference':sum(diffs),'wins':sum(d>0 for d in diffs),'losses':sum(d<0 for d in diffs),
            'cluster_mean_rate_difference':float(means.mean()),'descriptive_family_stratified_cluster_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
            'joint_n':len(joint),'faster_with_biomass_loss_le_0_5pct':sum(j['time_ratio']<1 and j['biomass_ratio']>=.995 for j in joint),
            'median_joint_time_ratio':float(np.median([j['time_ratio'] for j in joint])) if joint else None,
            'median_joint_biomass_ratio':float(np.median([j['biomass_ratio'] for j in joint])) if joint else None,'joint_pairs':joint}
    data={'episodes':len(rows),'summary':summary,'pairs':pairs,'timing_scope':'Parallel wall times and ratios are diagnostic only; isolated speed comparisons use timing_summary.json.'};save(ROOT/'results/summary.json',data)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
    fig,ax=plt.subplots(1,3,figsize=(12.6,4.1));xx=np.arange(len(methods));labels=[LABEL[m] for m in methods]
    ax[0].barh(xx,[summary[m]['success'] for m in methods],color=COLORS)
    ax[0].set_yticks(xx,labels);ax[0].invert_yaxis();ax[0].set_xlim(0,len(man['cases'])*1.2);ax[0].set_xlabel('Actual feasible episodes');ax[0].set_title('(a) Composition and physical bounds')
    for i,m in enumerate(methods):ax[0].text(summary[m]['success']+.7,i,f"{summary[m]['success']}/{summary[m]['n']}",va='center',fontsize=8)
    errors=[[r['result'].get('log_ratio_error',np.nan) for r in rows if r['method']==m] for m in methods]
    ax[1].boxplot(errors,labels=['CE5','CE12','S-open','S-feed','Managed'],showfliers=True)
    ax[1].axhline(ACCEPT,color='#b84444',ls='--',label='Acceptance band');ax[1].set_yscale('log');ax[1].set_ylabel('Absolute terminal log-ratio error');ax[1].set_title('(b) Continuous error');ax[1].legend(fontsize=8)
    ax[2].bar(xx,[summary[m]['model_calls']/1000 for m in methods],color=COLORS);ax[2].set_xticks(xx,['CE5','CE12','S-open','S-feed','Managed'],rotation=25)
    ax[2].set_ylabel('Trajectory evaluations (thousands)');ax[2].set_title('(c) Model-evaluation workload')
    fig.tight_layout();fig.savefig(ROOT/'figures/locked_outcomes.pdf');fig.savefig(ROOT/'figures/locked_outcomes.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for i,family in enumerate(('table','analytic')):
        for m,color in zip(methods,COLORS):
            rr=[r for r in rows if r['method']==m and r['case']['family']==family]
            cap=np.array([[1.8-s[0] for s in r['states']] for r in rr if 'states' in r])
            axes[i].plot(np.arange(8)*2,np.median(cap,axis=0),label=LABEL[m],color=color)
        axes[i].set(xlabel='Time (h)',ylabel='Median remaining volume (L)',title=family.capitalize()+' process family');axes[i].set_ylim(-.01,1.34)
    axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/'figures/remaining_volume.pdf');fig.savefig(ROOT/'figures/remaining_volume.png',dpi=170);plt.close(fig)
    print(json.dumps({m:{k:v for k,v in s.items() if k!='grouped'} for m,s in summary.items()},indent=2))
if __name__=='__main__':main()
