"""Serial timing evidence, independent of concurrent effectiveness wall times."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import ROOT
from run_study import save
from summarize import LABEL,COLORS
r=ROOT/'results/timing_serial';man=json.loads((r/'execution.json').read_text());main=ROOT/'results/locked';rows=[];compare=[]
for c in man['cases']:
    for m in man['methods']:
        name=f'{c["id"]}__{m}.json';d=json.loads((r/name).read_text());old=json.loads((main/name).read_text());rows.append(d)
        compare.append({'case':c['id'],'method':m,'same_classification':d['result']['terminal_feasible']==old['result']['terminal_feasible'],
            'absolute_error_difference':abs(d['result']['log_ratio_error']-old['result']['log_ratio_error']),
            'relative_biomass_difference':abs(d['result']['biomass_amount_gDW']/old['result']['biomass_amount_gDW']-1)})
summary={}
for m in man['methods']:
    rr=[x for x in rows if x['method']==m];seconds=[x['controller_seconds'] for x in rr]
    summary[m]={'n':len(rr),'total_seconds':sum(seconds),'median_seconds':float(np.median(seconds)),
        'p90_seconds':float(np.quantile(seconds,.9)),'estimator_seconds':sum(x['estimator_seconds'] for x in rr),
        'model_calls':sum(x['model_calls'] for x in rr),'success':sum(x['result']['terminal_feasible'] for x in rr)}
by={(x['case']['id'],x['method']):x for x in rows};paired={}
for base in ('scenario_feedback','ce_limited','ce_direct'):
    panel=[]
    for c in man['cases']:
        a,b=by[c['id'],'scenario_managed'],by[c['id'],base]
        panel.append({'case':c['id'],'jointly_feasible':a['result']['terminal_feasible'] and b['result']['terminal_feasible'],
            'time_ratio':a['controller_seconds']/b['controller_seconds'],
            'biomass_ratio':a['result']['biomass_amount_gDW']/b['result']['biomass_amount_gDW']})
    joint=[x for x in panel if x['jointly_feasible']]
    paired[base]={'all_pairs':panel,'joint_n':len(joint),
        'faster_with_biomass_loss_le_0_5pct':sum(x['time_ratio']<1 and x['biomass_ratio']>=.995 for x in joint),
        'median_joint_time_ratio':float(np.median([x['time_ratio'] for x in joint])) if joint else None}
save(ROOT/'results/timing_summary.json',{'summary':summary,'paired_managed_vs':paired,'verification':compare,
    'scope':'Eight preselected cases, one isolated serial repetition per method; no timing-distribution or universal speed claim.'})
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(1,2,figsize=(9.2,4));methods=man['methods'];idx=np.arange(len(methods))
ax[0].barh(idx,[summary[m]['total_seconds'] for m in methods],color=COLORS);ax[0].set_yticks(idx,[LABEL[m] for m in methods]);ax[0].invert_yaxis();ax[0].set_xlabel('Total isolated online time (s)');ax[0].set_title('(a) Serial eight-case panel')
for c in man['cases']:
    a,b=by[c['id'],'scenario_managed'],by[c['id'],'scenario_feedback'];joint=a['result']['terminal_feasible'] and b['result']['terminal_feasible']
    ax[1].scatter(a['controller_seconds']/b['controller_seconds'],a['result']['biomass_amount_gDW']/b['result']['biomass_amount_gDW'],
        marker='o' if joint else 'x',color='#177b89' if joint else '#b04444',s=40)
ax[1].axvline(1,color='gray',ls='--');ax[1].axhline(.995,color='gray',ls='--');ax[1].set_xlabel('Managed / feedback direct time');ax[1].set_ylabel('Managed / feedback direct biomass');ax[1].set_title('(b) Paired achieved tradeoff')
ax[1].text(.98,.98,'Circle: jointly feasible\nCross: at least one fails',transform=ax[1].transAxes,ha='right',va='top',fontsize=8,
    bbox={'facecolor':'white','edgecolor':'none','alpha':.95,'pad':3})
fig.tight_layout();fig.savefig(ROOT/'figures/serial_timing.pdf',bbox_inches='tight');fig.savefig(ROOT/'figures/serial_timing.png',dpi=170,bbox_inches='tight')
print(json.dumps(summary,indent=2));print('classification_differences',sum(not x['same_classification'] for x in compare))
