"""Recount saved independent outcomes and produce the selected-witness figure."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'audit/deployment_results.json').read_text());assert r['complete']
s=r['summary'];pairs=[]
for family in ('analytic','table'):
    a={ (v['root'],v['target']):v for v in s['exploratory']['per_method']['recovery_economic']['root_target_worst_cases'] if v['root'].startswith(family)}
    b={ (v['root'],v['target']):v for v in s['exploratory']['per_method']['recovery_filter']['root_target_worst_cases'] if v['root'].startswith(family)}
    for key,x in a.items():
        y=b[key];pairs.append(dict(root=key[0],target=key[1],family=family,economic_error=x['maximum_log_ratio_error'],filter_error=y['maximum_log_ratio_error'],economic_minB=x['minimum_biomass_gDW'],filter_minB=y['minimum_biomass_gDW'],filter_economic_minB_ratio=y['minimum_biomass_gDW']/x['minimum_biomass_gDW']))
out={'selected_witness_pairs':pairs,'conditional_minB_ratios':{f:dict(n=len(q:=[p for p in pairs if f=='all' or p['family']==f]),median=float(np.median([p['filter_economic_minB_ratio'] for p in q])),minimum=float(np.min([p['filter_economic_minB_ratio'] for p in q])),maximum=float(np.max([p['filter_economic_minB_ratio'] for p in q]))) for f in ('all','analytic','table')},'summary':s}
# Independent arithmetic recount: compare stored labels to terminal observables.
for scope in ('primary','ood','exploratory'):
    for ep in r[scope+'_episodes']:
        assert ep['complete'] and ep['physical_valid']
        for band,delta in [('inner',.00748+1e-10),('outer',.0075),('actual',np.log(1.01))]:assert ep['terminal_band_success'][band] == (ep['terminal_log_ratio_error']<=delta)
(ROOT/'results/deployment_analysis.json').write_text(json.dumps(out,indent=2)+'\n')
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
fig,axes=plt.subplots(2,1,figsize=(7.1,4.7),sharex=True,layout='constrained');x=np.arange(1,len(pairs)+1)
witnesses=json.loads((ROOT/'results/target_set_witnesses.json').read_text());w={(v['root'],v['target']):v for v in witnesses};ordered=[w[(p['root'],p['target'])]['choices']['recovery_filter'] for p in pairs]
axes[0].plot(x,[p['proposal']['quantized'] for p in ordered],'x--',lw=.8,ms=4,color='#999',label='Affine proposal (quantized)')
axes[0].plot(x,[p['feed'] for p in ordered],'o-',lw=1,ms=3,color='#20639B',label='Selected recovery action')
axes[0].set_ylabel('First feed (L/h)');axes[0].legend(frameon=False,fontsize=8,ncol=2);axes[0].grid(alpha=.15)
axes[1].plot(x,[p['filter_error'] for p in pairs],'o-',color='#278A76',ms=3,lw=1,label='Independent worst-path error');axes[1].axhline(.00748,color='#555',ls='--',lw=1,label='Inner band');axes[1].set_ylabel('Worst terminal\nlog-ratio error');axes[1].set_xlabel('Selected component midpoint (21-action target sets)');axes[1].grid(alpha=.15)
for ax in axes:ax.axvline(35.5,color='#888',ls=':',lw=1)
axes[1].text(.03,.1,'Analytic (35)',transform=axes[1].transAxes,fontsize=8);axes[1].text(.8,.1,'Table (6)',transform=axes[1].transAxes,fontsize=8)
for ext in ('pdf','png'):fig.savefig(ROOT/'figures'/f'selected_witnesses.{ext}',dpi=200)
print(json.dumps(out['conditional_minB_ratios'],indent=2))
