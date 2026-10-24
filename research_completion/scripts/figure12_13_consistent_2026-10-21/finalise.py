"""Validate, plot and describe the consistently solved reactor comparison."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RC=Path(__file__).resolve().parents[2];ROOT=RC.parent
OUT=RC/'runs/figure12_13_consistent_2026-10-21'
FIG=RC/'figures/figure12_13_consistent_2026-10-21';FIG.mkdir(parents=True,exist_ok=True)
KEYS=['biomass_g_L','glucose_g_L','ethanol_g_L','biomass_g','glucose_g','ethanol_g']
records={};trajectories={};summary={}
for mode in ['batch','fed_batch']:
    for method in ['concentration','direct']:
        records[mode,method]=json.loads((OUT/f'{mode}_{method}_audit.json').read_text())
        trajectories[mode,method]=np.genfromtxt(OUT/f'{mode}_{method}_trajectory.csv',delimiter=',',names=True)
        d=trajectories[mode,method];flow=0 if mode=='batch' else .5
        assert np.all(np.diff(d['time_h'])>0) and d['time_h'][-1]==14
        assert all(np.all(np.isfinite(d[k])) and np.min(d[k])>=0 for k in d.dtype.names)
        assert np.max(abs(d['volume_L']-(5+flow*d['time_h'])))<1e-12
        for name in ['biomass','glucose','ethanol']:
            assert np.max(abs(d[name+'_g']-d[name+'_g_L']*d['volume_L']))<1e-12
        assert np.all(d['glucose_g']<=112.5+flow*22.5*d['time_h']+1e-10)
        for row in records[mode,method]['sampled_checks']:
            assert row['minimum_slack']>0 and -1e-8<=row['LP_gap']<=row['nominal_gap_bound']+1e-8
    a=trajectories[mode,'concentration'];b=trajectories[mode,'direct']
    delta={key:float(np.max(abs(a[key]-PchipInterpolator(b['time_h'],b[key])(a['time_h']))))for key in KEYS}
    summary[mode]={'concentration_vs_amount_max_abs':delta,
        'adopted_terminal':[float(b[k][-1])for k in KEYS],
        'direct_events':records[mode,'direct']['events'],'concentration_events':records[mode,'concentration']['events'],
        'maximum_concentration_equality_residual':max(r['equality_residual']for r in records[mode,'concentration']['sampled_checks']),
        'maximum_concentration_primal_dual_residual':max(r['primal_dual_residual']for r in records[mode,'concentration']['sampled_checks']),
        'minimum_concentration_slack':min(r['minimum_slack']for r in records[mode,'concentration']['sampled_checks']),
        'maximum_direct_equality_residual':max(r['equality_residual']for r in records[mode,'direct']['sampled_checks']),
        'maximum_direct_primal_dual_residual':max(r['primal_dual_residual']for r in records[mode,'direct']['sampled_checks']),
        'maximum_LP_gap':max(r['LP_gap']for r in records[mode,'direct']['sampled_checks']),
    }
    assert summary[mode]['minimum_concentration_slack']>0
    assert summary[mode]['maximum_concentration_equality_residual']<1e-9
    assert summary[mode]['maximum_direct_primal_dual_residual']<1e-9
    assert max(delta[k] for k in KEYS[:3])<1e-3,summary[mode]

plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'consistent-reactor-2026-10-21'})
for number,keys,labels in [(12,KEYS[:3]+['volume_L'],['Biomass (gDW L$^{-1}$)','Glucose (g L$^{-1}$)','Ethanol (g L$^{-1}$)','Volume (L)']),
                           (13,KEYS[3:]+['volume_L'],['Biomass amount (gDW)','Glucose amount (g)','Ethanol amount (g)','Volume (L)'])]:
    fig,axes=plt.subplots(2,2,figsize=(7.2,6.1),sharex=True,constrained_layout=True)
    for i,(ax,key,label)in enumerate(zip(axes.ravel(),keys,labels)):
        for mode,style,color,marker,legend in [('batch','-','#1f4e79','s','Batch'),('fed_batch','--','#ad4e32','^','Fed-batch')]:
            d=trajectories[mode,'direct'];ax.plot(d['time_h'],d[key],style,color=color,lw=1.7,marker=marker,markevery=28,ms=4,markerfacecolor='white',label=legend)
        ax.set(xlim=(-.15,14.2),xticks=np.arange(0,15,2),ylim=(0,None),ylabel=label);ax.grid(axis='y',alpha=.2);ax.text(-.13,1.04,f'({chr(97+i)})',transform=ax.transAxes,fontweight='bold')
        if i>1:ax.set_xlabel('Time (h)')
    axes[0,0].legend(frameon=False)
    fig.savefig(FIG/f'figure_{number}.png',dpi=300);fig.savefig(FIG/f'figure_{number}.svg',metadata={'Date':None});plt.close(fig)

report={'adopted_method':'Direct KKT amount integration, independently checked against a separate concentration-balance implementation at different integration tolerances',
    'verification_checks':['Finite nonnegative output states','Strictly increasing time ending at14h','Analytic volume identity','Concentration-times-volume amount identities','Glucose inventory no greater than initial inventory plus feed','Positive sampled slacks','Nonnegative LP objective gaps within2nmu','Independent concentration/amount agreement','Sampled KKT and equality residuals'],
    'comparison':summary,
    'artifacts':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in sorted(OUT.glob('*_trajectory.csv'))]+[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in sorted(FIG.glob('*'))]}
(OUT/'validation_and_adoption.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(summary,indent=2))
