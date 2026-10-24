"""Publication figures and descriptive summaries; no new scientific selections."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/name).read_text())
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.bbox':'tight'})
COLORS={'early':'#20639B','uniform':'#278A76','late':'#C75B39'}
def save(fig,name):
    for extension in ('pdf','png'):fig.savefig(ROOT/'figures'/f'{name}.{extension}',dpi=200)
    plt.close(fig)

roots=read('results/roots.json');sets=read('results/target_sets.json');extra=read('results/target_set_sensitivity.json');primary=read('results/primary_design.json');grid=read('results/grid_sensitivity.json')
lookup={r['id']:r for r in roots};ss={r['root']:r for r in sets}
summary={'fixed_21':{'problems':len(primary),'causal_recoverable':sum(bool(r['values']['safe']) for r in primary),'common_open_recoverable':sum(r['values']['open']['safe_count']>0 for r in primary)},'components_21':sum(len(r['feedback']) for r in sets),'nonempty_roots_21':sum(bool(r['feedback']) for r in sets),'grids':{},'matched_volume':[]}
for n in (11,21,41):
    rows=[r for r in sets if r['root'].endswith('uniform')] if n==21 else [r for r in extra if r['grid']==n]
    summary['grids'][str(n)]={'uniform_roots':len(rows),'components':sum(len(r['feedback']) for r in rows),'feedback_log_length':sum(r['feedback_log_length'] for r in rows),'open_log_length':sum(r['open_log_length'] for r in rows),'per_root':[dict(root=r['root'],feedback_log_length=r['feedback_log_length'],open_log_length=r['open_log_length'],feedback_open_length_ratio=r['feedback_log_length']/r['open_log_length'] if r['open_log_length'] else None) for r in rows]}
for fam in ('analytic','table'):
    for vol in (.9,1.3,1.7):
        group=[r for r in roots if r['family']==fam and r['volume']==vol]
        summary['matched_volume'].append(dict(family=fam,volume=vol,states=[dict(timing=r['timing'],ratio=float(np.exp(r['state'][1]-r['state'][2])),substrate=r['state'][3]/r['state'][0],biomass=float(np.exp(r['state'][1])+np.exp(r['state'][2])),feedback_components=len(ss[r['id']]['feedback']),feedback_log_length=ss[r['id']]['feedback_log_length']) for r in group]))
(ROOT/'results/maps_summary.json').write_text(json.dumps(summary,indent=2)+'\n')

fig,axes=plt.subplots(2,2,figsize=(7.1,5.4),sharex=True,layout='constrained')
for j,fam in enumerate(('analytic','table')):
    for timing in ('early','uniform','late'):
        rows=sorted([r for r in roots if r['family']==fam and r['timing']==timing],key=lambda r:r['volume']);x=[r['volume'] for r in rows]
        axes[0,j].plot(x,[np.exp(r['state'][1]-r['state'][2]) for r in rows],'o-',color=COLORS[timing],label=timing.title(),lw=1.5,ms=4)
        axes[1,j].plot(x,[ss[r['id']]['feedback_log_length'] for r in rows],'o-',color=COLORS[timing],lw=1.5,ms=4)
    axes[0,j].set_title(('a  ' if j==0 else 'b  ')+fam.title()+' family')
    axes[1,j].set_xlabel('Volume after the common 10 h prefix (L)')
    axes[1,j].set_xticks([.9,1.3,1.7]);axes[1,j].ticklabel_format(axis='y',style='sci',scilimits=(-2,-2))
    for ax in axes[:,j]:ax.grid(alpha=.15)
axes[0,0].set_ylabel('Biomass ratio at recovery start');axes[1,0].set_ylabel('Admissible log-target length\n(21-action numerical model)')
axes[0,0].legend(frameon=False,ncol=3,loc='upper left',bbox_to_anchor=(-.02,1.02));axes[0,0].set_ylim(1,5.3);axes[0,1].set_ylim(.5,5.3)
save(fig,'matched_histories')

fig,axes=plt.subplots(2,3,figsize=(7.1,4.4),sharex=True,layout='constrained')
for j,r in enumerate([x for x in extra if x['grid']==41]):
    ax=axes.flat[j]
    for y,key,color in [(1,'feedback','#20639B'),(0,'open','#C75B39')]:
        for lo,hi in r[key]:ax.plot([lo,hi],[y,y],color=color,lw=8,solid_capstyle='butt')
        if not r[key]:ax.text(.75,y,'empty',ha='center',va='center',fontsize=8,color='#707070')
    root=lookup[r['root']];ax.set_title(f"{root['family'].title()}, {root['volume']:.1f} L")
    ax.set_yticks([0,1],['Open','Causal']);ax.set_ylim(-.5,1.5);ax.set_xlim(.25,1.5);ax.grid(axis='x',alpha=.15)
    if j>=3:ax.set_xlabel('Admitted log target, log(r*)')
save(fig,'refined_target_sets')

fig,axes=plt.subplots(1,2,figsize=(7.1,3.3),layout='constrained')
for fam,ax in zip(('analytic','table'),axes):
    for vol,color in zip((.9,1.3,1.7),('#20639B','#278A76','#C75B39')):
        name=f'{fam}_v{vol:.1f}_uniform'
        vals=[next(r for r in sets+extra if r['root']==name and r['grid']==n) for n in (11,21,41)]
        ax.plot([11,21,41],[r['feedback_log_length'] for r in vals],'o-',color=color,label=f'{vol:.1f} L causal',ms=4)
        ax.plot([11,21,41],[r['open_log_length'] for r in vals],'s--',color=color,ms=3,lw=1)
    ax.set_title(fam.title()+' family');ax.set_xlabel('Number of raw feed actions');ax.set_xticks([11,21,41]);ax.grid(alpha=.15)
axes[0].set_ylabel('Total admitted log-target length');axes[0].legend(frameon=False,fontsize=7)
axes[1].text(.03,.98,'Solid: causal\nDashed: common open',transform=axes[1].transAxes,va='top',fontsize=8)
save(fig,'grid_dependence')
print(json.dumps(summary,indent=2))
