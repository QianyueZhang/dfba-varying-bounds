"""Publication figure of every assigned continuous comparison, without selection."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'results/continuous/formal_report.json';report=json.loads(source.read_text())
    assert report['complete'] and len(report['paired'])==36
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,
                         'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
                         'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.8),sharex=True,sharey=True)
    minimum=min(min(r['open_error'],r['causal_error']) for r in report['paired'])
    maximum=max(max(r['open_error'],r['causal_error']) for r in report['paired'])
    for i,family in enumerate(['analytic','table']):
        for j,volume in enumerate([.9,1.3,1.7]):
            ax=axes[i,j];root=f'{family}_v{volume}_uniform'
            rows=sorted([r for r in report['paired'] if r['root']==root],key=lambda r:r['target'])
            assert len(rows)==6
            x=[r['target'] for r in rows]
            ax.plot(x,[r['open_error'] for r in rows],color='#24649b',marker='o',
                    markerfacecolor='white',markersize=4.5,lw=1.2,label='Common open')
            ax.plot(x,[r['causal_error'] for r in rows],color='#138974',marker='^',
                    markersize=3.8,lw=1.1,label='Causal deployment')
            ax.axhline(.00748,color='.20',ls='--',lw=.8)
            ax.axhline(np.log(1.01),color='.55',ls=':',lw=.8)
            ax.set_yscale('log');ax.set_ylim(minimum*.6,maximum*1.45)
            ax.set_xticks([.5,.75,1,1.25,1.5,2]);ax.set_xticklabels(['.5','.75','1','1.25','1.5','2'])
            ax.grid(axis='y',which='major',alpha=.15)
            ax.set_title(f'{chr(97+i*3+j)}  {family.capitalize()}, {volume:g} L',loc='left')
            if i==1:ax.set_xlabel('Requested biomass ratio')
            if j==0:ax.set_ylabel('Worst terminal log-ratio error')
    handles=[Line2D([],[],color='#24649b',marker='o',markerfacecolor='white',lw=1.2,label='Common open'),
             Line2D([],[],color='#138974',marker='^',lw=1.1,label='Causal deployment'),
             Line2D([],[],color='.20',ls='--',lw=.8,label='Inner band'),
             Line2D([],[],color='.55',ls=':',lw=.8,label='Actual acceptance')]
    fig.legend(handles=handles,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,1.02),fontsize=9)
    fig.tight_layout(rect=(0,0,1,.94));destination=ROOT/'figures';destination.mkdir(parents=True,exist_ok=True)
    for suffix in ['pdf','png']:fig.savefig(destination/f'continuous_comparison.{suffix}',dpi=220,bbox_inches='tight')
    manifest=dict(source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=36,
                  series='Worst independent terminal log-ratio error over all9 paths, all36 assigned problems. Causal values are deployed actual-state reoptimizations, not planned tree values.',
                  outputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.glob('continuous_comparison.*')})
    (ROOT/'results/continuous/figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
