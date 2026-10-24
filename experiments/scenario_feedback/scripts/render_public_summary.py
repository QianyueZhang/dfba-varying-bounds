import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[1]
p=json.loads((r/'public_validation/results/results.json').read_text())
s=json.loads((r/'public_validation/secondary_published_model/results.json').read_text())
print(s.keys())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(9,3.3))
for ax,mode,title in zip(axs,['free_run','one_step'],['(a) Complete free rollout','(b) One-step prediction']):
    primary=[next(x for x in p['summaries'] if x['model']==m and x['mode']==mode and x['window']=='full') for m in ['linear','dose_informed']]
    secondary=next(x for x in s['summaries'] if x['mode']==mode and x['window']=='full')
    values=[primary[0]['macro_baseline_rmse'],primary[0]['macro_rmse'],primary[1]['macro_rmse'],secondary['macro_rmse']]
    ax.barh(range(4),values,color=['#7c858e','#326481','#148c8c','#b98242']);ax.invert_yaxis()
    ax.set_yticks(range(4),['Simple baseline','Linear response','Dose informed','Published model*'])
    ax.set_xlim(0,max(values)*1.22);ax.set_xlabel('Mean per-run RMSE (fraction units)');ax.set_title(title)
    for i,v in enumerate(values):ax.text(v+max(values)*.02,i,f'{v:.4f}',va='center',fontsize=9)
fig.text(.5,.005,'All eight trajectories retained. *Secondary post hoc replay; unchanged published parameters.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.04,1,1));fig.savefig(r/'figures/public_prediction.pdf',bbox_inches='tight');fig.savefig(r/'figures/public_prediction.png',dpi=180,bbox_inches='tight')
