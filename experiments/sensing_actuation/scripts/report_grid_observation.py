"""Recount saved grid/observation evidence and generate scientific figures.

Reporting only: no flow integration or policy optimization occurs here.
"""
from pathlib import Path
import json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42})


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    primary=json.loads((ROOT/'results/grid_observation.json').read_text())
    records=list(primary['records']);convergence=list(primary['convergence'])
    roots=sorted(set(r['root'] for r in records))
    adaptive_path=ROOT/'results/grid_observation_adaptive.json'
    adaptive=json.loads(adaptive_path.read_text()) if adaptive_path.exists() else None
    frontiers={r['root']:r for r in primary['sensing_frontiers']}
    adaptive_status={root:'not_triggered_or_pending' for root in roots}
    if adaptive:
        for p in adaptive['results']:
            rid=p['root']['id'] if isinstance(p['root'],dict) else p['root']
            adaptive_status[rid]='complete' if p['complete'] else 'unresolved'
            if p['complete']:
                records.extend(p['records']);convergence.extend(p['comparisons'])
                frontiers[rid]={'root':rid,'grid':321,'targets':p['sensing_frontiers']}
    lookup={(r['root'],r['grid'],r['mode'],r['epsilon']):r for r in records}
    finest={root:max(r['grid'] for r in records if r['root']==root) for root in roots}
    counts=[]
    for n in sorted(set(r['grid'] for r in records)):
        for mode,eps in [('full',None),('open',None)]+[('ratio',e) for e in [0.,.001,.005,.01,.02]]:
            selected=[r for r in records if r['grid']==n and r['mode']==mode and r['epsilon']==eps]
            counts.append(dict(grid=n,mode=mode,epsilon=eps,root_count=len(selected),problems=6*len(selected),
                               inner=sum(r['all_path_fixed_inner'] for r in selected),outer=sum(r['all_path_fixed_outer'] for r in selected),actual=sum(r['all_path_fixed_actual'] for r in selected)))
    finest_counts=[]
    for mode,eps in [('full',None),('open',None)]+[('ratio',e) for e in [0.,.001,.005,.01,.02]]:
        selected=[r for r in records if r['grid']==finest[r['root']] and r['mode']==mode and r['epsilon']==eps]
        finest_counts.append(dict(mode=mode,epsilon=eps,root_count=len(selected),problems=6*len(selected),
                                 inner=sum(r['all_path_fixed_inner'] for r in selected),outer=sum(r['all_path_fixed_outer'] for r in selected),actual=sum(r['all_path_fixed_actual'] for r in selected),
                                 grids={r['root']:r['grid'] for r in selected}))
    convergence_counts=[]
    for fine in [81,161,321]:
        cs=[c for c in convergence if c['fine']==fine]
        if cs:convergence_counts.append(dict(coarse=(fine+1)//2,fine=fine,comparisons=len(cs),passed=sum(c['passed'] for c in cs),vacuous=sum(c['status']=='both_empty_vacuous' for c in cs),positive_pass=sum(c.get('positive_convergence_evidence',False) for c in cs),failed=sum(not c['passed'] for c in cs)))
    for f in frontiers.values():
        for t in f['targets']:
            if not t['zero_error_ratio_recoverable']:t['reported_epsilon_supremum']=None
            else:t['reported_epsilon_supremum']=t.get('epsilon_supremum')
    summary=dict(primary_complete=primary['complete'],primary_algebra_passed=primary['algebra_passed'],primary_seconds=primary['elapsed_seconds'],
                 adaptive_complete=None if adaptive is None else adaptive['complete'],adaptive_seconds=None if adaptive is None else adaptive['wall_seconds'],
                 adaptive_status=adaptive_status,finest_completed_grid=finest,
                 fixed_problem_counts=counts,convergence_counts=convergence_counts,
                 finest_completed_problem_counts=finest_counts,
                 finest_sets=[{k:r[k] for k in ['root','family','volume','grid','mode','epsilon','components','singletons','log_length','ratio_length']} for r in records if r['grid']==finest[r['root']]],
                 sensing_frontiers=[frontiers[r] for r in roots])
    costs=[]
    for p in sorted((ROOT/'results/trees').glob('*.json')):
        meta=json.loads(p.read_text())
        costs.append({k:meta[k] for k in ['raw_grid','complete','new_transitions','reused_transitions','accepted_steps','rejected_steps','seconds']})
        costs[-1]['root']=meta['root']['id']
    summary['tree_cost_ledger']=costs
    diagnostic_path=ROOT/'results/diagnostic_set_hausdorff.json'
    if diagnostic_path.exists():
        diagnostic=json.loads(diagnostic_path.read_text())
        summary['post_hoc_set_distance_counts']=[dict(fine=n,
            nonempty_comparisons=sum(r['fine']==n and r['set_hausdorff']['status']=='nonempty' for r in diagnostic['records']),
            nonempty_at_most_001=sum(r['fine']==n and r['set_hausdorff']['status']=='nonempty' and r['diagnostic_set_distance_at_most_001'] for r in diagnostic['records'])) for n in [81,161,321]]
    audit_summaries=[]
    for phase,name in [('primary','grid_observation_deployment.json'),('adaptive','grid_observation_adaptive_deployment.json')]:
        p=ROOT/'audit'/name
        if not p.exists():continue
        a=json.loads(p.read_text())
        if not a.get('complete'):continue
        differences=a['band_label_differences']
        audit_summaries.append(dict(phase=phase,complete=a['complete'],passed=a['passed'],
            cases=len(a['cases']),fixed_cases=sum(c['cohort'].endswith('fixed') for c in a['cases']),
            component_cases=sum(c['cohort'].endswith('component') for c in a['cases']),
            observation_path_episodes=sum(len(c['episodes']) for c in a['cases']),
            numerical=a['numerical_summary'],optimistic_differences=sum(x['predicted'] and not x['actual'] for x in differences),
            conservative_differences=sum(not x['predicted'] and x['actual'] for x in differences),
            inputs_unchanged=a['inputs_unchanged'],elapsed_seconds=a['elapsed_seconds']))
    summary['independent_audits']=audit_summaries
    write_json(ROOT/'results/grid_observation_summary.json',summary)
    figs=ROOT/'figures';figs.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(10,5.5),constrained_layout=True)
    for ax,root in zip(axes.flat,roots):
        ns=sorted(r['grid'] for r in records if r['root']==root and r['mode']=='full')
        for mode,eps,label,color,style in [('full',None,'Full state','#1261A0','-'),('ratio',.005,'Ratio, error ≤0.005','#D17700','-.'),('open',None,'Common open','#444444','--')]:
            ax.plot(ns,[lookup[root,n,mode,eps]['log_length'] for n in ns],style,marker='o',ms=4,color=color,label=label)
        title=root.replace('_uniform','').replace('_v',' | ')+' L'
        if adaptive_status[root]=='unresolved':title+=' (321 unresolved)'
        ax.set_title(title)
        ax.set_xticks(ns);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.18)
        ax.set_xlabel('Raw feed actions')
    axes[0,0].set_ylabel('Admitted log-target length');axes[1,0].set_ylabel('Admitted log-target length')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='outside upper center',frameon=False)
    fig.savefig(figs/'maturity_grid_resolution.pdf',bbox_inches='tight');fig.savefig(figs/'maturity_grid_resolution.png',dpi=170,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.4,4.4),constrained_layout=True)
    eps=[0.,.001,.005,.01,.02]
    colors=['#1261A0','#2991C0','#72B7D2','#A03C23','#D17700','#E7AF48']
    for root,color in zip(roots,colors):
        n=finest[root];base=lookup[root,n,'full',None]['log_length']
        if base==0:continue
        values=[lookup[root,n,'ratio',e]['log_length']/base for e in eps]
        ax.plot(eps,values,marker='o',ms=4,color=color,label=root.replace('_uniform','').replace('_v',' | ')+f' L ({n})')
        openfrac=lookup[root,n,'open',None]['log_length']/base
        ax.scatter([.0208],[openfrac],marker='s',s=20,color=color)
    ax.set_xlabel('Bound on log-ratio measurement error, ε')
    ax.set_ylabel('Admitted length / full-state admitted length')
    ax.set_xticks(eps,['0','0.001','0.005','0.01','0.02'])
    ax.get_xticklabels()[0].set_ha('right')
    ax.get_xticklabels()[1].set_ha('left')
    ax.set_ylim(-.03,1.05)
    ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,fontsize=8,loc='lower left',bbox_to_anchor=(1.01,0))
    ax.text(.0208,1.02,'Open',ha='center',va='bottom',fontsize=8)
    fig.savefig(figs/'maturity_observation_resolution.pdf',bbox_inches='tight');fig.savefig(figs/'maturity_observation_resolution.png',dpi=170,bbox_inches='tight');plt.close(fig)
    targets=[.5,.75,1.,1.25,1.5,2.]
    matrix=np.full((6,6),np.nan);texts=np.full((6,6),'—',dtype=object)
    for i,root in enumerate(roots):
        for j,t in enumerate(frontiers[root]['targets']):
            if t['status']=='common_open_succeeds':texts[i,j]='open';matrix[i,j]=.022
            elif t['zero_error_ratio_recoverable']:
                matrix[i,j]=t['epsilon_supremum'];texts[i,j]=f"{t['epsilon_supremum']:.5f}"
    fig,ax=plt.subplots(figsize=(9,3.4),constrained_layout=True)
    cmap=plt.colormaps['Blues'].copy();cmap.set_bad('#EFEFEF')
    ax.imshow(matrix,cmap=cmap,vmin=0,vmax=.025,aspect='auto')
    for i in range(6):
        for j in range(6):ax.text(j,i,texts[i,j],ha='center',va='center',fontsize=9,color='white' if matrix[i,j]>.014 else '#202020')
    ax.set_xticks(range(6),[str(t) for t in targets]);ax.set_xlabel('Requested biomass ratio')
    ax.set_yticks(range(6),[r.replace('_uniform','').replace('_v',' | ')+f' L ({finest[r]})' for r in roots])
    ax.set_title('Supremal log-ratio error ε (finite boundary excluded; values rounded)')
    fig.savefig(figs/'maturity_sensing_budget.pdf',bbox_inches='tight');fig.savefig(figs/'maturity_sensing_budget.png',dpi=170,bbox_inches='tight');plt.close(fig)
    lines=['# Grid and bounded-observation study','',
           'This report recounts the saved finite-tree results. It does not perform new integration or change policy selection.',
           '',f"Primary analysis complete: {primary['complete']}; all algebra inclusion checks passed: {primary['algebra_passed']}.",
           f"Primary parallel wall time: {primary['elapsed_seconds']:.1f} s.",
           f"Adaptive321 complete: {'pending' if adaptive is None else adaptive['complete']}; completed roots: {sum(v=='complete' for v in adaptive_status.values())}/{len(roots)}; additional wall time: {'pending' if adaptive is None else round(adaptive['wall_seconds'],1)} s.",'',
           '## Fixed-target outcomes','']
    for c in counts:
        name=c['mode'] if c['epsilon'] is None else f"ratio ε={c['epsilon']:g}"
        lines.append(f"- {c['grid']} actions, {name}: {c['inner']}/{c['problems']} inner-band, {c['actual']}/{c['problems']} actual-band all-path successes ({c['root_count']} roots).")
    lines+=['','## Finest completed grid per prescribed root','',
            'This aggregate uses the explicitly listed finest completed grid for each root. If any adaptive root is unresolved, it is not a complete 321-action panel.','',
            '; '.join(f'{root}: {finest[root]} actions ({adaptive_status[root]})' for root in roots)+'.','']
    for c in finest_counts:
        name=c['mode'] if c['epsilon'] is None else f"ratio ε={c['epsilon']:g}"
        lines.append(f"- {name}: {c['inner']}/{c['problems']} inner-band and {c['actual']}/{c['problems']} actual-band all-path successes.")
    lines+=['','## Descriptive grid-stability checks','']
    for c in convergence_counts:lines.append(f"- {c['coarse']}→{c['fine']}: {c['positive_pass']}/{c['comparisons']} nonvacuous passes, {c['vacuous']} empty/empty vacuous passes, {c['failed']} failures.")
    lines+=['','Both relative symmetric-difference length ≤5% and endpoint Hausdorff distance ≤0.001 are required. Passing is descriptive stability, not a convergence proof. Failing is an unresolved grid gap.','',
            '## Separate post hoc set-distance diagnostic','']
    for d in summary.get('post_hoc_set_distance_counts',[]):
        if d['nonempty_comparisons']:lines.append(f"- {(d['fine']+1)//2}→{d['fine']}: {d['nonempty_at_most_001']}/{d['nonempty_comparisons']} comparisons nonempty on both grids have target-set Hausdorff distance ≤0.001.")
    lines+=['','This diagnostic uses distance between the target sets themselves, rather than their component endpoints. It does not replace the frozen gates or change adaptive root selection.','',
            '## Sensing frontiers','']
    for root in roots:
        f=frontiers[root];success=[]
        for t in f['targets']:
            if t['status']=='common_open_succeeds':success.append(f"target {t['target']:g}: common-open plan succeeds, so the finite-model sensing radius is unbounded")
            elif t['zero_error_ratio_recoverable']:success.append(f"target {t['target']:g}: ε<{t['epsilon_supremum']:.10g}")
        lines.append(f"- {root}, {f['grid']} actions: "+('; '.join(success) if success else 'none of the six fixed targets has an admitted sensing budget')+'.')
    lines+=['','Finite frontier boundaries are excluded because observation supports are closed. The 1e−7 numerical compatibility pad has already been subtracted. Targets with no feasible nonnegative error allowance are reported as none; a raw zero-clamped display field must not be interpreted as feasibility at zero.','',
            '## Interpretation and verification','',
            'The frozen study evaluates target feasibility in a finite numerical model with bounded scalar composition measurements. Full-state feedback, ratio-only feedback and common-open plans have explicitly different information. The ratio-only policy accepts measured log ratio and chooses one feed shared by all compatible branches; it has no true scenario-label input.',
            '', 'Independent deployment and source checks are recorded separately in audit/grid_observation_*; a passed algebra check or development integrator gate is not a substitute for those deployment outcomes.',
            '', 'Figures: figures/maturity_grid_resolution.pdf; figures/maturity_observation_resolution.pdf; figures/maturity_sensing_budget.pdf. The sensing-budget matrix uses “—” for no admitted budget and “open” for a successful common-open plan; “open” is categorical, not a finite upper bound.']
    for a in audit_summaries:
        lines+=['',f"Independent {a['phase']} audit: {a['cases']} cases ({a['fixed_cases']} fixed, {a['component_cases']} component points); {a['observation_path_episodes']} logical observation-path episodes. Numerical comparisons passed {a['numerical']['comparisons_passed']}/{a['numerical']['comparisons']}; optimistic band-label differences {a['optimistic_differences']}, conservative differences {a['conservative_differences']}. Overall audit passed={a['passed']}; source hashes unchanged={a['inputs_unchanged']}. Counts describe deterministic checks, not independent experimental replicates."]
    (ROOT/'results/GRID_OBSERVATION_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'reported':True,'records':len(records),'finest':finest}))


if __name__=='__main__':main()
