"""Render manuscript figures from saved observations and calculation records.

No simulations or parameter fitting are performed. All points, including the
anaerobic failure, are retained. Run from any working directory.
"""
from pathlib import Path
import csv, json, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parent
MS = STUDY / 'manuscript'
DATA = MS / 'data' / 'empirical'
FIG = MS / 'figures'
DATA.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
old = ROOT / 'research_completion/runs/figure11_zampar_2026-10-21'
for name in ['trajectory_direct_barrier.csv', 'final_observations_predictions.csv', 'final_observation_metrics.json']:
    dest = DATA / name
    if (old / name).exists():
        shutil.copy2(old / name, dest)
    elif not dest.exists():
        raise FileNotFoundError(dest)

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.labelsize': 9,
                     'legend.fontsize': 8, 'figure.dpi': 160})
C = ['#1D5D85', '#D07A31', '#34836D']
def rows(path):
    return list(csv.DictReader(Path(path).open()))
def nums(records, key):
    return np.array([float(r[key]) for r in records])
def save(fig, name):
    fig.savefig(FIG / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(FIG / (name + '.png'), bbox_inches='tight', dpi=180)
    plt.close(fig)

tr = rows(DATA / 'trajectory_direct_barrier.csv')
ob = rows(DATA / 'final_observations_predictions.csv')
variables = ['biomass_OD600', 'glucose', 'ethanol']
labels = [r'Biomass (OD$_{600}$)', 'Glucose (g/L)', 'Ethanol (g/L)']
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.65), layout='constrained')
for j, (ax, var, key, label) in enumerate(zip(axs, variables, ['X_gDW_L', 'G_g_L', 'E_g_L'], labels)):
    y = nums(tr, key) / (0.486 if j == 0 else 1)
    ax.plot(nums(tr, 't_h'), y, c=C[0], lw=1.6, label='Finite-barrier model')
    for split, marker, fc, text in [('calibration', 'o', C[1], 'Calibration'), ('withheld_series', 'o', 'none', 'Withheld series')]:
        rr = [r for r in ob if r['variable'] == var and r['split'] == split]
        ax.scatter(nums(rr, 'time_relative_to_glucose_exhaustion_h'), nums(rr, 'observed_value_as_published'),
                   s=12, marker=marker, facecolors=fc, edgecolors=C[1] if split=='calibration' else '#737373',
                   linewidths=.65, alpha=.85, label=text)
    ax.axvline(0, c='#999999', lw=.7, ls=':')
    ax.set(xlabel='Time relative to exhaustion (h)', ylabel=label)
    ax.set_ylim(top=max(max(y),max(float(r['observed_value_as_published']) for r in ob if r['variable']==var))*1.12)
    ax.text(.02, .97, chr(97+j), transform=ax.transAxes, fontweight='bold', va='top')
axs[0].legend(loc='lower right', frameon=False, fontsize=6.8)
save(fig, 'figure4_yeast')

dc = rows(STUDY/'numerical/results/observation_decomposition.csv')
fig, axs = plt.subplots(2, 3, figsize=(7.1, 4.5), layout='constrained')
fields = ['expansion', 'barrier_difference', 'total']
names = ['Expansion', 'Finite minus expanded LP', 'Total']
cases = ['archived_adopted', 'fixed_expansion_small_barrier', 'tight_pair']
case_labels = ['Adopted', 'Smaller\n'+r'$\mu$', 'Joint\nrefinement']
for j, var in enumerate(variables):
    rr = [r for r in dc if r['variable']==var and r['split']=='withheld_series' and r['case']==cases[0]]
    rr.sort(key=lambda r:float(r['time_h']))
    for k, field in enumerate(fields):
        axs[0,j].plot(nums(rr,'time_h'), nums(rr,field), c=C[k], lw=1.3, label=names[k])
        values = [max(abs(float(r[field])) for r in dc if r['variable']==var and r['split']=='withheld_series' and r['case']==case) for case in cases]
        axs[1,j].bar(np.arange(3)+(k-1)*.24, values, width=.23, color=C[k], label=names[k])
    axs[0,j].axhline(0,c='#999999',lw=.6)
    axs[0,j].set(xlabel='Time (h)', ylabel='Signed difference', title=labels[j])
    axs[0,j].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axs[1,j].set_xticks(np.arange(3),case_labels,fontsize=8)
    axs[1,j].set(ylabel='Maximum absolute difference',yscale='log')
    for i in range(2):
        axs[i,j].text(.02,.97,chr(97+i*3+j),transform=axs[i,j].transAxes,fontweight='bold',va='top')
axs[0,0].legend(frameon=False,fontsize=6.5,loc='upper left',bbox_to_anchor=(.05,.91))
save(fig,'figure5_decomposition')

ex = rows(STUDY/'external/recalibration/results.csv')
ob2 = json.loads((STUDY/'external/observations.json').read_text())['observations']
conds = ['o2_20.9percent','o2_2.8percent','o2_1.0percent','o2_0.5percent','o2_0percent']
fig, axs = plt.subplots(1,2,figsize=(7.1,3.05),layout='constrained')
x=np.arange(5)
for ax in axs:
    ax.axvspan(-.45,.45,color='#F1EBDE',zorder=0)
    ax.set_xticks(x,['20.9','2.8','1.0','0.5','0'])
    ax.set_xlabel('Inlet oxygen (%)')
for tag, col, marker, lab, shift in [('original',C[0],'s','Batch parameter',-.12),('single_condition_recalibration',C[1],'^','One-condition adjustment',.12)]:
    rr=[next(r for r in ex if r['condition_id']==c and r['semantics']=='capacity' and r['maintenance_label']==tag) for c in conds]
    axs[0].scatter(x+shift,nums(rr,'growth_h-1'),c=col,marker=marker,s=30,label=lab,zorder=4)
    axs[1].scatter(x[:4]+shift,nums(rr[:4],'ethanol_mmol_gDW_h'),c=col,marker=marker,s=30,label=lab,zorder=4)
axs[0].plot(x,[.1]*5,'o',c='black',ms=4,label='Observed steady state')
source_y=[];source_sem=[]
for c in conds:
    record=next(r for r in ob2 if r['condition_id']==c)
    # Derived values retain the explicit carbon-to-compound conversion.
    entry=record['values']['ethanol_Cmmol_gDW_h']
    source_y.append(entry['mean']/2)
    source_sem.append((entry['sem'] or 0)/2)
axs[1].errorbar(x,source_y,yerr=source_sem,fmt='o',ms=4,c='black',capsize=2,label='Observed ethanol')
axs[1].annotate('Zero predicted growth;\nethanol FVA is uninformative',xy=(4,0),xytext=(2.0,5.0),fontsize=7,
                arrowprops={'arrowstyle':'-','color':'#777777'},ha='center')
axs[0].set_ylabel(r'Maximal growth (h$^{-1}$)')
axs[1].set_ylabel(r'Ethanol (mmol gDW$^{-1}$ h$^{-1}$)')
axs[0].set_ylim(-.007,.135)
for j,ax in enumerate(axs):
    ax.text(.02,.98,chr(97+j),transform=ax.transAxes,fontweight='bold',va='top')
axs[0].legend(frameon=False,fontsize=7,loc='lower left',bbox_to_anchor=(.02,.14))
save(fig,'figure6_external')
