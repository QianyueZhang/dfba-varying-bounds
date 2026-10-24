"""Presentation-only adaptation of the Paper 1 empirical renderer.

Scientific input files and manuscript sources are read-only. Plot arrays are
checked against the original renderer's selections and saved with input hashes.
"""
from pathlib import Path
import csv, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT=Path(__file__).resolve().parent/'figures'
OUT.mkdir(exist_ok=True)
ROOT=Path(__file__).resolve().parent/'payload'
STUDY=ROOT/'paper1_improvement_2026-10-24'
source_renderer=STUDY/'scripts/render_empirical_figures.py'
paths=[STUDY/'numerical/results/observation_decomposition.csv',
       STUDY/'external/recalibration/results.csv',STUDY/'external/observations.json',source_renderer]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before={str(p.relative_to(ROOT)):sha(p) for p in paths}
def rows(path):return list(csv.DictReader(path.open()))
def nums(records,key):return np.array([float(r[key]) for r in records])
def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(type(value))
plot_arrays={};checks=[]
def verify(observed,expected,name):
    passed=np.array_equal(np.asarray(observed),np.asarray(expected))
    checks.append({'name':name,'passed':bool(passed)})
    assert passed,name
def save(fig,name):
    for ext in ['pdf','png','svg']:fig.savefig(OUT/(name+'.'+ext),bbox_inches='tight',dpi=240)
    plt.close(fig)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42,'axes.labelsize':9,
                     'legend.fontsize':8,'figure.dpi':160,'axes.formatter.use_mathtext':True})
C=['#1D5D85','#D07A31','#34836D']
variables=['biomass_OD600','glucose','ethanol']
labels=[r'Biomass (OD$_{600}$)',r'Glucose (g L$^{-1}$)',r'Ethanol (g L$^{-1}$)']

dc=rows(paths[0])
fields=['expansion','barrier_difference','total']
names=['Expansion','Finite barrier − expanded LP','Total']
cases=['archived_adopted','fixed_expansion_small_barrier','tight_pair']
case_labels=['Adopted','Smaller\n'+r'$\mu$','Joint\nrefinement']
line_styles=['-','--','-.'];hatches=['','///','..']
fig,axs=plt.subplots(2,3,figsize=(7.2,4.65))
decomp=[]
for j,var in enumerate(variables):
    rr=[r for r in dc if r['variable']==var and r['split']=='withheld_series' and r['case']==cases[0]]
    rr.sort(key=lambda r:float(r['time_h']))
    for k,field in enumerate(fields):
        x=nums(rr,'time_h');y=nums(rr,field)
        line,=axs[0,j].plot(x,y,c=C[k],lw=1.25,ls=line_styles[k],label=names[k])
        verify(line.get_xdata(),x,f'{var}:{field}:time')
        verify(line.get_ydata(),y,f'{var}:{field}:signed')
        values=[max(abs(float(r[field])) for r in dc if r['variable']==var and r['split']=='withheld_series' and r['case']==case) for case in cases]
        bars=axs[1,j].bar(np.arange(3)+(k-1)*.24,values,width=.23,color=C[k],edgecolor='white',linewidth=.4,hatch=hatches[k],label=names[k])
        verify([b.get_height() for b in bars],values,f'{var}:{field}:maxima')
        decomp.append({'variable':var,'field':field,'time_h':x,'signed_difference':y,'max_absolute_by_case':values,'cases':cases})
    axs[0,j].axhline(0,c='#999999',lw=.6,zorder=0)
    axs[0,j].set(xlabel='Time (h)',ylabel='Signed difference')
    axs[0,j].set_title(f'({chr(97+j)}) '+labels[j],loc='left',fontsize=9)
    axs[0,j].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    axs[0,j].tick_params(labelsize=8)
    axs[1,j].set_xticks(np.arange(3),case_labels,fontsize=8)
    axs[1,j].set(ylabel='Maximum absolute difference',yscale='log')
    axs[1,j].set_title(f'({chr(100+j)}) '+labels[j],loc='left',fontsize=9)
    axs[1,j].tick_params(axis='y',labelsize=8)
    axs[1,j].grid(axis='y',which='major',alpha=.13,zorder=0)
handles=[Line2D([],[],color=C[k],lw=1.3,ls=line_styles[k],label=names[k]) for k in range(3)]
fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,1.01),fontsize=8)
fig.tight_layout(rect=(0,0,1,.94),h_pad=1.5,w_pad=.6)
plot_arrays['decomposition']=decomp
save(fig,'yeast_decomposition')

ex=rows(paths[1]);ob=json.loads(paths[2].read_text())['observations']
conds=['o2_20.9percent','o2_2.8percent','o2_1.0percent','o2_0.5percent','o2_0percent']
x=np.arange(5);fig,axs=plt.subplots(1,2,figsize=(7.2,3.25))
for ax in axs:
    ax.axvspan(-.45,.45,color='#F1EBDE',zorder=0)
    ax.set_xticks(x,['20.9','2.8','1.0','0.5','0']);ax.set_xlabel('Inlet oxygen (%)')
    ax.tick_params(labelsize=8)
external=[]
for tag,col,marker,lab,shift in [('original',C[0],'s','Batch maintenance parameter',-.12),('single_condition_recalibration',C[1],'^','One-condition calibration',.12)]:
    rr=[next(r for r in ex if r['condition_id']==c and r['semantics']=='capacity' and r['maintenance_label']==tag) for c in conds]
    growth=nums(rr,'growth_h-1');ethanol=nums(rr[:4],'ethanol_mmol_gDW_h')
    g=axs[0].scatter(x+shift,growth,c=col,marker=marker,s=28,label=lab,zorder=4)
    e=axs[1].scatter(x[:4]+shift,ethanol,c=col,marker=marker,s=28,label=lab,zorder=4)
    verify(g.get_offsets(),np.c_[x+shift,growth],f'{tag}:growth')
    verify(e.get_offsets(),np.c_[x[:4]+shift,ethanol],f'{tag}:ethanol')
    external.append({'parameter_set':tag,'conditions':conds,'growth_x':x+shift,'growth_h-1':growth,
                     'ethanol_x':x[:4]+shift,'ethanol_mmol_gDW_h':ethanol,
                     'anaerobic_ethanol_omitted_as_in_source_renderer':True})
nominal=np.full(5,.1)
line,=axs[0].plot(x,nominal,color='.25',linestyle=':',marker='o',ms=3.8,lw=.9,label='Nominal dilution rate',zorder=3)
verify(line.get_ydata(),[.1]*5,'nominal_dilution_rate')
source_y=[];source_sem=[]
for c in conds:
    entry=next(r for r in ob if r['condition_id']==c)['values']['ethanol_Cmmol_gDW_h']
    source_y.append(entry['mean']/2);source_sem.append((entry['sem'] or 0)/2)
err=axs[1].errorbar(x,source_y,yerr=source_sem,fmt='o',ms=3.8,c='black',capsize=2,label='Measured ethanol',zorder=5)
verify(err.lines[0].get_ydata(),source_y,'ethanol_observation_means')
segments=err.lines[2][0].get_segments()
verify([[s[0,1],s[1,1]] for s in segments],np.c_[np.array(source_y)-source_sem,np.array(source_y)+source_sem],'ethanol_SEM_endpoints')
# Label the omission in genuine empty space, without inventing a plotted zero.
axs[1].text(.04,.93,'0% oxygen: model ethanol omitted\nbecause predicted growth is zero.',
            transform=axs[1].transAxes,fontsize=7.4,ha='left',va='top',color='#555555',linespacing=1.25)
axs[0].set_ylabel(r'Maximal growth (h$^{-1}$)');axs[1].set_ylabel(r'Ethanol (mmol gDW$^{-1}$ h$^{-1}$)')
axs[0].set_ylim(-.007,.135);axs[1].set_ylim(-.45,10.4)
axs[0].set_title('(a) Growth and dilution rate',loc='left',fontsize=9)
axs[1].set_title('(b) Ethanol exchange',loc='left',fontsize=9)
handles=[Line2D([],[],color=C[0],marker='s',ls='',ms=5,label='Batch maintenance parameter'),
         Line2D([],[],color=C[1],marker='^',ls='',ms=5,label='One-condition calibration'),
         Line2D([],[],color='.25',marker='o',ls=':',ms=4,lw=.9,label='Nominal dilution rate'),
         Line2D([],[],color='black',marker='o',ls='',ms=4,label='Measured ethanol (±SEM)')]
fig.legend(handles=handles,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.03),frameon=False,fontsize=8,columnspacing=1.8)
fig.tight_layout(rect=(0,0,1,.86),w_pad=2.2)
plot_arrays['external']={'model_series':external,'conditions':conds,'nominal_dilution_rate':nominal,
                         'observed_ethanol_mean':source_y,'observed_ethanol_sem':source_sem,
                         'conversion':'Published ethanol C-mmol divided by two for compound mmol; same as source renderer.'}
save(fig,'yeast_external')

after={str(p.relative_to(ROOT)):sha(p) for p in paths}
assert before==after
(OUT/'yeast_plot_arrays.json').write_text(json.dumps(plot_arrays,indent=2,default=serial)+'\n')
audit={'input_hashes_before':before,'input_hashes_after':after,'source_inputs_unchanged':before==after,
       'renderer_origin':str(source_renderer),'array_checks':checks,'all_array_checks_passed':all(c['passed'] for c in checks),
       'scientific_changes':'None: same selected observation rows, signed difference arrays, maxima, growth/ethanol values and SEM conversion. Nominal 0.1 h^-1 is correctly labelled as the dilution-rate reference.',
       'presentation_changes':['distinct solid/dashed/dash-dot decomposition curves','hatches on decomposition bars','shared legends outside plot panels','nominal dilution-rate label','external-case annotation in empty upper-left space, without a leader to an invented zero','consistent panel headings and units'],
       'outputs':{p.name:sha(p) for name in ['yeast_decomposition','yeast_external'] for p in OUT.glob(name+'.*')}}
(OUT/'yeast_figure_style_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({'source_inputs_unchanged':audit['source_inputs_unchanged'],'array_checks':len(checks),'passed':audit['all_array_checks_passed'],'outputs':list(audit['outputs'])}))
