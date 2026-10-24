"""Render verified biomass-transfer calculations and prepare manuscript events."""
from pathlib import Path
import json,os,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3];R=ROOT/'research_completion';P=R/'runs/python_consistency/biomass_transfer'
os.environ.setdefault('MPLCONFIGDIR',str(P/'matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'lines.linewidth':1.6,'legend.fontsize':9,'svg.fonttype':'none'})
out=P/'figures';out.mkdir(parents=True,exist_ok=True)
a=np.load(P/'primary/trajectory.npz');b=np.load(P/'reference/trajectory.npz');t=a['time_h'];x=a['concentrations'];z=b['concentrations'];V=a['volume_L']
COL=['#0072BD','#D95319','#237A3B'];MS=40
def line(ax,y,label,colour,style='-',marker=None):ax.plot(t,y,label=label,color=colour,ls=style,marker=marker,markevery=MS,mfc='white',ms=4)
def finish(fig,name):
 fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=250);fig.savefig(out/(name+'.svg'))
 Image.open(out/(name+'.png')).convert('L').save(out/(name+'_greyscale.png'));plt.close(fig)
def base(ax,letter,ylabel):
 ax.set(xlabel='Time (h)',ylabel=ylabel,xlim=(0,20));ax.set_title('('+letter+')',loc='left');ax.grid(alpha=.18)
fig,axes=plt.subplots(2,2,figsize=(8,6.5))
ax=axes[0,0];line(ax,x[:,0],'Population 1',COL[0],'-','o');line(ax,x[:,1],'Population 2',COL[1],'--','s');line(ax,z[:,0],'Each, no transfer','0.25',':','^');base(ax,'a',r'Biomass (gDW L$^{-1}$)');ax.legend()
ax=axes[0,1];line(ax,x[:,2],'Transfer',COL[0],'-','o');line(ax,z[:,2],'No transfer','0.25',':','^');base(ax,'b',r'Glucose (g L$^{-1}$)');ax.legend()
ax=axes[1,0];line(ax,.1*x[:,0]*x[:,1],'Loss from population 1',COL[0],'-','o');line(ax,.05*x[:,0]*x[:,1],'Gain to population 2',COL[1],'--','s');base(ax,'c',r'Transfer (gDW L$^{-1}$ h$^{-1}$)');ax.legend()
ax=axes[1,1];line(ax,a['accumulators'][:,-1],'Unassimilated biomass',COL[2],'-.','^');base(ax,'d','Cumulative amount (gDW)');ax.legend()
finish(fig,'biomass_transfer_states')
fig,axes=plt.subplots(1,2,figsize=(8,3.5))
ax=axes[0];lower=-10*x[:,2]/(1+x[:,2]);upper=np.full_like(t,1e-4)
ax.fill_between(t,lower,upper,color='.5',alpha=.1);ax.plot(t,lower,'--',color='.5',lw=1,label='Lower bound');ax.plot(t,upper,':',color='.5',lw=1,label='Upper bound')
line(ax,a['flux'][:,27],'Both populations',COL[0],'-','o');base(ax,'a',r'Glucose flux (mmol gDW$^{-1}$ h$^{-1}$)');ax.legend(loc='lower right')
ax=axes[1];line(ax,a['flux'][:,12],'Transfer',COL[0],'-','o');line(ax,b['flux'][:,12],'No transfer','0.25',':','^');base(ax,'b',r'Metabolic growth (h$^{-1}$)');ax.legend()
finish(fig,'biomass_transfer_fluxes')
reports={n:json.loads((P/n/'audit.json').read_text()) for n in ['primary','refined','reference','barrier']}
comparisons={n:np.max(abs(a['concentrations']-np.load(P/n/'trajectory.npz')['concentrations']),axis=0).tolist() for n in ['refined','barrier']}
summary={'runs':reports,'maximum_concentration_differences':comparisons,'recipient_change_percent':float((x[-1,1]/z[-1,1]-1)*100),'all_checks_pass':all(r['passed'] for r in reports.values()),'interpretation':'Phenomenological outer biomass-transfer model; core intracellular network is a reference metabolic system, not an assertion of E. coli predation. Half the transferred material is inventoried as unassimilated biomass; no recycling or complete elemental accounting is claimed.'}
(P/'validation.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
