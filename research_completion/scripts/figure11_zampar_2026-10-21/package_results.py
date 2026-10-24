"""Render the experimental comparison from verified numerical results."""
from fit_batch import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def scientific_text(value):
 mantissa,exponent=f'{value:.1e}'.split('e')
 return mantissa+'×10'+str(int(exponent)).translate(str.maketrans('-0123456789','⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))

def main():
 audit=json.loads((OUT/'direct_audit.json').read_text());fit=json.loads((OUT/'fit_summary.json').read_text());tr=np.genfromtxt(OUT/'trajectory_direct_barrier.csv',delimiter=',',names=True);pred=[]
 for r in ROWS:
  j=VARIABLES.index(r['variable']);v=np.interp(float(r['time_relative_to_glucose_exhaustion_h']),tr['t_h'],tr[['X_gDW_L','G_g_L','E_g_L'][j]]);pred.append(v/OD_FACTOR if j==0 else v)
 metrics=summary_metrics(ROWS,np.array(pred));(OUT/'final_observation_metrics.json').write_text(json.dumps(metrics,indent=2))
 with (OUT/'final_observations_predictions.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(ROWS[0])+['finite_barrier_prediction','residual']);w.writeheader()
  for r,v in zip(ROWS,pred):w.writerow(r|{'finite_barrier_prediction':float(v),'residual':float(v-float(r['observed_value_as_published']))})
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.spines.top':False,'axes.spines.right':False})
 for grayscale in [False,True]:
  blue='#181818'if grayscale else '#0072B2';orange='#555555'if grayscale else '#D55E00';green='#777777'if grayscale else '#009E73'
  fig,axs=plt.subplots(2,2,figsize=(8.2,6.7),sharex=True);axs=axs.ravel()
  fields=['X_gDW_L','G_g_L','E_g_L'];ylabs=['Biomass (OD$_{600}$, AU)','Glucose (g L$^{-1}$)','Ethanol (g L$^{-1}$)']
  for j,ax in enumerate(axs[:3]):
   ax.plot(tr['t_h'],tr[fields[j]]/(OD_FACTOR if j==0 else 1),color=blue,lw=1.75,zorder=3)
   for split,marker,col in [('calibration','o',orange),('withheld_series','^','#8a8a8a')]:
    rr=[r for r in ROWS if r['variable']==VARIABLES[j]and r['split']==split];ax.scatter([float(r['time_relative_to_glucose_exhaustion_h'])for r in rr],[float(r['observed_value_as_published'])for r in rr],marker=marker,s=22 if split=='calibration'else 17,facecolors='none',edgecolors=col,lw=.8,alpha=.85,zorder=2)
   ax.set_ylabel(ylabs[j]);ax.set_ylim(bottom=-.06 if j==1 else 0);ax.text(.02,1.01,'('+chr(97+j)+')',transform=ax.transAxes,ha='left',va='bottom',fontweight='bold')
  ax=axs[3];ax.plot(tr['t_h'],-tr['v_glucose'],color=blue,lw=1.6,label='Glucose uptake, −v$_G$');ax.plot(tr['t_h'],tr['v_ethanol'],color=orange,lw=1.6,ls='--',label='Ethanol exchange, v$_E$');ax.plot(tr['t_h'],-tr['v_oxygen'],color=green,lw=1.3,ls=':',label='Oxygen uptake, −v$_O$');ax.axhline(0,color='#777777',lw=.6);ax.set_ylabel('Specific flux (mmol gDW$^{-1}$ h$^{-1}$)');ax.legend(loc='upper right',fontsize=8,frameon=False);ax.text(.02,1.01,'(d)',transform=ax.transAxes,ha='left',va='bottom',fontweight='bold')
  for ax in axs:ax.axvline(0,color='#aaaaaa',ls=':',lw=.8,zorder=0);ax.grid(alpha=.13);ax.set_xlim(T0-.12,TF+.12)
  for ax in axs[2:]:ax.set_xlabel('Time relative to glucose exhaustion (h)')
  handles=[Line2D([0],[0],color=blue,lw=1.75,label='Finite-barrier dFBA'),Line2D([0],[0],ls='',marker='o',mfc='none',mec=orange,label='Calibration observations'),Line2D([0],[0],ls='',marker='^',mfc='none',mec='#8a8a8a',label='Withheld observations')];fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.),ncol=3,frameon=False,fontsize=8.5)
  fig.tight_layout(rect=(0,0,1,.95));stem=OUT/('figure11_grayscale'if grayscale else'figure11_colour');fig.savefig(str(stem)+'.png',dpi=260);fig.savefig(str(stem)+'.svg');fig.savefig(str(stem)+'.pdf');plt.close(fig)

if __name__=='__main__':main()
