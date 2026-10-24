"""Independent saved-state checks and adoption payload for the diauxic case."""
from pathlib import Path
import json, sys, hashlib, os
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import simpson
from run import Model,solve,ROOT,OUT
os.environ['MPLCONFIGDIR']=str(OUT/'mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
primary=np.load(OUT/'primary.npz');meta=json.loads((OUT/'primary.json').read_text())
t=primary['t'];y=primary['y'];v=primary['flux'];checks=[]
def check(label,passed,details):checks.append({'check':label,'passed':bool(passed),'details':details})
check('Nonnegative concentrations and strict scalar-bound slacks',np.all(y>=0) and np.all(v[:,1:]>primary['lower']) and np.all(v[:,1:]<primary['upper']),{'minimum_concentrations':y.min(0).tolist(),'minimum_selected_flux_slack':float(min(np.min(v[:,1:]-primary['lower']),np.min(primary['upper']-v[:,1:])) )})
comparisons={}
for label in ['refined','small_barrier']:
 d=np.load(OUT/(label+'.npz'));m=json.loads((OUT/(label+'.json')).read_text());common=np.linspace(0,min(t[-1],d['t'][-1]),1501)
 diff=PchipInterpolator(t,y)(common)-PchipInterpolator(d['t'],d['y'])(common)
 comparisons[label]={'max_absolute_concentration_difference':np.max(abs(diff),axis=0).tolist(),'endpoint_difference':(d['y'][-1]-y[-1]).tolist(),'end_time_difference_h':float(d['t'][-1]-t[-1]),'parameters':{k:m[k] for k in ['mu','epsilon','rtol','atol']}}
check('Tighter ODE tolerances preserve the trajectory',max(comparisons['refined']['max_absolute_concentration_difference'])<1e-3 and abs(comparisons['refined']['end_time_difference_h'])<1e-4,comparisons['refined'])
check('Tenfold smaller barrier and margins preserve interpretation',max(comparisons['small_barrier']['max_absolute_concentration_difference'])<.01 and abs(comparisons['small_barrier']['end_time_difference_h'])<.002,comparisons['small_barrier'])
model=Model();static=[]
for i in np.unique(np.linspace(0,len(t)-1,31,dtype=int)):
 yy=y[i];s=solve(model,yy,mu=meta['mu'],tol=1e-10,complementarity_rtol=1e-7);lp=model.lp(yy);vf=s[0]
 static.append({'time_h':float(t[i]),'growth_gap':float(-lp.fun-vf[model.bio]),'raw_matrix_residual':float(np.max(abs(model.raw['S']@vf))), 'repeat_selected_flux_difference':float(np.max(abs(vf[[model.bio,model.g,model.xy,model.o]]-v[i]))),**s[-1]})
check('Raw stoichiometry and independent cold-start states',max(x['raw_matrix_residual'] for x in static)<1e-8 and max(x['repeat_selected_flux_difference'] for x in static)<1e-5,{'samples':len(static),'max_full_matrix_residual':max(x['raw_matrix_residual'] for x in static),'max_selected_flux_difference':max(x['repeat_selected_flux_difference'] for x in static)})
check('LP objective gaps obey the finite-barrier bound',min(x['growth_gap'] for x in static)>-1e-8 and max(x['growth_gap'] for x in static)<2*len(model.c)*meta['mu'],{'max_growth_gap_h_inv':max(x['growth_gap'] for x in static),'bound':2*len(model.c)*meta['mu']})
rates=np.c_[v[:,0]*y[:,0],.180156*v[:,1]*y[:,0],.15013*v[:,2]*y[:,0]]
balances=y[-1]-y[0]-simpson(rates,x=t,axis=0)
check('Saved-flux quadrature agrees with extracellular balances',max(abs(balances))<.001,{'residual_g_L':balances.tolist(),'scope':'Independent Simpson quadrature of501 saved flux/state samples; not a full elemental balance.'})
oldcomp={'scope':'Public release uses the declared consistent-model trajectories.'}
Gtime=float(np.interp(.15,y[::-1,1],t[::-1]));observed={'glucose_99_percent_depleted_h':Gtime,'terminal_biomass_gDW_L':float(y[-1,0]),'terminal_xylose_g_L':float(y[-1,2]),'end_h':float(t[-1]),'biomass_increment_per_consumed_carbon_substrate_g_g':float((y[-1,0]-y[0,0])/(23-y[-1,1]-y[-1,2]))}

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'legend.fontsize':8})
for grey in [False,True]:
 colors=['.1','.4','.65'] if grey else ['#176b9a','#a84b14','#3b8057']
 fig,axes=plt.subplots(2,2,figsize=(7.5,5.9),sharex=True);axes=axes.ravel()
 for c,style,label,col in [(0,'-',r'$\it{E.\ coli}$ biomass',colors[0]),(1,'--','Glucose',colors[1]),(2,'-.','Xylose',colors[2])]:axes[0].plot(t,y[:,c],style,color=col,lw=1.7,label=label)
 axes[0].set_ylabel('Concentration (g L$^{-1}$)');axes[0].legend(loc='upper right',frameon=False);axes[0].set_ylim(bottom=0)
 for ax,k,name in [(axes[1],0,'Glucose'),(axes[2],2,'Oxygen'),(axes[3],1,'Xylose')]:
  ax.fill_between(t,primary['lower'][:,k],primary['upper'][:,k],color='0.5',alpha=.12)
  ax.plot(t,primary['lower'][:,k],'--',color='0.35',lw=1.1,label='Lower bound')
  ax.plot(t,primary['upper'][:,k],':',color='0.5',lw=1.1,label='Upper bound')
  ax.plot(t,v[:,k+1],'-',color=colors[0],lw=1.7,label='Exchange flux')
  ax.set_ylabel(name+' flux\n(mmol gDW$^{-1}$ h$^{-1}$)');ax.legend(frameon=False,loc='lower left' if name=='Xylose' else 'upper left')
 for i,ax in enumerate(axes):
  ax.set_xlim(0,t[-1]);ax.grid(axis='y',alpha=.2,lw=.5);ax.text(.01,1.015,f'({chr(97+i)})',transform=ax.transAxes,fontweight='bold')
  if i>=2:ax.set_xlabel('Time (h)')
 fig.tight_layout(w_pad=1.7,h_pad=1.5)
 for ext in ['png','svg','pdf']:fig.savefig(OUT/('figure10_consistent'+('_greyscale' if grey else '')+'.'+ext),dpi=300,bbox_inches='tight')
 plt.close(fig)
asset=OUT/'figure10_consistent.png'
report={'adoption_ready':all(c['passed'] for c in checks),'scope':'Direct finite-barrier iJR904 dynamic calculation over the stated active-growth interval, with integration and barrier refinement. Independent biological calibration is outside the scope of this case.','checks':checks,'primary':meta,'comparisons':comparisons,'static_checks':static,'scientific_outputs':observed,'source_case_context':oldcomp,'figure':{'path':str(asset.relative_to(ROOT)),'sha256':sha(asset)},'input_hashes':{str((OUT/n).relative_to(ROOT)):sha(OUT/n) for n in ['primary.npz','refined.npz','small_barrier.npz']}}
(OUT/'adoption_validation.json').write_text(json.dumps(report,indent=2)+'\n')
if not report['adoption_ready']:raise RuntimeError([x for x in checks if not x['passed']])
