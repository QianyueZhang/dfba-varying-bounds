"""Validate completed inhibition runs and render the adopted two-figure case."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.integrate import cumulative_simpson
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0,str(Path(__file__).parent))
from product_inhibition_check import ROOT,OUT,Model,MU,EPS

NAMES={'adopted':'flow_1p0_refined_balanced','reference':'flow_1p0_refined_reference_balanced','primary':'flow_1p0_primary_balanced'}

def load(tag):
 p=OUT/'separate'/tag
 return dict(np.load(p/'trajectory.npz')),json.loads((p/'audit.json').read_text())

def check(tag):
 x,r=load(tag);m=[Model('separate',s,1e-11,inhibition=r['operating_definition']['inhibition'])for s in ['S1','S2']]
 t=x['time_h'];y=x['amounts'];flux=x['flux'];minmargin=np.inf;maxeq=0.;maxbound=0.;minslack=np.inf;maxgap=0.
 for j,time in enumerate(t):
  V=5+time
  for i,model in enumerate(m):
   model.level=x['secretion_level'][j] if i==0 else 1.
   env=(y[j,2+i]/V,y[j,4]/V);l,u=model.bounds(env);v=flux[j,i]
   maxeq=max(maxeq,float(abs(model.rawA@v).max()));maxbound=max(maxbound,float((l-v).max()),float((v-u).max()));minslack=min(minslack,float(min((v-l).min(),(u-v).min())))
   minmargin=min(minmargin,model.margin(env));lp=model.lp(env);assert lp.success;maxgap=max(maxgap,float(-lp.fun-v[12]))
 expected=np.c_[5+y[:,5],5+y[:,6],50+10*t+y[:,7],50+10*t+y[:,8],y[:,9]+y[:,10]]
 balance=np.max(abs(y[:,:5]-expected),axis=0)
 result={'run':tag,'full_10h':bool(t[-1]==10),'sample_count':len(t),'maximum_raw_stoichiometric_residual':maxeq,'maximum_bound_excess':maxbound,'minimum_slack':minslack,'minimum_LP_feasibility_margin_mmol_gDW_h':minmargin,'maximum_LP_growth_gap':maxgap,'nominal_gap_bound':190*MU,'tracked_pool_balance_errors_g':balance.tolist(),'minimum_biomass_flux_h_inv':float(flux[:,:,12].min()),'minimum_tracked_concentration':float((y[:,:5]/(5+t[:,None])).min())}
 assert result['full_10h'] and minmargin>0 and minslack>0 and maxeq<1e-9 and maxbound<1e-10 and maxgap<190*MU+1e-9 and max(balance)<1e-9 and result['minimum_biomass_flux_h_inv']>0
 result['passed']=True
 return result

def common(x):
 tt=np.linspace(0,10,401);ix=np.array([np.flatnonzero(np.isclose(x['time_h'],t,rtol=0,atol=1e-12))[0]for t in tt]);return tt,x['amounts'][ix],x['flux'][ix]

def style(ax,title,ylabel):
 ax.set(xlim=(0,10),xlabel='Time (h)',ylabel=ylabel);ax.set_title(title,loc='left',fontsize=9.5);ax.grid(axis='y',alpha=.16);ax.spines[['top','right']].set_visible(False)

def save(fig,name):
 d=OUT/'figures';d.mkdir(parents=True,exist_ok=True)
 for ext in ['png','svg']:
  fig.savefig(d/f'{name}.{ext}',dpi=300,bbox_inches='tight',metadata={'Date':None}if ext=='svg'else None)
 plt.close(fig);Image.open(d/f'{name}.png').convert('L').save(d/f'{name}_greyscale.png')
 return {ext:{'path':str((d/f'{name}.{ext}').relative_to(ROOT)),'sha256':hashlib.sha256((d/f'{name}.{ext}').read_bytes()).hexdigest()}for ext in ['png','svg']}

def figures():
 x,r=load(NAMES['adopted']);z,rz=load(NAMES['reference']);t,y,v=common(x);_,ry,rv=common(z);V=5+t;C=y[:,:5]/V[:,None];R=ry[:,:5]/V[:,None]
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'svg.fonttype':'none','svg.hashsalt':'product_inhibition_adopted','lines.linewidth':1.6})
 donor=dict(color='#175b8c',ls='-',marker='s',markevery=50,ms=3);recipient=dict(color='#ad511b',ls='--',marker='^',markevery=50,ms=3);reference=dict(color='#303030',ls=':',marker='o',mfc='white',markevery=50,ms=3)
 fig,ax=plt.subplots(2,2,figsize=(7.2,5.8),constrained_layout=True)
 ax[0,0].plot(t,C[:,0],**donor,label='Donor (both cases)');ax[0,0].plot(t,C[:,1],**recipient,label='Recipient, inhibition');ax[0,0].plot(t,R[:,1],**reference,label='Recipient, no inhibition');style(ax[0,0],'(a) Biomass concentrations',r'Biomass (gDW L$^{-1}$)');ax[0,0].legend(frameon=False,fontsize=7.5,loc='lower right')
 ax[0,1].plot(t,y[:,0],**donor);ax[0,1].plot(t,y[:,1],**recipient);ax[0,1].plot(t,ry[:,1],**reference);style(ax[0,1],'(b) Biomass amounts',r'Biomass amount (gDW)')
 ax[1,0].plot(t,C[:,2],**donor,label='Donor glucose');ax[1,0].plot(t,C[:,3],**recipient,label='Recipient, inhibition');ax[1,0].plot(t,R[:,3],**reference,label='Recipient, no inhibition');style(ax[1,0],'(c) Separate glucose pools',r'Glucose (g L$^{-1}$)');ax[1,0].legend(frameon=False,fontsize=7.5,loc='upper right')
 ax[1,1].plot(t,C[:,4],color='#2e6f4e',ls='-',marker='D',markevery=50,ms=3,label='Inhibition');ax[1,1].plot(t,R[:,4],**reference,label='No inhibition');style(ax[1,1],'(d) Shared ethanol pool',r'Ethanol (g L$^{-1}$)');ax[1,1].legend(frameon=False,fontsize=7.5,loc='upper right')
 files={'states':save(fig,'product_inhibition_states')}
 fig,ax=plt.subplots(2,2,figsize=(7.2,5.8),constrained_layout=True)
 models=[Model('separate',s)for s in ['S1','S2']]
 for i in [0,1]:
  lower=np.array([model.bounds((c[2+i],c[4]))[0][27] for model,c in [(models[i],c)for c in C]])
  ax[0,i].fill_between(t,lower,EPS,color='#c6cbd0',alpha=.3,label='Inhibited-case bound interval')
  ax[0,i].plot(t,lower,color='#202020',ls='--',lw=1.1,label='Lower bound')
  ax[0,i].axhline(EPS,color='#606060',ls=':',lw=1.1,label='Upper bound')
  ax[0,i].plot(t,v[:,i,27],**(donor if i==0 else recipient),label='Selected flux')
  if i==1:ax[0,i].plot(t,rv[:,1,27],**reference,label='No-inhibition flux')
  style(ax[0,i],'(a) Donor glucose exchange'if i==0 else'(b) Recipient glucose exchange',r'Flux (mmol gDW$^{-1}$ h$^{-1}$)')
  ax[0,i].legend(frameon=False,fontsize=6.8,loc='lower right')
 et=x['time_h'];lower=x['secretion_level']-EPS
 ax[1,0].step(et,lower,where='post',color='#202020',ls='--',lw=1.1,label='Secretion lower bound')
 ax[1,0].step(et,x['flux'][:,0,23],where='post',color='#175b8c',ls='-',marker='s',markevery=50,ms=3,label='Donor exchange flux');style(ax[1,0],'(c) Donor ethanol exchange',r'Flux (mmol gDW$^{-1}$ h$^{-1}$)');ax[1,0].legend(frameon=False,fontsize=7.2,loc='upper right')
 atp=cumulative_simpson(y[:,1]*v[:,1,10],x=t,initial=0);ratp=cumulative_simpson(ry[:,1]*rv[:,1,10],x=t,initial=0)
 ax[1,1].plot(t,atp,**recipient,label='Recipient, inhibition');ax[1,1].plot(t,ratp,**reference,label='Recipient, no inhibition');style(ax[1,1],'(d) Cumulative ATP maintenance',r'ATPM turnover (mmol)');ax[1,1].legend(frameon=False,fontsize=7.2,loc='upper left')
 files['fluxes']=save(fig,'product_inhibition_fluxes')
 return files

def main():
 checks=[check(n)for n in NAMES.values()];x,r=load(NAMES['adopted']);z,rz=load(NAMES['reference']);p,rp=load(NAMES['primary']);t,y,v=common(x);_,ry,rv=common(z);_,py,pv=common(p);V=5+t
 early=np.flatnonzero(t==2)[0];delta=(y[-1,1]-ry[-1,1])/15
 comparison={'recipient_2h_inhibition_gDW_L':float(y[early,1]/V[early]),'recipient_2h_reference_gDW_L':float(ry[early,1]/V[early]),'recipient_10h_inhibition_gDW_L':float(y[-1,1]/15),'recipient_10h_reference_gDW_L':float(ry[-1,1]/15),'terminal_difference_gDW_L':float(delta),'terminal_percent_difference':float(100*(y[-1,1]/ry[-1,1]-1)),'maximum_donor_concentration_difference':float(np.max(abs(y[:,0]-ry[:,0])/V)),'max_direct_tolerance_concentration_difference':float(np.max(abs(y[:,:5]-py[:,:5])/V[:,None]))}
 report={'checks':checks,'comparison':comparison,'differentiated_verification':json.loads((OUT/'separate/differentiated/audit.json').read_text()),'figures':figures(),'adoption_ready':True,'adoption_scope':'One separate-resource fed-batch inhibition benchmark; F=1 L/h; initial volume5 L;10h; original source kinetics/ATPM; both ethanol exchanges included; matched inhibition-disabled reference. No claim of terminal biomass suppression or physiological calibration.','source_history':'Source F0.1 runs and F1 shared-resource diagnostic stop near active-model feasibility limits. Those runs are research records only and are not source plots for either adopted figure.','limits':['Population-specific glucose pools assume selective resource access.','Ki=.1g/L and donor secretion are benchmark parameters, not calibrated E.coli ethanol physiology.','Finite-bound expansion epsilon1e-4 allows small nonzero fluxes on otherwise blocked directions.','Only tracked-pool balances and intracellular stoichiometry are certified, not complete reactor elemental-carbon recovery.','Transient inhibition does not imply a lower terminal biomass objective.']}
 (OUT/'adoption_validation.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(comparison,indent=2))
if __name__=='__main__':main()
