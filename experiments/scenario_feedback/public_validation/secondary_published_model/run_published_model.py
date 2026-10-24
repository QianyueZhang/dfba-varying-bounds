#!/usr/bin/env python3
"""Post hoc, fixed-author-parameter replay. No parameter estimation."""
import csv, hashlib, io, json, os, time, zipfile
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from scipy.special import expit, logit
ROOT=Path(__file__).resolve().parent
VALIDATION=ROOT.parent
STUDY=VALIDATION.parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def write_csv(p,rows):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 started=time.perf_counter()
 protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
 lock=json.loads((ROOT/'LOCK.json').read_text())
 assert sha(__file__)==lock['script_sha256']
 assert sha(ROOT/'PROTOCOL.json')==lock['protocol_sha256']
 assert not (ROOT/'results.json').exists(),'Do not overwrite completed secondary analysis'
 source=STUDY/'literature_sources'/'Kumar2022_supplementary_software.zip'
 assert sha(source)==protocol['software_sha256']
 z=zipfile.ZipFile(source);prefix='SupplementarySoftwareFile/Co-Culture_Model/'
 pars=loadmat(io.BytesIO(z.read(prefix+'opt_params.mat')),variable_names=['x','p_fix_struct'],simplify_cells=True)
 x=pars['x'];f=pars['p_fix_struct'];gr_const=1.58369475/60
 aT,aC,KG,hmin,hmax,KL,nL,nG,hC,mol2fluo,L0=x
 Np,KD,KC,k,nr,nT,nC,g0,phimax,phi0,dcell,nu,AE=[f[k] for k in ['N_p','K_D','K_C','k','n_r','n_T','n_C','g_0','phi_max','phi_0','d_cell','nu','A_ext']]
 diagnostics={'negative_fractional_operand_rhs_evaluations':0,'minimum_trial_operand':0.,'integrations':0,'nfev':0,'accepted_state_minimum':1e100}
 def rhs(t,y,L,internal_only=False):
  TT,TD,C=y[:3]
  if TD<0 or C<0:
   diagnostics['negative_fractional_operand_rhs_evaluations']+=1
   diagnostics['minimum_trial_operand']=min(diagnostics['minimum_trial_operand'],float(TD),float(C))
  # Numerical extension at solver trial points only. Accepted-state sign checked.
  TDp=max(TD,0.);Cp=max(C,0.)
  gon=Np*TDp**nG/(TDp**nG+KG**nG)
  phis=(nT*TT+nC*C)/dcell
  gr=(phimax-phi0-phis)*nu*g0/(g0+nu*(1+(AE/KD)/(1+(Cp/(k*KC))**hC)))
  hon=hmin+(hmax-hmin)*L**nL/(L**nL+KL**nL)
  ru=dcell*gr/(nr*g0)
  deriv=[aT*ru*gr/nu-gr*TT,hon*(TT-2*(TD+gon))**2-gr*(TD+gon),aC*ru*gon/nu-gr*C]
  return deriv if internal_only else deriv+[gr-gr_const]
 def integrate(y,L,dt,t_eval=None,internal_only=False,rtol=1e-9,atol=1e-10):
  r=solve_ivp(lambda t,y:rhs(t,y,L,internal_only),[0,dt],y,method='BDF',rtol=rtol,atol=atol,t_eval=t_eval)
  assert r.success,r.message
  diagnostics['integrations']+=1;diagnostics['nfev']+=r.nfev
  m=float(np.min(r.y[:3]));diagnostics['accepted_state_minimum']=min(diagnostics['accepted_state_minimum'],m)
  assert m>=-1e-7,f'Negative accepted internal state {m}'
  return r
 init=integrate(np.zeros(3),float(L0),100000.,internal_only=True).y[:,-1]
 steady_residual=np.asarray(rhs(0,init,float(L0),True))
 numerical=[]
 for light,name in [(0,'OL_Simulation_0L.mat'),(800,'OL_Simulation_800L.mat')]:
  ref=loadmat(io.BytesIO(z.read(prefix+name)),variable_names=['t_OL','y_OL'],simplify_cells=True)
  t=ref['t_OL'];y=ref['y_OL'];start=np.r_[y[0,:3],logit(y[0,3])]
  r=integrate(start,light,float(t[-1]),t_eval=t)
  numerical.append(dict(file=name,light=light,points=len(t),max_internal_abs_error=np.max(np.abs(r.y[:3].T-y[:,:3]),axis=0).tolist(),max_fraction_abs_error=float(np.max(abs(expit(r.y[3])-y[:,3]))),initial_internal_max_abs_difference=float(np.max(abs(init-y[0,:3]))),qualification='Archived numerical outputs, not measured biology; publisher-supplied solver tolerances not recorded in data.'))
 rows=list(csv.DictReader((VALIDATION/'data'/'measured_trajectories.csv').open()))
 ids=list(dict.fromkeys(r['trajectory'] for r in rows if r['split']=='holdout'))
 records=[];predictions=[];metrics=[];tolerance=[]
 for name in ids:
  a=[r for r in rows if r['trajectory']==name]
  t=np.array([float(r['time_h']) for r in a]);y=np.array([float(r['photophilic_fraction']) for r in a]);L=np.array([float(r['light_digital_au']) for r in a]);target=np.array([float(r['setpoint']) for r in a])
  free=np.empty(len(t));free[0]=y[0];one=np.full(len(t),np.nan);state=np.r_[init,logit(y[0])];internal=[init.tolist()]
  for i,dt in enumerate(np.diff(t)*60):
   oldz=state[3]
   state=integrate(state,float(L[i]),float(dt)).y[:,-1]
   free[i+1]=expit(state[3]);one[i+1]=expit(logit(y[i])+state[3]-oldz);internal.append(state[:3].tolist())
  # Numerical convergence check uses the same fixed parameters and initial state.
  fine=np.empty(len(t));fine[0]=y[0];s=np.r_[init,logit(y[0])]
  for i,dt in enumerate(np.diff(t)*60):
   s=integrate(s,float(L[i]),float(dt),rtol=1e-10,atol=1e-11).y[:,-1];fine[i+1]=expit(s[3])
  tolerance.append(dict(trajectory=name,max_free_run_fraction_change=float(np.max(abs(free-fine)))))
  for window,mask in [('full',t>0),('through40h',(t>0)&(t<=40)),('after40h',t>40)]:
   for mode,pred,baseline in [('free_run',free,np.full(y.size,y[0])),('one_step',one,np.r_[np.nan,y[:-1]])]:
    e=pred[mask]-y[mask];be=baseline[mask]-y[mask]
    metrics.append(dict(trajectory=name,window=window,mode=mode,n=int(sum(mask)),rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(abs(e))),baseline_rmse=float(np.sqrt(np.mean(be*be))),max_abs_error=float(np.max(abs(e)))))
  for i in range(len(t)):
   predictions.append(dict(trajectory=name,time_h=t[i],observed_fraction=y[i],light_digital_au=L[i],free_run_fraction=free[i],one_step_fraction='' if i==0 else one[i],TT=internal[i][0],TD=internal[i][1],C=internal[i][2]))
  records.append(dict(name=name,t=t,y=y,L=L,target=target,free=free,one=one))
 summaries=[]
 for mode in ['free_run','one_step']:
  for window in ['full','through40h','after40h']:
   m=[a for a in metrics if a['mode']==mode and a['window']==window]
   summaries.append(dict(mode=mode,window=window,trajectories=len(m),macro_rmse=float(np.mean([a['rmse'] for a in m])),macro_baseline_rmse=float(np.mean([a['baseline_rmse'] for a in m])),better_than_baseline_count=sum(a['rmse']<a['baseline_rmse'] for a in m)))
 write_csv(ROOT/'metrics.csv',metrics);write_csv(ROOT/'predictions.csv',predictions)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(2,4,figsize=(12.4,6.),sharey=True,layout='constrained')
 for ax,r in zip(axes.flat,records):
  ax.plot(r['t'],r['y'],color='#202020',lw=1.2,label='Measured fraction')
  ax.plot(r['t'],r['free'],color='#a24229',lw=1.,label='Fixed published-model rollout')
  ax.plot(r['t'],r['target'],color='#888888',ls=':',lw=.9,label='Published setpoint')
  ax.axvspan(40,r['t'][-1],color='#d6d6d6',alpha=.3)
  ax.set_title(r['name'].replace('closedloop_','').replace('Setpoint',' target').replace('_',' '),fontsize=9)
  ax.set_ylim(-.025,1.025);ax.set_xlabel('Time (h)');ax.grid(alpha=.18)
 axes[0,0].set_ylabel('Photophilic cell fraction');axes[1,0].set_ylabel('Photophilic cell fraction')
 handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
 fig.suptitle('Secondary post hoc replay: published parameters, assumed ambient internal state',fontsize=12)
 fig.savefig(ROOT/'heldout_rollouts.pdf');fig.savefig(ROOT/'heldout_rollouts.png',dpi=180);plt.close(fig)
 result=dict(analysis_status='Secondary post hoc; no holdout refit; not a new-policy intervention experiment',protocol_sha256=sha(ROOT/'PROTOCOL.json'),script_sha256=sha(__file__),software_sha256=sha(source),published_variable_parameters=x.tolist(),published_fixed_parameters=f,constitutive_growth_per_min=gr_const,assumed_ambient_light_digital_au=float(L0),assumed_initial_internal_state=init.tolist(),initial_steady_rhs=steady_residual.tolist(),initial_fraction='Measured initial fraction per run; not fitted',numerical_references=numerical,tolerance_checks=tolerance,summaries=summaries,diagnostics=diagnostics,elapsed_seconds=time.perf_counter()-started)
 js(ROOT/'results.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
