"""Differentiated finite-barrier KKT verification of the adopted inhibition case."""
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.sparse import bmat,diags
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp
sys.path.insert(0,str(Path(__file__).parent))
import product_inhibition_check as src
src.F=1.
from product_inhibition_check import Model,solve_pd,MU,EPS,MWG,MWE,KI,ROOT,OUT

def consistent(m,env):
 z=solve_pd(m,env,mu=MU,tol=1e-12,warm=None);v,lam=z[:2];l,u=m.bounds(env)
 for k in range(12):
  sl=v-l;su=u-v;r=-m.c+m.A.T@lam-MU/sl+MU/su;rp=m.A@v-m.b
  if max(abs(r).max(),abs(rp).max())<1e-9:return v,lam
  H=MU/sl**2+MU/su**2;K=bmat([[diags(H),m.A.T],[m.A,None]],format='csc')
  d=spsolve(K,-np.r_[r,rp]);dv=d[:95];alpha=1.
  for slack,delta in [(sl,dv),(su,-dv)]:
   jj=delta<0
   if np.any(jj):alpha=min(alpha,.99*float(np.min(-slack[jj]/delta[jj])))
  v=v+alpha*dv;lam=lam+alpha*d[95:]
 raise RuntimeError('Stationarity polishing did not converge')

class Differential:
 def __init__(self):
  self.mods=[Model('separate',s,1e-12)for s in ['S1','S2']];self.calls=0;self.trial_slack=np.inf
 def initial(self,amounts,t,level):
  self.mods[0].level=level;out=[amounts]
  for i,m in enumerate(self.mods):
   env=(amounts[2+i]/(5+t),amounts[4]/(5+t));v,lam=consistent(m,env);out.extend([v,lam])
  return np.concatenate(out)
 def unpack(self,y):return [y[11:106],y[173:268]],[y[106:173],y[268:335]]
 def rhs(self,t,y):
  self.calls+=1;V=5+t;v,lam=self.unpack(y);a=y[:11]
  growth=np.array([v[i][12]*a[i]for i in range(2)]);gg=np.array([MWG*v[i][27]*a[i]for i in range(2)]);ee=np.array([MWE*v[i][23]*a[i]for i in range(2)])
  da=np.r_[growth,np.ones(2)*10+gg,ee.sum(),growth,gg,ee]
  out=[da];E=a[4]/V;dE=(da[4]-E)/V
  for i,m in enumerate(self.mods):
   G=a[2+i]/V;dG=(da[2+i]-G)/V;l,u=m.bounds((G,E));sl=v[i]-l;su=u-v[i]
   self.trial_slack=min(self.trial_slack,float(min(sl.min(),su.min())))
   dl=np.zeros(95);dq=10*dG/(1+G)**2
   if i==1:dq=dq/(1+E/KI)-10*G/(1+G)*dE/KI/(1+E/KI)**2
   dl[27]=-dq
   zl=MU/sl**2;zu=MU/su**2;K=bmat([[diags(zl+zu),m.A.T],[m.A,None]],format='csc')
   derivative=spsolve(K,np.r_[zl*dl,np.zeros(67)])
   out.append(derivative)
  return np.concatenate(out)

def main():
 m=Differential();amounts=np.r_[5.,5.,50.,50.,0.,np.zeros(6)];t0=0.;level=1.;segments=[];events=[];start=time.perf_counter()
 while t0<10-1e-12:
  initial=m.initial(amounts,t0,level);threshold=7.5 if level==1 else 2.
  def change(t,y):return y[2]/(5+t)-threshold
  change.terminal=True;change.direction=-1
  sol=solve_ivp(m.rhs,(t0,10.),initial,method='DOP853',rtol=1e-10,atol=1e-12,max_step=.02,events=change if level>0 else None,dense_output=True)
  assert sol.success,sol.message
  segments.append({'t0':t0,'tf':float(sol.t[-1]),'level':level,'solution':sol})
  t0=float(sol.t[-1]);amounts=sol.y[:11,-1]
  if t0>=10-1e-12:break
  events.append({'time_h':t0,'G1':threshold});level-=.5
  print('secretion event',t0,threshold,flush=True)
 accepted=[]
 for seg in segments:
  m.mods[0].level=seg['level']
  for t,y in zip(seg['solution'].t,seg['solution'].y.T):
   v,lam=m.unpack(y)
   for i,model in enumerate(m.mods):
    l,u=model.bounds((y[2+i]/(5+t),y[4]/(5+t)))
    accepted.append([float(min((v[i]-l).min(),(u-v[i]).min())),float(abs(model.rawA@v[i]).max())])
 accepted=np.asarray(accepted)
 tt=np.linspace(0,10,401);aa=[];vv=[];res=[];gaps=[]
 for t in tt:
  seg=next((s for s in segments if s['t0']<=t<s['tf']),segments[-1]);m.mods[0].level=seg['level'];y=seg['solution'].sol(t);v,lam=m.unpack(y);aa.append(y[:11]);vv.append(v);rr=[]
  for i,model in enumerate(m.mods):
   env=(y[2+i]/(5+t),y[4]/(5+t));l,u=model.bounds(env);sl=v[i]-l;su=u-v[i]
   rr.append([float(abs(model.rawA@v[i]).max()),float(abs(-model.c+model.A.T@lam[i]-MU/sl+MU/su).max()),float(min(sl.min(),su.min()))])
   lp=model.lp(env);assert lp.success;gaps.append(float(-lp.fun-v[i][12]))
  res.append(rr)
 aa=np.asarray(aa);vv=np.asarray(vv);res=np.asarray(res);direct=dict(np.load(OUT/'separate/flow_1p0_refined_balanced/trajectory.npz'));ix=[int(np.where(np.isclose(direct['time_h'],t,rtol=0,atol=1e-12))[0][0])for t in tt]
 errors=np.max(abs(aa[:,:5]-direct['amounts'][ix,:5])/(5+tt[:,None]),axis=0)
 report={'model':'Separate glucose pools; F=1 L/h; inhibition enabled; shared ethanol balances both species; no biomass death or quiescent mode.','method':'Differentiated finite-barrier KKT integrated using DOP853; consistent initialisation and event restart; no flux clipping.','rtol':1e-10,'atol':1e-12,'max_step_h':.02,'mu':MU,'full_10h':True,'events':events,'maximum_raw_stoichiometric_residual':float(res[:,:,0].max()),'maximum_barrier_stationarity_residual':float(res[:,:,1].max()),'minimum_sampled_slack':float(res[:,:,2].min()),'minimum_internal_trial_slack':m.trial_slack,'max_difference_to_direct_concentrations':errors.tolist(),'max_difference_to_direct_flux':float(abs(vv-direct['flux'][ix]).max()),'max_LP_growth_gap':max(gaps),'nominal_gap_bound':190*MU,'elapsed_s':time.perf_counter()-start,'rhs_calls':m.calls}
 assert report['minimum_sampled_slack']>0
 report['accepted_output_species_states_checked']=len(accepted)
 report['minimum_accepted_output_slack']=float(accepted[:,0].min())
 report['maximum_accepted_output_stoichiometric_residual']=float(accepted[:,1].max())
 report['trial_state_scope']='Intermediate Runge-Kutta trial-stage approximations are not reported as feasible intracellular solutions. Accepted step outputs and all saved states have independently checked positive slacks; no clipping is applied.'
 assert report['minimum_accepted_output_slack']>0
 assert report['maximum_raw_stoichiometric_residual']<1e-9
 assert max(errors)<2e-6
 assert report['max_LP_growth_gap']<190*MU+1e-8
 dest=OUT/'separate/differentiated';dest.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(dest/'trajectory.npz',time_h=tt,amounts=aa,flux=vv,residuals=res)
 (dest/'audit.json').write_text(json.dumps(report,indent=2));(dest/'run_script.py').write_text(Path(__file__).read_text())
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
