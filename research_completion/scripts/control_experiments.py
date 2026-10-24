"""Matched 7-interval genome-scale fed-batch control and held-out surrogate study."""
from pathlib import Path
import sys,os,json,time,platform,hashlib,pickle,warnings
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.exceptions import ConvergenceWarning
from flux_model import ROOT
O=ROOT/'research_completion/runs';F=ROOT/'research_completion/figures';F.mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(O/'matplotlib_cache')
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
warnings.filterwarnings('ignore',category=ConvergenceWarning)
class Reactor:
 def __init__(self,epsilon=1e-6,table_path=None,boundary_ramp=1e-5):
  self.boundary_ramp=boundary_ramp
  self.path=Path(table_path) if table_path else O/f'flux_table_eps{epsilon:g}.json';self.table=json.loads(self.path.read_text());self.curves=[]
  for name in ['SC','KP']:
   rows=self.table[name]['rows'];self.curves.append(PchipInterpolator([r['S'] for r in rows],[[r['growth'],r['glucose_flux']] for r in rows],axis=0))
 def simulate(self,u,rtol=1e-7,atol=1e-9,dense=False,initial=None,horizon=14):
  y=np.array([.5,.001,.001,20.]) if initial is None else np.array(initial,float);ts=[0.];ys=[y.copy()];minimum=20.
  for j,feed in enumerate(u):
   def rhs(t,y):
    V,x1,x2,S=y;Suse=max(S,0.)
    a,b=(c(Suse) for c in self.curves)
    # Boundary convention: arrest growth and uptake at zero available glucose.
    # Smoothly vanish rates only below 1e-5 g/L; sensitivity is recorded separately.
    factor=min(1.,Suse/self.boundary_ramp)
    a=a*factor;b=b*factor
    return [feed,(a[0]-feed/V)*x1,(b[0]-feed/V)*x2,feed/V*(100.-S)+.18*(a[1]*x1+b[1]*x2)]
   span=(j*horizon/len(u),(j+1)*horizon/len(u));sol=solve_ivp(rhs,span,y,rtol=rtol,atol=atol,method='DOP853',dense_output=dense,max_step=.25)
   if not sol.success:raise RuntimeError(sol.message)
   if np.min(sol.y[3]) < -1e-6:raise RuntimeError('Negative substrate beyond numerical tolerance')
   y=sol.y[:,-1];minimum=min(minimum,float(np.min(sol.y[3])))
   if dense:
    tt=np.linspace(*span,51)[1:];ts.extend(tt);ys.extend(sol.sol(tt).T)
  J=float((1-y[1]/y[2])**2) if y[2]>0 else None
  result={'objective':J,'terminal':y.tolist(),'minimum_glucose':minimum,'volume_analytic':.5+horizon/len(u)*float(np.sum(u))}
  if dense:result.update(time=np.array(ts),state=np.array(ys))
  return result
class BudgetEnd(Exception):pass
class Evaluator:
 def __init__(self,fun,budget):self.fun=fun;self.budget=budget;self.n=0;self.best=np.inf;self.u=None;self.trace=[]
 def __call__(self,u):
  if self.n>=self.budget:raise BudgetEnd()
  value=float(self.fun(np.asarray(u)));self.n+=1
  if value<self.best:self.best=value;self.u=np.array(u).copy()
  self.trace.append(self.best);return value

def swarm(e,seed,secant=False):
 rng=np.random.default_rng(seed);pop=rng.uniform(0,.2,(20,7));v=rng.uniform(-.03,.03,(20,7));fit=np.array([e(x) for x in pop]);pb=pop.copy();pf=fit.copy();g=pop[np.argmin(fit)].copy();gf=min(fit)
 try:
  while True:
   for j in range(20):
    old=pop[j].copy();vold=v[j].copy();fold=fit[j]
    v[j]=np.clip(.8*v[j]+1.49445*rng.random()*(pb[j]-pop[j])+1.49445*rng.random()*(g-pop[j]),-.03,.03)
    pop[j]=np.clip(pop[j]+v[j],0,.2);fit[j]=e(pop[j])
    if secant:
     den=old-pop[j]
     td=np.divide(fold-fit[j],den,out=np.zeros(7),where=np.abs(den)>1e-14)
     v[j]=np.clip(.8*vold+1.49445*rng.random()*(pb[j]-old)+1.49445*rng.random()*(g-old)-1.49445*.8*rng.random()*td,-.03,.03)
     pop[j]=np.clip(pop[j]+v[j],0,.2);fit[j]=e(pop[j])
    if fit[j]<pf[j]:pf[j]=fit[j];pb[j]=pop[j].copy()
    if fit[j]<gf:gf=fit[j];g=pop[j].copy()
 except BudgetEnd:pass
 return e.u

def run():
 reactor=Reactor(table_path=O/'flux_table_control_experiment_snapshot.json');runs=[];seeds=[1701,1702,1703,1704,1705];budget=400
 summary={'protocol':{'horizon_h':14,'initial_volume_L':.5,'initial_biomass_gDW_L':[.001,.001],'initial_glucose_g_L':20,'inlet_glucose_g_L':100,'intervals':7,'feed_bounds_L_h':[0,.2],'objective':'(1-X_SC(tf)/X_KP(tf))^2','mu':1e-6,'epsilon':1e-6,'rtol':1e-7,'atol':1e-9,'evaluation_budget':budget,'seeds':seeds,'boundary_rate_ramp_g_L':1e-5,'model_policy':'Non-interacting, no forced ethanol production; raw archived bounds relaxed by epsilon; state-dependent glucose lower bound -q(S)-epsilon; source matrices retained','solver':'DOP853 on PCHIP reconstruction of consistent barrier states'},'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':__import__('scipy').__version__,'sklearn':__import__('sklearn').__version__},'baselines':{},'runs':runs}
 for label,u in [('no_feed',np.zeros(7)),('constant_0.1',np.ones(7)*.1),('constant_0.2',np.ones(7)*.2),('late_feed',np.r_[np.zeros(4),np.ones(3)*.2])]:summary['baselines'][label]={'control':u.tolist(),**reactor.simulate(u)}
 for method in ['SLSQP','PSO','Secant-PSO']:
  for seed in seeds:
   e=Evaluator(lambda u:reactor.simulate(u)['objective'],budget);start=time.perf_counter();status='budget'
   if method=='SLSQP':
    x0=np.random.default_rng(seed).uniform(0,.2,7)
    try:
     res=minimize(e,x0,method='SLSQP',bounds=[(0,.2)]*7,options={'maxiter':400,'ftol':1e-10,'eps':1e-5});status=str(res.message)
    except BudgetEnd:pass
   else:swarm(e,seed,secant=method=='Secant-PSO')
   row={'method':method,'seed':seed,'evaluations':e.n,'seconds':time.perf_counter()-start,'objective':e.best,'control':e.u.tolist(),'trace':e.trace,'status':status,'tight_validation':reactor.simulate(e.u,rtol=1e-9,atol=1e-11)};runs.append(row)
   (O/'control_results.json').write_text(json.dumps(summary,indent=2));print(method,seed,e.best,e.n,flush=True)
 best=min(runs,key=lambda r:r['objective']);summary['best']=best;trajectory=reactor.simulate(best['control'],rtol=1e-9,atol=1e-11,dense=True)
 np.savetxt(O/'best_control_trajectory.csv',np.c_[trajectory['time'],trajectory['state']],delimiter=',',header='time_h,volume_L,X_SC_gDW_L,X_KP_gDW_L,glucose_g_L',comments='')
 # Independent held-out data. Targets are terminal log ratios, transformed to a nonnegative objective.
 inputs=.2*qmc.LatinHypercube(7,seed=731).random(1000);targets=[];terminals=[];start=time.perf_counter()
 for i,u in enumerate(inputs):
  r=reactor.simulate(u);terminals.append(r['terminal']);targets.append(np.log(r['terminal'][1]/r['terminal'][2]))
  if i%100==0:print('dataset',i,flush=True)
 targets=np.array(targets);order=np.random.default_rng(732).permutation(1000);train,val,test=order[:700],order[700:850],order[850:]
 np.savez(O/'surrogate_dataset.npz',controls=inputs,target_log_ratio=targets,terminal_states=terminals,train=train,validation=val,test=test)
 candidates=[]
 for seed in [733,734,735]:
  net=make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(15,),activation='logistic',solver='lbfgs',alpha=1e-4,max_iter=1500,tol=1e-7,random_state=seed))
  net.fit(inputs[train],targets[train]);candidates.append((float(np.sqrt(mean_squared_error(targets[val],net.predict(inputs[val])))),seed,net))
 vr,seed,net=min(candidates,key=lambda x:x[0]);pred=net.predict(inputs[test]);pred_obj=np.expm1(pred)**2;true_obj=np.expm1(targets[test])**2
 nn={'target':'log(X_SC(tf)/X_KP(tf)); objective = expm1(prediction)^2','samples':1000,'train':700,'validation':150,'test':150,'data_seconds':time.perf_counter()-start,'network':'7-15-1 logistic, linear output; L-BFGS training; alpha=1e-4; max_iter=1500','selected_seed':seed,'validation_log_ratio_rmse':vr,'candidate_validation_rmse':[[a,b] for a,b,n in candidates],'test_log_ratio_rmse':float(np.sqrt(mean_squared_error(targets[test],pred))),'test_objective_rmse':float(np.sqrt(mean_squared_error(true_obj,pred_obj))),'test_objective_mae':float(mean_absolute_error(true_obj,pred_obj)),'constant_baseline_objective_rmse':float(np.sqrt(mean_squared_error(true_obj,np.repeat(np.mean(np.expm1(targets[train])**2),len(test)))))}
 pickle.dump(net,open(O/'surrogate_model.pkl','wb'))
 e=Evaluator(lambda u:np.expm1(net.predict(np.array(u).reshape(1,-1))[0])**2,1000);start=time.perf_counter();swarm(e,736);nn['optimisation']={'predicted_objective':e.best,'control':e.u.tolist(),'seconds':time.perf_counter()-start,'evaluations':e.n,'simulator_validation':reactor.simulate(e.u,rtol=1e-9,atol=1e-11)}
 np.savetxt(O/'surrogate_heldout.csv',np.c_[test,true_obj,pred_obj,targets[test],pred],delimiter=',',header='row_id,true_objective,predicted_objective,true_log_ratio,predicted_log_ratio',comments='')
 summary['neural']=nn
 # Bound-relaxation sensitivity uses exactly the selected controls.
 try:
  coarse=Reactor(1e-3);summary['epsilon_sensitivity']=[{'epsilon':eps,**react.simulate(best['control'],rtol=1e-9,atol=1e-11)} for eps,react in [(1e-6,reactor),(1e-3,coarse)]]
 except Exception as ex:summary['epsilon_sensitivity_error']=str(ex)
 (O/'control_results.json').write_text(json.dumps(summary,indent=2))
 make_plots(reactor,summary,trajectory,targets[test],pred)
 print('complete control and surrogate study',flush=True)

def save(name):
 plt.tight_layout();plt.savefig(F/(name+'.png'),dpi=240);plt.savefig(F/(name+'.pdf'));plt.close()
def make_plots(reactor,summary,tr,true,pred):
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(2,2,figsize=(9,6));t=tr['time'];y=tr['state']
 for i,label in [(1,'S. cerevisiae'),(2,'K. phaffii')]:ax[0,0].plot(t,y[:,i],label=label)
 ax[0,0].set(xlabel='Time (h)',ylabel='Biomass (gDW L$^{-1}$)');ax[0,0].legend()
 ax[0,1].plot(t,y[:,3]);ax[0,1].set(xlabel='Time (h)',ylabel='Glucose (g L$^{-1}$)')
 ax[1,0].stairs(summary['best']['control'],np.linspace(0,14,8));ax[1,0].set(xlabel='Time (h)',ylabel='Feed rate (L h$^{-1}$)',ylim=(-.005,.205))
 ax[1,1].plot(t,y[:,0]);ax[1,1].set(xlabel='Time (h)',ylabel='Volume (L)');save('control_best')
 fig,ax=plt.subplots(1,2,figsize=(10,4))
 for method in ['SLSQP','PSO','Secant-PSO']:
  rows=[r for r in summary['runs'] if r['method']==method];series=np.array([np.pad(r['trace'],(0,400-len(r['trace'])),constant_values=r['trace'][-1]) for r in rows]);med=np.median(series,axis=0);lo,hi=np.quantile(series,[.2,.8],axis=0)
  ax[0].semilogy(np.arange(1,401),np.maximum(med,1e-14),label=method);ax[0].fill_between(np.arange(1,401),np.maximum(lo,1e-14),np.maximum(hi,1e-14),alpha=.15)
 ax[0].set(xlabel='Simulator evaluations',ylabel='Best dimensionless ratio objective');ax[0].legend()
 methods=['SLSQP','PSO','Secant-PSO'];ax[1].boxplot([[r['seconds'] for r in summary['runs'] if r['method']==m] for m in methods],labels=methods);ax[1].set(ylabel='Wall time (s)');save('optimizer_comparison')
 plt.figure(figsize=(5,4));plt.scatter(np.expm1(true)**2,np.expm1(pred)**2,s=14,alpha=.6);lim=max(max(np.expm1(true)**2),max(np.expm1(pred)**2));plt.plot([0,lim],[0,lim],color='black',ls='--');plt.xlabel('Simulator objective (dimensionless)');plt.ylabel('Held-out predicted objective');save('neural_heldout')
 fig,ax=plt.subplots(1,2,figsize=(9,4))
 for name,i in [('S. cerevisiae',0),('K. phaffii',1)]:
  initial=[.5,.001 if i==0 else 0,.001 if i==1 else 0,20];r=reactor.simulate([0],initial=initial,horizon=12,dense=True);np.savetxt(O/('batch_'+str(i)+'.csv'),np.c_[r['time'],r['state']],delimiter=',',header='time_h,V_L,X_SC_gDW_L,X_KP_gDW_L,S_g_L',comments='');ax[0].plot(r['time'],r['state'][:,i+1],label=name);ax[1].plot(r['time'],r['state'][:,3],label=name)
 for a in ax:a.set_xlabel('Time (h)');a.legend()
 ax[0].set_ylabel('Biomass (gDW L$^{-1}$)');ax[1].set_ylabel('Glucose (g L$^{-1}$)');save('isolated_genome_scale_batch')
if __name__=='__main__':run()
