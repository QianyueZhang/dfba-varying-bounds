"""Independent balance, data and direct-KKT verification for Chapters 5–6.
Writes only to runs/revision_validation. Command modes can run independently.
"""
from control_experiments import *
from flux_model import FluxModel
from primal_dual import solve
from scipy.integrate import simpson
import argparse
R=O/'revision_validation';R.mkdir(parents=True,exist_ok=True)
def dump(name,data):(R/(name+'.json')).write_text(json.dumps(data,indent=2))
def diagnostics(model,state,out,mu=1e-6):
 x,lam,zl,zu,_=out;l,u=model.bounds(state)
 return {'equality':float(np.max(np.abs(model.A@x-model.b))),'stationarity':float(np.max(np.abs(-model.c+model.A.T@lam-zl+zu))),'lower_complementarity':float(np.max(np.abs((x-l)*zl-mu))),'upper_complementarity':float(np.max(np.abs((u-x)*zu-mu))),'minimum_slack':float(min(np.min(x-l),np.min(u-x))),'minimum_dual':float(min(zl.min(),zu.min()))}
def quick():
 data=np.load(O/'surrogate_dataset.npz');X=data['controls'];z=data['target_log_ratio'];Y=data['terminal_states'];split=[data[k] for k in ['train','validation','test']];net=pickle.load(open(O/'surrogate_model.pkl','rb'));r=json.loads((O/'control_results.json').read_text());rows=[]
 def check(n,a,b,tol):rows.append({'check':n,'value':float(a),'expected':float(b),'absolute_tolerance':tol,'pass':bool(abs(a-b)<=tol)})
 check('1000 distinct controls',len(np.unique(X,axis=0)),1000,0);check('split disjoint coverage',len(np.unique(np.concatenate(split))),1000,0);check('all controls within bounds',np.all((X>=0)&(X<=.2)),1,0);check('log-ratio target matches terminal states',np.max(abs(z-np.log(Y[:,1]/Y[:,2]))),0,1e-14);check('volume identity all training trajectories',np.max(abs(Y[:,0]-(.5+2*X.sum(axis=1)))),0,1e-12)
 check('scaler learned training mean only',np.max(abs(net[0].mean_-X[split[0]].mean(axis=0))),0,1e-14)
 held=split[2];p=net.predict(X[held]);check('test log RMSE',np.sqrt(np.mean((p-z[held])**2)),r['neural']['test_log_ratio_rmse'],1e-14);check('test objective RMSE',np.sqrt(np.mean((np.expm1(p)**2-np.expm1(z[held])**2)**2)),r['neural']['test_objective_rmse'],1e-14)
 frozen=Reactor(table_path=O/'flux_table_control_experiment_snapshot.json');recomputed=[]
 for i in np.random.default_rng(2811).choice(1000,30,replace=False):
  y=np.array(frozen.simulate(X[i])['terminal']);recomputed.append({'row':int(i),'max_absolute_terminal_difference':float(np.max(abs(y-Y[i]))),'log_ratio_difference':float(np.log(y[1]/y[2])-z[i])})
 check('30 independently rerun dataset entries',max(q['max_absolute_terminal_difference'] for q in recomputed),0,1e-10)
 for method in ['SLSQP','PSO','Secant-PSO']:
  runs=[q for q in r['runs'] if q['method']==method]
  check(method+' repeat count',len(runs),5,0);check(method+' budget',max(q['evaluations'] for q in runs)<=400,1,0)
  for row in runs:
   val=frozen.simulate(row['control'])['objective'];check(method+str(row['seed'])+' objective rerun',val,row['objective'],1e-13)
 ref=Reactor();recomputed_baselines={name:ref.simulate(row['control'],1e-9,1e-11) for name,row in r['baselines'].items()};best_biomass=min(json.loads((O/'biomass_objective_results.json').read_text()),key=lambda d:d['terminal_KP_objective']);recomputed_biomass=ref.simulate(best_biomass['control'],1e-9,1e-11)
 flux=[]
 for name in ['SC','KP']:
  model=FluxModel(name,1e-6)
  for S in [.02,.2,1.,20.,60.]:
   out=solve(model,S,tol=1e-11);lp=model.lp(S);gap=-lp.fun-float(model.c@out[0]);barrier_bound=2*len(model.c)*1e-6
   d=diagnostics(model,S,out);flux.append({'species':name,'S_g_L':S,**d,'lp_growth':-lp.fun,'barrier_growth':float(model.c@out[0]),'growth_gap':gap,'barrier_duality_bound':barrier_bound})
   check(name+str(S)+' feasible KKT',max(d['equality'],d['stationarity'],d['lower_complementarity'],d['upper_complementarity']),0,1.1e-11);check(name+str(S)+' positive primal dual',d['minimum_slack']>0 and d['minimum_dual']>0,1,0);check(name+str(S)+' LP growth gap bound',-1e-8<=gap<=barrier_bound+1e-8,1,0)
 dump('quick_audit',{'checks':rows,'passed':sum(x['pass'] for x in rows),'failed':[x for x in rows if not x['pass']],'dataset_reruns':recomputed,'baselines_refined':recomputed_baselines,'biomass_objective_refined':recomputed_biomass,'flux_KKT_and_LP_checks':flux})
 print('quick audit',sum(x['pass'] for x in rows),'/',len(rows),flush=True)
def direct(label):
 old=json.loads((O/'control_results.json').read_text());u=old['best']['control'] if label=='best' else next(r['polished_control'] for r in json.loads((R/'neural_extension.json').read_text())['hybrid'] if r['seed']==736)
 models=[FluxModel(n,1e-6) for n in ['SC','KP']];warm=[None,None];calls=0;records=[];ts=[0.];ys=[np.array([.5,.001,.001,20.])];q=np.array([.0005,.0005,10.]);V0=.5;t0=0;start=time.perf_counter()
 # Independent amount balances. Volume is analytic, not an ODE state.
 for j,F in enumerate(u):
  def rhs(t,q):
   nonlocal calls
   V=V0+F*(t-t0);S=max(q[2]/V,0);out=[]
   for i,m in enumerate(models):
    try:a=solve(m,S,tol=1e-11,warm=warm[i])
    except RuntimeError:a=solve(m,S,tol=1e-11)
    warm[i]=a[:4];out.append(a[0]);records.append(diagnostics(m,S,a))
   calls+=1;factor=min(1,S/1e-5);v,w=out
   return [v[models[0].bio]*factor*q[0],w[models[1].bio]*factor*q[1],F*100+.18*factor*(v[models[0].g]*q[0]+w[models[1].g]*q[1])]
  a,b=2*j,2*j+2;sol=solve_ivp(rhs,(a,b),q,method='DOP853',rtol=5e-8,atol=5e-11,max_step=.25,dense_output=True)
  if not sol.success:raise RuntimeError(sol.message)
  tt=np.linspace(a,b,51)[1:];qq=sol.sol(tt).T;VV=V0+F*(tt-a);ts.extend(tt);ys.extend(np.c_[VV,qq/VV[:,None]]);q=sol.y[:,-1];V0+=2*F;t0=b
  print('direct',label,'interval',j,'calls',calls,flush=True)
 y=np.r_[V0,q/V0];np.savetxt(R/f'direct_{label}_trajectory.csv',np.c_[ts,ys],delimiter=',',header='time_h,V_L,X_SC_gDW_L,X_KP_gDW_L,S_g_L',comments='')
 ref=Reactor().simulate(u,1e-10,1e-12)
 d={'control':u,'terminal':y.tolist(),'objective':float((1-y[1]/y[2])**2),'absolute_ratio_error_percent':float(abs(1-y[1]/y[2])*100),'minimum_glucose':float(np.asarray(ys)[:,3].min()),'refined_table':ref,'terminal_relative_difference_from_table':((y-np.array(ref['terminal']))/np.maximum(np.abs(y),1e-12)).tolist(),'max_residuals':{k:max(q[k] for q in records) for k in ['equality','stationarity','lower_complementarity','upper_complementarity']},'minimum_slack':min(q['minimum_slack'] for q in records),'rhs_evaluations':calls,'wall_seconds':time.perf_counter()-start,'protocol':'Amount balances dM_i/dt=g_i M_i; d(VS)/dt=F Sin+0.18 sum(vGi Mi); analytic volume, DOP853 rtol5e-8,atol5e-11,maxstep0.25,KKTtol1e-11. Independent balance implementation; same KKT solver/matrices.'}
 if label=='best':
  prior=json.loads((O/'control_validation.json').read_text())['selected_controls']['best']['direct_KKT_integration'];d['previous_direct_result']=prior;d['previous_terminal_difference']=(y-prior['terminal']).tolist()
 dump('direct_'+label,d);print('complete',label,d['objective'],flush=True)
def cstr():
 from cstr_experiments import CSTRModel
 from revision_precision_kkt import solve
 ms=[CSTRModel('SC'),CSTRModel('RC')];warm=[None,None];count=0;records=[];y=np.array([125,1.1,1.86,.014,3,.000653,1.06,8.21]);ts=[0];ys=[y.copy()];then=time.perf_counter();eqmax=0
 def rhs(t,y):
  nonlocal count
  out=[]
  for i,m in enumerate(ms):
   try:a=solve(m,(t,y),tol=2e-12,warm=warm[i])
   except RuntimeError:a=solve(m,(t,y),tol=2e-12)
   warm[i]=a[:4];out.append(a[0]);records.append(diagnostics(m,(t,y),a))
  count+=1;v,w=out;V,X,Y,G,A,O2,C,E=y;D=1/125.;khO=55560/(25131+709.2*25);khC=1000*np.exp(-8.1403+842.9/176.5)
  # Explicitly form inflow, outflow, intracellular production, and gas transfer.
  cin=np.array([.014,30,.000653,1.06,0]);con=np.array([G,A,O2,C,E]);rates=np.array([v[436]*X,v[713]*X+w[77]*Y,v[5]*X+(w[50]+w[91])*Y,v[415]*X+w[54]*Y,v[428]*X+w[19]*Y]);gas=np.array([0,0,.25*(khO*.21-O2),.25*(khC*.00032-C),0]);solute=D*(cin-con)+rates+gas
  return np.r_[0,(v[1264]-D)*X,(w[47]-D)*Y,solute]
 for a,b in [(0,5),(5,12),(12,19),(19,24)]:
  sol=solve_ivp(rhs,(a,b),y,method='BDF',rtol=2e-8,atol=2e-11,max_step=.125,dense_output=True)
  if not sol.success:raise RuntimeError(sol.message)
  tt=np.linspace(a,b,101)[1:];ts.extend(tt);ys.extend(sol.sol(tt).T);y=sol.y[:,-1];print('cstr segment',b,'rhs',count,flush=True)
 prior=json.loads((O/'cstr_results.json').read_text());dump('cstr_finest',{'terminal':y.tolist(),'previous_terminal':prior['terminal'],'absolute_terminal_difference':(y-prior['terminal']).tolist(),'max_residuals':{k:max(q[k] for q in records) for k in ['equality','stationarity','lower_complementarity','upper_complementarity']},'minimum_slack':min(q['minimum_slack'] for q in records),'minimum_states':np.array(ys).min(axis=0).tolist(),'rhs_evaluations':count,'wall_seconds':time.perf_counter()-then,'protocol':'Independent explicit vector balances, same networks and light/medium; BDF rtol2e-8,atol2e-11,maxstep.125,KKTcomp_tol2e-12,equality_and_dual_tol1e-10'})
 np.savetxt(R/'cstr_finest.csv',np.c_[ts,ys],delimiter=',',header='t_h,V_L,X_SC_gDW_L,X_RC_gDW_L,G_mmol_L,Ac_mmol_L,O2_mmol_L,CO2_mmol_L,E_mmol_L',comments='');print('cstr complete',flush=True)
class EthanolModel(FluxModel):
 def __init__(self):super().__init__('SC',1e-6);self.minimum_ethanol=10
 def bounds(self,S):
  l,u=super().bounds(S);l[2707]=self.minimum_ethanol-self.epsilon;return l,u
class InhibitedKP(FluxModel):
 def __init__(self,Ki):super().__init__('KP',1e-6);self.Ki=Ki
 def bounds(self,state):
  S,E=state;l,u=super().bounds(S);l[self.g]=-6*max(S,0)/(.385+max(S,0))/(1+max(E,0)/self.Ki)-self.epsilon;return l,u
 def lp(self,state):
  from scipy.optimize import linprog
  l,u=self.bounds(state);r=linprog(-self.c,A_eq=self.A,b_eq=self.b,bounds=np.c_[l,u],method='highs',options={'dual_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8})
  if not r.success:raise RuntimeError(r.message)
  return r

def inhibition():
 allresults=[]
 for Ki in [5.,float('inf')]:
  ms=[EthanolModel(),InhibitedKP(Ki)];warm=[None,None];rec=[];count=0;y=np.array([.001,.001,20.,0.]);start=0.;ts=[0];ys=[y.copy()];then=time.perf_counter();events=[]
  for etoh,threshold in [(10,7.5),(2,2),(0,1e-5)]:
   ms[0].minimum_ethanol=etoh;warm=[None,None]
   def rhs(t,y):
    nonlocal count
    X,Y,S,E=y;inputs=[max(S,threshold),(max(S,0),max(E,0))];v=[]
    for i,m in enumerate(ms):
     try:a=solve(m,inputs[i],tol=1e-10,warm=warm[i])
     except RuntimeError:a=solve(m,inputs[i],tol=1e-10)
     warm[i]=a[:4];v.append(a[0]);rec.append(diagnostics(m,inputs[i],a))
    count+=1;v,w=v
    return [v[ms[0].bio]*X,w[ms[1].bio]*Y,.18*(v[ms[0].g]*X+w[ms[1].g]*Y),.046*v[2707]*X]
   def event(t,y):return y[2]-threshold
   event.terminal=True;event.direction=-1
   sol=solve_ivp(rhs,(start,14),y,method='DOP853',rtol=5e-7,atol=1e-9,max_step=.25,events=event,dense_output=True)
   if not sol.success:raise RuntimeError(sol.message)
   end=sol.t[-1];tt=np.linspace(start,end,101)[1:];ts.extend(tt);ys.extend(sol.sol(tt).T);y=sol.y[:,-1];events.append({'threshold':threshold,'time':end});start=end;print('inhibition',Ki,etoh,end,'rhs',count,flush=True)
   if end>=14:break
  if start<14:ts.append(14);ys.append(y.copy())
  key='inhibited' if Ki==5 else 'uninhibited';prior=next(x for x in json.loads((O/'inhibition_results.json').read_text()) if x['case']==key);out={'case':key,'terminal':y.tolist(),'previous_terminal':prior['terminal'],'terminal_difference':(y-prior['terminal']).tolist(),'events':events,'max_residuals':{k:max(q[k] for q in rec) for k in ['equality','stationarity','lower_complementarity','upper_complementarity']},'minimum_slack':min(q['minimum_slack'] for q in rec),'rhs_evaluations':count,'wall_seconds':time.perf_counter()-then,'protocol':'Both organisms solved directly by KKT; KP bound directly includes ethanol rather than table substitution; DOP853rtol5e-7,atol1e-9,maxstep.25,KKTtol1e-10.'};allresults.append(out);dump('inhibition_direct',allresults);np.savetxt(R/f'{key}_direct.csv',np.c_[ts,ys],delimiter=',',header='t_h,X_SC_gDW_L,X_KP_gDW_L,S_g_L,E_g_L',comments='')
 print('inhibition complete',flush=True)
def batch():
 res=[]
 for i,n in enumerate(['SC','KP']):
  m=FluxModel(n,1e-6);warm=None;count=0;di=[];then=time.perf_counter()
  def rhs(t,y):
   nonlocal warm,count
   X,S=y;S=max(S,0)
   try:a=solve(m,S,tol=1e-10,warm=warm)
   except RuntimeError:a=solve(m,S,tol=1e-10)
   warm=a[:4];di.append(diagnostics(m,S,a));count+=1;v=a[0];f=min(1,S/1e-5)
   return [v[m.bio]*X*f,.18*v[m.g]*X*f]
  sol=solve_ivp(rhs,(0,12),[.001,20],method='DOP853',rtol=1e-7,atol=1e-10,max_step=.25,dense_output=True)
  if not sol.success:raise RuntimeError(sol.message)
  previous=np.loadtxt(O/f'batch_{i}.csv',delimiter=',',skiprows=1)[-1];prior=previous[[i+2,4]];y=sol.y[:,-1];res.append({'species':n,'terminal':y.tolist(),'previous_terminal':prior.tolist(),'difference':(y-prior).tolist(),'rhs_evaluations':count,'wall_seconds':time.perf_counter()-then,'max_residuals':{k:max(q[k] for q in di) for k in ['equality','stationarity','lower_complementarity','upper_complementarity']},'protocol':'Direct KKT at each RHS; DOP853rtol1e-7,atol1e-10,maxstep.25,KKTtol1e-10, same depletion-ramp convention.'});dump('batch_direct',res);tt=np.linspace(0,12,241);np.savetxt(R/f'batch_{n}_direct.csv',np.c_[tt,sol.sol(tt).T],delimiter=',',header='t_h,X_gDW_L,S_g_L',comments='');print('batch',n,'complete',y,flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['quick','best','polished','cstr','inhibition','batch']);a=parser.parse_args();{'quick':quick,'best':lambda:direct('best'),'polished':lambda:direct('polished'),'cstr':cstr,'inhibition':inhibition,'batch':batch}[a.mode]()
