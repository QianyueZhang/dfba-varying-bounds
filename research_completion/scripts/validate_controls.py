from control_experiments import *
from flux_model import FluxModel
from primal_dual import solve
from scipy.integrate import solve_ivp
r=json.loads((O/'control_results.json').read_text());ref=Reactor();old=Reactor(table_path=O/'flux_table_control_experiment_snapshot.json')
report={'refined_table_sha256':hashlib.sha256(ref.path.read_bytes()).hexdigest(),'experiment_table_sha256':hashlib.sha256(old.path.read_bytes()).hexdigest(),'selected_controls':{},'direct_flux_checks':[],'mu_sensitivity':[]}
for label,u in [('best',r['best']['control']),('neural',r['neural']['optimisation']['control'])]:
 report['selected_controls'][label]={'refined_tight':ref.simulate(u,1e-9,1e-11),'refined_tighter':ref.simulate(u,1e-10,1e-12),'experiment_table':old.simulate(u,1e-9,1e-11),'boundary_ramp_1e_4':Reactor(boundary_ramp=1e-4).simulate(u,1e-9,1e-11)}
tr=ref.simulate(r['best']['control'],1e-9,1e-11,dense=True)
np.savetxt(O/'best_control_refined_trajectory.csv',np.c_[tr['time'],tr['state']],delimiter=',',header='time_h,V_L,X_SC_gDW_L,X_KP_gDW_L,S_g_L',comments='')
for i,name in enumerate(['SC','KP']):
 model=FluxModel(name,1e-6)
 for S in np.unique(tr['state'][np.linspace(0,len(tr['time'])-1,12).astype(int),3]):
  d=solve(model,S,tol=1e-11)[-1];pred=ref.curves[i](S);report['direct_flux_checks'].append({'species':name,'S':float(S),**d,'growth_interpolation_error':float(pred[0]-d['growth']),'uptake_interpolation_error':float(pred[1]-d['glucose_flux'])})
 for S in [.2,1,20]:
  for mu in [1e-5,1e-6,1e-7]:
   d=solve(model,S,mu=mu,tol=1e-11)[-1];report['mu_sensitivity'].append({'species':name,'S':S,'mu':mu,**d})
 print('flux check complete',name,flush=True)
(O/'control_validation.json').write_text(json.dumps(report,indent=2))
# Independent trajectory integration: solve the genome-scale KKT systems at every RHS evaluation.
models=[FluxModel(n,1e-6) for n in ['SC','KP']]
for label,u in [('best',r['best']['control']),('neural',r['neural']['optimisation']['control'])]:
 warm=[None,None];calls=[0];y=np.array([.5,.001,.001,20.]);start=time.perf_counter();minS=20
 def rhs(t,y):
  V,x1,x2,S=y;a=[]
  for i,model in enumerate(models):
   try:out=solve(model,max(S,0),tol=1e-10,warm=warm[i])
   except RuntimeError:out=solve(model,max(S,0),tol=1e-10)
   warm[i]=out[:4];d=out[-1];a.append(np.array([d['growth'],d['glucose_flux']])*min(1,max(S,0)/1e-5))
  calls[0]+=1
  return [feed,(a[0][0]-feed/V)*x1,(a[1][0]-feed/V)*x2,feed/V*(100-S)+.18*(a[0][1]*x1+a[1][1]*x2)]
 for j,feed in enumerate(u):
  sol=solve_ivp(rhs,(2*j,2*j+2),y,method='DOP853',rtol=2e-7,atol=2e-9,max_step=.5)
  if not sol.success:raise RuntimeError(sol.message)
  y=sol.y[:,-1];minS=min(minS,float(min(sol.y[3])));print('direct',label,j,'rhs',calls[0],flush=True)
 report['selected_controls'][label]['direct_KKT_integration']={'terminal':y.tolist(),'objective':float((1-y[1]/y[2])**2),'minimum_glucose':minS,'seconds':time.perf_counter()-start,'rhs_evaluations':calls[0],'rtol':2e-7,'atol':2e-9,'KKT_tolerance':1e-10}
 (O/'control_validation.json').write_text(json.dumps(report,indent=2))
print('validation complete',flush=True)
