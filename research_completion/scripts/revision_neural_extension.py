"""Predefined data-size study and mechanistically corrected neural control.
No original result files are modified. Models selected only on validation data.
"""
from control_experiments import *
import warnings
from sklearn.exceptions import ConvergenceWarning
R=O/'revision_validation';R.mkdir(parents=True,exist_ok=True)
D=np.load(O/'surrogate_dataset.npz');X=D['controls'];z=D['target_log_ratio'];train=D['train'];val=D['validation'];test=D['test']
def metrics(y,p):
 return {'log_ratio_rmse':float(np.sqrt(np.mean((y-p)**2))),'objective_rmse':float(np.sqrt(np.mean((np.expm1(y)**2-np.expm1(p)**2)**2))),'objective_mae':float(np.mean(np.abs(np.expm1(y)**2-np.expm1(p)**2)))}
def main():
 protocol={'training_sizes':[175,350,700],'network_hidden':[15],'activation':'logistic','alpha':1e-4,'max_iter':1500,'tol':1e-7,'initialisation_seeds':[733,734,735],'selection':'Minimum validation log-ratio RMSE; independent audit data never used for selection.','audit_seed':2731,'audit_samples':200,'surrogate_search_seeds':[736,737,738,739,740],'polish_method':'SLSQP on refined mechanistic response table; rtol=1e-9,atol=1e-11,ftol=1e-12,FD step=1e-5,bounds=[0,0.2]^7','source_dataset_sha256':hashlib.sha256((O/'surrogate_dataset.npz').read_bytes()).hexdigest()}
 (R/'neural_protocol.json').write_text(json.dumps(protocol,indent=2))
 ref=Reactor();frozen=Reactor(table_path=O/'flux_table_control_experiment_snapshot.json');models=[];rows=[]
 for size in protocol['training_sizes']:
  candidates=[]
  for seed in protocol['initialisation_seeds']:
   net=make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(15,),activation='logistic',solver='lbfgs',alpha=1e-4,max_iter=1500,tol=1e-7,random_state=seed))
   with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always',ConvergenceWarning);net.fit(X[train[:size]],z[train[:size]])
   score=metrics(z[val],net.predict(X[val]));candidates.append((score['log_ratio_rmse'],seed,net))
   rows.append({'training_size':size,'seed':seed,'validation':score,'training_iterations':net[-1].n_iter_,'convergence_warnings':[str(w.message) for w in caught]})
   print('trained',size,seed,score['log_ratio_rmse'],flush=True)
  _,seed,net=min(candidates,key=lambda q:q[0]);models.append((size,seed,net))
 # Independent additional validation controls are generated after candidate selection.
 audit_X=.2*qmc.LatinHypercube(7,seed=2731).random(200);audit_z=[];audit_ref=[]
 for u in audit_X:
  y=frozen.simulate(u)['terminal'];audit_z.append(np.log(y[1]/y[2]));y=ref.simulate(u,1e-9,1e-11)['terminal'];audit_ref.append(np.log(y[1]/y[2]))
 audit_z=np.array(audit_z);audit_ref=np.array(audit_ref)
 np.savez(R/'neural_fresh_audit.npz',controls=audit_X,target_log_ratio=audit_z,refined_target_log_ratio=audit_ref,seed=2731)
 selected=[]
 for size,seed,net in models:
  pickle.dump(net,open(R/f'neural_training_size_{size}.pkl','wb'))
  selected.append({'training_size':size,'selected_seed':seed,'validation':metrics(z[val],net.predict(X[val])),'existing_test':metrics(z[test],net.predict(X[test])),'fresh_audit':metrics(audit_z,net.predict(audit_X)),'fresh_audit_refined':metrics(audit_ref,net.predict(audit_X))})
 np.savez(R/'neural_audit_predictions.npz',target_log_ratio=audit_z,refined_target_log_ratio=audit_ref,predictions=np.array([net.predict(audit_X) for _,_,net in models]),training_sizes=np.array([size for size,_,_ in models]),selected_seeds=np.array([seed for _,seed,_ in models]))
 np.savetxt(R/'neural_audit_predictions.csv',np.c_[audit_z,*[net.predict(audit_X) for _,_,net in models],audit_ref],delimiter=',',header='true_log_ratio,pred_log_ratio_n175,pred_log_ratio_n350,pred_log_ratio_n700,refined_true_log_ratio',comments='')
 net=models[-1][2];pickle.dump(net,open(R/'neural_selected.pkl','wb'));hybrid=[]
 for seed in protocol['surrogate_search_seeds']:
  e=Evaluator(lambda u:np.expm1(net.predict(np.asarray(u).reshape(1,-1))[0])**2,1000);swarm(e,seed);before=ref.simulate(e.u,1e-9,1e-11);calls=[0];trace=[]
  def fun(u):
   calls[0]+=1;r=ref.simulate(u,1e-9,1e-11);trace.append(r['objective']);return r['objective']
  polish=minimize(fun,e.u,method='SLSQP',bounds=[(0,.2)]*7,options={'maxiter':80,'ftol':1e-12,'eps':1e-5})
  after=ref.simulate(polish.x,1e-10,1e-12)
  hybrid.append({'seed':seed,'surrogate_control':e.u.tolist(),'surrogate_predicted_objective':e.best,'before':before,'polished_control':polish.x.tolist(),'after':after,'mechanistic_calls':calls[0],'status':str(polish.message),'success':bool(polish.success),'trace':trace})
  print('polished',seed,before['objective'],after['objective'],calls[0],flush=True)
 report={'protocol':protocol,'candidates':rows,'selected':selected,'fresh_audit_table_log_ratio_rmse':float(np.sqrt(np.mean((audit_z-audit_ref)**2))),'hybrid':hybrid,'selected_direct_validation_seed':736,'note':'Selection of seed736 for direct KKT validation fixed before examining direct results. Learning-curve audit sets are evaluation-only; full-size seed selected solely by validation.'}
 (R/'neural_extension.json').write_text(json.dumps(report,indent=2))
 fig,ax=plt.subplots(1,2,figsize=(10,4));sizes=np.array(protocol['training_sizes'])
 ax[0].plot(sizes,[r['validation']['log_ratio_rmse'] for r in selected],'o-',label='Validation (150)');ax[0].plot(sizes,[r['fresh_audit']['log_ratio_rmse'] for r in selected],'s-',label='Independent audit (200)');ax[0].set(xlabel='Training trajectories',ylabel='Log-ratio RMSE (dimensionless)');ax[0].legend()
 a=[100*np.sqrt(r['before']['objective']) for r in hybrid];b=[100*np.sqrt(r['after']['objective']) for r in hybrid]
 for i,(v,w) in enumerate(zip(a,b)):ax[1].semilogy([0,1],[v,w],'o-',label=str(hybrid[i]['seed']))
 ax[1].set(xticks=[0,1],xticklabels=['Surrogate proposal','Mechanistic correction'],ylabel='Absolute terminal ratio error (%)');ax[1].legend(title='Search seed',fontsize=8)
 plt.tight_layout();plt.savefig(F/'revision_neural_extension.png',dpi=240);plt.savefig(F/'revision_neural_extension.pdf');plt.close()
 print('neural extension complete',flush=True)
if __name__=='__main__':main()
