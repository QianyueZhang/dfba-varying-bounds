from control_experiments import *
r=Reactor(table_path=O/'flux_table_control_experiment_snapshot.json');out=[]
for seed in [1801,1802,1803]:
 e=Evaluator(lambda u:-r.simulate(u)['terminal'][2],400);start=time.perf_counter();swarm(e,seed);v=r.simulate(e.u,1e-9,1e-11);out.append({'seed':seed,'control':e.u.tolist(),'evaluations':e.n,'seconds':time.perf_counter()-start,'terminal_KP_objective':e.best,'validation':v});print(out[-1],flush=True)
(O/'biomass_objective_results.json').write_text(json.dumps(out,indent=2))
