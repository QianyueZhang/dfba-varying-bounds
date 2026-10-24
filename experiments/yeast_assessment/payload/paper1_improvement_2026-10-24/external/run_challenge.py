"""Execute the frozen, no-refitting Wiebe2008 resource-constraint challenge."""
from pathlib import Path
import csv,hashlib,itertools,json,platform,sys,time
import numpy as np
import scipy
from scipy.io import loadmat
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

HERE=Path(__file__).resolve().parent

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    if (HERE/'results.json').exists(): raise RuntimeError('Results already exist; preserve frozen outcome')
    protocol=json.loads((HERE/'protocol.json').read_text())
    for name,h in protocol['input_sha256'].items(): assert sha(HERE/name)==h
    assert sha(HERE/'observations.json')==protocol['observations_sha256']
    assert sha(Path(__file__))==protocol['run_script_sha256']
    expected=(HERE/'PROTOCOL_SHA256.txt').read_text().split()[0]
    assert sha(HERE/'protocol.json')==expected
    source=json.loads((HERE/'observations.json').read_text())
    proc=loadmat(HERE/'inputs/SC_Model.mat',simplify_cells=True)['Struct']
    raw=loadmat(HERE/'inputs/SCerevisiae_Data.mat',simplify_cells=True)['SCerevisiae_Data']
    meta=loadmat(HERE/'inputs/SCerevisiae_Model.mat',simplify_cells=True)['model']
    A=csc_matrix(proc['A']);b=np.asarray(proc['b'],float).ravel();c=np.asarray(proc['c'],float).ravel()
    lower=np.asarray(raw['lb'],float).ravel().copy();upper=np.asarray(raw['ub'],float).ravel().copy()
    lower[np.isneginf(lower)]=-1000.;upper[np.isposinf(upper)]=1000.
    lookup={str(name):i for i,name in enumerate(meta['rxns'])}
    ids={'glucose':'r_1714','oxygen':'r_1992','ethanol':'r_1761','growth':'r_2111',
         'maintenance':'r_0227','ergosterol':'r_1757','oleate':'r_2189'}
    idx={name:lookup[rid] for name,rid in ids.items()}
    assert np.count_nonzero(c)==1 and c[idx['growth']]==1
    assert (csc_matrix(meta['S'])-csc_matrix(raw['A'][:-1,:])).nnz==0
    assert lower[idx['ethanol']]==0 and upper[idx['ethanol']]==1000
    for lipid in ['ergosterol','oleate']:
        assert lower[idx[lipid]]==0 and upper[idx[lipid]]==1000
    lower[idx['maintenance']]=protocol['maintenance_mmol_gDW_h']
    options={k:protocol['solver'][k] for k in ['primal_feasibility_tolerance','dual_feasibility_tolerance']}
    vectors={};records=[];start=time.perf_counter();lp_calls=0
    def residuals(v,lo,hi):
        return {'max_equality_residual':float(np.max(np.abs(A@v-b))),
                'max_bound_violation':float(max(0,np.max(lo-v),np.max(v-hi)))}
    def run(obj,lo,hi,face=None):
        nonlocal lp_calls
        lp_calls+=1
        args={} if face is None else {'A_ub':csc_matrix(-c.reshape(1,-1)),'b_ub':[-face]}
        return linprog(obj,A_eq=A,b_eq=b,bounds=np.c_[lo,hi],method='highs-ds',options=options,**args)
    for medium in protocol['media']:
      for semantics in protocol['resource_semantics']:
       for condition in source['observations']:
        qg=condition['converted']['glucose_mmol_gDW_h'];qo=condition['values']['oxygen_mmol_gDW_h'];qe=condition['converted']['ethanol_mmol_gDW_h']
        gvals=sorted(set(max(0,qg['mean']+sign*(qg['sem'] or 0)) for sign in [-1,1]))
        ovals=sorted(set(max(0,qo['mean']+sign*(qo['sem'] or 0)) for sign in [-1,1]))
        scenarios=[('nominal',qg['mean'],qo['mean'])]+[(f'sem_corner_{j}',g,o) for j,(g,o) in enumerate(itertools.product(gvals,ovals))]
        for case,g,o in scenarios:
          lo=lower.copy();hi=upper.copy()
          if medium=='unlimited_lipid_availability':
              lo[[idx['ergosterol'],idx['oleate']]]=-1000
          lo[idx['glucose']]=-g;lo[idx['oxygen']]=-o
          hi[idx['glucose']]=0 if semantics=='capacity' else -g
          hi[idx['oxygen']]=0 if semantics=='capacity' else -o
          key=f'{medium}__{semantics}__{condition["condition_id"]}__{case}'
          rec={'case_id':key,'medium':medium,'resource_semantics':semantics,
               'condition_id':condition['condition_id'],'case_kind':case,
               'o2_inlet_percent':condition['values']['o2_inlet_percent']['mean'],
               'input_glucose_uptake_mmol_gDW_h':g,'input_oxygen_uptake_mmol_gDW_h':o,
               'reference_growth_h-1':.1,'reference_ethanol_mmol_gDW_h':qe['mean'],
               'reference_ethanol_sem_mmol_gDW_h':qe['sem'],
               'anaerobic_applicability_boundary':condition['anaerobic_applicability_boundary']}
          opt=run(-c,lo,hi);rec.update({'solver_status':int(opt.status),'solver_message':opt.message,'optimal':bool(opt.success)})
          if opt.success:
              x=opt.x;mu=float(c@x);res=residuals(x,lo,hi);vectors[key]=x
              rec.update(res);rec['numerically_valid']=all(res[n]<=v for n,v in protocol['numerical_acceptance'].items())
              rec.update({'growth_h-1':mu,'growth_error_h-1':mu-.1,
                          'realized_glucose_uptake_mmol_gDW_h':float(-x[idx['glucose']]),
                          'realized_oxygen_uptake_mmol_gDW_h':float(-x[idx['oxygen']]),
                          'ethanol_one_optimum_mmol_gDW_h':float(x[idx['ethanol']]),
                          'ergosterol_uptake_mmol_gDW_h':float(-x[idx['ergosterol']]),
                          'oleate_uptake_mmol_gDW_h':float(-x[idx['oleate']])})
              fv={};face=mu-protocol['biomass_fva_slack_h-1'];e=np.zeros_like(c);e[idx['ethanol']]=1
              for kind,obj in [('min',e),('max',-e)]:
                  f=run(obj,lo,hi,face)
                  fv[kind]={'status':int(f.status),'message':f.message,'success':bool(f.success)}
                  if f.success:
                      fr=residuals(f.x,lo,hi);gap=max(0,face-float(c@f.x))
                      fv[kind].update({'value':float(f.x[idx['ethanol']]),'growth_h-1':float(c@f.x),
                                       'growth_face_violation':gap,**fr,
                                       'numerically_valid':gap<=1e-7 and all(fr[n]<=v for n,v in protocol['numerical_acceptance'].items())})
                      vectors[key+'__ethanol_'+kind]=f.x
              rec['ethanol_fva']=fv
              if all(fv[k]['success'] and fv[k]['numerically_valid'] for k in ['min','max']):
                  rec['ethanol_fva_min_mmol_gDW_h']=fv['min']['value'];rec['ethanol_fva_max_mmol_gDW_h']=fv['max']['value']
                  # Missing SEM remains a point-comparison at its reported zero, not a CI.
                  elo=max(0,qe['mean']-(qe['sem'] or 0));ehi=qe['mean']+(qe['sem'] or 0)
                  rec['ethanol_reference_interval_distance_mmol_gDW_h']=float(max(0,elo-fv['max']['value'],fv['min']['value']-ehi))
          records.append(rec)
    assert len(records)==protocol['expected_primary_lp_count']
    out={'protocol_sha256':expected,'run_script_sha256':sha(Path(__file__)),
         'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()},
         'model':{'processed_shape':list(A.shape),'indices_zero_based':idx,
                  'ids':ids,'raw_stoichiometry_matches_metadata':True,
                  'processed_b_max_abs':float(np.max(np.abs(b)))},
         'records':records,'primary_lp_count':len(records),'total_lp_calls':lp_calls,
         'elapsed_seconds':time.perf_counter()-start}
    (HERE/'results.json').write_text(json.dumps(out,indent=2))
    columns=list(dict.fromkeys(k for r in records for k in r if k!='ethanol_fva'))
    with (HERE/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        writer.writerows({k:v for k,v in r.items() if k!='ethanol_fva'} for r in records)
    np.savez_compressed(HERE/'flux_vectors.npz',**vectors)
    print(json.dumps({'primary_lp_count':len(records),'total_lp_calls':lp_calls,
                      'optimal':sum(r['optimal'] for r in records),'elapsed_seconds':out['elapsed_seconds']},indent=2))
    for r in records:
        if r['case_kind']=='nominal':
            print(r['medium'],r['resource_semantics'],r['condition_id'],r['solver_status'],r.get('growth_h-1'),r.get('ethanol_fva_min_mmol_gDW_h'),r.get('ethanol_fva_max_mmol_gDW_h'),flush=True)

if __name__=='__main__':
    main()
