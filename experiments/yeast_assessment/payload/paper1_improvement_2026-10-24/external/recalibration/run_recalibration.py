"""Retrospective maintenance-only calibration, one oxygen condition only."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,sys,time
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix
from scipy.optimize import linprog,brentq

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE))
from cached_scipy_driver import cached_wrapper
import scipy.optimize._linprog_highs as lm

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    if (HERE/'results.json').exists():raise RuntimeError('Preserve saved outcomes')
    protocol=json.loads((HERE/'protocol.json').read_text())
    assert protocol['script_sha256']==sha(Path(__file__))
    for f,h in protocol['input_sha256'].items():assert sha(BASE/f)==h
    lm._highs_wrapper=cached_wrapper()[0]
    obs=json.loads((BASE/'observations.json').read_text())['observations']
    raw=loadmat(BASE/'inputs/SCerevisiae_Data.mat',simplify_cells=True)['SCerevisiae_Data']
    proc=loadmat(BASE/'inputs/SC_Model.mat',simplify_cells=True)['Struct']
    A=csc_matrix(proc['A']);b=np.asarray(proc['b'],float).ravel();c=np.asarray(proc['c'],float).ravel()
    lb=np.asarray(raw['lb'],float).ravel().copy();ub=np.asarray(raw['ub'],float).ravel().copy()
    lb[np.isneginf(lb)]=-1000;ub[np.isposinf(ub)]=1000
    options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}
    calls=[];vectors={};t0=time.perf_counter()
    def solve(matp,condition,semantics='capacity',objective=None,face=None,tag=''):
        lo=lb.copy();hi=ub.copy();lo[203]=matp
        g=condition['converted']['glucose_mmol_gDW_h']['mean'];o=condition['values']['oxygen_mmol_gDW_h']['mean']
        lo[2664]=-g;lo[2892]=-o;hi[2664]=0 if semantics=='capacity' else -g;hi[2892]=0 if semantics=='capacity' else -o
        kw={} if face is None else {'A_ub':csc_matrix(-c.reshape(1,-1)),'b_ub':[-face]}
        q=linprog(-c if objective is None else objective,A_eq=A,b_eq=b,bounds=np.c_[lo,hi],method='highs-ds',options=options,**kw)
        rr={'sequence':len(calls),'tag':tag,'condition_id':condition['condition_id'],'semantics':semantics,
            'maintenance_mmol_gDW_h':float(matp),'status':int(q.status),'success':bool(q.success),'message':q.message}
        if q.success:
            rr.update({'growth_h-1':float(c@q.x),'ethanol_mmol_gDW_h':float(q.x[2706]),
                       'max_equality_residual':float(np.max(abs(A@q.x-b))),
                       'max_bound_violation':float(max(0,np.max(lo-q.x),np.max(q.x-hi))),
                       'growth_face_violation':0 if face is None else float(max(0,face-c@q.x))})
            vectors[f'call_{len(calls):03d}']=q.x
        calls.append(rr);return rr
    calibration=obs[0];assert calibration['values']['o2_inlet_percent']['mean']==20.9
    lower,upper=protocol['maintenance_bounds_mmol_gDW_h'];target=.1
    left=solve(lower,calibration,tag='calibration_lower_bound')
    right=solve(upper,calibration,tag='calibration_upper_bound')
    assert left['success'], 'Lower-bound feasibility failed; do not silently repair'
    calibration_ok=False;chosen=None;reason=None
    if left['growth_h-1']<target-protocol['growth_attainment_tolerance_h-1']:
        chosen=lower;reason='Target growth exceeds maximum over allowed maintenance interval (monotone feasible-set nesting); lower-bound diagnostic retained'
    elif abs(left['growth_h-1']-target)<=protocol['growth_attainment_tolerance_h-1']:
        chosen=lower;calibration_ok=True;reason='Target attained at lower bound'
    elif right['success'] and right['growth_h-1']>target+protocol['growth_attainment_tolerance_h-1']:
        chosen=upper;reason='Target growth below endpoint value over allowed interval; upper-bound diagnostic retained'
    else:
        feasible_upper=upper;feasible_right=right
        if not right['success']:
            good=lower;bad=upper
            for _ in range(50):
                mid=(good+bad)/2;r=solve(mid,calibration,tag='feasibility_bisection')
                if r['success']:
                    good=mid;feasible_right=r
                    if r['growth_h-1']<=target:break
                else:bad=mid
            feasible_upper=good
        if not feasible_right['success'] or feasible_right['growth_h-1']>target:
            chosen=lower;reason='No feasible lower-growth bracket found within frozen50bisections; lower-bound diagnostic retained'
        else:
            def f(v):
                r=solve(v,calibration,tag='calibration_brentq');assert r['success'];return r['growth_h-1']-target
            chosen=float(brentq(f,lower,feasible_upper,xtol=1e-9,maxiter=50))
            check=solve(chosen,calibration,tag='calibration_attainment_check')
            calibration_ok=abs(check['growth_h-1']-target)<=protocol['growth_attainment_tolerance_h-1']
            reason='Target attained by maintenance-only root' if calibration_ok else 'Root returned but frozen target-attainment tolerance not met'
    records=[]
    original=protocol['original_maintenance_mmol_gDW_h']
    for maintenance_label,m in [('original',original),('single_condition_recalibration' if calibration_ok else 'unattainable_target_boundary_diagnostic',chosen)]:
      for semantics in ['capacity','prescribed']:
       for condition in obs:
        r=solve(m,condition,semantics,tag=maintenance_label)
        rec=r.copy();rec['maintenance_label']=maintenance_label;rec['role']='calibration_condition' if condition is calibration else 'retrospective_heldout_condition'
        rec['o2_inlet_percent']=condition['values']['o2_inlet_percent']['mean'];rec['reference_growth_h-1']=.1
        rec['reference_ethanol_mmol_gDW_h']=condition['converted']['ethanol_mmol_gDW_h']['mean']
        if r['success']:
            rec['growth_error_h-1']=r['growth_h-1']-.1
            e=np.zeros_like(c);e[2706]=1
            fv=[]
            for name,obj in [('min',e),('max',-e)]:
                rr=solve(m,condition,semantics,obj,r['growth_h-1']-1e-9,tag=maintenance_label+'_ethanol_'+name)
                fv.append(rr)
                if rr['success']:rec['ethanol_fva_'+name+'_mmol_gDW_h']=rr['ethanol_mmol_gDW_h']
            rec['fva_calls']=[z['sequence'] for z in fv]
        records.append(rec)
    out={'protocol_sha256':sha(HERE/'protocol.json'),'calibration_success':calibration_ok,
         'selected_maintenance_mmol_gDW_h':chosen,'calibration_reason':reason,
         'all_solver_calls':calls,'records':records,'elapsed_seconds':time.perf_counter()-t0,
         'label':'Retrospective single-condition calibration assessment; original external outcomes were already inspected'}
    (HERE/'results.json').write_text(json.dumps(out,indent=2))
    keys=list(dict.fromkeys(k for r in records for k in r if k!='fva_calls'))
    with (HERE/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:v for k,v in r.items() if k!='fva_calls'} for r in records)
    np.savez_compressed(HERE/'flux_vectors.npz',**vectors)
    print(json.dumps({k:v for k,v in out.items() if k not in ['records','all_solver_calls']},indent=2))
    for r in records:
        print(r['maintenance_label'],r['semantics'],r['condition_id'],r['status'],r.get('growth_h-1'),r.get('ethanol_fva_min_mmol_gDW_h'),r.get('ethanol_fva_max_mmol_gDW_h'))

if __name__=='__main__':main()
