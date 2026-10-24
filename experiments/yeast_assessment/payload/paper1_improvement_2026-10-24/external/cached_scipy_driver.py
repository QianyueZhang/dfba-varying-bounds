"""Cache two immutable result arrays in SciPy1.15.1 postprocessing only.

No installed dependency is modified. Solver calls and numerical inputs/options
remain those of frozen run_challenge.py. The original wrapper repeatedly copies
whole pybind vectors inside its per-column loop. This driver caches those same
vectors once after the solver returns. Run --equivalence before --run.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,inspect,json,runpy,time
import numpy as np
from scipy.io import loadmat
from scipy.optimize import linprog
from scipy.sparse import csc_matrix
import scipy.optimize._highspy._highs_wrapper as wrapper_module
import scipy.optimize._linprog_highs as linprog_module

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def cached_wrapper():
    original=inspect.getsource(wrapper_module._highs_wrapper)
    assert original.count('basis = highs.getBasis()')==1
    assert original.count('basis.col_status[ii]')==2
    assert original.count('solution.col_dual[ii]')==2
    modified=original.replace('basis = highs.getBasis()',
        'basis = highs.getBasis()\n    basis_col_status = basis.col_status\n    solution_col_dual = solution.col_dual')
    modified=modified.replace('basis.col_status[ii]','basis_col_status[ii]')
    modified=modified.replace('solution.col_dual[ii]','solution_col_dual[ii]')
    namespace=dict(vars(wrapper_module))
    exec(compile(modified,'<cached_scipy_highs_postprocessing>','exec'),namespace)
    return namespace['_highs_wrapper'],original,modified

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--equivalence',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    fast,original,modified=cached_wrapper()
    if args.equivalence:
        if (HERE/'runtime_equivalence.json').exists():raise RuntimeError('Equivalence record exists')
        protocol=json.loads((HERE/'protocol.json').read_text())
        p=loadmat(HERE/'inputs/SC_Model.mat',simplify_cells=True)['Struct']
        r=loadmat(HERE/'inputs/SCerevisiae_Data.mat',simplify_cells=True)['SCerevisiae_Data']
        A=csc_matrix(p['A']);b=np.asarray(p['b'],float).ravel();c=np.asarray(p['c'],float).ravel()
        lo=np.asarray(r['lb'],float).ravel().copy();hi=np.asarray(r['ub'],float).ravel().copy()
        lo[np.isneginf(lo)]=-1000;hi[np.isposinf(hi)]=1000
        lo[203]=protocol['maintenance_mmol_gDW_h'];lo[2664]=-6.6/6;hi[2664]=0;lo[2892]=-2.7;hi[2892]=0
        kw={'A_eq':A,'b_eq':b,'bounds':np.c_[lo,hi],'method':'highs-ds',
            'options':{'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}}
        t=time.perf_counter();a=linprog(-c,**kw);ta=time.perf_counter()-t
        linprog_module._highs_wrapper=fast
        t=time.perf_counter();z=linprog(-c,**kw);tz=time.perf_counter()-t
        checks={'status_equal':a.status==z.status,'success_equal':a.success==z.success,
                'objective_exactly_equal':a.fun==z.fun,'flux_vectors_bitwise_equal':np.array_equal(a.x,z.x),
                'equality_residuals_bitwise_equal':np.array_equal(A@a.x-b,A@z.x-b),
                'bound_residuals_bitwise_equal':np.array_equal(np.r_[a.x-lo,hi-a.x],np.r_[z.x-lo,hi-z.x]),
                'bound_marginals_bitwise_equal':np.array_equal(np.r_[a.lower.marginals,a.upper.marginals],np.r_[z.lower.marginals,z.upper.marginals])}
        out={'date_utc':datetime.now(timezone.utc).isoformat(),'case':'archived medium, capacity,20.9%O2 nominal',
             'checks':checks,'pass':all(checks.values()),'original_wrapper_seconds':ta,'cached_wrapper_seconds':tz,
             'driver_sha256':sha(Path(__file__)),'original_wrapper_source_sha256':hashlib.sha256(original.encode()).hexdigest(),
             'modified_wrapper_source_sha256':hashlib.sha256(modified.encode()).hexdigest(),
             'protocol_sha256':sha(HERE/'protocol.json')}
        (HERE/'runtime_equivalence.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True)
        assert out['pass']
    elif args.run:
        eq=json.loads((HERE/'runtime_equivalence.json').read_text());assert eq['pass'] and eq['driver_sha256']==sha(Path(__file__))
        assert eq['protocol_sha256']==sha(HERE/'protocol.json')
        linprog_module._highs_wrapper=fast
        runpy.run_path(str(HERE/'run_challenge.py'),run_name='__main__')
    else:parser.error('select --equivalence or --run')

if __name__=='__main__':main()
