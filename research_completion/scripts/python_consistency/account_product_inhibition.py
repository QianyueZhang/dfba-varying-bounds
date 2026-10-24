"""Calculate fixed-dual maintenance accounting from saved inhibition fluxes.

No dynamic simulation is performed. Actual endpoint amounts provide cumulative
glucose consumption and biomass formation; saved fluxes provide the remaining
Simpson integrals. The fixed-dual identity is not a globally affine optimal law.
"""
from pathlib import Path
import hashlib, json, sys
import numpy as np
from scipy.integrate import simpson
from scipy.io import loadmat
from product_inhibition_check import Model, OUT, MWG

model = Model('separate', 'S2', inhibition=False)
lp = model.lp((1., 0.))
assert lp.success
w = lp.lower.marginals + lp.upper.marginals
assert np.max(abs(model.c + model.A.T @ lp.eqlin.marginals + w)) < 1e-12
alpha, beta = float(w[27]), float(w[10])
other = w.copy(); other[[27, 10]] = 0.
rxns = loadmat(model.source/'e_coli_core.mat', simplify_cells=True)['e_coli_core']['rxns']
affine = []
for q in np.linspace(.65, 9.1, 31):
    check = model.lp((q/(10-q), 0.))
    assert check.success
    ww = check.lower.marginals + check.upper.marginals
    affine.append(dict(q=float(q), growth=float(-check.fun),
                       dual_difference=float(np.max(abs(ww-w))),
                       intercept=float(-check.fun-alpha*q)))
records = {}
inputs = {}
for variant in ['flow_1p0_refined_balanced', 'flow_1p0_refined_reference_balanced']:
    path = OUT/'separate'/variant/'trajectory.npz'
    inputs[str(path.relative_to(OUT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    z = np.load(path); rawt = z['time_h']; common = np.linspace(0., 10., 401)
    indices = np.array([np.argmin(abs(rawt-t)) for t in common])
    assert np.max(abs(rawt[indices]-common)) < 1e-12
    t = rawt[indices]; y = z['amounts'][indices]; v = z['flux'][indices, 1]
    m = y[:, 1]; weighted = v*m[:, None]
    integ = simpson(weighted, x=t, axis=0)
    atpm = float(integ[10]); rem = float(simpson(-weighted@other, x=t))
    consumed = float(-y[-1, 8]); formed = float(y[-1, 1]-y[0, 1])
    reconstructed = alpha/MWG*consumed-beta*atpm+rem
    records[variant] = {
        'recipient_glucose_consumed_g': consumed,
        'recipient_biomass_formed_gDW': formed,
        'biomass_exposure_gDW_h': float(simpson(m, x=t)),
        'ATPM_turnover_mmol': atpm,
        'glucose_biomass_contribution_gDW': alpha/MWG*consumed,
        'ATPM_biomass_cost_gDW': beta*atpm,
        'other_dual_weighted_biomass_contribution_gDW': rem,
        'identity_reconstructed_biomass_gDW': reconstructed,
        'quadrature_identity_error_gDW': reconstructed-formed,
        'coarsened_ATPM_difference_mmol': float(simpson(weighted[::2, 10], x=t[::2])-atpm),
        'coarsened_other_difference_gDW': float(simpson(-weighted[::2]@other, x=t[::2])-rem),
        'max_pointwise_dual_identity_error': float(np.max(abs(v[:, 12]+v@w))),
        'recipient_exchange_integrals_mmol': {str(r): float(integ[j]) for j, r in enumerate(rxns) if str(r).startswith('EX_')}
    }
a, b = records.values()
delta_keys = ['recipient_glucose_consumed_g', 'recipient_biomass_formed_gDW',
              'biomass_exposure_gDW_h', 'ATPM_turnover_mmol',
              'glucose_biomass_contribution_gDW', 'ATPM_biomass_cost_gDW',
              'other_dual_weighted_biomass_contribution_gDW']
result = {
    'derivation': 'c = -A.T @ LP_equality_marginals - lower_marginals - upper_marginals. For Av=0, vbio=-w.T v; split glucose and ATPM components. Integrate using biomass amount and the actual selected intracellular fluxes.',
    'glucose_growth_coefficient': alpha,
    'glucose_yield_gDW_g': alpha/MWG,
    'ATPM_growth_cost_coefficient': beta,
    'LP_affine_checks': affine, 'records': records,
    'quadrature': 'Composite Simpson at 401 common times; 201-point coarsening reported. No rate or coefficient fitted to obtain the endpoint ordering.',
    'inhibited_minus_reference': {k: a[k]-b[k] for k in delta_keys},
    'interpretation': 'LP growth is piecewise affine over the sampled uptake range, not one exact global affine law. A fixed dual stationarity identity is nonetheless exact for any feasible flux. It decomposes the biomass difference into a negative glucose contribution, a larger positive reduction in ATP-maintenance cost, and a smaller remainder from other fluxes. This is a stoichiometric/dual accounting, not a complete reactor elemental-carbon inventory.',
    'input_sha256': inputs,
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
}
assert max(abs(v['quadrature_identity_error_gDW']) for v in records.values()) < 4e-6
assert max(v['max_pointwise_dual_identity_error'] for v in records.values()) < 1e-12
(OUT/'maintenance_mechanism.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result['inhibited_minus_reference'], indent=2))
