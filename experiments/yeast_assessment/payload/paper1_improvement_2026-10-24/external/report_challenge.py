"""Descriptive reporting only; leaves frozen numerical inputs/results intact."""
from pathlib import Path
import hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    result=json.loads((HERE/'results.json').read_text());rows=result['records']
    summary={'protocol_sha256':result['protocol_sha256'],'results_sha256':sha(HERE/'results.json'),
             'primary_lp_count':len(rows),'total_lp_calls':result['total_lp_calls'],
             'elapsed_seconds':result['elapsed_seconds'],'groups':{}}
    for medium in ['archived_medium','unlimited_lipid_availability']:
      for sem in ['capacity','prescribed']:
        group=[r for r in rows if r['medium']==medium and r['resource_semantics']==sem]
        nominal=[r for r in group if r['case_kind']=='nominal']
        valid=[r for r in nominal if r.get('numerically_valid',False)]
        key=medium+'__'+sem
        entry={'nominal_count':len(nominal),'nominal_optimal':sum(r['optimal'] for r in nominal),
               'nominal_numerically_valid':len(valid),'all_input_cases':len(group),
               'all_optimal':sum(r['optimal'] for r in group),
               'all_numerically_valid':sum(r.get('numerically_valid',False) for r in group),
               'solver_status_counts':{str(s):sum(r['solver_status']==s for r in group) for s in sorted(set(r['solver_status'] for r in group))},
               'conditions':[]}
        if valid:
            entry['growth_mae_nominal_h-1']=float(np.mean([abs(r['growth_error_h-1']) for r in valid]))
            entry['growth_rmse_nominal_h-1']=float(np.sqrt(np.mean([r['growth_error_h-1']**2 for r in valid])))
        for nom in nominal:
            match=[r for r in group if r['condition_id']==nom['condition_id']]
            ok=[r for r in match if r.get('numerically_valid',False)]
            fvaok=[r for r in ok if 'ethanol_fva_min_mmol_gDW_h' in r]
            rr={k:v for k,v in nom.items() if k!='ethanol_fva'}
            rr['all_input_scenarios']=len(match);rr['valid_input_scenarios']=len(ok)
            if ok:rr['growth_input_sensitivity_envelope_h-1']=[min(r['growth_h-1'] for r in ok),max(r['growth_h-1'] for r in ok)]
            if fvaok:rr['ethanol_input_and_fva_envelope_mmol_gDW_h']=[min(r['ethanol_fva_min_mmol_gDW_h'] for r in fvaok),max(r['ethanol_fva_max_mmol_gDW_h'] for r in fvaok)]
            entry['conditions'].append(rr)
        summary['groups'][key]=entry
    valid_all=[r for r in rows if r.get('numerically_valid',False)]
    summary['max_primary_equality_residual']=max((r['max_equality_residual'] for r in valid_all),default=None)
    summary['max_primary_bound_violation']=max((r['max_bound_violation'] for r in valid_all),default=None)
    fvas=[v for r in rows for v in r.get('ethanol_fva',{}).values()]
    summary['fva_lp_count']=len(fvas);summary['fva_optimal_count']=sum(v['success'] for v in fvas)
    summary['fva_valid_count']=sum(v.get('numerically_valid',False) for v in fvas)
    summary['limits']=[
      'No fit, biological replicate inference, or dynamic kinetic validation.',
      'Measured realized qG/qO used as inputs; these are not measured uptake capacities.',
      'Growth reference is nominal chemostat D0.10, not a new fitted growth parameter.',
      'Input SEM corners and reported output SEM comparisons are descriptive, not statistical acceptance tests.',
      'Unlimited lipid sensitivity changes potential carbon/energy and is not quantitative medium validation.',
      'Infeasible constraints and unsuccessful solver outcomes remain in the complete denominator.']
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Independent steady-state constraint-transfer challenge','',
      'This experiment uses the original yeast network, archived growth objective and frozen ATP-maintenance lower bound. It supplies measured glucose/oxygen fluxes from Wiebe et al. (2008), Table 1, as input constraints and compares predicted growth and ethanol secretion with the published outputs. There is no parameter refitting.','',
      f"Protocol SHA-256: `{result['protocol_sha256']}`.",
      f"Completed primary LPs: {len(rows)}; total LP calls including ethanol variability: {result['total_lp_calls']}; elapsed computation: {result['elapsed_seconds']:.1f} s.",'']
    for key,grp in summary['groups'].items():
        lines.extend(['## '+key.replace('__',' / '),'',
          f"Nominal optima: {grp['nominal_optimal']}/{grp['nominal_count']}; optima across nominal and SEM-corner cases: {grp['all_optimal']}/{grp['all_input_cases']}. Numerical-validity checks pass for {grp['all_numerically_valid']} cases.",''])
        for r in grp['conditions']:
            label=f"{r['o2_inlet_percent']:g}% inlet oxygen"
            if not r['optimal']:
                lines.append(f"- {label}: {r['solver_message']} All input scenarios retained ({r['valid_input_scenarios']}/{r['all_input_scenarios']} numerically valid optima).")
                continue
            fva=('unavailable' if 'ethanol_fva_min_mmol_gDW_h' not in r else f"[{r['ethanol_fva_min_mmol_gDW_h']:.6g}, {r['ethanol_fva_max_mmol_gDW_h']:.6g}]")
            lines.append(f"- {label}: growth {r['growth_h-1']:.6g} h⁻¹ (reference0.10), ethanol near-optimal range {fva} mmol gDW⁻¹ h⁻¹ (reference{r['reference_ethanol_mmol_gDW_h']:.6g}). Valid input scenarios {r['valid_input_scenarios']}/{r['all_input_scenarios']}.")
        lines.append('')
    lines.extend(['## Interpretation limits','']+['- '+s for s in summary['limits']]+['',
      '## Files and reproduction','',
      '`observations.json` records source Table1/body-row/column and all unit conversions. `protocol.json` and its checksum were frozen before solving. `results.json` includes every nominal/corner case, statuses and FVA diagnostics; `results.csv` provides a flat summary. `flux_vectors.npz` retains every successful primary and FVA vector. `summary.json` is descriptive aggregation.','',
      'In a fresh copied directory, remove only the generated results files and execute `python run_challenge.py` followed by `python report_challenge.py` with NumPy/SciPy installed. Keep protocol, observations, inputs and numerical script unchanged; their hashes are verified. `prepare_inputs.py` documents original provenance extraction and is not required to rerun the already frozen study.','',
      'Source: [Wiebe et al. (2008), Table1](https://doi.org/10.1111/j.1567-1364.2007.00234.x). Source-download metadata and private full-text audit are retained separately. The published means are from CEN.PK113-1A, whereas maintenance was fit to Zampar FY4.',''])
    (HERE/'README.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in summary.items() if k!='groups'},indent=2))

if __name__=='__main__':main()
