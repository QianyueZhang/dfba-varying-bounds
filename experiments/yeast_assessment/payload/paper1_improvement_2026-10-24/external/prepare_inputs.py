"""Freeze primary-source observations and input protocol before any LP solves."""
from pathlib import Path
import hashlib, json, re, shutil
from html import unescape
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'paper1_improvement_2026-10-24/audit/sources/wiebe_2008_article.html'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    if (HERE / 'protocol.json').exists():
        raise RuntimeError('Frozen protocol already exists; do not overwrite')
    (HERE / 'inputs').mkdir(exist_ok=True)
    text = SOURCE.read_text()
    tables = [t for t in re.findall(r'<table\b[^>]*>.*?</table>', text, re.S)
              if 'Specific OUR' in t and '37.1' in t]
    decoded = []
    for table in tables:
        rows = []
        for row in re.findall(r'<tr\b[^>]*>(.*?)</tr>', table, re.S):
            cells = [unescape(re.sub(r'<[^>]+>', '', c)).strip()
                     for c in re.findall(r'<td\b[^>]*>(.*?)</td>', row, re.S)]
            rows.append(cells)
        decoded.append(rows)
    assert len(decoded) == 4 and all(t == decoded[0] for t in decoded)
    raw = decoded[0]
    assert len(raw) == 6 and len(raw[0]) == 9
    columns = ['o2_inlet_percent','o2_water_solubility_uM','biomass_gDW_L',
               'carbon_yield_Cmol_Cmol','oxygen_mmol_gDW_h','co2_mmol_gDW_h',
               'glucose_Cmmol_gDW_h','ethanol_Cmmol_gDW_h','glycerol_Cmmol_gDW_h']
    def measured(s):
        if s == 'ND':
            return {'reported': s, 'mean': None, 'sem': None, 'censored': True}
        parts = s.split('±')
        return {'reported': s, 'mean': float(parts[0]),
                'sem': float(parts[1]) if len(parts) == 2 else None,
                'censored': False}
    observations = []
    for i, row in enumerate(raw[1:], 1):
        values = {col: measured(val) | {'source_table': 1, 'source_body_row': i,
                   'source_column': j + 1} for j, (col, val) in enumerate(zip(columns, row))}
        converted = {}
        for name, carbon in [('glucose',6),('ethanol',2),('glycerol',3)]:
            v = values[name + '_Cmmol_gDW_h']
            converted[name + '_mmol_gDW_h'] = {
                'mean': v['mean']/carbon if v['mean'] is not None else None,
                'sem': v['sem']/carbon if v['sem'] is not None else None,
                'conversion': f'original Cmmol divided by {carbon}',
                'source_column': v['source_column']}
        observations.append({'condition_id': f'o2_{row[0]}percent', 'values': values,
                             'converted': converted, 'growth_reference_h-1': .1,
                             'growth_reference_kind': 'reported nominal chemostat dilution rate',
                             'anaerobic_applicability_boundary': float(row[0]) == 0})
    data = {'doi':'10.1111/j.1567-1364.2007.00234.x',
            'citation':'Wiebe et al., FEMS Yeast Research 8 (2008), 140–154, Table 1',
            'source_url':'https://academic.oup.com/femsyr/article/8/1/140/562580',
            'download_url':'https://oup.silverchair-cdn.com/article-minimal/562580',
            'source_sha256':sha(SOURCE), 'html_duplicate_tables_identical':len(decoded),
            'source_header':raw[0], 'observations':observations,
            'strain':'Saccharomyces cerevisiae CEN.PK113-1A',
            'conditions':{'nominal_dilution_h-1':.1, 'reported_dilution_variation_h-1':.02,
                          'temperature_C':30,'pH':5,'gas_vvm':1.5,
                          'feed_glucose_g_L':10,'ergosterol_mg_L':10,'Tween80_mg_L':420},
            'uncertainty_notes':[
                'Published means and compound SEM; no replicate-level raw data supplied here.',
                'Sample counts differ across variables and combine repeated measurements from two or four cultivations.',
                'Reported zero without SEM stays zero with unknown SEM; it is not a zero-variance estimate.',
                'ND glycerol stays censored, not a measured zero.',
                'Dilution variation ±0.02 is not used as a biological confidence interval.']}
    (HERE/'observations.json').write_text(json.dumps(data,indent=2))
    copies = {
      'SC_Model.mat': ROOT/'14_control_pso/09_Sc_Kp/SC_Model.mat',
      'SCerevisiae_Data.mat': ROOT/'14_control_pso/09_Sc_Kp/SCerevisiae_Data.mat',
      'SCerevisiae_Model.mat': ROOT/"recovered/local/Downloads/One drive/research_MATLAB/Ta's work/Problem_3/SCerevisiae_Model.mat",
      'zampar_fit_summary.json': ROOT/'research_completion/runs/figure11_zampar_2026-10-21/fit_summary.json'}
    for name, path in copies.items():
        shutil.copyfile(path,HERE/'inputs'/name)
    protocol = {
      'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
      'purpose':'External steady-state resource-constraint transfer challenge; not dynamic kinetic validation',
      'selection':'All five Table 1 oxygen conditions, including anaerobic applicability boundary',
      'fitting':'None; no experimental outputs used to select or calibrate model parameters',
      'network':'Same archived processed yeast A,b,c and reconstructed bounds used by frozen Zampar model',
      'epsilon':0, 'barrier_parameter':0, 'solution_route':'LP limit',
      'objective':'Maximize archived biomass objective r_2111',
      'maintenance_mmol_gDW_h':5.565032869183548,
      'glucose_oxygen_roles':'Positive measured uptake means are external input constraints, not outputs or true uptake capacity measurements',
      'resource_semantics':{
        'capacity':'-qG <= vG <= 0; -qO <= vO <= 0',
        'prescribed':'vG=-qG and vO=-qO'},
      'ethanol':'Keep original exchange lower bound zero and upper bound1000; allow secretion. No measured ethanol or growth imposed. Internal ethanol transport remains original.',
      'media':{
        'archived_medium':'Original frozen uptake permissions and finite inherited bounds',
        'unlimited_lipid_availability':'Open existing r_1757 ergosterol and r_2189 oleate lower bounds to-1000; no new reactions. Inherited1000 is nonlimiting sensitivity assumption, not measured flux. Oleate is an imperfect Tween80 proxy. Additional lipid carbon/energy can confound growth; feasibility/confounding diagnostic only.'},
      'input_sensitivity':'Nominal plus all unique corners of qG mean±1 SEM and qO mean±1 SEM, clipped at0. Missing SEM gives unchanged input; no joint confidence interpretation.',
      'expected_primary_lp_count':92,
      'solver':{'method':'scipy.optimize.linprog(method=highs-ds)',
                'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9},
      'ethanol_fva':'Min/max ethanol on cTv >= optimal_growth-1e-9; not conditioned on measured growth. Report failed ranges and never use arbitrary LP vertex as unique prediction.',
      'biomass_fva_slack_h-1':1e-9,
      'numerical_acceptance':{'max_equality_residual':1e-7,'max_bound_violation':1e-7},
      'reported_outputs':['solver status','growth','growth error vs D=0.1','realized glucose/oxygen uptake','one optimal ethanol flux','near-optimal ethanol FVA interval','ethanol mean±SEM interval overlap (descriptive)','lipid uptake','numerical residuals'],
      'limits':['Cross-strain FY4-to-CEN.PK113-1A and medium transfer',
                'SEM corners are descriptive input perturbations, not confidence bands',
                'No residual glucose or measured capacity data, so qmax/K kinetics are not tested',
                'No biological replicate validation, no full dynamic trajectory validation',
                'No post-outcome constraint repair or model selection'],
      'input_sha256':{str((HERE/'inputs'/name).relative_to(HERE)):sha(HERE/'inputs'/name) for name in copies},
      'source_paths':{name:str(p.relative_to(ROOT)) for name,p in copies.items()},
      'observations_sha256':sha(HERE/'observations.json'),
      'run_script_sha256':sha(HERE/'run_challenge.py'),
      'prepare_script_sha256':sha(Path(__file__))}
    (HERE/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (HERE/'PROTOCOL_SHA256.txt').write_text(sha(HERE/'protocol.json')+'  protocol.json\n')
    print(json.dumps({'protocol_sha256':sha(HERE/'protocol.json'), 'observations':len(observations),
                      'expected_primary_lp_count':92},indent=2))

if __name__ == '__main__':
    main()
