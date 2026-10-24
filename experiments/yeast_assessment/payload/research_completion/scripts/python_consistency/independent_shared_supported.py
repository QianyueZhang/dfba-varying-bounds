"""Three source resource cases under one maintenance-supported feed specification.

All physical parameters except the common inlet rate are unchanged. The rate
1 L/h is selected from a conservative substrate/maintenance bound, not a search
for a preferred outcome. This is not same-input reproduction of source plots.
"""
from pathlib import Path
import sys,json,argparse
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0,str(Path(__file__).resolve().parent))
import independent_shared_audit as a

def support_bound():
 records=[];Y=.5;h=5.;G=.10591730126842593;total_G0=100.;total_B0=10.;Gin=20.
 for key,c in a.CASES.items():
  for m in a.make_models(c):
   lower,upper=m.bounds(1e20);lower[m.g]=-m.vmax
   objective=m.c.copy();objective[m.g]+=Y*a.MW
   lp=linprog(-objective,A_eq=m.A,b_eq=m.b,bounds=list(zip(lower,upper)),method='highs')
   assert lp.success and -lp.fun<0
   records.append({'case':key,'species':m.name,'yield_bound_gDW_g':Y,'max_growth_minus_Y_glucose_mass_uptake_h_inv':float(-lp.fun)})
 # At the higher of the two active-domain thresholds, every organism's
 # uptake magnitude is at most q_high. This upper bound plus the verified
 # yield bound guarantees that the substrate derivative points inward.
 q_high=10*G/(1+G);c=a.MW*q_high
 numerator=c*(total_B0+Y*(total_G0-a.V0*G))
 denominator=Gin-G-c*Y*h*(Gin-G)
 threshold=numerator/denominator
 return {'chosen_common_feed_L_h':1.,'unchanged_source_feed_L_h':.1,'basis':'Bound on inward substrate derivative at the larger active threshold over the full5h shared-resource horizon; includes both species in one mass budget. The10h independent pool bound is weaker.','yield_LP_certificates':records,'shared_threshold_glucose_g_L':G,'maximum_per_biomass_glucose_consumption_g_gDW_h':c,'conservative_feed_lower_bound_L_h':threshold,'shared_parameters':{'B0_total_gDW':total_B0,'initial_total_glucose_g':total_G0,'inlet_glucose_g_L':Gin,'horizon_h':h},'formula':'F >= c [B0+Y(NG0−V0 G*)] / [(Gin−G*) (1−c Y tf)], c=MW vmax_high G*/(Ks+G*)','interpretation':'For any feasible active trajectory, Btotal<=B0+Y(NG0+F Gin t−V(t)G). At G=G*, total metabolic glucose demand<=c Btotal. The chosen feed makes F(Gin−G*) exceed that demand bound through tf. This is an operating specification, not a biological parameter change or a claim of matching source inputs.'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--case',choices=list(a.CASES)+['matched_controls']);args=p.parse_args()
 a.F=1.;a.MU=1e-6;a.OUT=a.OUT/'supported';a.OUT.mkdir(parents=True,exist_ok=True)
 design=support_bound();(a.OUT/'operating_condition_design.json').write_text(json.dumps(design,indent=2)+'\n')
 if args.case=='matched_controls':
  cases={}
  for i in range(2):
   case=dict(a.CASES['shared_competition']);case['vmax']=[case['vmax'][i]];case['species_indices']=[i];case['shared']=False;case['figures']=[]
   case['parameter_basis']='Matched isolated population with the same20g/L initial and inlet glucose,1gDW/L initial biomass,5L volume,1L/h feed, kinetic capacity and intracellular constraints as its counterpart in the shared system. Internal validation control, not an additional thesis case.'
   cases['matched_control_species'+str(i+1)]=case
 else:cases={args.case:a.CASES[args.case]}if args.case else a.CASES
 for key,case in cases.items():
  report=a.run(key,case)
  report['physical_parameters']={'V0_L':a.V0,'F_L_h':a.F,'initial_biomass_gDW_L':[1.]*len(case['vmax']),'glucose_MW_g_mmol':a.MW,'biomass_death_rate_h_inv':0.}
  report['numerical']['reason_for_mu']='The source finite-barrier parameter1e-6 is retained; the substrate-supported operating condition avoids the low-substrate feasibility boundary.'
  report['scope']='Direct-KKT check of an explicitly specified substrate-supported operating condition. Same source models, kinetics, initial states and original duration; common feed changed from0.1to1L/h using the conservative substrate/maintenance budget. No dormancy extension or claim of same-input reproduction.'
  report['operating_condition_design']=design;report['supported_runner_sha256']=a.sha(Path(__file__))
  report['adoption_ready']=False
  (a.OUT/key/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
  assert report['full_horizon_completed']
if __name__=='__main__':main()
