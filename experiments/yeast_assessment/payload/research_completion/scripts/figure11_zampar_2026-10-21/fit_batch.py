"""Reproducible series-withheld dFBA comparison with Zampar et al. (2013).
Exact uptake/respiration LP fit; finite-barrier verification is a separate script.
No source observation is transformed except a stated OD observation operator.
"""
from pathlib import Path
import sys,time,csv,json,hashlib,platform
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'research_completion/runs/figure11_zampar_2026-10-21';OUT.mkdir(exist_ok=True)
sys.path[:0]=[str(ROOT/'research_completion/scripts'),str(OUT/'_deps')]
from flux_model import FluxModel
import highspy,scipy
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.io import loadmat
SOURCE=ROOT/'research_completion/runs/figure11_source_review_2026-10-21/candidates/zampar2013/zampar2013_observations.csv'
ALL=list(csv.DictReader(SOURCE.open()));VARIABLES=['biomass_OD600','glucose','ethanol'];T0=-6.8;TF=8.0583333335;OD_FACTOR=.486
# Initial glucose/ethanol: first Exp1 observation; initial OD: log interpolation
# between two bracketing Biomass_Exp1 samples. No cross-workbook culture pairing
# is assumed: this defines a population-level representative initial condition.
OD0=float(np.exp(np.interp(T0,[-7.5,-6.5],np.log([.16,.2245]))));Y0=np.array([OD_FACTOR*OD0,4.84384260862773,.22730440275683])
ROWS=[r for r in ALL if float(r['time_relative_to_glucose_exhaustion_h'])>=T0-1e-8]
for r in ROWS:r['split']='calibration' if r['replicate_label_as_published'].replace(' ','')=='Exp1' else 'withheld_series'
class BatchModel(FluxModel):
 def __init__(self,epsilon=0):
  super().__init__('SC',epsilon);self.e=2706;self.et=2707;self.o=2892
  self.h=highspy.Highs();self.h.setOptionValue('output_flag',False);self.h.setOptionValue('threads',1);self.h.setOptionValue('primal_feasibility_tolerance',1e-9);self.h.setOptionValue('dual_feasibility_tolerance',1e-9)
  a=self.A.tocsc();p=highspy.HighsLp();p.num_col_=a.shape[1];p.num_row_=a.shape[0];p.col_cost_=-self.c;p.col_lower_=self.l;p.col_upper_=self.u;p.row_lower_=self.b;p.row_upper_=self.b;p.a_matrix_.format_=highspy.MatrixFormat.kColwise;p.a_matrix_.start_=a.indptr;p.a_matrix_.index_=a.indices;p.a_matrix_.value_=a.data;self.h.passModel(p)
  self.calls=0;self.max_residual=0.;self.max_bound_violation=0.
 def bounds_state(self,y,p):
  qg,qo,qe,matp=p;kg=.01;g=max(float(y[1]),0.);e=max(float(y[2]),0.);l=self.l.copy();u=self.u.copy()
  l[self.g]=-qg*g/(kg+g)
  eth=qe*e/(.1+e)/(1+(g/.05)**4)
  l[self.e]=l[self.et]=-eth;l[self.o]=-qo-self.epsilon;l[203]=matp-self.epsilon
  return l,u
 def flux(self,y,p):
  l,u=self.bounds_state(y,p)
  inds=np.array([self.g,self.e,self.et,self.o,203],np.int32);self.h.changeColsBounds(5,inds,l[inds],u[inds]);self.h.run();status=self.h.getModelStatus()
  if status!=highspy.HighsModelStatus.kOptimal:raise RuntimeError(str(status)+" state="+str(y)+" params="+str(p))
  x=np.array(self.h.getSolution().col_value);self.calls+=1
  self.max_residual=max(self.max_residual,float(np.max(abs(self.A@x-self.b))));self.max_bound_violation=max(self.max_bound_violation,float(max(np.max(l-x),np.max(x-u),0.)))
  return x
 def rhs(self,t,y,p):
  x=self.flux(y,p);return y[0]*np.array([x[self.bio],.18*x[self.g],.046*x[self.e]])
 def simulate(self,p,rtol=2e-6,atol=2e-8):
  return solve_ivp(lambda t,y:self.rhs(t,y,p),(T0,TF),Y0,rtol=rtol,atol=atol,dense_output=True,max_step=.25)

def summary_metrics(rows,pred):
 result={}
 for split in ['calibration','withheld_series']:
  result[split]={}
  for j,v in enumerate(VARIABLES):
   ii=[i for i,r in enumerate(rows) if r['split']==split and r['variable']==v];obs=np.array([float(rows[i]['observed_value_as_published']) for i in ii]);res=pred[ii]-obs
   result[split][v]={'n':len(ii),'rmse':float(np.sqrt(np.mean(res**2))),'mae':float(np.mean(abs(res))),'range_normalised_rmse_percent':float(100*np.sqrt(np.mean(res**2))/(np.max(obs)-np.min(obs))),'mean_bias':float(res.mean()),'observed_range':[float(obs.min()),float(obs.max())]}
 return result

def run():
 model=BatchModel();train=[r for r in ROWS if r['split']=='calibration'];times=np.array([float(r['time_relative_to_glucose_exhaustion_h']) for r in train]);inds=np.array([VARIABLES.index(r['variable']) for r in train]);obs=np.array([float(r['observed_value_as_published']) for r in train]);scales=np.array([2.,5.,2.])[inds]
 # Equal contribution for each measured variable; fixed scale is not a fitted uncertainty.
 counts=np.array([sum(inds==j) for j in inds]);w=1/scales/np.sqrt(counts)
 ncall=0;start=time.perf_counter()
 def residual(logp):
  nonlocal ncall
  p=np.array([np.exp(logp[0]),np.exp(logp[1]),10.,np.exp(logp[2])])
  try:sol=model.simulate(p)
  except RuntimeError:return np.full(len(obs),10.)
  vals=sol.sol(times);vals[0]/=OD_FACTOR;res=(vals[inds,np.arange(len(times))]-obs)*w;ncall+=1
  if ncall%10==0:print('eval',ncall,'params',p,'norm',np.linalg.norm(res),'elapsed',time.perf_counter()-start,flush=True)
  return res
 fit=least_squares(residual,np.log([11.,3.8,5.]),bounds=(np.log([3.,.2,.001]),np.log([30.,15.,20.])),diff_step=2e-3,xtol=2e-6,ftol=2e-6,gtol=2e-6,max_nfev=80)
 p=np.array([np.exp(fit.x[0]),np.exp(fit.x[1]),10.,np.exp(fit.x[2])]);sol=model.simulate(p,rtol=2e-8,atol=2e-10);tt=np.linspace(T0,TF,1001);yy=sol.sol(tt);fluxes=np.array([model.flux(y,p)[[model.bio,model.g,model.e,model.o]] for y in yy.T]);pred=[]
 for r in ROWS:
  j=VARIABLES.index(r['variable']);v=float(sol.sol(float(r['time_relative_to_glucose_exhaustion_h']))[j]);pred.append(v/OD_FACTOR if j==0 else v)
 with (OUT/'observations_predictions.csv').open('w',newline='') as f:
  fields=list(ROWS[0])+['prediction','residual'];writer=csv.DictWriter(f,fields);writer.writeheader()
  for r,v in zip(ROWS,pred):writer.writerow(r|{'prediction':v,'residual':v-float(r['observed_value_as_published'])})
 np.savetxt(OUT/'trajectory_lp.csv',np.c_[tt,yy.T,yy[0]/OD_FACTOR,fluxes],delimiter=',',header='t_h,X_gDW_L,G_g_L,E_g_L,OD600,growth_h-1,v_glucose_mmol_gDW_h,v_ethanol_mmol_gDW_h,v_oxygen_mmol_gDW_h',comments='')
 rawmeta=ROOT/"recovered/local/Downloads/One drive/research_MATLAB/Ta's work/Problem_3/SCerevisiae_Model.mat";raw=loadmat(rawmeta,simplify_cells=True)['model'];from scipy.sparse import csc_matrix
 source=loadmat(ROOT/'new_model copy/13_Sc_single_simu_compared_with_ex/Batch/SCerevisiae_Data.mat',simplify_cells=True)['SCerevisiae_Data'];assert (csc_matrix(raw['S'])-csc_matrix(source['A'][:-1,:])).nnz==0
 metrics=summary_metrics(ROWS,np.array(pred));result={'date':'2026-10-21','parameters':dict(zip(['qGmax_mmol_gDW_h','qOmax_mmol_gDW_h','qEmax_mmol_gDW_h','ATP_maintenance_mmol_gDW_h'],p.tolist())),'fixed_parameters':{'qEmax_mmol_gDW_h':10.,'KG_g_L':.01,'KE_g_L':.1,'glucose_repression_scale_g_L':.05,'glucose_repression_exponent':4,'OD_gDW_L_per_OD':OD_FACTOR,'MWG_g_mmol':.18,'MWE_g_mmol':.046,'growth_scaling':1},'initial':{'time_h':T0,'X_gDW_L':Y0[0],'G_g_L':Y0[1],'E_g_L':Y0[2],'OD600':OD0,'biomass_rule':'log-linear interpolation between Biomass_Exp1 at -7.5 and -6.5h; not a culture pairing claim'},'calibration':{'selection':'All Exp1-labelled series samples with t >= -6.8h, label is workbook-specific','withheld':'All other workbook series with t >= -6.8h; no claim of cross-variable replicate matching or statistically independent biological validation','residual_scales':[2.,5.,2.],'objective':'sum over variables of mean squared residual divided by fixed scale squared','parameters_fitted':3,'evaluations':ncall,'optimiser_success':bool(fit.success),'optimiser_message':fit.message,'cost':float(2*fit.cost)},'metrics':metrics,'numerics':{'method':'HiGHS simplex with reusable basis + adaptive RK45','highs_version':model.h.version(),'LP_tolerance':1e-9,'final_rtol':2e-8,'final_atol':2e-10,'max_step_h':.25,'max_equality_residual':model.max_residual,'max_bound_violation':model.max_bound_violation,'lp_calls':model.calls,'elapsed_seconds':time.perf_counter()-start,'epsilon':0},'model_identity':{'metadata_source':str(rawmeta.relative_to(ROOT)),'S_exactly_equal':True,'growth_id':'r_2111','biomass_assembly_id':'r_4041','glucose_id':'r_1714','ethanol_exchange_id':'r_1761','ethanol_transport_id':'r_1762','oxygen_id':'r_1992','biomass_normalization':'biomass-normalized growth flux used directly; kappa=1; no .56 multiplier'},'source_data':{'doi':'10.1038/msb.2013.11','path':str(SOURCE.relative_to(ROOT)),'sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'eligible_observations':len(ROWS),'all_observations':len(ALL),'excluded':'Before representative initialization at -6.8h, not omitted based on residual'},'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()},'terminal':dict(zip(['X_gDW_L','G_g_L','E_g_L'],yy[:,-1].tolist()))}
 (OUT/'fit_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':run()
