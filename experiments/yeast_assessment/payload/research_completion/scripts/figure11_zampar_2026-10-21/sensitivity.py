"""Conditional kinetic and regularization checks; never refit on withheld data."""
from fit_batch import *
class Variant(BatchModel):
 def __init__(self,epsilon=0,kg=.01,ke=.1,repression=.05):super().__init__(epsilon);self.kg=kg;self.ke=ke;self.rep=repression
 def bounds_state(self,y,p):
  qg,qo,qe,matp=p;g=max(y[1],0);e=max(y[2],0);l=self.l.copy();u=self.u.copy();l[self.g]=-qg*g/(self.kg+g);eth=qe*e/(self.ke+e)/(1+(g/self.rep)**4);l[self.e]=l[self.et]=-eth;l[self.o]=-qo-self.epsilon;l[203]=matp-self.epsilon;return l,u

def run():
 p=np.array(list(json.loads((OUT/'fit_summary.json').read_text())['parameters'].values()));t=np.linspace(T0,TF,1001);m=BatchModel();base=m.simulate(p,rtol=2e-9,atol=2e-11);ref=base.sol(t);records=[]
 for label,kwargs,qe in [('epsilon_1e-3',{'epsilon':1e-3},10),('epsilon_1e-4',{'epsilon':1e-4},10),('epsilon_1e-5',{'epsilon':1e-5},10),('epsilon_1e-6',{'epsilon':1e-6},10),('KG_half',{'kg':.005},10),('KG_double',{'kg':.02},10),('KE_half',{'ke':.05},10),('KE_double',{'ke':.2},10),('repression_half',{'repression':.025},10),('repression_double',{'repression':.1},10),('ethanol_capacity_half',{},5),('ethanol_capacity_double',{},20)]:
  mod=Variant(**kwargs);pp=p.copy();pp[2]=qe;sol=mod.simulate(pp,rtol=2e-8,atol=2e-10);diff=sol.sol(t)-ref;records.append({'case':label,'options':kwargs,'ethanol_capacity':qe,'max_abs_state_change':dict(zip(['X_gDW_L','G_g_L','E_g_L'],np.max(abs(diff),axis=1).tolist())),'terminal':sol.y[:,-1].tolist(),'min_state':np.min(sol.sol(t),axis=1).tolist()});print(label,records[-1],flush=True)
 # The alternate source conversion modifies both the initial dry weight and
 # observation operator; all kinetic parameters remain unchanged.
 import fit_batch
 fit_batch.Y0[0]=.485*OD0
 alt=BatchModel().simulate(p,rtol=2e-8,atol=2e-10);fit_batch.Y0[0]=.486*OD0
 altobs=alt.sol(t);altobs[0]/=.485;refobs=ref.copy();refobs[0]/=.486
 report={'method':'One-factor-at-a-time forward sensitivity with calibrated capacities fixed; no withheld-data optimisation','variants':records,'ODE_refinement':{'reference_rtol':2e-9,'reference_atol':2e-11,'base_rtol':2e-8,'base_atol':2e-10,'max_state_difference':dict(zip(['X_gDW_L','G_g_L','E_g_L'],np.max(abs(BatchModel().simulate(p,rtol=2e-8,atol=2e-10).sol(t)-ref),axis=1).tolist()))},'OD_0p485_conversion':{'max_observation_change':dict(zip(VARIABLES,np.max(abs(altobs-refobs),axis=1).tolist())),'note':'Changes initial dry weight and biomass observation operator consistently; no data rescaling'},'parameter_fitting_scope':'All parameters remain from calibration series Exp1. Sensitivity is conditional, not a parameter confidence interval.'}
 (OUT/'sensitivity.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':run()
