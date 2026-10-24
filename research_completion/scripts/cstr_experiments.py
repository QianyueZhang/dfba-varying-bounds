from flux_model import *
from primal_dual import solve
from scipy.integrate import solve_ivp
import sys
import os
O=ROOT/'research_completion/runs';F=ROOT/'research_completion/figures';os.environ['MPLCONFIGDIR']=str(O/'matplotlib_cache')
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
RAW=ROOT/'data/models/cstr';ARCH=ROOT/'data/models/yeast_alga_cstr'
class CSTRModel(FluxModel):
 def __init__(self,name):
  self.name=name;self.epsilon=1e-6
  d=loadmat(ARCH/f'{name}_Model.mat',simplify_cells=True)['Struct'];self.A=csc_matrix(d['A']);self.b=np.array(d['b'],float).ravel();self.c=np.array(d['c'],float).ravel()
  fn='iND750.mat' if name=='SC' else 'iRC1080.mat';raw=next(v for k,v in loadmat(RAW/fn,simplify_cells=True).items() if not k.startswith('__'));self.l=np.array(raw['lb'],float).ravel()-self.epsilon;self.u=np.array(raw['ub'],float).ravel()+self.epsilon;self.l[np.isneginf(self.l)]=-1000;self.u[np.isposinf(self.u)]=1000
  self.l=np.maximum(self.l,-1000.);self.u=np.minimum(self.u,1000.)
  # Resolve extracellular exchanges by reaction identity, never by an index
  # copied from another reconstruction. Column order is retained in Struct.A.
  ids={str(r):i for i,r in enumerate(raw['rxns'])}
  assert self.A.shape[1]==len(ids)==len(raw['c'])
  self.exchange={k:ids[r] for k,r in {'oxygen':'EX_o2_e','carbon_dioxide':'EX_co2_e','ethanol':'EX_etoh_e','acetate':'EX_ac_e'}.items()}
  if name=='SC':self.exchange['glucose']=ids['EX_glc__D_e']
  else:self.exchange['photosynthetic_oxygen']=ids['DM_o2D_u']
  self.g=self.exchange['glucose'] if name=='SC' else self.exchange['acetate'];self.bio=ids['BIOMASS_SC4_bal'] if name=='SC' else ids['BIOMASS_Chlamy_auto']
  # The algal raw file has no preset objective; the processed model selects
  # its autotrophic biomass reaction explicitly.
  assert np.flatnonzero(self.c).tolist()==[self.bio] and self.c[self.bio]==1
  self.photon=np.array([12,13,14,17,18,19,24,25,26,32,33,34])-1;self.photon_max=np.array(raw['lb'],float).ravel()[self.photon]
  if name=='RC':
   starch=ids['SK_starch300_h'];self.l[starch]=-self.epsilon;self.u[starch]=self.epsilon
 def bounds(self,state):
  t,y=state;V,x1,x2,G,Ac,O2,CO2,E=y;l=self.l.copy();u=self.u.copy();eps=self.epsilon
  def q(S,qmax,K):return qmax*max(S,0)/(K+max(S,0))
  if self.name=='SC':
   g,o=self.exchange['glucose'],self.exchange['oxygen']
   l[g]=-q(G,.3,.5)-eps;u[g]=eps;l[o]=-q(O2,1,.5)-eps;u[o]=eps
  else:
   co,ac,o=(self.exchange[k] for k in ['carbon_dioxide','acetate','oxygen'])
   l[co]=-q(CO2,.3,.1)-eps;u[co]=1000+eps;l[ac]=-q(Ac,.3,.1)-eps;u[ac]=eps;l[o]=-q(O2,10,.05)-eps;u[o]=eps
   # Continuous incident-light fraction. No undocumented threshold or attenuation.
   ratio=max(0,(np.sin(np.pi*t/24)**2-np.sin(5*np.pi/24)**2)/(1-np.sin(5*np.pi/24)**2))
   # One solar spectrum; alternative lamps are inactive. Available light is a capacity.
   l[self.photon]=-eps;u[self.photon]=eps
   u[self.photon[0]]=ratio*self.photon_max[0]+eps
  return l,u

def run():
 models=[CSTRModel('SC'),CSTRModel('RC')];warm=[None,None];count=[0];record=[];y0=np.array([125,1.1,1.86,.014,3.,.000653,1.06,8.21])
 for m in models:
  for t in [0,12]:
   lp=m.lp((t,y0));d=solve(m,(t,y0),tol=1e-10)[-1];print(m.name,t,lp.fun,d,flush=True)
 def rhs(t,y):
  vals=[]
  for i,m in enumerate(models):
   try:out=solve(m,(t,y),tol=1e-11,warm=warm[i])
   except RuntimeError:out=solve(m,(t,y),tol=1e-11)
   warm[i]=out[:4];vals.append(out[0]);record.append(out[-1])
  v,w=vals;V,x1,x2,G,Ac,O2,CO2,E=y;D=1/V
  khO=1000*55.56/(25131+709.2*25);khC=1000*np.exp(-8.1403+842.9/(25+151.5))
  count[0]+=1
  if count[0]%100==0:print('rhs',count[0],t,flush=True)
  sc,rc=(m.exchange for m in models)
  return [0,(v[models[0].bio]-D)*x1,(w[models[1].bio]-D)*x2,D*(.014-G)+v[sc['glucose']]*x1,D*(30.-Ac)+v[sc['acetate']]*x1+w[rc['acetate']]*x2,D*(.000653-O2)+v[sc['oxygen']]*x1+(w[rc['oxygen']]+w[rc['photosynthetic_oxygen']])*x2+.25*(khO*.21-O2),D*(1.06-CO2)+v[sc['carbon_dioxide']]*x1+w[rc['carbon_dioxide']]*x2+.25*(khC*.00032-CO2),-D*E+v[sc['ethanol']]*x1+w[rc['ethanol']]*x2]
 ts=[];ys=[];y=y0.copy();start=time.perf_counter()
 for a,b in [(0,5),(5,12),(12,19),(19,24)]:
  sol=solve_ivp(rhs,(a,b),y,method='BDF',rtol=1e-7,atol=1e-10,max_step=.25,dense_output=True)
  if not sol.success:raise RuntimeError(sol.message)
  tt=np.linspace(a,b,51);ts.extend(tt);ys.extend(sol.sol(tt).T);y=sol.y[:,-1];print('segment',b,y,flush=True)
 yy=np.array(ys);tt=np.array(ts);np.savetxt(O/'cstr_trajectory.csv',np.c_[tt,yy],delimiter=',',header='t_h,V_L,X_SC_gDW_L,X_RC_gDW_L,G_mmol_L,Ac_mmol_L,O2_mmol_L,CO2_mmol_L,E_mmol_L',comments='')
 summary={'terminal':y.tolist(),'seconds':time.perf_counter()-start,'rhs_calls':count[0],'minimum_states':yy.min(axis=0).tolist(),'max_equality_residual':max(d['equality_residual'] for d in record),'max_primal_dual_residual':max(d['primal_dual_residual'] for d in record),'protocol':{'V_L':125,'Fin_L_h':1,'Fout_L_h':1,'mu':1e-6,'epsilon':1e-6,'rtol':1e-7,'atol':1e-10,'KKT_tolerance':1e-11,'light':'single solar spectrum capacity; continuous normalised incident-light curve, no biomass attenuation or threshold jump','initial_acetate_mmol_L':3.,'inlet_mmol_L':[.014,30.,.000653,1.06,0],'artificial_infinite_bound_cap':1000,'selection':'consistent finite-barrier KKT at every RHS; no lexicographic secondaries','scenario_change':'The 24 h demonstration uses 3 mmol/L initial acetate and 30 mmol/L inlet acetate with fixed ATP maintenance. These concentrations define this operating condition. The adopted 72 h case is specified separately in cases.json.'}}
 summary['reaction_mapping']={m.name:{k:v+1 for k,v in m.exchange.items()} for m in models}
 summary['protocol']['oxygen_accounting']='Algal EX_o2_e plus DM_o2D_u: the latter removes dummy oxygen generated by PSIIblue/PSIIred, which is not produced in the ordinary oxygen pool.'
 summary['protocol']['mapping_revision']='S. cerevisiae EX_co2_e is reaction416, not reaction55; acetate includes both species exchange terms.'
 (O/'cstr_results.json').write_text(json.dumps(summary,indent=2))
 fig,ax=plt.subplots(2,2,figsize=(10,6))
 for k,n in [(1,'S. cerevisiae'),(2,'C. reinhardtii')]:ax[0,0].plot(tt,yy[:,k],label=n)
 ax[0,0].set_ylabel('Biomass (gDW L$^{-1}$)')
 for k,n in [(3,'Glucose'),(4,'Acetate')]:ax[0,1].plot(tt,yy[:,k],label=n)
 ax[0,1].set_ylabel('Concentration (mmol L$^{-1}$)')
 for k,n in [(5,'Oxygen'),(6,'Carbon dioxide')]:ax[1,0].plot(tt,yy[:,k],label=n)
 ax[1,0].set_ylabel('Concentration (mmol L$^{-1}$)');ax[1,1].plot(tt,yy[:,7],label='Ethanol');ax[1,1].set_ylabel('Concentration (mmol L$^{-1}$)')
 for a in ax.ravel():a.set_xlabel('Time (h)');a.legend()
 plt.tight_layout();plt.savefig(F/'cstr_recomputed.png',dpi=240);plt.savefig(F/'cstr_recomputed.pdf')
if __name__=='__main__':run()
