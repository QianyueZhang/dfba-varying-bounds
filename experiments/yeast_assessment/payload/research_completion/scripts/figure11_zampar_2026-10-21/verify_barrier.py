"""Direct and differentiated finite-barrier verification; no fitted data altered."""
from pathlib import Path
import sys,time,json
import numpy as np
from scipy.sparse import bmat,diags
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp
from types import SimpleNamespace
from fit_batch import BatchModel,ROOT,OUT,Y0,T0,TF
class BarrierBatch(BatchModel):
 def __init__(self,p,epsilon,mu):super().__init__(epsilon);self.params=p;self.mu=mu;self.rhs_calls=0
 def bounds(self,state):return self.bounds_state(state,self.params)
 def lp(self,state):
  x=self.flux(state,self.params);s=self.h.getSolution();return SimpleNamespace(x=x,eqlin=SimpleNamespace(marginals=np.array(s.row_dual)))
 def state_rhs(self,y,v):return y[0]*np.array([v[self.bio],.18*v[self.g],.046*v[self.e]])
 def dl(self,y,dy):
  qg,qo,qe,matp=self.params;kg=.01;g=max(y[1],0);e=max(y[2],0);r=1/(1+(g/.05)**4);dl=np.zeros(len(self.c));dl[self.g]=-qg*kg/(kg+g)**2*dy[1] if y[1]>0 else 0
  de=-qe*.1/(.1+e)**2*r*dy[2]+qe*e/(.1+e)*r*r*4*g**3/.05**4*dy[1] if y[2]>0 else 0
  dl[self.e]=dl[self.et]=de;return dl
 def solve_barrier(self,y,mu=None):
  mu=self.mu if mu is None else mu
  lp=self.lp(y);l,u=self.bounds(y);x=np.clip(lp.x,l+np.minimum(1e-5,(u-l)*.1),u-np.minimum(1e-5,(u-l)*.1));lam=-lp.eqlin.marginals
  return self.barrier(y,mu=mu,warm=(x,lam),tol=2e-7,maxiter=180)
 def initial(self,y):
  v,lam,info=self.solve_barrier(y);return np.r_[y,v,lam],info
 def rhs(self,t,z):
  self.rhs_calls+=1;n=len(self.c);y=z[:3];v=z[3:3+n];lam=z[3+n:];dy=self.state_rhs(y,v);l,u=self.bounds(y);sl=v-l;su=u-v;zl=self.mu/sl**2;zu=self.mu/su**2;K=bmat([[diags(zl+zu),self.A.T],[self.A,None]],format='csc');dvdl=spsolve(K,np.r_[zl*self.dl(y,dy),np.zeros(len(self.b))]);return np.r_[dy,dvdl]

def direct_check():
 p=np.array(list(json.loads((OUT/'fit_summary.json').read_text())['parameters'].values()));tr=np.genfromtxt(OUT/'trajectory_lp.csv',delimiter=',',names=True);times=np.array([-6.8,-4,-2,-.5,0,.5,2,5,8.]);states=np.array([[np.interp(t,tr['t_h'],tr[n])for n in ['X_gDW_L','G_g_L','E_g_L']]for t in times]);records=[]
 for eps,mu in [(1e-4,1e-6),(1e-5,1e-7),(1e-6,1e-8)]:
  model=BarrierBatch(p,eps,mu);start=time.perf_counter()
  for t,y in zip(times,states):
   x,lam,info=model.solve_barrier(y,mu=mu);lp=model.lp(y).x;records.append({'epsilon':eps,'mu':mu,'time_h':float(t),**info,'LP_growth':float(lp[model.bio]),'growth_gap':float(lp[model.bio]-x[model.bio]),'glucose_difference':float(x[model.g]-lp[model.g]),'ethanol_difference':float(x[model.e]-lp[model.e])});print('point',eps,mu,t,info,'elapsed',time.perf_counter()-start,flush=True)
  (OUT/'barrier_point_checks.json').write_text(json.dumps(records,indent=2))
if __name__=='__main__':direct_check()
