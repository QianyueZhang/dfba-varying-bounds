"""Compute the adopted LP uptake landscape and check its closed forms."""
from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'data/models/edwards'
OUT=ROOT/'research_completion/runs/python_consistency/edwards'
class LP:
 def __init__(self):
  d=loadmat(SOURCE/'Edwards_Model.mat',simplify_cells=True)['Struct'];self.A=d['A'].toarray();self.c=np.asarray(d['c'],float)
 def fixed(self,g,o,epsilon=0):
  lo=np.full(13,-epsilon,dtype=float);up=np.full(13,100.+epsilon);lo[[0,9]]=[g,o];up[[0,9]]=[g,o]
  return linprog(-self.c,A_eq=self.A,b_eq=np.zeros(8),bounds=list(zip(lo,up)),method='highs')
 def capacity(self,g,epsilon=0):
  lo=np.full(13,-epsilon,dtype=float);up=np.full(13,100.+epsilon);up[[0,9]]=[12*g/(2+g),15.]
  return linprog(-self.c,A_eq=self.A,b_eq=np.zeros(8),bounds=list(zip(lo,up)),method='highs')

def analytic(g,o):
 """Exact LP optimum for the 8x13 matrix and the plotted uptake domain."""
 if g==0:return np.array([0.,0.,0.,0.]) if o==0 else np.full(4,np.nan)
 r=o/g
 if r<1-1e-10 or r>13+1e-10:return np.full(4,np.nan)
 if r>=4:return np.array([(13*g-o)/10,0.,0.,3*o-12*g])
 if r>=3:return np.array([(23*g+19*o)/110,9*(4*g-o)/11,0.,0.])
 if r>=2:return np.array([(g+5*o)/22,9*(o-2*g)/11,1.5*(3*g-o),0.])
 return np.array([(o-g)/2,0.,1.5*(3*g-o),6*g-3*o])

def main():
 OUT.mkdir(parents=True,exist_ok=True);lp=LP()
 # Compute LP landscape and test closed forms independently over the whole grid.
 gg=np.linspace(0,10,101);oo=np.linspace(0,20,101);grid=np.full((len(oo),len(gg),4),np.nan);maxeq=0.;formulaerr=0.;nfeasible=0
 for i,ov in enumerate(oo):
  for j,gv in enumerate(gg):
   s=lp.fixed(gv,ov);a=analytic(gv,ov)
   if s.success:
    nfeasible+=1;grid[i,j]=s.x[[8,11,12,7]];maxeq=max(maxeq,float(abs(lp.A@s.x).max()));assert np.all(np.isfinite(a));formulaerr=max(formulaerr,float(abs(grid[i,j]-a).max()))
   else:assert np.all(np.isnan(a)),(gv,ov,a)
 np.savez_compressed(OUT/'phase_grid.npz',glucose=gg,oxygen=oo,optimal_biomass_D_E_ATP_dissipation=grid)
 phase={'grid_shape':list(grid.shape),'feasible_points':nfeasible,'infeasible_points':int(len(oo)*len(gg)-nfeasible),'maximum_equality_residual':maxeq,'maximum_closed_form_vs_LP_error':formulaerr,'bounds':'All 13 fluxes nonnegative and at most100; glucose and oxygen are fixed to each grid pair. Blank areas are infeasible exact-input pairs, not zero-growth solutions.','region_definition':'r = oxygen uptake / glucose uptake. P4:1<=r<=2; P3:2<=r<=3; P2:3<=r<=4; P1:4<=r<=13. Origin is feasible.','biomass_planes':{'P1':'(13g-o)/10','P2':'(23g+19o)/110','P3':'(g+5o)/22','P4':'(o-g)/2'},'LO':'o=4g; maximum biomass flux per unit glucose is0.9. This is a resource-yield ridge, not the maximum absolute biomass rate for independently bounded oxygen and glucose.','flux_region_semantics':{'P1':'D=E=0; ATP dissipation positive except LO; positive growth inside region.','P2':'D>0,E=0','P3':'D>0,E>0','P4':'D=0,E>0; ATP dissipation positive except o=2g.'},'approximate_old_vertices_vs_exact':{'o20_lower_feasibility_g':[1.63,20/13],'o20_LO_g':[4.89,5.0],'o20_P2_P3_g':[6.59,20/3],'o20_LO_biomass':[4.5,4.5],'o20_P2_P3_biomass':[4.8,160/33]},'assertions_passed':formulaerr<1e-8 and maxeq<1e-8}
 (OUT/'phase_grid_validation.json').write_text(json.dumps(phase,indent=2)+'\n')

if __name__=='__main__':main()
