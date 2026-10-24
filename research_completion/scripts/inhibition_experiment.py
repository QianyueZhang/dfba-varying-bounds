from control_experiments import *
from flux_model import FluxModel
from primal_dual import solve
class EthanolModel(FluxModel):
 def __init__(self):super().__init__('SC',1e-6);self.minimum_ethanol=10
 def bounds(self,S):
  l,u=super().bounds(S);l[2707]=self.minimum_ethanol-self.epsilon;return l,u
m=EthanolModel();kp=Reactor().curves[1];output=[]
for Ki in [5,float('inf')]:
 warm=[None];count=[0];rec=[];y=np.array([.001,.001,20.,0.]);ts=[0.];ys=[y.copy()];start=0.;then=time.perf_counter()
 for etoh,threshold in [(10,7.5),(2,2),(0,1e-5)]:
  m.minimum_ethanol=etoh;warm[0]=None
  def rhs(t,y):
   X1,X2,S,E=y
   try:res=solve(m,max(S,threshold),tol=1e-10,warm=warm[0])
   except RuntimeError:res=solve(m,max(S,threshold),tol=1e-10)
   warm[0]=res[:4];v=res[0];rec.append(res[-1]);q=6*max(S,0)/(.385+max(S,0))/(1+max(E,0)/Ki);se=.385*q/(6-q) if q<6 else 100.;b=kp(se);count[0]+=1
   return [v[m.bio]*X1,b[0]*X2,.18*(v[m.g]*X1+b[1]*X2),.046*v[2707]*X1]
  def event(t,y):return y[2]-threshold
  event.terminal=True;event.direction=-1
  sol=solve_ivp(rhs,(start,14),y,method='DOP853',rtol=2e-6,atol=1e-8,max_step=.5,events=event,dense_output=True)
  if not sol.success:raise RuntimeError(sol.message)
  end=sol.t[-1];tt=np.linspace(start,end,81)[1:];ts.extend(tt);ys.extend(sol.sol(tt).T);y=sol.y[:,-1];start=end;print('Ki',Ki,'phase',etoh,end,y,flush=True)
  if end>=14:break
 if start<14:ts.append(14);ys.append(y.copy())
 arr=np.c_[ts,ys];key='inhibited' if Ki==5 else 'uninhibited';np.savetxt(O/f'genome_scale_{key}.csv',arr,delimiter=',',header='t_h,X_SC_gDW_L,X_KP_gDW_L,S_g_L,E_g_L',comments='')
 output.append({'case':key,'Ki_g_L':Ki if Ki==5 else None,'terminal':y.tolist(),'seconds':time.perf_counter()-then,'rhs_evaluations':count[0],'max_equality_residual':max(d['equality_residual'] for d in rec),'max_primal_dual_residual':max(d['primal_dual_residual'] for d in rec),'protocol':'Batch, 14 h, initial X=(0.001,0.001),S=20,E=0; SC ethanol minimum10/2/0 at S>7.5,2<S<=7.5,S<=2; epsilon=mu=1e-6; restart algebraic solve at each switch; near-exhaustion stop at S=1e-5; KP uses refined one-dimensional flux response at equivalent inhibited uptake capacity'})
 (O/'inhibition_results.json').write_text(json.dumps(output,indent=2))
fig,ax=plt.subplots(1,3,figsize=(12,3.6))
for key,style in [('inhibited','-'),('uninhibited','--')]:
 a=np.loadtxt(O/f'genome_scale_{key}.csv',delimiter=',',skiprows=1)
 for i,n in [(1,'S. cerevisiae'),(2,'K. phaffii')]:ax[0].plot(a[:,0],a[:,i],ls=style,label=n+(' (inhibition)' if key=='inhibited' else ' (reference)'))
 ax[1].plot(a[:,0],a[:,3],ls=style,label=key);ax[2].plot(a[:,0],a[:,4],ls=style,label=key)
for a in ax:a.set_xlabel('Time (h)')
ax[0].set_ylabel('Biomass (gDW L$^{-1}$)');ax[0].legend(fontsize=7);ax[1].set_ylabel('Glucose (g L$^{-1}$)');ax[2].set_ylabel('Ethanol (g L$^{-1}$)');ax[2].legend(fontsize=8);save('genome_scale_inhibition')
