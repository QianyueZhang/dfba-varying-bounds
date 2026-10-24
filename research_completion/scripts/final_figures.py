from control_experiments import *
r=json.loads((O/'control_results.json').read_text());react=Reactor();tr=react.simulate(r['best']['control'],1e-9,1e-11,dense=True);t=tr['time'];y=tr['state'];a,b=[c(y[:,3]) for c in react.curves]
fig,ax=plt.subplots(1,3,figsize=(12,3.5))
for c,name in [(a,'S. cerevisiae'),(b,'K. phaffii')]:
 ax[0].plot(t,c[:,0],label=name);ax[1].plot(t,-c[:,1],label=name)
ax[2].plot(t,y[:,1]/y[:,2],label='Biomass ratio');ax[2].axhline(1,ls='--',color='black',lw=.8,label='Target')
for aa in ax:aa.set_xlabel('Time (h)');aa.legend(fontsize=7)
ax[0].set_ylabel('Specific growth rate (h$^{-1}$)');ax[1].set_ylabel('Glucose uptake (mmol gDW$^{-1}$ h$^{-1}$)');ax[2].set_ylabel('X$_{SC}$/X$_{KP}$');save('control_fluxes')
out=json.loads((O/'biomass_objective_results.json').read_text());best=min(out,key=lambda z:z['terminal_KP_objective']);tr=react.simulate(best['control'],1e-9,1e-11,dense=True);np.savetxt(O/'biomass_control_trajectory.csv',np.c_[tr['time'],tr['state']],delimiter=',',header='time_h,V_L,X_SC_gDW_L,X_KP_gDW_L,S_g_L',comments='');best['refined_validation']=react.simulate(best['control'],1e-9,1e-11);(O/'biomass_objective_results.json').write_text(json.dumps(out,indent=2))
fig,ax=plt.subplots(1,2,figsize=(9,3.6));ax[0].stairs(best['control'],np.linspace(0,14,8),label='Maximise K. phaffii');ax[0].stairs(r['best']['control'],np.linspace(0,14,8),label='Target equal biomass');ax[0].set(xlabel='Time (h)',ylabel='Feed rate (L h$^{-1}$)');ax[0].legend(fontsize=8)
ax[1].plot(tr['time'],tr['state'][:,1],label='S. cerevisiae');ax[1].plot(tr['time'],tr['state'][:,2],label='K. phaffii');ax[1].set(xlabel='Time (h)',ylabel='Biomass (gDW L$^{-1}$)');ax[1].legend();save('biomass_control')
# Refresh isolated-culture figures using the final response table.
held=np.loadtxt(O/'surrogate_heldout.csv',delimiter=',',skiprows=1);tr=react.simulate(r['best']['control'],1e-9,1e-11,dense=True);make_plots(react,r,tr,held[:,3],held[:,4])
