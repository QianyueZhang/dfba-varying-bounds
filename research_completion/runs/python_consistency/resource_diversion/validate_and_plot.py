"""Independent numerical checks and final figures for resource diversion."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from resource_diversion import Model,HERE,ROOT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

CASES={'fedbatch':('fedbatch_refined','fedbatch_no_diversion','fedbatch_differentiated'),
       'batch':('batch_refined_2p4','batch_no_diversion_2p4','batch_differentiated_2p4')}

def load(name):
    return dict(np.load(HERE/name/'trajectory.npz')),json.loads((HERE/name/'audit.json').read_text())

def validate(name):
    a,r=load(name);p=r['parameters'];mu=r['numerical']['mu'];mods=[Model(p['case'],s,p['Kp'],mu)for s in ['S1','S2']]
    eq=stat=pd=gap=bound=0.;minslack=np.inf;mingap=np.inf;mingrowth=np.inf
    for j,y in enumerate(a['amounts']):
        env=y[1:5]/y[0]
        for i,m in enumerate(mods):
            v=a['flux'][j,i];lam=a['lambda_'][j,i];zl=a['zl'][j,i];zu=a['zu'][j,i];l,u=m.bounds(env)
            sl=v-l;su=u-v;minslack=min(minslack,float(min(sl.min(),su.min())))
            eq=max(eq,float(abs(m.rawA@v).max()))
            stat=max(stat,float(abs(-m.c+m.A.T@lam-mu/sl+mu/su).max()))
            pd=max(pd,float(abs(-m.c+m.A.T@lam-zl+zu).max()),float(abs(sl*zl-mu).max()),float(abs(su*zu-mu).max()))
            bound=max(bound,float(np.max(l-v)),float(np.max(v-u)))
            lp=m.lp(env);assert lp.success
            delta=float(-lp.fun-v[12]);gap=max(gap,delta);mingap=min(mingap,delta);mingrowth=min(mingrowth,float(v[12]))
    Y=a['amounts'];M0=p['initial_volume_L']*p['initial_biomass_gDW_L']
    expected=np.c_[M0+Y[:,5]-Y[:,7],M0+Y[:,6]-Y[:,8],100+Y[:,9]+Y[:,10]+Y[:,13],Y[:,11]+Y[:,12]]
    balance=np.max(abs(Y[:,1:5]-expected),axis=0)
    volume=float(abs(Y[:,0]-(5+p['feed_L_h']*a['time_h'])).max())
    rec=dict(run=name,full_horizon=r['full_requested_horizon'],time_h=float(a['time_h'][-1]),
             max_raw_stoichiometric_residual=eq,max_explicit_stationarity_residual=stat,max_primal_dual_residual=pd,
             max_bound_excess=bound,min_slack=minslack,max_growth_gap_to_LP=gap,min_growth_gap_to_LP=mingap,
             nominal_gap_bound=190*mu,min_growth_h_inv=mingrowth,min_concentration=float((Y[:,1:5]/Y[:,:1]).min()),
             tracked_pool_balance_errors_g=balance.tolist(),volume_error_L=volume,
             terminal_concentrations=(Y[-1,1:5]/Y[-1,0]).tolist(),terminal_amounts=Y[-1,:5].tolist(),
             cumulative_pyruvate_export_and_exchange_g=Y[-1,11:13].tolist(),
             cumulative_growth_gDW=Y[-1,5:7].tolist(),cumulative_biomass_loss_gDW=Y[-1,7:9].tolist())
    rec['passed']=all([rec['full_horizon'],eq<1e-9,pd<1e-10,bound<1e-10,minslack>0,gap<=190*mu+1e-9,mingap>=-1e-9,
                       mingrowth>0,float(balance.max())<1e-8,volume<1e-9,rec['min_concentration']>=-1e-12])
    assert rec['passed'],rec
    return rec

def comparisons(case):
    names=CASES[case];a,ra=load(names[0]);b,rb=load(names[1]);d,rd=load(names[2])
    assert np.array_equal(a['time_h'],b['time_h']) and np.array_equal(a['time_h'],d['time_h'])
    c=a['amounts'][:,1:5]/a['amounts'][:,:1];bc=b['amounts'][:,1:5]/b['amounts'][:,:1];dc=d['amounts'][:,1:5]/d['amounts'][:,:1]
    diff=np.max(abs(c-dc),axis=0);gap=d['LP_growth_gaps'];res=d['residuals']
    # Independently check the complete chain rule for representative states.
    derivative_errors=[]
    for j in [0,100,250,400,500]:
        x1,x2,G,P=c[j];V=a['amounts'][j,0];v=a['flux'][j];F=ra['parameters']['feed_L_h'];kd=ra['parameters']['death_h_inv']
        xd=np.array([(v[0,12]-F/V-kd)*x1,(v[1,12]-F/V-kd)*x2,
                     .18*(v[0,27]*x1+v[1,27]*x2)+(20-G)*F/V,
                     .088*(v[0,37]*x1+v[1,37]*x2)-P*F/V])
        analytic=.1*(xd[2]*x1*x2+G*xd[0]*x2+G*x1*xd[1])
        h=1e-5;f=lambda s:.1*s[2]*s[0]*s[1]
        fd=(f(c[j]+h*xd)-f(c[j]-h*xd))/(2*h)
        derivative_errors.append(float(abs(fd-analytic)/max(1,abs(analytic))))
    out=dict(case=case,horizon_h=float(a['time_h'][-1]),
             differentiated_vs_direct_max_concentration_difference=diff.tolist(),
             differentiated_vs_direct_max_flux_difference=float(abs(a['flux']-d['flux']).max()),
             differentiated_max_stoichiometric_residual=float(res[:,:,0].max()),
             differentiated_max_explicit_stationarity_residual=float(res[:,:,1].max()),
             differentiated_min_slack=float(res[:,:,2].min()),
             differentiated_max_growth_gap_to_LP=float(gap.max()),
             directional_finite_difference_relative_error=max(derivative_errors),
             donor_change_gDW_L=float(c[-1,0]-bc[-1,0]),recipient_change_gDW_L=float(c[-1,1]-bc[-1,1]),
             donor_change_percent=float(100*(c[-1,0]/bc[-1,0]-1)),recipient_change_percent=float(100*(c[-1,1]/bc[-1,1]-1)),
             counterfactual_terminal_concentrations=bc[-1].tolist(),
             counterfactual_population_difference_max_gDW_L=float(abs(bc[:,0]-bc[:,1]).max()))
    out['passed']=all([diff.max()<1e-6,res[:,:,0].max()<1e-9,res[:,:,2].min()>0,gap.min()>-1e-8,gap.max()<190e-8+1e-8,
                       max(derivative_errors)<1e-7])
    assert out['passed'],out
    return out

def style(ax,title,ylabel,horizon):
    ax.set(xlim=(0,horizon),xlabel='Time (h)',ylabel=ylabel)
    if abs(horizon-2.4)<1e-9:ax.set_xticks([0,.6,1.2,1.8,2.4])
    ax.set_title(title,loc='left',fontsize=9.5);ax.grid(axis='y',alpha=.16);ax.spines[['top','right']].set_visible(False)

def save(fig,name):
    p=HERE/'figures';p.mkdir(parents=True,exist_ok=True)
    for ext in ['png','svg']:
        fig.savefig(p/f'{name}.{ext}',dpi=300,bbox_inches='tight',metadata={'Date':None}if ext=='svg'else None)
    plt.close(fig)
    Image.open(p/f'{name}.png').convert('L').save(p/f'{name}_greyscale.png')
    return {ext:dict(path=str((p/f'{name}.{ext}').relative_to(ROOT)),sha256=hashlib.sha256((p/f'{name}.{ext}').read_bytes()).hexdigest())for ext in ['png','svg']}

def figures(case):
    a,r=load(CASES[case][2]);b,br=load(CASES[case][1]);t=a['time_h'];H=t[-1];Y=a['amounts'];c=Y[:,1:5]/Y[:,:1];bc=b['amounts'][:,1:5]/b['amounts'][:,:1]
    colors=['#175b8c','#ad511b'];marks=['s','^'];styles=['-','--']
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.8),constrained_layout=True)
    for i in range(2):axs[0,0].plot(t,c[:,i],color=colors[i],ls=styles[i],marker=marks[i],ms=3,markevery=65,label=['Donor','Recipient'][i])
    axs[0,0].plot(t,bc[:,0],color='#303030',ls=':',marker='o',mfc='white',ms=3,markevery=65,label='No diversion (both)')
    style(axs[0,0],'(a) Biomass concentrations',r'Biomass (gDW L$^{-1}$)',H);axs[0,0].legend(frameon=False,fontsize=8)
    axs[0,1].plot(t,c[:,2],color='#175b8c',marker='s',ms=3,markevery=65,label='Recipient-dependent diversion')
    axs[0,1].plot(t,bc[:,2],color='#303030',ls=':',marker='o',mfc='white',ms=3,markevery=65,label='No diversion')
    style(axs[0,1],'(b) Shared glucose',r'Glucose (g L$^{-1}$)',H);axs[0,1].legend(frameon=False,fontsize=7.5)
    axs[1,0].plot(t,c[:,3],color='#2e6f4e',marker='D',ms=3,markevery=65,label='Recipient-dependent diversion')
    axs[1,0].plot(t,bc[:,3],color='#303030',ls=':',marker='o',mfc='white',ms=3,markevery=65,label='No diversion')
    style(axs[1,0],'(c) Shared pyruvate',r'Pyruvate (g L$^{-1}$)',H);axs[1,0].legend(frameon=False,fontsize=7.5)
    for i in range(2):axs[1,1].plot(t,Y[:,1+i],color=colors[i],ls=styles[i],marker=marks[i],ms=3,markevery=65,label=['Donor','Recipient'][i])
    axs[1,1].plot(t,b['amounts'][:,1],color='#303030',ls=':',marker='o',mfc='white',ms=3,markevery=65,label='No diversion (both)')
    style(axs[1,1],'(d) Biomass amounts',r'Biomass amount (gDW)',H);axs[1,1].legend(frameon=False,fontsize=8)
    out={'states':save(fig,f'{case}_diversion_states')}
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.8),constrained_layout=True)
    specs=[(0,27,'(a) Donor glucose exchange'),(1,27,'(b) Recipient glucose exchange'),(0,37,'(c) Donor pyruvate secretion'),(1,37,'(d) Recipient pyruvate exchange')]
    for ax,(i,k,title)in zip(axs.flat,specs):
        low=a['lower'][:,i,k];up=a['upper'][:,i,k];fl=a['flux'][:,i,k]
        if not(i==0 and k==37):
            ax.fill_between(t,low,up,color='#c6cbd0',alpha=.3,label='Scalar bound interval')
            ax.plot(t,up,color='#666666',ls=':',lw=1.1,label='Upper bound')
        ax.plot(t,low,color='#202020',ls='--',lw=1.1,label='Lower bound')
        ax.plot(t,fl,color=colors[i],ls='-',marker=marks[i],markevery=60,ms=3,label='Exchange flux')
        style(ax,title,r'Flux (mmol gDW$^{-1}$ h$^{-1}$)',H)
        ax.legend(frameon=False,fontsize=7.2,loc='best')
    out['fluxes']=save(fig,f'{case}_diversion_fluxes')
    return out

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'svg.fonttype':'none','svg.hashsalt':'resource_diversion','lines.linewidth':1.6})
    checks=[validate(name)for names in CASES.values()for name in names[:2]]
    comp=[comparisons(case)for case in CASES]
    files={case:figures(case)for case in CASES}
    r=dict(validation=checks,comparisons=comp,figures=files,adoption_ready=True,
           figure_data_sources={case:{'diversion':names[2],'no_diversion_reference':names[1]}for case,names in CASES.items()},
           scope='Correct finite-barrier resource-diversion model during active growth. No starvation continuation, physiological calibration or global ecological generalisation.',
           accounting='Tracked glucose, pyruvate and biomass balances include feed and biomass loss; no complete elemental-carbon recovery claim.')
    (HERE/'VALIDATION_AND_ADOPTION.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))

if __name__=='__main__':main()
