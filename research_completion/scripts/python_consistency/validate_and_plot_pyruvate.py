"""Independent saved-array checks and publication-style pyruvate figures."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(Path(__file__).parent))
from pyruvate_crossfeeding import Model,OUT,VOLUME,MW_G,MW_P
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(name):
    p=OUT/name
    return dict(np.load(p/'trajectory.npz')),json.loads((p/'audit.json').read_text())


def validate(name):
    a,r=load(name);models=[Model(s,r['numerical']['mu'],1e-11,r['parameters']['pyruvate_recipient_uptake_enabled'])for s in ['S1','S2']]
    maxeq=0.;maxbound=0.;maxgap=0.;mingrowth=np.inf;inactive_nonzero=0.;minslack=np.inf
    flux=a['flux'];y=a['amounts'];modes=a['active'];t=a['time_h']
    for j in range(len(t)):
        for i,m in enumerate(models):
            v=flux[j,i]
            maxeq=max(maxeq,float(abs(m.rawA@v).max()))
            if not modes[j,i]:
                inactive_nonzero=max(inactive_nonzero,float(abs(v).max()));continue
            m.level=a['donor_secretion_level'][j]
            env=(y[j,2+i]/VOLUME,y[j,4]/VOLUME);l,u=m.bounds(env)
            maxbound=max(maxbound,float(np.max(l-v)),float(np.max(v-u)))
            minslack=min(minslack,float(min(np.min(v-l),np.min(u-v))))
            lp=m.lp(env)
            assert lp.success,(name,t[j],i,lp.message)
            gap=float(-lp.fun-v[m.bio]);maxgap=max(maxgap,gap)
            assert gap>=-1e-8,(name,t[j],gap)
            mingrowth=min(mingrowth,float(v[m.bio]))
    expected=np.c_[5+y[:,5]-y[:,7],5+y[:,6]-y[:,8],50+y[:,9],50+y[:,10],y[:,11]+y[:,12]]
    mass_error=np.max(abs(y[:,:5]-expected),axis=0)
    event_checks=[]
    for event in r['events']:
        if event['event']!='quiescent':continue
        i=event['species']-1;m=models[i];m.level=0.;m.warm=None
        state=event['concentrations'];env=(state[2+i],state[4])
        lp=m.lp(env);v=m.flux(env)
        event_checks.append({'species':i+1,'time_h':event['time_h'],
            'maximum_LP_growth_h_inv':float(-lp.fun),'specified_threshold_h_inv':r['numerical']['entry'],
            'active_barrier_growth_immediately_before_switch_h_inv':float(v[12]),
            'max_stoichiometric_residual_before_switch':float(abs(m.rawA@v).max())})
        assert abs(-lp.fun-r['numerical']['entry'])<1e-9
        assert v[12]>0
    result={'run':name,'full_5h':bool(t[0]==0 and t[-1]==5),
            'max_raw_stoichiometric_residual':maxeq,'max_active_bound_excess':maxbound,
            'min_active_slack':minslack,'max_active_growth_gap_to_LP':maxgap,'nominal_gap_bound':190*r['numerical']['mu'],
            'min_active_growth_h_inv':mingrowth,'inactive_max_abs_flux':inactive_nonzero,
            'max_tracked_pool_balance_errors_g':mass_error.tolist(),
            'minimum_concentration_g_L':float((y[:,:5]/VOLUME).min()),
            'terminal_biomass_loss_amounts_gDW':y[-1,7:9].tolist(),
            'terminal_cumulative_pyruvate_export_g':float(y[-1,11]),
            'terminal_cumulative_recipient_pyruvate_exchange_g':float(y[-1,12]),'quiescence_event_checks':event_checks,
            'mass_accounting_scope':'Biomass growth and loss, both glucose pools and shared pyruvate are balanced. Biomass loss is recorded as unreturned necromass. Other metabolic exchange products/reservoirs are not dynamically tracked, so this is not a complete elemental-carbon recovery calculation.'}
    checks=[result['full_5h'],maxeq<1e-9,maxbound<1e-10,minslack>0,maxgap<190*r['numerical']['mu']+1e-9,mingrowth>0,inactive_nonzero==0,float(mass_error.max())<1e-9,result['minimum_concentration_g_L']>=-1e-10]
    result['passed']=all(checks)
    assert result['passed'],result
    np.savetxt(OUT/name/'pool_accounting.csv',np.c_[t,y[:,5:13]],delimiter=',',
               header='time_h,cumulative_growth1_gDW,cumulative_growth2_gDW,unreturned_necromass1_gDW,unreturned_necromass2_gDW,glucose1_net_exchange_g,glucose2_net_exchange_g,donor_pyruvate_net_exchange_g,recipient_pyruvate_net_exchange_g',comments='')
    return result


def savediff(a,b):
    aa,ar=load(a);bb,br=load(b)
    assert np.array_equal(aa['time_h'],bb['time_h'])
    return np.max(abs(aa['amounts'][:,:5]-bb['amounts'][:,:5])/VOLUME,axis=0).tolist()


def style(ax,title,ylabel):
    ax.set(xlim=(0,5),xlabel='Time (h)',ylabel=ylabel)
    ax.set_title(title,loc='left',fontsize=9.5)
    ax.grid(axis='y',alpha=.16)
    ax.spines[['top','right']].set_visible(False)


def save(fig,stem):
    dest=OUT/'figures';dest.mkdir(parents=True,exist_ok=True)
    for ext in ['png','svg']:
        fig.savefig(dest/f'{stem}.{ext}',dpi=300,bbox_inches='tight',metadata={'Date':None}if ext=='svg'else None)
    plt.close(fig)
    return {ext:{'path':str((dest/f'{stem}.{ext}').relative_to(ROOT)),
                 'sha256':hashlib.sha256((dest/f'{stem}.{ext}').read_bytes()).hexdigest()}for ext in ['png','svg']}


def figures():
    a,r=load('refined');b,rb=load('reference');t=a['time_h'];y=a['amounts'][:,:5]/VOLUME;ref=b['amounts'][:,:5]/VOLUME
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'svg.fonttype':'none','svg.hashsalt':'coherent_pyruvate_batch','lines.linewidth':1.6})
    fig=plt.figure(figsize=(7.2,5.8),constrained_layout=True)
    grid=fig.add_gridspec(2,2)
    axes=[fig.add_subplot(grid[0,:]),fig.add_subplot(grid[1,0]),fig.add_subplot(grid[1,1])]
    axes[0].plot(t,y[:,0],color='#175b8c',ls='-',marker='s',markevery=65,ms=3,label='Donor (both cases)')
    axes[0].plot(t,y[:,1],color='#ad511b',ls='--',marker='^',markevery=65,ms=3,label='Recipient, transfer enabled')
    axes[0].plot(t,ref[:,1],color='#303030',ls=':',marker='o',mfc='white',markevery=65,ms=3,label='Recipient, uptake disabled')
    style(axes[0],'(a) Biomass concentrations',r'Biomass (gDW L$^{-1}$)')
    axes[0].legend(frameon=False,fontsize=8,loc='lower right')
    axes[1].plot(t,y[:,2],color='#175b8c',ls='-',marker='s',markevery=65,ms=3,label='Donor glucose (both cases)')
    axes[1].plot(t,y[:,3],color='#ad511b',ls='--',marker='^',markevery=65,ms=3,label='Recipient glucose, enabled')
    axes[1].plot(t,ref[:,3],color='#303030',ls=':',marker='o',mfc='white',markevery=65,ms=3,label='Recipient glucose, disabled')
    style(axes[1],'(b) Separate glucose pools',r'Glucose (g L$^{-1}$)')
    axes[1].legend(frameon=False,fontsize=7.5,loc='upper right')
    axes[2].plot(t,y[:,4],color='#2e6f4e',ls='-',marker='D',markevery=65,ms=3,label='Transfer enabled')
    axes[2].plot(t,ref[:,4],color='#303030',ls='--',marker='o',mfc='white',markevery=65,ms=3,label='Recipient uptake disabled')
    style(axes[2],'(c) Shared pyruvate pool',r'Pyruvate (g L$^{-1}$)')
    axes[2].legend(frameon=False,fontsize=8,loc='lower right')
    files={'states':save(fig,'pyruvate_crossfeeding_states')}
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.8),constrained_layout=True)
    configs=[(0,27,'(a) Donor glucose exchange'),(1,27,'(b) Recipient glucose exchange'),(0,37,'(c) Donor pyruvate secretion'),(1,37,'(d) Recipient pyruvate exchange')]
    models=[Model(s,r['numerical']['mu'],1e-11)for s in ['S1','S2']]
    for ax,(i,k,title)in zip(axes.flat,configs):
        lows=[];ups=[]
        for j in range(len(t)):
            m=models[i];m.level=a['donor_secretion_level'][j]
            l,u=m.bounds((y[j,2+i],y[j,4]));active=a['active'][j,i]
            lows.append(l[k] if active else 0.);ups.append(u[k] if active else 0.)
        lows=np.asarray(lows);ups=np.asarray(ups)
        if not(i==0 and k==37):ax.fill_between(t,lows,ups,color='#c6cbd0',alpha=.3,label='Scalar bound interval')
        ax.plot(t,lows,color='#202020',ls='--',lw=1.1,label='Lower bound')
        if not(i==0 and k==37):ax.plot(t,ups,color='#606060',ls=':',lw=1.1,label='Upper bound')
        ax.plot(t,a['flux'][:,i,k],color='#175b8c' if i==0 else '#ad511b',ls='-',marker='s' if i==0 else '^',markevery=60,ms=3,label='Exchange flux')
        style(ax,title,r'Flux (mmol gDW$^{-1}$ h$^{-1}$)')
        # Panels (a) and (b) define the shared bound/flux conventions. Repeating
        # this legend in (d) would obscure the switch to zero exchange.
        if not(i==1 and k==37):
            ax.legend(frameon=False,fontsize=7.5,loc='upper right' if k==37 else 'lower right')
        if i==0 and k==37:ax.set_ylim(-.05,2.2)
    files['fluxes']=save(fig,'pyruvate_crossfeeding_fluxes')
    return files


def main():
    names=['primary','refined','reference','threshold','barrier_sensitivity']
    checks=[validate(n)for n in names]
    diffs={'integration_RK45_vs_DOP853':savediff('primary','refined'),
           'barrier_1e_8_vs_1e_9':savediff('refined','barrier_sensitivity'),
           'quiescence_threshold_1e_5_vs_1e_6':savediff('barrier_sensitivity','threshold')}
    assert max(diffs['integration_RK45_vs_DOP853'])<2e-6
    assert max(diffs['barrier_1e_8_vs_1e_9'])<1e-5
    assert max(diffs['quiescence_threshold_1e_5_vs_1e_6'])<5e-5
    a,ar=load('refined');b,br=load('reference')
    final_a=a['amounts'][-1,:5]/VOLUME;final_b=b['amounts'][-1,:5]/VOLUME
    donor_diff=float(np.max(abs(a['amounts'][:,0]-b['amounts'][:,0]))/VOLUME)
    assert donor_diff<2e-6
    report={'checked_runs':checks,'max_concentration_differences_g_L':diffs,
            'counterfactual':{'donor_max_trajectory_difference_gDW_L':donor_diff,
                'recipient_terminal_with_uptake_gDW_L':float(final_a[1]),'recipient_terminal_without_uptake_gDW_L':float(final_b[1]),
                'recipient_terminal_gain_gDW_L':float(final_a[1]-final_b[1]),
                'recipient_terminal_gain_percent':float(100*(final_a[1]/final_b[1]-1)),
                'difference':'Only recipient pyruvate uptake is disabled. The donor export policy, initial conditions, network, numerical tolerances and quiescence law are otherwise identical.'},
            'figure_files':figures(),'adoption_ready':True,
            'adoption_condition':'Present as a chemically consistent pyruvate cross-feeding model with explicitly stated growth/quiescence extension and matched counterfactual. Never label an equivalent port of the ethanol-labelled Figure27/28 calculation.',
            'limits':['No measured cultivation data calibrate the uptake, secretion or quiescence policies.','The zero-metabolism quiescent state is an idealisation; it is distinct from satisfying the active-cell ATP-maintenance bound.','Tracked pool balances plus stoichiometric flux residuals do not establish complete reactor elemental-carbon recovery.','Trajectories follow the stated equations, switching thresholds and parameters; no trajectory-fitting objective is imposed.']}
    (OUT/'validation_and_adoption.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
