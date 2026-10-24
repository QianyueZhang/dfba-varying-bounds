"""Render benchmark figures from the saved calculations without rerunning them."""
from pathlib import Path
from hashlib import sha256
import json
import os

HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'mpl-cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
RES=HERE/'results'
data=json.loads((RES/'validation.json').read_text())
if not data['passed']:raise RuntimeError('Calculation checks did not pass')
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':11,
                     'axes.labelsize':11,'legend.fontsize':9,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.linewidth':.7,'lines.linewidth':1.6,
                     'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
BLUE='#2d617e';TEAL='#47877d';GOLD='#ae862d';RED='#b55b49';INK='#26373f'
assets=[]


def finish(fig,name,caption):
    fig.canvas.draw()
    for suffix in ['pdf','svg','png']:
        path=OUT/f'{name}.{suffix}';fig.savefig(path,dpi=400,bbox_inches='tight',pad_inches=.08)
        assets.append({'path':str(path.relative_to(HERE)),'sha256':sha256(path.read_bytes()).hexdigest(),'caption':caption})
    path=OUT/f'{name}_preview.png';fig.savefig(path,dpi=135,bbox_inches='tight',pad_inches=.08)
    with Image.open(path) as im:im.convert('L').save(OUT/f'{name}_greyscale.png')
    plt.close(fig)


base=np.loadtxt(RES/'analytic_direct.csv',delimiter=',',skiprows=1)
tr=np.load(RES/'trajectory_complete.npz')
sens=np.loadtxt(RES/'bound_sensitivities.csv',delimiter=',',skiprows=1)
fig,ax=plt.subplots(3,1,figsize=(7.1,8.2),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[1.25,1,1]})
t=base[:,0]
ax[0].fill_between(t,base[:,1],base[:,2],color=TEAL,alpha=.09,label='Common bound interval')
ax[0].plot(t,base[:,1],color=GOLD,ls='--',label='Lower bound')
ax[0].plot(t,base[:,2],color=GOLD,ls='-.',label='Upper bound')
ax[0].plot(t,base[:,3],color=INK,label='Analytic finite-barrier solution')
ax[0].plot(tr['time_h'][::30],tr['state'][::30,0],ls='none',marker='o',ms=4,mfc='white',mec=BLUE,label='Differentiated KKT solution')
ax[0].set_ylabel('Flux (dimensionless)');ax[0].legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,-.02),frameon=False,fontsize=8.5)
ax[0].set_title('(a)  Moving bounds and coincident fluxes $v_1=v_2$',loc='left',pad=5)
ax[1].plot(t,sens[:,1]*sens[:,3],color=TEAL,ls='--',label='Lower-bound contribution')
ax[1].plot(t,sens[:,2]*sens[:,4],color=GOLD,ls='-.',label='Upper-bound contribution')
ax[1].plot(t,sens[:,5],color=INK,label='Total flux rate')
ax[1].axhline(0,color='#bbbbbb',lw=.5)
ax[1].set_ylabel(r'Flux rate (h$^{-1}$)');ax[1].set_title('(b)  Chain-rule contributions',loc='left')
ax[1].legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.02),frameon=False,fontsize=8.5)
err_ode=np.max(abs(tr['state'][:,:2]-tr['reference'][:,:2]),axis=1)
err_root=np.max(abs(base[:,6:8]-base[:,3:5]),axis=1)
ax[2].plot(t,err_ode,color=BLUE,label='Differentiated versus analytic')
ax[2].plot(t,err_root,color=RED,ls='--',label='Independent root versus analytic')
ax[2].set_ylabel('Absolute flux error');ax[2].set_xlabel('Synthetic time (h)')
ax[2].ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
ax[2].set_title('(c)  Independent numerical comparisons',loc='left')
ax[2].legend(frameon=False,loc='upper right')
for a in ax:a.set_xlim(0,12);a.grid(axis='y',color='#dddddd',lw=.5,alpha=.65)
finish(fig,'moving_bound_verification','Two-flux verification problem with moving lower and upper bounds: (a) common scalar bound interval and coincident fluxes; (b) contributions to the differentiated flux; (c) errors relative to the closed-form finite-barrier solution. The barrier parameter is 0.2; all fluxes and bounds are dimensionless.')

fig,ax=plt.subplots(2,2,figsize=(7.4,6.5),layout='constrained')
fd=data['finite_difference_sensitivity'];h=np.array([x['step']for x in fd])
for key,label,color,marker in [('lower_sensitivity_max_abs_error','Lower-bound sensitivity',TEAL,'o'),('upper_sensitivity_max_abs_error','Upper-bound sensitivity',GOLD,'s')]:
    ax[0,0].loglog(h,[x[key]for x in fd],marker=marker,mfc='white',color=color,label=label)
ax[0,0].set(xlabel='Central-difference step',ylabel='Maximum absolute sensitivity error')
ax[0,0].set_title('(a)  Independent sensitivity check',loc='left');ax[0,0].legend(frameon=False,fontsize=8)
br=data['barrier_refinement'];mu=[x['mu']for x in br]
ax[0,1].loglog(mu,[x['maximum_LP_objective_gap']for x in br],'-o',mfc='white',color=BLUE,label='Maximum LP objective gap')
ax[0,1].loglog(mu,[x['nominal_primal_dual_gap_bound']for x in br],'--',color=INK,label=r'$4\mu$ certificate')
ax[0,1].set(xlabel=r'Barrier parameter $\mu$',ylabel='Objective difference (dimensionless)')
ax[0,1].set_title('(b)  Barrier refinement; fixed bounds',loc='left');ax[0,1].legend(frameon=False,fontsize=8)
rr=data['tolerance_refinement']
ax[1,0].loglog([x['rtol']for x in rr],[x['max_abs_flux_error']for x in rr],'-o',mfc='white',color=BLUE)
ax[1,0].set(xlabel='ODE relative tolerance',ylabel='Maximum absolute flux error')
ax[1,0].set_title(r'(c)  ODE refinement; fixed $\mu=0.2$',loc='left')
for key,label,color,ls in [('omit_lower','Lower-bound term omitted',GOLD,'--'),('omit_upper','Upper-bound term omitted',RED,'-.')]:
    z=np.load(RES/f'trajectory_{key}.npz');error=np.max(abs(z['state'][:,:2]-z['reference'][:,:2]),axis=1)
    ax[1,1].plot(z['time_h'],error,color=color,ls=ls,label=label)
    if key=='omit_upper':
        ax[1,1].plot(z['time_h'][-1],error[-1],'x',color=color,ms=7)
        ax[1,1].annotate('Interior guard',xy=(z['time_h'][-1],error[-1]),xytext=(4.2,.22),fontsize=8,arrowprops={'arrowstyle':'-','color':color,'lw':.7})
ax[1,1].set(xlabel='Synthetic time (h)',ylabel='Absolute flux error',xlim=(0,12))
ax[1,1].set_title('(d)  Deliberately incorrect-term diagnostics',loc='left',fontsize=10)
ax[1,1].legend(frameon=False,fontsize=8,loc='upper right',bbox_to_anchor=(1,.75))
for a in ax.flat:a.grid(which='major',color='#dddddd',lw=.5,alpha=.65)
finish(fig,'moving_bound_refinement','Numerical checks for the two-flux verification problem: (a) central differences of the closed-form solution with respect to each bound; (b) objective gap under barrier refinement with unchanged bounds; (c) integration-tolerance refinement at fixed barrier; (d) deliberate omission of one moving-bound term. The upper-term diagnostic terminates at the stated positive-slack guard; the diagnostic formulas are not alternative solution methods.')
(OUT/'manifest.json').write_text(json.dumps({'validation_sha256':sha256((RES/'validation.json').read_bytes()).hexdigest(),'renderer_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'figures':assets,'quantitative_source':'results/*.csv and results/*.npz; no generated or digitised numeric data.'},indent=2)+'\n')
print('Rendered two figures as PNG, SVG and PDF with colour and greyscale previews.')
