"""Plot adopted model trajectories and independently sample their KKT fluxes.

This script never integrates a trajectory. All fluxes are evaluated at saved
states, using the same finite-barrier model as the adopted calculations. Full
flux and bound vectors, selected diagnostics, and source hashes are retained.
"""
from pathlib import Path
import csv
import hashlib
import json
import platform
import sys
import time

import numpy as np
import scipy

from flux_model import FluxModel, ROOT
from primal_dual import solve
from cstr_experiments import CSTRModel

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = ROOT / 'research_completion/runs/model_plot_revision'
FIG = ROOT / 'research_completion/figures'
OUT.mkdir(parents=True,exist_ok=True)
FIG.mkdir(parents=True,exist_ok=True)
MU, TOL = 1e-6, 1e-11


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def light(t):
    return np.maximum(0, (np.sin(np.pi * t / 24) ** 2 - np.sin(5 * np.pi / 24) ** 2)
                      / (1 - np.sin(5 * np.pi / 24) ** 2))


def load_unique(path):
    a = np.loadtxt(path, delimiter=',', skiprows=1)
    _, idx = np.unique(a[:, 0], return_index=True)
    return a[np.sort(idx)]


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample(model, trajectory, *, cstr):
    rows, vectors, lower, upper, diagnostics = [], [], [], [], []
    warm = None
    cold_retries = 0
    for i, r in enumerate(trajectory):
        t = float(r[0])
        state = (t, r[1:]) if cstr else float(r[-1])
        try:
            out = solve(model, state, mu=MU, tol=TOL, warm=warm)
        except RuntimeError:
            cold_retries += 1
            out = solve(model, state, mu=MU, tol=TOL)
        warm = out[:4]
        v, lam, zl, zu, diag = out
        lo, hi = model.bounds(state)
        assert np.isfinite(v).all() and np.isfinite(lo).all() and np.isfinite(hi).all()
        assert np.min(v-lo) > 0 and np.min(hi-v) > 0
        assert np.min(zl) > 0 and np.min(zu) > 0
        assert diag['equality_residual'] < TOL and diag['primal_dual_residual'] < TOL
        row = {'time_h': t, 'growth_h-1': float(v[model.bio])}
        exchanges = model.exchange if cstr else {'glucose': model.g}
        for name, j in exchanges.items():
            row[f'{name}_flux'] = float(v[j])
            row[f'{name}_lower'] = float(lo[j])
            row[f'{name}_upper'] = float(hi[j])
        if cstr:
            row['light_fraction'] = float(light(t))
            if model.name == 'RC':
                row['net_oxygen_flux'] = row['oxygen_flux'] + row['photosynthetic_oxygen_flux']
                solar = model.photon[0]
                row['solar_flux'] = float(v[solar])
                row['solar_lower'] = float(lo[solar])
                row['solar_upper'] = float(hi[solar])
        else:
            row['glucose_g_L'] = float(state)
            row['depletion_factor'] = float(min(1., max(0., state)/1e-5))
            row['effective_glucose_flux'] = row['glucose_flux']*row['depletion_factor']
            row['effective_growth_h-1'] = row['growth_h-1']*row['depletion_factor']
        # The generic solver calls its selected exchange 'glucose_flux'; in
        # the CSTR algal model that selected exchange is acetate, so retain
        # reaction-specific names above and add diagnostics without that alias.
        row.update({k: value for k, value in diag.items() if k not in ['growth', 'glucose_flux']})
        rows.append(row)
        vectors.append(v)
        lower.append(lo)
        upper.append(hi)
        diagnostics.append(diag)
        if (i+1) % 50 == 0:
            print(f'{"CSTR" if cstr else "isolated"} {model.name}: {i+1}/{len(trajectory)} samples', flush=True)
    tag = f'{"cstr" if cstr else "isolated"}_{model.name.lower()}'
    write_csv(OUT/f'{tag}_flux_bounds.csv', rows)
    np.savez_compressed(OUT/f'{tag}_full_flux_bounds.npz', time_h=trajectory[:,0],
                        states=trajectory[:,1:], flux=np.array(vectors),
                        lower=np.array(lower), upper=np.array(upper))
    summary = {
        'samples': len(rows), 'cold_retries': cold_retries,
        'model_shape': list(model.A.shape),
        'exchange_indices_1based': {k: int(v)+1 for k,v in exchanges.items()},
        'max_equality_residual': max(d['equality_residual'] for d in diagnostics),
        'max_primal_dual_residual': max(d['primal_dual_residual'] for d in diagnostics),
        'max_explicit_barrier_stationarity_residual': max(d['stationarity_residual'] for d in diagnostics),
        'minimum_slack': min(d['minimum_slack'] for d in diagnostics),
        'all_bounds_satisfied': True,
        'selected_flux_ranges': {
            k: [min(r[k] for r in rows), max(r[k] for r in rows)]
            for k in rows[0] if k.endswith('_flux')
        },
        'max_lower_bound_distance': {
            k: max(r[f'{k}_flux']-r[f'{k}_lower'] for r in rows)
            for k in exchanges
        },
    }
    return rows, summary


BLUE, ORANGE, GREEN, PURPLE = '#17629A', '#C16920', '#267A50', '#8856A7'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10.5,
    'axes.labelsize': 10.5, 'legend.fontsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .7, 'lines.linewidth': 1.8,
    'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
    'savefig.facecolor': 'white',
})


def finish(fig, axes, key, xmax):
    for i, ax in enumerate(np.ravel(axes)):
        ax.set_xlim(0, xmax)
        ax.grid(axis='y', alpha=.18, linewidth=.6)
        ax.text(-.13, 1.02, f'({chr(97+i)})', transform=ax.transAxes,
                fontweight='bold', va='bottom', fontsize=10)
    np.ravel(axes)[-1].set_xlabel('Time (h)')
    fig.savefig(FIG/f'{key}.png', dpi=320, bbox_inches='tight')
    fig.savefig(FIG/f'{key}.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_states(trajectory):
    t, y = trajectory[:,0], trajectory[:,1:]
    fig, ax = plt.subplots(5, 1, figsize=(6.45,8.9), sharex=True,
                           gridspec_kw={'height_ratios':[.75,1,1.2,1,1]},
                           layout='constrained')
    ax[0].plot(t,light(t),c=GREEN)
    ax[0].fill_between(t,0,light(t),color=GREEN,alpha=.12)
    ax[0].set_ylabel('Available light\n(fraction)')
    ax[0].set_ylim(-.03,1.05)
    ax[1].plot(t,y[:,1],c=BLUE,label=r'$S.\ cerevisiae$')
    ax[1].plot(t,y[:,2],c=ORANGE,label=r'$C.\ reinhardtii$')
    ax[1].set_ylabel('Biomass\n(gDW L$^{-1}$)')
    ax[1].legend(loc='best',ncols=2)
    ax[2].plot(t,y[:,3],c=BLUE,label='Glucose')
    ax[2].plot(t,y[:,4],c=ORANGE,label='Acetate')
    ax[2].set_yscale('log')
    ax[2].set_ylabel('Carbon substrates\n(mmol L$^{-1}$; log scale)')
    ax[2].legend(loc='best',ncols=2)
    ax[3].plot(t,y[:,5],c=BLUE,label='Dissolved oxygen')
    ax[3].plot(t,y[:,6],c=ORANGE,label='Dissolved carbon dioxide')
    ax[3].set_ylabel('Dissolved gases\n(mmol L$^{-1}$)')
    ax[3].legend(loc='best',ncols=2)
    ax[4].plot(t,y[:,7],c=PURPLE,label='Ethanol')
    ax[4].set_ylabel('Ethanol\n(mmol L$^{-1}$)')
    ax[4].set_xticks([0,5,12,19,24])
    finish(fig,ax,'revision_cstr_states_light',24)


def col(rows, name):
    return np.array([r[name] for r in rows])


def bound_panel(ax, rows, name, title, *, clipped_upper=False):
    t=col(rows,'time_h');v=col(rows,f'{name}_flux')
    lo=col(rows,f'{name}_lower');hi=col(rows,f'{name}_upper')
    ax.plot(t,lo,c=ORANGE,ls='--',label='Lower bound',lw=1.4)
    if not clipped_upper:
        ax.plot(t,hi,c=GREEN,ls=':',label='Upper bound',lw=1.5)
        ax.fill_between(t,lo,hi,color=BLUE,alpha=.075,label='Admissible interval')
    else:
        ymin=min(lo.min(),v.min())
        ymax=max(v.max(),0.)
        span=max(ymax-ymin,.05)
        ax.set_ylim(ymin-.08*span,ymax+.16*span)
        ax.fill_between(t,lo,ymax+.16*span,color=BLUE,alpha=.075)
        ax.text(.98,.94,r'Upper bound: $1000+\varepsilon$ (off-axis)',
                transform=ax.transAxes,ha='right',va='top',fontsize=8.7)
    ax.plot(t,v,c=BLUE,label='KKT flux',lw=1.9)
    ax.set_title(title,loc='left',fontsize=10.5,pad=5)
    ax.set_ylabel('Flux\n(mmol gDW$^{-1}$ h$^{-1}$)')


def plot_sc_fluxes(sc):
    fig,ax=plt.subplots(3,1,figsize=(6.45,6.7),sharex=True,layout='constrained')
    bound_panel(ax[0],sc,'glucose',r'$S.\ cerevisiae$: glucose exchange')
    bound_panel(ax[1],sc,'oxygen',r'$S.\ cerevisiae$: ordinary oxygen exchange')
    handles,labels=ax[0].get_legend_handles_labels()
    ax[0].legend(handles,labels,loc='best',ncols=2)
    ax[2].plot(col(sc,'time_h'),col(sc,'oxygen_flux'),c=BLUE,label='Same oxygen flux as (b)')
    ax[2].set_title('Ordinary oxygen exchange: expanded flux scale',loc='left',fontsize=10.5,pad=5)
    ax[2].set_ylabel('Flux\n(mmol gDW$^{-1}$ h$^{-1}$)')
    ax[2].ticklabel_format(axis='y',style='sci',scilimits=(-3,-3),useMathText=True)
    ax[2].legend(loc='best')
    ax[2].set_xticks([0,5,12,19,24])
    finish(fig,ax,'revision_cstr_sc_flux_bounds',24)


def plot_fluxes(sc, rc, isolated):
    plot_sc_fluxes(sc)
    fig,ax=plt.subplots(4,1,figsize=(6.45,8.6),sharex=True,layout='constrained')
    bound_panel(ax[0],rc,'acetate',r'$C.\ reinhardtii$: acetate exchange')
    ax[0].legend(loc='best',ncols=2)
    bound_panel(ax[1],rc,'carbon_dioxide',r'$C.\ reinhardtii$: carbon-dioxide exchange',clipped_upper=True)
    bound_panel(ax[2],rc,'oxygen',r'$C.\ reinhardtii$: ordinary oxygen exchange')
    t=col(rc,'time_h')
    ax[3].plot(t,col(rc,'photosynthetic_oxygen_flux'),c=GREEN,label='Photosynthetic oxygen')
    ax[3].plot(t,col(rc,'net_oxygen_flux'),c=PURPLE,ls='--',label='Net algal oxygen')
    ax[3].axhline(0,color='grey',lw=.7)
    ax[3].set_title('Photosynthetic and net oxygen contributions',loc='left',fontsize=10.5,pad=5)
    ax[3].set_ylabel('Flux\n(mmol gDW$^{-1}$ h$^{-1}$)')
    ax[3].legend(loc='best',ncols=2)
    ax[3].set_xticks([0,5,12,19,24])
    finish(fig,ax,'revision_cstr_rc_flux_bounds',24)

    fig,ax=plt.subplots(2,1,figsize=(6.45,5),sharex=True,layout='constrained')
    for a,(name,rows) in zip(ax,isolated.items()):
        title=r'$S.\ cerevisiae$: glucose exchange' if name=='SC' else r'$K.\ phaffii$: glucose exchange'
        bound_panel(a,rows,'glucose',title)
        if np.any(col(rows,'depletion_factor')<1):
            a.plot(col(rows,'time_h'),col(rows,'effective_glucose_flux'),c=PURPLE,ls='-.',
                   label='Effective flux at depletion',lw=1.4)
        a.legend(loc='best',ncols=2)
    finish(fig,ax,'revision_isolated_glucose_bounds',12)


def plot_isolated(isolated):
    fig,ax=plt.subplots(2,1,figsize=(6.45,5),sharex=True,layout='constrained')
    for a,(name,rows) in zip(ax,isolated.items()):
        title=r'$S.\ cerevisiae$: glucose exchange' if name=='SC' else r'$K.\ phaffii$: glucose exchange'
        bound_panel(a,rows,'glucose',title)
        if np.any(col(rows,'depletion_factor')<1):
            a.plot(col(rows,'time_h'),col(rows,'effective_glucose_flux'),c=PURPLE,ls='-.',
                   label='Effective flux at depletion',lw=1.4)
        a.legend(loc='best',ncols=2)
    finish(fig,ax,'revision_isolated_glucose_bounds',12)


def main():
    isolated,diagnostics={},{}
    for i,name in enumerate(['SC','KP']):
        trajectory=load_unique(ROOT/'research_completion/runs'/f'batch_{i}.csv')
        isolated[name],diagnostics[name]=sample(FluxModel(name,1e-6),trajectory,cstr=False)
    plot_isolated(isolated)
    (OUT/'isolated_flux_audit.json').write_text(json.dumps(diagnostics,indent=2))

if __name__=='__main__':main()
