"""Render the recorded feedback study without rerunning or altering calculations.

Use Python 3.11 with NumPy and Matplotlib. Inputs are read only. All plotted
values are checked against the individual episode records before export.
"""
from pathlib import Path
import hashlib
import json
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np
from PIL import Image, ImageOps

SOURCE = Path(__file__).resolve().parents[1]
ROOT = SOURCE
OUT = SOURCE / 'local_results/figures'
OUT.mkdir(parents=True, exist_ok=True)
METHODS = ['open_loop', 'nominal_mpc', 'adaptive_direct_mpc',
           'adaptive_limited_direct', 'adaptive_ungated', 'adaptive_verified']
REGIMES = ['nominal', 'constant_mismatch', 'step_mismatch', 'noisy_mismatch']
LABELS = ['Open loop', 'Nominal feedback', 'Adaptive direct',
          'Limited direct', 'Unchecked local', 'Checked local']
REGIME_LABELS = ['Nominal', 'Fixed mismatch', 'Parameter shift', 'Mismatch + noise']
COLOURS = ['#747474', '#B47743', '#365D81', '#835E91', '#4E856F', '#202020']
STYLES = [(0, (2, 2)), (0, (5, 2)), '-', (0, (5, 2, 1, 2)),
          (0, (7, 3)), (0, (4, 1, 1, 1, 1, 1))]
MARKERS = ['o', 's', '^', 'D']
REGIME_COLOURS = ['#365D81', '#B47743', '#4E856F', '#835E91']
load = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
summary = load(SOURCE / 'results/summary.json')
protocol = load(SOURCE / 'protocol.json')
episodes = {(c['id'], m): load(SOURCE / 'results/frozen_evaluation' / f'{c["id"]}__{m}.json')
            for c in protocol['cases'] for m in METHODS}
assert len(episodes) == 144
assert all(d['status'] == 'completed' for d in episodes.values())
inputs = [SOURCE / 'results/summary.json', SOURCE / 'protocol.json',
          SOURCE / 'scripts/plot_results.py', SOURCE / 'scripts/model.py',
          SOURCE / 'scripts/summarize.py']
inputs += sorted((SOURCE / 'results/frozen_evaluation').glob('cluster_*__*.json'))
input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}

# Reconcile every aggregate displayed in the figures with all 144 source runs.
counts = np.zeros((6, 4), dtype=int)
totals = np.zeros(6)
for i, method in enumerate(METHODS):
    group = [d for (c, m), d in episodes.items() if m == method]
    totals[i] = sum(d['controller_seconds'] for d in group)
    assert np.isclose(totals[i], summary['methods'][method]['controller_seconds']['sum'],
                      rtol=1e-13, atol=1e-10)
    for j, regime in enumerate(REGIMES):
        counts[i, j] = sum(d['result']['terminal_feasible'] for d in group
                           if d['case']['regime'] == regime)
        assert counts[i, j] == summary['regimes'][regime][method]['feasible']
for method in METHODS[3:]:
    pair = summary['paired'][method + '__vs__adaptive_direct_mpc']
    expected_cases = {c['id'] for c in protocol['cases']
                      if episodes[c['id'], method]['result']['terminal_feasible']
                      and episodes[c['id'], 'adaptive_direct_mpc']['result']['terminal_feasible']}
    assert {p['case'] for p in pair['pairs']} == expected_cases
    for p in pair['pairs']:
        a, b = episodes[p['case'], method], episodes[p['case'], 'adaptive_direct_mpc']
        assert p['time_ratio'] == a['controller_seconds'] / b['controller_seconds']
        assert p['biomass_difference_percent'] == 100 * (
            a['result']['biomass_amount_gDW'] / b['result']['biomass_amount_gDW'] - 1)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'mathtext.fontset': 'stix', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.linewidth': .7, 'grid.linewidth': .4,
    'savefig.facecolor': 'white', 'lines.dash_capstyle': 'butt',
})
manifest = []
plot_data = {'acceptance_counts': counts, 'controller_seconds_total': totals}


def save(fig, number, name, short, caption, sources, notes):
    files = {}
    for ext in ['png', 'pdf', 'svg']:
        p = OUT / f'{name}.{ext}'
        kwargs = {'bbox_inches': 'tight', 'pad_inches': .06, 'dpi': 400}
        if ext == 'pdf':
            kwargs['metadata'] = {'Title': short, 'Author': '',
                                  'Subject': 'Recorded feedback-control simulation results'}
        fig.savefig(p, **kwargs)
        files[ext] = {'sha256': sha(p), 'bytes': p.stat().st_size}
    plt.close(fig)
    with Image.open(OUT / f'{name}.png') as im:
        preview = im.copy()
        preview.thumbnail((1500, 1200))
        preview.save(OUT / f'{name}_preview.png')
        ImageOps.grayscale(preview).save(OUT / f'{name}_greyscale.png')
    manifest.append({'number': number, 'semantic_id': name,
                     'png': f'figures/{name}.png', 'pdf': f'figures/{name}.pdf',
                     'svg': f'figures/{name}.svg', 'short_title': short,
                     'caption_markdown': caption, 'files': files,
                     'source_patterns': sources, 'presentation_notes': notes})


# Cluster 0 is the same illustrative cluster used by the source figure.
# Every stored adaptive time node and every applied piecewise feed is retained.
fig, axes = plt.subplots(2, 4, figsize=(7.15, 4.45), sharex=True,
                         gridspec_kw={'height_ratios': [1.12, 1]})
for col, (regime, label) in enumerate(zip(REGIMES, REGIME_LABELS)):
    for i, method in enumerate(METHODS):
        d = episodes[f'cluster_0_{regime}', method]
        tt, zz = [], []
        for stage in d['stages']:
            ys = np.asarray(stage['plant_trajectory'])
            ts = np.asarray(stage['plant_times_local']) + stage['time_h']
            tt.extend(ts.tolist())
            zz.extend((ys[:, 1] - ys[:, 2] - np.log(d['case']['target_ratio'])).tolist())
        tt, zz = np.asarray(tt), np.asarray(zz)
        feed_time = np.arange(8) * 2.
        feed = np.r_[d['controls'], d['controls'][-1]]
        plot_data[f'{regime}__{method}__time'] = tt
        plot_data[f'{regime}__{method}__log_ratio_deviation'] = zz
        plot_data[f'{regime}__{method}__feed_time'] = feed_time
        plot_data[f'{regime}__{method}__feed'] = feed
        axes[0, col].plot(tt, zz, color=COLOURS[i], ls=STYLES[i], lw=1.25,
                          label=LABELS[i], zorder=3 + i / 10)
        axes[1, col].step(feed_time, feed, where='post', color=COLOURS[i],
                          ls=STYLES[i], lw=1.25, zorder=3 + i / 10)
    axes[0, col].axhspan(-np.log(1.01), np.log(1.01), color='#C5CDD0', alpha=.5, zorder=0)
    axes[0, col].set_title(f'({chr(97 + col)}) {label}', loc='left', fontsize=8.3)
    axes[0, col].set_ylim(-.25, 1.2)
    axes[0, col].set_yticks([-.2, 0, .4, .8, 1.2])
    axes[1, col].set_ylim(-.006, .212)
    axes[1, col].set_yticks([0, .1, .2])
    for ax in axes[:, col]:
        ax.set_xlim(0, 14)
        ax.set_xticks([0, 7, 14])
        ax.grid(axis='y', alpha=.15, zorder=0)
        if col:
            ax.tick_params(labelleft=False)
    axes[1, col].set_xlabel('Time (h)')
axes[0, 0].set_ylabel('Log-ratio deviation\nfrom terminal target')
axes[1, 0].set_ylabel(r'Applied feed (L h$^{-1}$)')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
           bbox_to_anchor=(.52, -.008), handlelength=3.1, columnspacing=1.5)
fig.subplots_adjust(left=.105, right=.992, bottom=.22, top=.93, hspace=.18, wspace=.12)
save(fig, 64, 'mismatch_trajectories', 'Feedback trajectories under kinetic mismatch',
     'Simulated composition and applied feeds for condition cluster 0 under (a) nominal dynamics, '
     '(b) constant affinity mismatch, (c) a parameter change at 6 h and (d) constant mismatch with observation noise. '
     'Upper panels show the deviation of log biomass ratio from its terminal target; lower panels show the recorded feed commands. '
     'The shaded band denotes the terminal composition tolerance, not a path constraint. '
     'All six controllers use the same initial condition and target within each regime; coincident curves may overlap.',
     ['results/frozen_evaluation/cluster_0_*__*.json', 'protocol.json'],
     ['Same cluster and all exact adaptive nodes as source figure; no interpolation or smoothing.',
      'Common vertical scales across regimes; colour and distinct line patterns identify controllers.',
      'A four-column layout preserves the temporal correspondence of each composition/feed pair.'])

# Shared controller ordering links the heatmap and complete-cost bars directly.
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.35),
                         gridspec_kw={'width_ratios': [1.35, 1]})
cmap = LinearSegmentedColormap.from_list('thesis_acceptance', ['#F3F6F8', '#B3CBD9', '#365D81'])
axes[0].imshow(counts, vmin=0, vmax=6, cmap=cmap, aspect='auto')
for i in range(6):
    for j in range(4):
        axes[0].text(j, i, f'{counts[i, j]}/6', ha='center', va='center', fontsize=9,
                     color='white' if counts[i, j] >= 5 else '#202020')
axes[0].set_yticks(range(6), LABELS)
axes[0].set_xticks(range(4), ['Nominal', 'Fixed\nmismatch', 'Parameter\nshift', 'Mismatch\n+ noise'])
axes[0].set_title('(a) Actual terminal acceptance', loc='left')
axes[0].tick_params(length=0, pad=5)
axes[0].spines['left'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)
axes[1].barh(range(6), totals, height=.66, color=COLOURS,
             edgecolor='#333333', linewidth=.35)
axes[1].set_ylim(5.5, -.5)
axes[1].set_yticks(range(6), [''] * 6)
axes[1].tick_params(axis='y', length=0)
axes[1].spines['left'].set_visible(False)
axes[1].set_xlim(0, 760)
axes[1].set_xticks([0, 200, 400, 600])
axes[1].set_xlabel('Total controller time (s)')
axes[1].set_title('(b) Complete online cost', loc='left')
axes[1].grid(axis='x', alpha=.15)
axes[1].set_axisbelow(True)
for i, value in enumerate(totals):
    axes[1].text(value + 10, i, f'{value:.3g}', va='center', fontsize=8)
fig.subplots_adjust(left=.19, right=.985, bottom=.2, top=.89, wspace=.16)
save(fig, 65, 'mismatch_feasibility_cost', 'Terminal acceptance and online controller cost',
     'Terminal acceptance and complete online calculation time for six feed-control methods. '
     '(a) Accepted outcomes among the six condition clusters in each prescribed regime. '
     '(b) Total controller time over all 24 runs per method, including estimation, sensitivity calculations, '
     'local proposals, checking, correction, fallback and unsuccessful searches. Common response-table preparation, '
     'simulated-process integration and subsequent audits are excluded. The four regimes reuse the same six clusters.',
     ['results/frozen_evaluation/cluster_*__*.json', 'results/summary.json'],
     ['All 144 episodes are included; counts and exact elapsed times independently reconciled with source episodes.',
      'Counts are printed in each heatmap cell, so colour is not needed for interpretation.',
      'Time values are displayed to three significant figures; underlying values remain unrounded.'])

fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.4), sharey=True)
for k, (ax, method, title) in enumerate(zip(axes, METHODS[3:], LABELS[3:])):
    pairs = summary['paired'][method + '__vs__adaptive_direct_mpc']['pairs']
    for regime, marker, colour, label in zip(REGIMES, MARKERS, REGIME_COLOURS, REGIME_LABELS):
        subset = [p for p in pairs if p['regime'] == regime]
        xx = np.array([p['time_ratio'] for p in subset])
        yy = np.array([p['biomass_difference_percent'] for p in subset])
        plot_data[f'{method}__{regime}__paired_cost'] = xx
        plot_data[f'{method}__{regime}__paired_biomass'] = yy
        ax.scatter(xx, yy, marker=marker, facecolors='none', edgecolors=colour,
                   s=29, linewidths=1.15, label=label, zorder=3)
    ax.axvline(1, color='#686868', ls=':', lw=.95)
    ax.axhline(-.5, color='#686868', ls='--', lw=.95)
    ax.axhline(0, color='#D7D7D7', lw=.5, zorder=0)
    ax.set_xscale('log')
    ax.set_xlim(.03, 3)
    ax.set_ylim(-7.6, 3.6)
    ax.set_yticks([-6, -4, -2, 0, 2])
    ax.set_title(f'({chr(97 + k)}) {title}', loc='left')
    ax.set_xlabel('Controller time ratio')
    ax.xaxis.set_major_locator(FixedLocator([.03, .1, .3, 1, 3]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis='y', alpha=.13, zorder=0)
axes[0].set_ylabel('Terminal biomass difference\nrelative to adaptive direct (%)')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=4, loc='lower center', frameon=False,
           bbox_to_anchor=(.51, -.008), fontsize=7.5, columnspacing=1.2)
fig.subplots_adjust(left=.105, right=.985, bottom=.25, top=.9, wspace=.19)
save(fig, 66, 'mismatch_paired_quality_cost', 'Paired biomass and computation comparisons',
     'Paired terminal-biomass and controller-time comparisons with full adaptive direct control: '
     '(a) limited direct, (b) unchecked local and (c) checked local control. Each panel contains the 17 cases '
     'in which both methods satisfy the final acceptance conditions. The horizontal dashed line marks the '
     'prespecified 0.5% biomass-loss threshold; the vertical dotted line marks equal computation time. '
     'Symbols distinguish the four regimes. All panels use identical axes, with a logarithmic time ratio; '
     'unsuccessful cases remain included in the complete acceptance counts.',
     ['results/frozen_evaluation/cluster_*__*.json', 'results/summary.json'],
     ['Every jointly feasible pair is included; pair identities and exact values reconciled with episode records.',
      'Markers distinguish regimes in greyscale. No jitter, smoothing or selective removal is applied.'])

np.savez(OUT / 'plotted_values.npz', **plot_data)
assert input_hashes == {str(p.relative_to(ROOT)): sha(p) for p in inputs}
(OUT / 'manifest.json').write_text(json.dumps({
    'figures': manifest, 'selection': 'Retain all three: trajectories, complete outcomes/cost, and paired quality/cost answer different questions.',
    'additional_workflow': 'No further workflow is proposed: the control-method prose describes the feedback loop and another schematic would add length without replacing quantitative evidence.',
    'input_sha256': input_hashes, 'plotted_values_sha256': sha(OUT / 'plotted_values.npz'),
    'render_script_sha256': sha(Path(__file__)),
    'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
    'data_audit': {'episodes': len(episodes), 'all_completed': True,
                   'aggregate_values_reconciled': True, 'jointly_feasible_pairs_reconciled': True,
                   'source_files_unchanged': True, 'source_trajectory_nodes_preserved': True}
}, indent=2) + '\n')
print('Rendered and reconciled three result figures from 144 immutable episode records.')
