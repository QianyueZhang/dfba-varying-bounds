# Terminal recovery under sensing and actuation limits

This study calculates admissible terminal-composition targets and constructive feed decisions for a finite-volume coculture. Six prescribed states, six target ratios, two two-hour control intervals and nine affinity paths define the common comparison panel. One model uses paired dFBA growth/uptake response tables; the auxiliary model uses constant-yield Monod kinetics. They are mathematical process models, not two independently calibrated biological experiments.

## Inspect the evidence

```sh
python experiments/sensing_actuation/run.py verify
```

This read-only scientific check verifies distributed bytes, numerical-source locks, exact outcome arithmetic, all 72 continuous method/problem assignments and all 141,537 continuous candidate records. It writes local reports under ignored `local_results/`. It does not launch the full numerical campaign or turn a retained failed scientific audit into a pass.

At 161 raw feed actions, full-state feedback meets the inner band in 9/36 fixed problems and committed open-loop control in 0/36. Ratio feedback gives 9, 9, 8, 3 and 0 inner-band successes at measurement-error allowances 0, 0.001, 0.005, 0.01 and 0.02. A shared continuation action must serve all states compatible with the measured ratio. The controller is never supplied a hidden disturbance label.

The continuous comparison uses the same physical problem and fixed targets, but its causal controller observes the full state. Independently executed inner-band counts are 1/36 open-loop and 9/36 causal; operational-band counts are 2/36 and 12/36. These are best-found candidates from the recorded searches, not globally certified optima or an equal-budget speed comparison.

Independent amount integration preserves all 756 primary fixed-policy classifications. All 99 selected component midpoints meet the inner band. Exact endpoint classifications remain qualified: 63 predicted-feasible endpoints fail the strict inner band by at most 2.23e-10, although all selected component cases pass the wider bands. The completed five-state adaptive panel similarly retains 53 strict positive-to-failure flips. Both overall strict endpoint audits remain failed.

The 81-to-161 stability comparison has no nonvacuous pass. Five of six states complete the budgeted 321-action study; the tabulated 0.9 L state remains unresolved. Its partial cache is not interpreted as a completed target set. Mixed-finest summaries retain their mixed-grid denominator and labels.

## Contents

- `data/`: prescribed states and paired-rate table, sufficient for these process calculations.
- `scripts/`, `vendor/`, `audit/`: numerical models, finite-game algorithms, continuous searches and independently assembled amount checks.
- `protocols/`: decision rules, thresholds, seeds, budgets and recorded source identities.
- `results/`: complete fixed-target results, target sets, solver statuses and every continuous candidate ledger; `audit/` also retains all assigned deployment episodes and failed endpoint classifications.
- `figures/`: scientific plots drawn from those results.
- `SOURCE_MANIFEST.json`: the distributed study content and its hashes.
- `OPTIONAL_CACHE_MANIFEST.json`: hashes and sizes of omitted transition caches. These are regenerable acceleration caches, not omitted case outcomes. Full saved caches remain in the private research record; public users can generate them from the supplied inputs and scripts.

The 119.8 MB of optional NPZ transition caches are excluded to reduce download size. All continuous candidate archives (20.9 MB), independent path records and aggregate/per-grid outcomes remain included. No thesis/manuscript drafts, reconstruction matrices, installed dependencies or executable model serialisations are included.

## Run calculations

Install the separately pinned study requirements before numerical execution:

```sh
python -m pip install -r experiments/sensing_actuation/requirements.txt
python experiments/sensing_actuation/run.py numerical-smoke
python experiments/sensing_actuation/run.py prepare /path/to/new/control-study
```

The numerical installation check compares two model-family intervals with independent amount integration. The preparation command refuses an existing destination and copies source and scientific inputs without outcomes. Follow the ordered commands in the generated `RUN_COMMANDS.json`, using one BLAS/OpenMP thread. The independent integrator preflight must pass before the grid or continuous campaigns. Full searches are substantial calculations and may retain budget-limited states. Do not change thresholds to erase a failed numerical or feasibility check.

The supplied preparation helper retains its frozen protocol-interface names. New calculations bind their own inputs and actual execution metadata. Identical floating-point endpoint labels, candidate ordering and timing are not guaranteed across environments.

## Scope and licence

The rate-response table is author-generated numerical input from the interior-point dFBA model. It is not experimental data and does not relicense the underlying metabolic reconstruction. Author-owned code and numerical records use MIT within the root licence scope; dependencies retain their own terms. Fixed targets, selected feasible-set points and observation/disturbance episodes are distinct denominators. The results quantify conditional sensing/actuation effects; they do not validate a noisy neural controller or a biological intervention.
