# Terminal recovery and admissible composition targets

This study asks which terminal species ratios remain attainable from a given culture state when feed rate, vessel capacity and future affinity disturbances are constrained. It distinguishes states with equal volume and total past feed but different feeding histories. The calculation uses a finite two-step feedback game with exact state observations, a finite disturbance set and a stated feed-action grid.

The state is `[V, log(M_SC), log(M_KP), Q]`: volume in L, logarithms of species biomass amounts relative to 1 gDW, and glucose amount in g. Each interval lasts 2 h, feeds lie between 0 and 0.2 L/h, and vessel capacity is 1.8 L. The paired response table and separate analytic rate family are both supplied. Equations, numerical tolerances and all method definitions are specified in `PROTOCOL.json` and the executable model sources.

## Evidence and interpretation

The fixed primary panel contains 18 matched-history states, six target ratios, five methods and nine disturbance paths: **4,860 completed episodes**. Under the 21-action grid, no method satisfies the inner band on every path for any of the 108 fixed root–target problems. Under the wider actual acceptance band, four methods satisfy all paths in two problems; nominal replanning does so in none. These complete denominators are retained.

An explicitly exploratory analysis calculates admissible target intervals. At 41 actions, the six uniform-history states yield 161 selected interval midpoints. All 1,449 independently integrated paths through those midpoints satisfy the specified conditions. This verifies selected witnesses; it is not an unbiased efficacy rate over arbitrary targets. The separate sensitivity check of the original 36 fixed targets gives one all-path inner-band success and five actual-band successes. Continuous open-loop local optimisation provides a separate baseline and sometimes outperforms finite-grid feedback. No universal superiority or convergence claim follows.

The initial deployment audit retains ten comparison flags (seven prefix comparisons and three matched intervals). Subsequent diagnostics examine them without deleting the failed audit or relaxing its thresholds. The refined-target and continuous-minimum checks pass their own declared criteria. Finite actions, finite disturbances, exact sensing and numerical integration remain explicit assumptions; the study does not prove biological validity or continuous-control infeasibility.

## Files

- `scripts/engine.py`, `policies.py`, `campaign.py`: dynamics, finite-action recursion, and the fixed evaluation panel.
- `scripts/target_sets.py`: unions/intersections of admissible log-ratio intervals and selected witnesses.
- `scripts/continuous_baseline.py`, `run_continuous.py`: the continuous open-loop local baseline.
- `audit/`: independent physical-amount integrator, protocols, complete deployed episodes and numerical diagnostics.
- `results/`: all finite trees, target sets, fixed-target outcomes and continuous-baseline records.
- `figures/`: matched-history, grid-sensitivity and witness figures. Plotting scripts retain all assigned cases.
- `SOURCE_MANIFEST.json`: source/release identities and the limited portable-path/check-label changes. Numerical values, original execution identities and audit findings are preserved.

## Commands

From the repository root, verify the supplied records without running an optimisation:

```sh
python experiments/terminal_recovery/run.py verify
```

For numerical work, install this study's pinned Numba/llvmlite dependencies in a Python 3.11 environment, then run two short model-family checks:

```sh
python -m pip install -r experiments/terminal_recovery/requirements.txt
python experiments/terminal_recovery/run.py numerical-smoke
```

Prepare a new directory for a full calculation:

```sh
python experiments/terminal_recovery/run.py prepare /path/to/new/recovery_run
```

Preparation copies code and input definitions, verifies their hashes, and copies no outcomes. It refuses an existing destination. `RUN_COMMANDS.json` lists the scientific commands in order; run them from that new directory using one BLAS/OpenMP thread. No long calculation is started automatically. Some audits deliberately retain failure records; inspect those records rather than altering tolerances to obtain a pass. Supplied results remain unchanged.

The short numerical check compares one 2 h feed interval in each model family against independently coded physical-amount equations. It does not repeat the complete 4,860-episode campaign. The complete command sequence is provided for longer work; platform-dependent numerical borderline decisions and execution times must be reported under the fresh run's own identity.

## Scientific context

Finite-horizon dynamic programming supplies the min–max recursion; the contribution evaluated here is its specified use for culture-state-dependent terminal composition and feed-proposal checking. Related reference-governor and safety-filter methods include Garone, Di Cairano and Kolmanovsky (2017), DOI [10.1016/j.automatica.2016.08.013](https://doi.org/10.1016/j.automatica.2016.08.013), and Wabersich and Zeilinger (2021), DOI [10.1016/j.automatica.2021.109597](https://doi.org/10.1016/j.automatica.2021.109597). The dFBA formulation builds on Scott et al. (2018), DOI [10.1016/j.compchemeng.2018.08.041](https://doi.org/10.1016/j.compchemeng.2018.08.041). Full bibliographic identities and further control sources are in [control_references.bib](../../docs/control_references.bib). The accompanying scenario study is in [`../scenario_feedback/`](../scenario_feedback/); its independent outcomes are not relabelled as terminal-recovery evidence.
