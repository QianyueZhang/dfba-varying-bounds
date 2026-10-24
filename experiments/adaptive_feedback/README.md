# Adaptive feed control under model mismatch

This study assesses feedback control of a glucose-fed *Saccharomyces cerevisiae*–*Komagataella phaffii* culture. Six initial-condition/target combinations are each evaluated under nominal dynamics, constant affinity mismatch, a mid-cultivation affinity change and observation noise. Six control methods give 144 complete episodes and 1008 control intervals. The reactor lasts 14 h, with seven 2 h constant-feed decisions.

The objective is terminal biomass amount subject to composition, feed and volume constraints. An online affine approximation of terminal log ratio and log biomass is constructed from variational sensitivities. It is distinct from the terminal neural models in the other control study. The comparison separates parameter adaptation, a limited direct optimisation budget, local approximation and nonlinear candidate checking.

## Files and identities

- `scripts/model.py`, `controllers.py` and `run_study.py` contain the amount balances, sensitivity equations, estimators, six controllers and execution loop.
- `protocol.json` retains all 24 case definitions, seeds, numerical settings and acceptance criteria. Its descriptive wording is independent of the numerical payload.
- `data/flux_table_eps1e-06.json` supplies paired growth/uptake response curves. No additional model input is required to run this table-based study.
- `results/frozen_evaluation/` contains all 144 byte-identical episode records. Feeds, predictions, realised states, observation errors, iteration traces and measured costs remain available, including unsuccessful cases.
- `independent_audit/` contains the numerical outcome audit and all 58 retrospective final-interval diagnostics. `figures/` contains the three thesis result figures and their exact plotted arrays.
- `SOURCE_MANIFEST.json` maps the original source/result hashes to release hashes. The three runtime sources preserve their executable ASTs; only their module descriptions differ. Portable summary, audit and plotting interfaces record their separate transformations and unchanged scientific functions.
- `reference/executed_lock.json` identifies the code and protocol used for the supplied outcomes. `LOCK.json` identifies the current release files for fresh calculations. Historical episode hashes are not replaced with current hashes. Dates in the execution records describe those computations.

## Commands

Use Python 3.11 and the repository's pinned requirements. Run from the repository root:

```sh
python experiments/adaptive_feedback/run.py verify --numerical
python experiments/adaptive_feedback/run.py summary --label local_summary
python experiments/adaptive_feedback/run.py plots
```

Verification checks release hashes, original numeric payloads, source/function ASTs, all episode identities, independently recomputed endpoints and aggregate counts, and preservation of the nine strict numerical flags. `--numerical` adds one complete 14 h trajectory, analytic-sensitivity/finite-difference comparison and a separately coded physical-amount integration. This is a targeted installation check, not another 144-episode evaluation. AST identities use Python 3.11 syntax representation.

Summary and plot commands write to `local_results/`; supplied evidence is retained. The figures reconcile every plotted count, cost and paired difference against the episode records. Plotting needs NumPy, Matplotlib and Pillow. The input data for this study are included; direct intracellular calculations belong to the model studies elsewhere in the repository.

For a fresh serial evaluation with one BLAS/OpenMP thread:

```sh
python experiments/adaptive_feedback/run.py simulate --label local_evaluation
python experiments/adaptive_feedback/run.py summary --campaign local_evaluation --label local_evaluation
python experiments/adaptive_feedback/run.py audit --campaign local_evaluation --label local_evaluation
python experiments/adaptive_feedback/run.py posthoc --campaign local_evaluation --label local_evaluation
```

Fresh simulation labels must start with `local_`; existing runs are not overwritten. A complete campaign takes substantially longer than verification. Execution records include versions, platform, current source hashes and UTC start time. Controller timing includes parameter estimation, sensitivities, local proposals, nonlinear checks, corrections, fallback and failed evaluations; process simulation and independent audit time are reported separately.

The audit and retrospective commands can also use `--campaign frozen_evaluation` with a fresh `--label`. The audit returns a nonzero status when strict flags remain; for the supplied campaign, nine flags are an expected scientific outcome, not a packaging failure. Numerical libraries and hardware can alter borderline decisions and measured times in fresh runs. Fresh outcomes should be reported under their own execution identity.

## Recorded outcomes

Open loop, nominal feedback, full adaptive direct, limited adaptive direct, unchecked local and checked local control satisfy the final acceptance criteria in 6, 6, 17, 20, 20 and 17 of their 24 cases, respectively. All methods succeed in all six nominal cases. Under constant mismatch, all adaptive methods succeed in six cases, while open-loop and nominal feedback succeed in none. The experiment therefore supports parameter adaptation under the tested persistent mismatch.

Checked local control takes approximately 215 s across 24 cases, compared with 520 s for full adaptive direct and 119 s for limited direct. On the 17 jointly feasible pairs, the prespecified criterion of no more than 0.5% biomass loss together with lower cost holds in 9 cases against full direct and 3 against limited direct. These are paired computational observations; they do not establish uniform speed or quality dominance.

The independent strict audit covers all 144 episodes and 1008 intervals. It retains nine flags: three near-boundary controller-feasibility label changes, three glucose-amount discrepancies above audit tolerance and three slightly excessive planned volumes. One realised volume exceeds its numerical allowance and is counted unsuccessful. Final acceptance classifications agree with the separate physical-amount integration. The audit's `passed: false` is retained.

A retrospective 101-point final-feed search with local refinement finds feasible terminal alternatives in 18 of the 58 unsuccessful cases. It uses actual late states and hidden affinity parameters only after the actions have been taken. The remaining finite-search failures do not prove unreachability; a final alternative cannot undo an earlier path violation. Its 6899 model calls and approximately 342 s are excluded from controller timings.

## Model scope and sources

The reactor uses paired finite-barrier response curves, substrate-axis affinity factors, a stated low-substrate closure and synthetic measurement errors. It is a computational model-mismatch study of one species pair, not cultivation validation, a general robustness proof, or a safety guarantee. The six initial-condition clusters are not 144 independent biological replicates. Nonlinear candidate checking assesses the current controller model; it cannot guarantee the response of an unknown process.

The model settings build on the metabolic and control sources in the main [study guide](../../docs/studies.md#benchmark-source). Related candidate-checking and model-mismatch methods include Wabersich and Zeilinger (2021), [10.1016/j.automatica.2021.109597](https://doi.org/10.1016/j.automatica.2021.109597), and del Rio Chanona et al. (2021), [10.1016/j.compchemeng.2021.107249](https://doi.org/10.1016/j.compchemeng.2021.107249). Their formal guarantees are not assumed for the present controller. FBA-based surrogate predictive control is discussed by de Oliveira et al. (2021), [10.1021/acs.iecr.1c01242](https://doi.org/10.1021/acs.iecr.1c01242).

Author-owned source, documentation and generated numerical outputs fall within the repository's [MIT Licence](../../LICENSE) to the extent the author holds rights in them. Underlying reconstructions and external publications retain their own terms; see [licence scope](../../TERMS.md) and [third-party notices](../../THIRD_PARTY_NOTICES.md). No manuscript, installed dependency or model-reconstruction source is bundled here.
