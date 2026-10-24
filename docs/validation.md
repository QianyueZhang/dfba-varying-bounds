# Numerical verification and extension

## Short suite

With the pinned dependencies installed, run:

```sh
python tools/verify.py
```

The suite executes the installation/control checks, the analytic moving-bound benchmark, the independent calculation of neural-audit statistics, constrained-control record/predictor checks, and terminal-recovery/scenario-feedback source and result-record checks. It writes `local_verification/suite.json`, one log per stage, and the detailed reports at their normal experiment paths. A failed command or a failed numerical check gives a nonzero exit status. Prior stage reports are removed before execution so a stale success cannot be mistaken for a new result. The suite does not rerun the long biological studies or optimisation searches.

Once all model inputs are prepared, require their hashes and static LP/KKT checks too:

```sh
python tools/prepare_inputs.py --check
python tools/verify.py --require-models
```

The short suite covers numerical control trajectories from supplied tables, dataset allocation, experimental observation allocation, an analytic KKT derivative calculation and fixed neural prediction errors. With exact model inputs it also evaluates reduced-network finite-barrier states and selected genome-scale LP growth values. There is no claim that these checks exhaust every scientific claim in the thesis.

The lightweight case-runner interface checks are separate:

```sh
python -m unittest discover -s tests -p 'test_*.py'
```

## Full case calculations

The short suite and the table-based constrained-control campaign use supplied numerical inputs. They do not establish that every culture workflow runs from a clean clone: the 38 model-input paths are excluded, and direct intracellular calculations require the relevant subset. See the [input guide](model_inputs.md) for the exact case dependencies and source links. Hash-based preparation checks identity; it does not grant redistribution rights or obtain the files.

Use the ordered commands in `cases.json` through `tools/run_case.py`. Keep the supplied numerical tables unchanged when evaluating the stated control comparison. Rebuilding a response table defines a new numerical setting and requires its own checks before interpreting differences.

For a model extension:

1. Specify the research question, reaction bounds, mass balances, initial conditions, units and control objective before running comparisons.
2. Record the exact model-input hashes and all parameter changes. Preserve a baseline checkout when changing parameters or algorithms.
3. Check stoichiometric residuals, bound slack and KKT residuals where applicable; check the integrated material balances independently when possible.
4. Vary integration tolerances at fixed barrier and bounds. Then assess the barrier/relaxation choices separately. Similar final objective values need not imply identical state trajectories.
5. Compare methods on the same model, inputs, stopping criteria and achieved accuracy. Include seeds/repeated runs when the method is random, and avoid ranking solvers from one elapsed-time observation.
6. For surrogate control, keep training/validation/test roles separate and evaluate the final selected policy using the mechanistic model.
7. Save trajectories, numerical diagnostics and figure data with the run. Do not infer experimental validity from numerical feasibility alone.

The repository contains mechanisms and examples rather than a universal parameter set. A change in uptake, maintenance, inlet supply or state-switching policy can produce a different biological scenario, even when the numerical framework is unchanged.

## Scope of further research

The existing case set spans moving-bound differentiation, monocultures, several interaction mechanisms, genome-scale communities and composition control. Adding another similar trajectory is less informative than an independently motivated test. Useful extensions include matched achieved-accuracy/cost comparisons on larger systems, uncertainty in biological parameters, control objectives that jointly constrain feed use and production, and validation against additional cultivation conditions. These are prospective extensions, not completed results in this software version.

## Constrained-control extension

Run `python experiments/constrained_control/run.py verify --integrate` to evaluate all 180 supplied controls at the recorded tighter ODE tolerances and check final feasibility, alongside budget arithmetic, grouped dataset allocation and JSON predictor agreement. The normal short suite includes record/predictor checks without those integrations. The [experiment guide](../experiments/constrained_control/README.md) documents the full campaign, its separate direct KKT checks and limitations.

## Adaptive feedback study

Run `python experiments/adaptive_feedback/run.py verify --numerical` for release/source identities, all 144 primary records, independently recomputed terminal quantities and costs, preserved numerical audit flags, one trajectory and a sensitivity/independent-amount check. These package checks can pass while the recorded strict campaign audit remains `passed: false` with nine flags. The public [study guide](../experiments/adaptive_feedback/README.md) describes full audit and retrospective commands. Recorded outcomes preserve original source hashes; fresh runs use the separately declared current release lock. No historic outcome is relabelled with a current-source hash.

## Terminal recovery and scenario feedback

Run `python experiments/terminal_recovery/run.py verify` and `python experiments/scenario_feedback/run.py verify` for complete source/result identities, executable syntax, core source locks and independent arithmetic across all supplied primary outcomes. The root short suite includes these two record checks. Recovery retains ten original strict audit flags and scenario feedback retains eighteen comparison flags; the check ensures they have not been removed or relabelled. Distinct later numerical checks retain their own criteria and outcomes.

Each package also exposes `numerical-smoke`, requiring its separately pinned Numba/llvmlite dependencies. Recovery compares a 2 h interval in each model family with independent physical-amount equations. Scenario feedback compares native and ordinary state/gradient predictions. These checks assess installation and selected numerical identities; they do not repeat the complete campaigns or establish a new success rate.

Use `prepare /path/to/new/directory` for a fresh full calculation. The destination must not exist; it receives scientific code/input definitions, hash checks, empty output locations and `RUN_COMMANDS.json`. No supplied outcomes are copied, and no long command runs implicitly. See each study guide for execution order and the original audit qualifications. The published-data prediction check is separate and its optional fixed-model calculation requires an external software archive that is not bundled.

## Yeast assessment and sensing/actuation records

The root short suite includes `experiments/yeast_assessment/run.py verify` and `experiments/sensing_actuation/run.py verify`. These check content hashes, frozen numerical-source locks, complete reported case counts and independently recounted policy/solver evidence. They do not run long fitting or optimisation campaigns. Scientific failures retained in the records remain failures even when package verification succeeds.

Use `python experiments/yeast_assessment/run.py figures` to generate the assessment plots and their 34-array preservation checks. `python experiments/sensing_actuation/run.py numerical-smoke` additionally compares one interval from each of two process model families against independent amount equations; install that study's pinned requirements first. Its full preparation command writes the ordered calculation protocol in a new directory, with no campaign outcomes. The yeast preparation command preserves supplied reference records separately from the selected calculation outputs.
