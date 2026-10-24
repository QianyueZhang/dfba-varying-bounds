# Causal scenario feedback under model mismatch

This study compares certainty-equivalent feed decisions, common open-loop scenario plans, causal scenario feedback, and a local-proposal manager with nonlinear checking and direct fallback. Both the supplied yeast response-table model and a separately specified analytic rate family are evaluated. Scenario sets are design choices, not calibrated confidence regions.

## Complete recorded evidence

The locked primary campaign contains 88 designed cases and five methods, with all **440 episodes** retained. Actual terminal successes are 53/88 for five-iteration certainty-equivalent control, 54/88 for twelve-iteration certainty-equivalent control, 68/88 for scenario-open planning, 61/88 for scenario feedback and 69/88 for managed scenario feedback. These are finite computational-panel counts, not biological success probabilities.

The independent amount-state audit preserves the actual physical, terminal and operational classifications but fails its original strict numerical criteria: 18 comparison records correspond to six physical checkpoints in six episodes. The original failure and its targeted diagnostics remain available. All 135 selected forecast tails retain their inner-band labels, with no accepted-policy outer-band violation among their 513 scenario integrations. These finite checks do not establish recursive feasibility or a guarantee under arbitrary mismatch.

The separately timed serial panel has eight preselected cases per method, with all **40 result objects** identical to their corresponding primary results. Managed feedback takes about 133 s in total versus 139 s for direct scenario feedback, but its median time is larger. Among six jointly feasible comparisons, none satisfies both faster execution and at most 0.5% biomass loss. The package therefore makes no general speed or matched-quality superiority claim. Primary parallel-run timings are diagnostic; use the isolated serial panel for the stated timing comparison.

## Source and data layout

- `scripts/`: process models, native RHS, sensitivity equations, estimators, controllers, campaign and plotting commands.
- `protocol.json` and `LOCK.json`: the fixed case design and the executed core-source identities.
- `results/locked/`: all 440 episodes, execution manifest and independent audit.
- `results/timing_serial/`: all 40 serial records and source snapshots. Development attempts are separately retained under their pilot labels.
- `audit/`: sensitivity/failure-routing checks, forecast transport, numerical discrepancy diagnostics and optimisation-budget records.
- `public_validation/`: a distinct prediction study of published bacterial composition observations. It does not validate yeast glucose-feed interventions.
- `SOURCE_MANIFEST.json`: source/release hashes, complete source selection and documented portable-path/check-label changes. The original numerical source lock and original outcome identities are not replaced with an invented history.

## Commands

From the repository root:

```sh
python experiments/scenario_feedback/run.py verify
python -m pip install -r experiments/scenario_feedback/requirements.txt
python experiments/scenario_feedback/run.py numerical-smoke
python experiments/scenario_feedback/run.py prepare /path/to/new/scenario_run
```

Verification is a read-only inventory and arithmetic check of all primary and serial results, source locks, full denominators and preserved audit findings. The numerical smoke check compares native and NumPy/SciPy prediction and gradient paths in both model families. Neither command reruns the full campaign.

Preparation refuses existing destinations, copies scientific code/input definitions without outcomes, and writes `RUN_COMMANDS.json`. Run its commands from the new directory in order with one BLAS/OpenMP thread. The main campaign uses four worker processes, followed by a separate serial timing panel. Full computations take substantially longer than verification. Generated results have their own execution metadata; software or platform changes can affect borderline classifications and timings.

The original failed numerical audit remains a scientific outcome. A fresh run should retain its own failures and diagnostic thresholds. Do not regenerate an execution lock merely to suppress a mismatch or replace failed records with successful reruns.

## Published-data prediction test

The separately supplied observations are from Gutiérrez Mena, Kumar and Khammash (2022), *Dynamic cybergenetic control of bacterial co-culture composition via optogenetic feedback*, Nature Communications 13, 4808, DOI [10.1038/s41467-022-32392-z](https://doi.org/10.1038/s41467-022-32392-z). These are engineered **E. coli** cell fractions under recorded blue-light inputs, not yeast biomass amounts under glucose feeds.

All eight held-out input/output trajectories are retained. Both reduced predictors fail their full-trajectory and through-40-h macro baseline comparisons. The separately declared fixed published-model check also fails its full-run baseline comparisons. Negative prediction results are retained; they do not substantiate a replacement controller or validate the yeast model.

The CC BY 4.0 source-data archive, attributed extracted observations, predictions and metrics are included. The publisher's supplementary software archive is **not redistributed** because a separate software licence was not established. The optional fixed published-model calculation requires the user to obtain that external archive under its source terms. See [`public_validation/DATA_AND_METHODS.md`](public_validation/DATA_AND_METHODS.md) for exact inputs and commands. Manuscript drafts and publisher literature PDFs are not part of this public package.
