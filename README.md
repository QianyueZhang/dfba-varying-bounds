# Dynamic flux balance analysis with varying bounds

Python simulation and analysis software accompanying **Qianyue Zhang's PhD thesis**, *Dynamic Flux Balance Analysis with Varying Bounds: Simulation and Optimal Control of Microbial Systems*, Department of Chemical Engineering and Biotechnology, University of Cambridge.

The software connects intracellular metabolic optimisation to reactor mass balances, changing uptake capacities and feed decisions. Its 20 case workflows cover finite-barrier and differentiated KKT formulations, single cultures, microbial interactions, genome-scale models, optimal control and neural surrogate models. Numerical tables and selected simulation records are supplied alongside the source code. The thesis manuscript is maintained separately.

## Installation

Python 3.11 is recommended. From the repository root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/verify.py
```

On Windows, activate the environment with `.venv\Scripts\activate`. The pinned dependencies are NumPy, SciPy, scikit-learn, Matplotlib, Pillow, openpyxl and HiGHS.

The short verification suite checks the installation, six control trajectories, dataset allocation, the analytic moving-bound problem, neural prediction statistics, and the supplied records of the constrained-control, terminal-recovery, scenario-feedback, yeast-assessment and sensing/actuation studies. It writes `local_verification/suite.json` with per-stage logs and reports. It uses supplied data and does not run the long culture studies or optimisation searches. See the [verification guide](docs/validation.md) for its scope and optional model checks.

The [study guide](docs/studies.md) explains the scientific questions and links between the 20 cases. The [data dictionary](docs/data_dictionary.md) defines state ordering, units, neural arrays and experimental observation fields.

## First calculations

These two examples use only files included in the repository:

```sh
python tools/run_case.py moving-bound-analytic --execute
python tools/run_case.py neural-audit-statistics --execute
```

The first integrates a two-flux problem with simultaneous lower- and upper-bound motion and compares it with an analytic solution. It produces trajectory, sensitivity and convergence plots in `experiments/moving_bound_benchmark/figures/` and numerical records in the adjacent `results/` directory.

The second calculates prediction errors and residual plots for neural models from a fixed set of 200 independent simulator controls. Results are written to `experiments/neural_audit_diagnostics/results/`. It analyses the supplied predictions; training and feed-policy optimisation belong to `control-neural`.

## Running the culture and control studies

List the cases, inspect the commands for a case, and then run them:

```sh
python tools/run_case.py
python tools/run_case.py control-neural
python tools/run_case.py control-neural --execute
```

To run one numbered command after its prerequisite outputs exist:

```sh
python tools/run_case.py control-neural --step 4 --execute
```

[`cases.json`](cases.json) defines the commands and output directories. The runner executes Python scripts from the repository root with one BLAS/OpenMP thread. Full dynamic studies and repeated optimisations take longer than the installation check. Running a case again replaces outputs at its declared locations.

### Case map

| Case | Scientific purpose | Principal inputs |
| --- | --- | --- |
| `moving-bound-analytic` | Verify both moving-bound derivatives, finite differences and separate barrier/integration refinements | Defined analytic problem |
| `edwards` | Compare uptake capacity with prescribed uptake; compare direct and differentiated KKT trajectories | Reduced-network MAT input |
| `diauxic` | Simulate sequential glucose–xylose use with numerical refinement | iJR904 MAT inputs |
| `yeast-experiment` | Fit a yeast batch model to published observations and assess withheld observations | Yeast MAT input and supplied Zampar data |
| `yeast-batch-fedbatch` | Compare batch and fed-batch cultures using concentration and amount balances | Yeast MAT input |
| `independent-shared` | Distinguish independent substrate pools from shared-resource competition | Core-model MAT inputs |
| `product-inhibition` | Assess an inhibition mechanism with a complete ethanol balance and matched reference | Core-model MAT inputs |
| `resource-diversion` | Couple recipient biomass to a donor secretion bound and its time derivative | Core-model MAT inputs |
| `biomass-transfer` | Account for growth, dilution, transfer and loss | Core-model MAT inputs |
| `pyruvate-crossfeeding` | Simulate chemically specified transfer with active and quiescent states | Core-model MAT inputs |
| `genome-scale-yeasts` | Simulate isolated yeast and an inhibited yeast pair | Yeast MAT inputs |
| `cstr` | Compare 72 h light/dark cultivation with numerical and initial-loading sensitivities | iND750/iRC1080 MAT inputs |
| `control-neural` | Optimise feed, train/select neural models, and verify selected policies with direct KKT calculations | Supplied response tables/arrays and yeast MAT inputs for direct checks |
| `constrained-control` | Maximise biomass under composition/volume constraints; compare direct, database and surrogate methods across held-out conditions | Supplied grouped data, safe JSON predictors, 180 policy records and refined response table |
| `adaptive-feedback` | Assess parameter adaptation and local-affine candidate checking under affinity mismatch and observation noise | Supplied paired response table, fixed protocol, 144 episode records and independent diagnostics |
| `terminal-recovery` | Assess attainable terminal composition under finite feed actions and affinity disturbances | Two model families, locked protocol, full primary panel, target intervals and independent amount checks |
| `scenario-feedback` | Compare certainty-equivalent, open-loop scenario and causal feedback decisions | Two model families, 440 episodes, 40 isolated serial outcomes and complete numerical audits |
| `yeast-assessment` | Separate finite-barrier numerical error, chronological prediction and independent oxygen-limited yeast observations | Frozen numerical/experimental records and attributed observations |
| `sensing-actuation` | Assess terminal target recovery under feed resolution and ratio-observation uncertainty; compare continuous full-state control | Paired response table, finite-game outcomes, deployment episodes and continuous candidate ledgers |
| `neural-audit-statistics` | Calculate MSE, RMSE, MAE and R² for saved audit predictions | Supplied audit arrays |

### Model inputs

The complete culture workflows require 38 case-specific MAT input paths (30 distinct files by content), which are not bundled. A clean clone can run the analytic moving-bound study, fixed-prediction neural diagnostics and the table-based constrained-control campaign. It cannot run every culture study or the optional direct intracellular checks without the exact model inputs. [`data/model_inputs.json`](data/model_inputs.json) records their exact destinations, sizes and SHA-256 hashes; the [input guide](docs/model_inputs.md) maps them to cases and identifies source publications and unresolved distribution details. Prepare an existing local collection with:

```sh
python tools/prepare_inputs.py --search-root /path/to/local/model/files
python tools/prepare_inputs.py --check
python tests/verify_release.py --require-models
```

The utility searches an existing local collection for exact matching files, copies them into the expected locations and reports missing inputs. It performs no download and does not create processed matrices from a published reconstruction. The source papers identify model families, but a complete public acquisition/conversion route for these exact inputs is not yet supplied. Obtain reconstructions and processed matrices under their applicable source terms; a model with the same name need not contain the same matrix, reaction ordering or bounds. Consequently, numerical records can be inspected publicly even where a full independent culture calculation still requires separately supplied inputs.

Model inputs are stored under `data/models/` in case-specific directories created by the preparation utility. SciPy reads the MAT-format arrays directly. These locally supplied files are ignored by Git; the input manifest identifies the exact content required by each calculation.

### Outputs and repository layout

- `research_completion/scripts/` contains reactor models, optimisation routines and plotting code.
- `research_completion/runs/` contains supplied numerical arrays/tables and case-specific calculations. New case outputs include trajectories, solver settings, residuals, source hashes and summary metrics.
- `research_completion/figures/` receives many generated plots; other cases write figures beside their numerical results, as specified in `cases.json`.
- `experiments/` contains the analytic moving-bound example, fixed-prediction neural analysis and five control studies, together with yeast numerical/experimental assessment, each with its own method description and numerical records.
- `data/` contains input metadata and attributed experimental observations; locally prepared models are stored under `data/models/`.
- `tools/` contains the input-preparation, case and verification runners; `tests/` contains the short numerical checks, their expected values and command-interface checks.
- `docs/` contains the study guide, data dictionary and validation/extension guidance.

Directory names are stable to retain command and data-path compatibility.

## Numerical methods

The intracellular problem combines stoichiometric equalities with lower and upper flux bounds. A logarithmic barrier defines an interior flux solution; direct primal–dual solves and differentiated KKT equations are used in the cases that assess those formulations. Extracellular ODEs track biomass, substrates, products and reactor volume. Individual studies specify uptake policies, bound relaxation, barrier parameter, initial conditions and solver tolerances.

The analytic example tests simultaneous lower- and upper-bound motion against a closed form and a separate scalar root. Biological cases add case-specific checks such as stoichiometric residuals, LP objective gaps, independent amount balances and integration refinement. These checks address numerical consistency within the stated model; agreement with experimental observations is assessed separately in the yeast study.

Feed optimisation uses piecewise-constant controls and matched problem definitions for SLSQP, PSO and Secant-PSO. Neural models predict the terminal log biomass ratio, `z = ln(X_SC / X_KP)`, from which the ratio objective `J = expm1(z)^2` is calculated. Selected feed policies are evaluated with the mechanistic reactor model and direct finite-barrier calculations.

### Data and numerical settings

The control comparison uses `flux_table_control_experiment_snapshot.json`. The distinct `flux_table_eps1e-06.json` supplies the refined response table used for validation; their grids must not be interchanged silently. The 700/150/150 training/validation/test split, independent audit controls, random seeds and target definitions are retained with the relevant scripts and arrays.

The installation check evaluates the supplied baseline values. With all exact model inputs present, it also checks reduced-network finite-barrier solutions and selected genome-scale LP values. It does not rerun every long simulation or optimisation. The analytic and neural-statistics commands provide additional focused checks. Consult each case's numerical settings and refinement outputs when assessing its results.

## Constrained biomass/composition control

The [constrained-control study](experiments/constrained_control/README.md) extends the equal-composition case to twelve target cases: four held-out initial conditions and three desired species ratios. Five methods and three algorithm seeds give 180 optimised policies. Training, validation and test conditions are disjoint. The package includes full traces, JSON predictor weights, explicit failure/evaluation budgets, offline/online cost accounting, initial-condition sensitivity, two direct KKT checks and figure generation.

```sh
python tools/run_case.py constrained-control --execute
python experiments/constrained_control/run.py study --output local_verification/control_study_01
```

The first checks all supplied policies and renders their figures. The second runs data generation, fitting and the complete optimisation campaign in a new directory. All nominal selected policies pass the stated numerical checks; the fixed-policy sensitivity and measured complete costs qualify what those results establish. The neural model predicts held-out outcomes more accurately than the quadratic model, while direct single-start optimisation has the lowest measured complete cost for the 36-decision workload.

## Adaptive feedback under model mismatch

The [adaptive-feedback study](experiments/adaptive_feedback/README.md) compares six controllers over 24 scenarios, retaining all 144 episodes, 1008 control intervals, exact plot arrays, independent numerical audit and retrospective final-interval searches. Parameter adaptation succeeds in all six constant-mismatch cases. Checked local control does not uniformly outperform a limited direct solve, and nine strict numerical audit flags remain explicitly recorded. Current runtime identities are distinguished from the original execution hashes; all primary episode JSONs are unchanged.

```sh
python experiments/adaptive_feedback/run.py verify --numerical
python tools/run_case.py adaptive-feedback
```

## Terminal recovery and scenario feedback

The [terminal-recovery study](experiments/terminal_recovery/README.md) evaluates composition targets from matched-history culture states using a finite two-step feedback game. The complete fixed panel contains 4,860 paths. A separate exploratory analysis constructs admissible target intervals and verifies selected witnesses on a refined action grid. Complete fixed-target denominators, continuous open-loop baselines and failed strict audit comparisons remain available; selected feasible witnesses do not establish an arbitrary-target success rate.

The [scenario-feedback study](experiments/scenario_feedback/README.md) retains all 440 episodes across 88 designed cases and five methods, plus 40 separately timed serial outcomes. It assesses causal decisions under finite affinity scenarios and distinguishes terminal success, numerical verification and complete measured cost. The study includes a separately scoped published bacterial-data prediction test; those observations do not validate yeast glucose-feed interventions.

```sh
python tools/run_case.py terminal-recovery --execute
python tools/run_case.py scenario-feedback --execute
```

These two case commands verify supplied records without starting optimisation. Each study provides a `prepare` command that creates a fresh calculation directory with input definitions, source checks and an ordered command list, but no outcomes. Install its separate pinned requirements before numerical calculations. The Numba/llvmlite checks and complete campaigns are not part of the root installation suite. The recorded strict audits contain failures, which remain visible even when record-identity verification passes.

## Numerical and experimental assessment of yeast models

The [yeast-assessment study](experiments/yeast_assessment/README.md) separates bound expansion, finite-barrier effects, integration error and LP solver agreement. Its experimental analysis includes chronological prediction from Zampar observations and a separate oxygen-limited continuous-culture challenge from Wiebe et al. (2008). All 92 primary external cases, 276 LP calls, 20 recalibration cases and 72 follow-up calls are retained. The incomplete temporal forecast and unsuccessful external-growth predictions remain explicit scientific outcomes; numerical verification alone does not establish biological validity.

```sh
python tools/run_case.py yeast-assessment --execute
python experiments/yeast_assessment/run.py figures
```

## Sensing and actuation in terminal composition control

The [sensing-actuation study](experiments/sensing_actuation/README.md) characterises attainable target sets under finite feed grids, bounded ratio-observation error and causal decisions. It retains complete per-grid results, all independent deployment episodes and all 141,537 continuous candidate records across 72 method/problem assignments. The continuous comparator observes the full state; it is distinct from ratio-only observation tests. Strict endpoint failures, mixed grid resolution and budget-limited searches are preserved. Optional transition caches are omitted with their exact hashes and regeneration commands.

```sh
python tools/run_case.py sensing-actuation --execute
python experiments/sensing_actuation/run.py prepare /path/to/new/control-study
```

## Experimental data and scientific citation

The yeast observations are from Zampar et al. (2013), *Temporal system-level organization of the switch from glycolytic to gluconeogenic operation in yeast*, **Molecular Systems Biology** 9:651, DOI [10.1038/msb.2013.11](https://doi.org/10.1038/msb.2013.11), Supplementary File 1. The extracted dataset contains 266 observations; 256 fall within the analysis interval, of which 45 are used for calibration and 211 are withheld. [`data/zampar2013/`](data/zampar2013/) retains the extraction script, source links, checksums and attribution. The published workbook labels do not establish matched biological replicate identities across variables.

Cite the exact software version or commit using [`CITATION.cff`](CITATION.cff), together with the reconstruction, method and experimental publications relevant to the calculation. The code implements a research framework building on published interior-point dFBA methods and metabolic reconstructions; the software citation does not replace those scientific sources. Model identities and third-party terms are summarised in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The interior-point formulation builds on Scott, Wilson, Conejeros and Vassiliadis (2018), *Simulation and optimization of dynamic flux balance analysis models using an interior point method reformulation*, **Computers & Chemical Engineering**, DOI [10.1016/j.compchemeng.2018.08.041](https://doi.org/10.1016/j.compchemeng.2018.08.041). The analytic moving-bound example checks the specified derivative terms; it does not claim a new general theory of optimisation sensitivity.

The yeast study settings and equal-composition fed-batch benchmark are adapted from Prakitr Srisuma (2018), *Dynamic Simulation and Optimisation of Biochemical Engineering Systems Using the Interior Point Approach*, MPhil dissertation in Advanced Chemical Engineering, University of Cambridge. The present code uses concentration-dependent uptake capacities and the terminal ratio `X_SC / X_KP`; the source uses prescribed glucose exchange and the reciprocal ratio `X_KP / X_SC`. The shared target is equal composition, but the constraints and off-target losses are not identical. The [study guide](docs/studies.md#benchmark-source) identifies the source sections and the case-specific adaptations. No DOI or public full-text link is asserted for this dissertation.

## Licence and versions

Author-owned Python source and repository documentation are available under the [MIT Licence](LICENSE). Experimental observations, external model inputs and dependencies retain their own terms, as specified in [TERMS.md](TERMS.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Version **0.7.0** is a release snapshot of the 20 simulation and analysis workflows described above. Use its tagged version or commit for a fixed software reference. The release contents and computational scope are documented in [CHANGELOG.md](CHANGELOG.md).
