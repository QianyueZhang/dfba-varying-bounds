# Data and result conventions

Paths below are relative to the repository root. Use the model-input and source-data manifests for exact hashes and attribution. Numeric values must be interpreted with their case-specific units; different studies use different concentration scales.

## Metabolic inputs

`data/model_inputs.json` lists 38 required case-specific input destinations, byte sizes and SHA-256 hashes. The files are supplied locally under `data/models/` and read using SciPy. Reusing a reconstruction name does not guarantee identical reaction ordering, bounds or preprocessing. The manifest is the identity check.

`python tools/prepare_inputs.py --check` checks existing inputs without copying. `--search-root` locates exact byte matches in an authorised local collection and prepares the expected paths. These inputs are not downloaded or redistributed by the utility.

## Control state and feed

For `control_experiments.Reactor`, the state order is `[V, X_SC, X_KP, S]`:

- `V`: reactor liquid volume, L.
- `X_SC`, `X_KP`: biomass concentrations, gDW L⁻¹.
- `S`: glucose concentration, g L⁻¹.
- The seven controls are piecewise-constant inlet rates in L h⁻¹, each applied for 2 h over the 14 h horizon.
- The inlet glucose concentration is 100 g L⁻¹. The control bounds are 0–0.2 L h⁻¹.
- `z = ln(X_SC(tf)/X_KP(tf))` and `J = (1 − X_SC(tf)/X_KP(tf))² = expm1(z)²` are dimensionless.

The CSV files `batch_0.csv`, `batch_1.csv` and `revision_validation/direct_polished_trajectory.csv` store time followed by this state order. Headers include the units. Batch and selected-control trajectories have different horizons; use their time column.

The response tables `flux_table_control_experiment_snapshot.json` and `flux_table_eps1e-06.json` each contain `SC` and `KP` records. Each row records substrate `S` (g L⁻¹), growth (h⁻¹), glucose exchange (mmol gDW⁻¹ h⁻¹), residuals and minimum slack. The factor 0.18 converts glucose exchange to mass concentration rate. Negative glucose exchange represents uptake. The first table defines the fixed control comparison; the second is a distinct refined table. Keep each trajectory linked to its actual table.

## Neural arrays

Load NPZ data with `numpy.load(path, allow_pickle=False)`.

`research_completion/runs/surrogate_dataset.npz` contains:

- `controls`: shape `(1000, 7)`, feed policies in L h⁻¹.
- `terminal_states`: shape `(1000, 4)`, in the control-state order above.
- `target_log_ratio`: shape `(1000,)`, dimensionless `z`.
- `train`, `validation`, `test`: integer row indices with lengths 700, 150 and 150. These are disjoint and together cover all 1000 controls.

`research_completion/runs/revision_validation/neural_fresh_audit.npz` contains 200 separate controls, `target_log_ratio`, `refined_target_log_ratio` and the sampling `seed`. Both target arrays have shape `(200,)` and refer to the fixed and refined mechanistic tables respectively.

`neural_audit_predictions.npz` contains the aligned target arrays, `predictions` with shape `(3, 200)`, `training_sizes = [175, 350, 700]` and three `selected_seeds`. Each row of predictions belongs to its stated training size. The scalar predicted response is `z`; apply the stated transformation before calculating objective errors.

`neural_protocol.json` records architecture, training sizes, initialisation seeds, audit seed and policy-search protocol. `neural_extension.json` records the model-selection and policy results. `direct_polished.json` gives the mechanistic policy check, residuals, minimum slack and numerical settings. The data dictionary does not replace these recorded settings.

## Published yeast observations

`data/zampar2013/zampar2013_observations.csv` retains 266 observations extracted from Zampar et al. (2013), DOI [10.1038/msb.2013.11](https://doi.org/10.1038/msb.2013.11), Supplementary File 1. The source workbook links, hashes and licence are in the adjacent manifest and [third-party notices](../THIRD_PARTY_NOTICES.md).

`variable` identifies `biomass_OD600`, `glucose` or `ethanol`. `observed_value_as_published` must be paired with `observed_unit`; source workbook, sheet, row and cell fields locate the observation. `time_relative_to_glucose_exhaustion_h` defines the time origin. `replicate_label_as_published` preserves the workbook label; it does not establish paired biological replicate identities between variables. `secondary_*`, `lod_*` and `censoring_status` preserve distinctions between secondary values, detection-limit information and missingness.

The model interval starts at −6.8 h. There are 256 observations in this interval: 45 designated calibration observations (`Exp1`) and 211 withheld observations. Ten earlier observations remain in the raw CSV but are outside the model interval. Biomass is modelled in gDW L⁻¹; the observation operator uses 0.486 gDW L⁻¹ per OD600 unit for comparison with the published optical-density values.

The yeast trajectory `figure11_zampar_2026-10-21/trajectory_direct_barrier.csv` records biomass in gDW L⁻¹, glucose and ethanol in g L⁻¹, growth in h⁻¹, fluxes in mmol gDW⁻¹ h⁻¹, and numerical residuals/slack. The small positive slack and nonzero finite-barrier residual scales are numerical quantities, not experimental measurement uncertainties.

## Diagnostic outputs

`local_verification/suite.json` lists the checks actually executed, exit status, report hashes and elapsed time. Its per-stage logs preserve failures for inspection. The analytic example records its complete problem definition, tolerances, barrier values and environment. Neural statistics include MSE, RMSE, MAE and R² with separate rows for `z` and `J`; all these responses are dimensionless. A software check passing does not establish biological generality beyond the underlying study.

## Constrained-control data

`experiments/constrained_control/reference/data/` contains 1152 training, 192 validation and 256 test rows partitioned by distinct initial-condition groups. `X` has seven feed rates (L/h), log initial SC/KP ratio and glucose (g/L); `Y` has terminal log SC/KP ratio and log biomass amount relative to 1 gDW. `condition_index` identifies a group within that split, and `controls` repeats the seven feed components for auditability. The JSON predictors contain input/output means/scales, quadratic powers/coefficients or neural weights/biases. `reference/results/policies/` contains all 180 optimiser traces with controls, [V, X_SC, X_KP, S] terminal states, distinct attempted-evaluation counts, failures, ordinary/tighter feasibility and timing. Raw values retain full precision; display tables round independently.

## Recovery and scenario-feedback records

`experiments/terminal_recovery/` and `experiments/scenario_feedback/` use the log-amount state `[V, log(M_SC), log(M_KP), Q]`: volume in L, species biomass amounts relative to 1 gDW, and extracellular glucose amount in g. Their independent audit integrates physical amounts directly. Concentrations follow as amount divided by volume. Feed is in L/h; each decision interval lasts 2 h. Affinity multipliers and terminal log species ratios are dimensionless. Protocols specify the separate inner and actual composition acceptance bands; neither should be inferred from a rounded table entry.

Recovery records identify root/history, target, method and disturbance path. The primary set has 18 roots, six targets, five methods and nine paths (4,860 episodes); success on a root–target problem requires every path to meet the stated criteria. Exploratory selected interval midpoints and fixed-target sensitivity records have separate cohort labels. Scenario-feedback records identify 88 cases and five methods (440 episodes), with a separately timed eight-case panel per method (40 records). Timings in seconds belong to their recorded execution environment, not to an assumed current machine.

`SOURCE_MANIFEST.json` contains source/release hashes and documented portable-label changes. Original execution locks identify the numerical sources used for supplied outcomes; current portable sources have separate identities where necessary. A successful package-verification report confirms identities and arithmetic, including retained failed strict audit records. It is not a scientific audit pass.

The distinct bacterial observations under `scenario_feedback/public_validation/` are E. coli cell fractions and digital blue-light intensities (0–800 arbitrary units), not yeast dry-weight fractions or glucose feeds. Their exact source mapping, held-out split, time-window metrics and CC BY 4.0 attribution are in [DATA_AND_METHODS.md](../experiments/scenario_feedback/public_validation/DATA_AND_METHODS.md).

## Yeast assessment records

`experiments/yeast_assessment/payload/paper1_improvement_2026-10-24/` contains numerical, temporal and external substudies. `numerical/results/decomposition.json` retains signed error contributions and their algebraic closure. `external/observations.json` records Wiebe Table 1 conditions and reported measurements; dilution rates are prescribed h⁻¹, biomass is gDW L⁻¹, oxygen solubility is μM and carbon-specific glucose/ethanol/glycerol rates are C-mmol gDW⁻¹ h⁻¹. The comparison converts carbon-specific glucose and ethanol rates to compound mmol gDW⁻¹ h⁻¹ by dividing by six and two, respectively; reported SEM receives the same conversion. Oxygen and carbon dioxide rates are already mmol gDW⁻¹ h⁻¹. The nominal dilution rate is not an experimentally measured growth ordinate. Zero-growth model cases have undefined growth-normalised yield and are retained explicitly. Numerical records retain full precision; figure labels use rounded presentation values.

## Sensing/actuation records

`experiments/sensing_actuation/` uses the same `[V, log(M_SC), log(M_KP), Q]` state convention as terminal recovery. `results/grid_observation_*_grid*.json` supplies per-root/mode/measurement-error target sets and fixed-policy outcomes. `audit/` holds independently integrated physical-amount paths and exact endpoint results. `results/continuous/` retain all gzipped candidate ledgers. Continuous controls observe full state; ratio-observation errors and their target-set counts belong to the finite-grid campaign. `OPTIONAL_CACHE_MANIFEST.json` identifies omitted regenerable NPZ transition caches.
