# Guide to the simulation studies

The studies address three linked questions: how to integrate dynamic flux balance analysis with changing admissible fluxes, how that formulation behaves in microbial cultures and interactions, and how feed decisions can control a community. The [case runner](../tools/run_case.py) exposes the full command sequences in [cases.json](../cases.json). Commands within each case are ordered: simulation precedes validation and plotting.

## Where to begin

`moving-bound-analytic` needs no external model files and is the shortest route into the mathematical formulation. `neural-audit-statistics` uses the supplied arrays to explain prediction assessment. The culture cases require the exact matrices listed in [data/model_inputs.json](../data/model_inputs.json). Prepare these before starting a long run; input hashes identify the actual matrices, bounds and ordering.

Run a command preview before execution:

```sh
python tools/run_case.py edwards
python tools/run_case.py edwards --execute
```

A preview reads the configuration without creating result directories. A numbered `--step` can resume a case when all outputs required by that step already exist. Invalid steps are rejected. Existing outputs may be replaced during execution, so use a separate checkout for parameter exploration.

## Moving-bound formulation

**`moving-bound-analytic`** isolates simultaneous motion of both bounds in a two-flux equality-constrained problem. The direct scalar root, analytic central path and integrated KKT derivative provide independent comparisons. Barrier refinement and integration refinement vary different numerical parameters. This tests the derivative terms; the synthetic example does not represent a cultivation experiment. Its detailed problem statement is in the [experiment guide](../experiments/moving_bound_benchmark/README.md).

**`edwards`** uses a reduced metabolic network to distinguish an uptake-capacity inequality from a prescribed uptake equality. The workflow calculates the direct finite-barrier path, differentiated KKT path, LP uptake map and figures. Comparisons refer to their specified mathematical problems; capacity and prescribed uptake need not produce identical growth or product trajectories.

**`diauxic`** applies the sequential glucose–xylose uptake policy to iJR904. Primary, tighter-integration and smaller-barrier calculations precede the summary and plots. Changes in integration tolerance, barrier parameter and bound relaxation are recorded separately. The substrate switch is a property of the stated uptake policy and metabolic model.

## Single-culture studies and observations

**`yeast-experiment`** fits the yeast batch model to the designated calibration observations of Zampar et al. (2013), evaluates withheld observations, checks a direct finite-barrier trajectory and analyses numerical sensitivity. Time is measured relative to glucose exhaustion. The observation operator distinguishes optical density from dry biomass. Withheld observations from this cultivation condition assess predictive agreement within that condition; they are not independent-condition biological validation. The [data dictionary](data_dictionary.md) records the observation allocation and units.

**`yeast-batch-fedbatch`** checks batch and fed-batch calculations with concentration and amount balances. In fed-batch, the changing volume affects both biomass and substrate concentrations; concentration decrease alone is not evidence of biomass loss. The independent balance implementation checks this coupling before the figures are assembled.

The three-level ethanol-secretion rule adapts the construction in [Srisuma (2018)](#benchmark-source). This illustration uses levels of 10, 5 and 0 mmol gDW⁻¹ h⁻¹; the source's inhibitory co-culture example uses 10, 2 and 0 mmol gDW⁻¹ h⁻¹.

## Interactions

**`independent-shared`** separates identical independent pools, different independent pools and a shared-resource competition case. Matched control calculations isolate resource sharing. Direct and differentiated calculations assess numerical agreement for the explicitly stated maintenance-supported feed specification.

**`product-inhibition`** couples species through ethanol inhibition. The complete ethanol balance is retained, and a matched case disables the inhibition mechanism. The comparison isolates the declared inhibition law while preserving the other model choices.

**`resource-diversion`** makes a donor secretion capacity depend on recipient biomass. Batch and fed-batch calculations include corresponding no-diversion controls and differentiated KKT calculations. The derivative of the biomass-dependent bound is part of the formulation being assessed.

**`biomass-transfer`** accounts for biomass transfer, dilution and loss. Balance checks distinguish redistribution from new biomass production. Interpret species trajectories together with total amounts and the transfer terms.

**`pyruvate-crossfeeding`** models chemically specified transfer, with active and quiescent states. The growth/quiescence mechanism is an explicit biological modelling assumption. Cross-feeding is assessed from the transferred metabolite, its exchange fluxes and the coupled balances.

## Genome-scale communities

**`genome-scale-yeasts`** evaluates isolated yeast cultures and an inhibited pair. Network scale and reactor coupling extend the smaller examples; they do not remove the need to state uptake, maintenance and inhibition assumptions.

The reconstruction choices, effective uptake parameters and the inhibited pair's three-level secretion schedule follow the examples in [Srisuma (2018)](#benchmark-source). The present calculations represent glucose uptake through concentration-dependent capacity bounds rather than the source's prescribed exchange fluxes. The isolated-culture calculations impose no ethanol-production lower-bound policy.

**`cstr`** evaluates a 72 h light/dark yeast–algae cultivation with initial-loading and numerical sensitivities. The light input, substrate trajectories, biomasses and feasible flux intervals should be read together. The full case includes light and dark comparisons plus loading checks, rather than treating a single terminal biomass value as the result. The adopted commands specify the horizon and solver settings explicitly.

## Feed control and neural models

**`control-neural`** compares SLSQP, PSO and Secant-PSO for the same seven-interval feed problem. The objective is a terminal biomass-composition error, not total biomass yield or product productivity. The simulation comparison uses one fixed response table; refined tables and direct KKT calculations check selected policies. The random seeds, function-evaluation budgets and numerical tolerances are retained in the scripts and result records.

The shared-glucose fed-batch settings, kinetic parameter choices and equal-composition target are adapted from [Srisuma (2018)](#benchmark-source). The current problem uses uptake-capacity constraints and defines the terminal ratio as `X_SC / X_KP`; the source prescribes glucose exchange and uses `X_KP / X_SC`. Both ratios equal one at the target, but their squared errors differ away from it. The algorithm comparison therefore concerns the current problem as specified in the code, without assuming equivalence to the source's optimisation problem.

The neural response is `z = ln(X_SC(tf)/X_KP(tf))`; the composition objective follows as `J = expm1(z)^2`. Training, validation, test and independent audit sets have distinct roles. Validation selects the model; the audit set evaluates it. A low prediction error on sampled controls does not by itself validate the feed chosen by an optimiser: that policy is also evaluated using the mechanistic model.

**`neural-audit-statistics`** calculates MSE, RMSE, MAE and R² for fixed predictions on the common 200-control audit. It does not retrain a network. Log-ratio errors and composition-objective errors are reported separately because the transformation is nonlinear. See the [audit guide](../experiments/neural_audit_diagnostics/README.md).

## What each output establishes

Numerical residuals assess the equations actually solved. Refinement checks assess sensitivity to numerical settings. Matched biological controls isolate the specified mechanism. Experimental agreement assesses a model against observations. Surrogate statistics assess agreement with simulator outputs. These types of evidence are complementary and should retain their separate meanings when extending the studies.

## Benchmark source

Srisuma, Prakitr (2018). *Dynamic Simulation and Optimisation of Biochemical Engineering Systems Using the Interior Point Approach*. MPhil dissertation in Advanced Chemical Engineering, Department of Chemical Engineering and Biotechnology, University of Cambridge, August 2018. The isolated-yeast examples are on pp. 21–25, the secretion schedule is Equation 37i on p. 27, and the fed-batch control benchmark is on pp. 29–31, including Table 6 and Equation 39. No DOI or public full-text link is asserted.

## Constrained biomass and composition design

The [constrained-control extension](../experiments/constrained_control/README.md) maximises terminal biomass amount while constraining species ratio and volume over twelve target cases. Direct single-start/multistart, database, quadratic and neural strategies share the same 300-attempt budget. The distributed record contains all 180 optimisations, grouped surrogate datasets, fixed-policy initial-condition sensitivity and two prospectively specified direct checks. Nominal feasibility, prediction error, complete cost and sensitivity are distinct endpoints. See the experiment guide for the design, seeds, inputs, command interface and measured results.

## Adaptive feedback under affinity mismatch

**`adaptive-feedback`** compares open-loop, nominal feedback, full and limited adaptive direct optimisation, and unchecked/checked local-affine controllers. Six initial-condition/target combinations and four mismatch/noise regimes give 144 episodes. The objective is biomass amount under terminal composition and volume constraints. An affinity estimator uses past observations; sensitivities define a local affine approximation at each decision. This model is distinct from the previously described terminal neural surrogate. The [experiment guide](../experiments/adaptive_feedback/README.md) explains the fixed protocol, exact primary records, separate execution identities, commands and qualified outcomes. All response-table inputs required for this computational assessment are included. The full strict numerical audit retains nine flags; the finite retrospective feed search is a diagnostic, not a controller baseline.

## Terminal recovery and admissible targets

**`terminal-recovery`** asks whether terminal composition targets remain attainable after different feeding histories under the specified feed/volume constraints. A finite two-step min–max recursion uses exact observed states, 21 or 41 candidate feed rates and three affinity disturbances per step. The fixed evaluation panel, selected interval witnesses and continuous open-loop local baseline retain separate meanings and denominators. The [study guide](../experiments/terminal_recovery/README.md) links all 4,860 primary paths, matched-history target sets, action-grid sensitivities and independent amount-state checks. Package verification confirms the complete records, including original failed audit comparisons; it does not turn those comparisons into a pass.

## Causal scenario feedback

**`scenario-feedback`** compares certainty-equivalent control, common scenario-open plans, scenario-tree feedback and local proposals with nonlinear checks and direct fallback. The complete panel is 88 cases by five methods, evaluated with two model families. All 440 episodes and 40 isolated serial outcomes are supplied. Overall outcome counts, numerical discrepancies and measured costs qualify the method comparisons. The [study guide](../experiments/scenario_feedback/README.md) also identifies a distinct bacterial light-input prediction test with attributed published observations. That test concerns a different organism, actuator and response; it provides no validation of yeast feed-control benefit.

Both cases expose `verify` by default. Their `prepare` commands copy scientific inputs into a new directory, preserve source identities and supply the ordered long-calculation commands without copying results. Supporting control publications are collected in [control_references.bib](control_references.bib).

## Yeast numerical and experimental assessment

`yeast-assessment` separates bound-expansion, barrier and integration contributions and compares independently solved LP problems. It also asks whether an early-time calibration predicts later observations and whether yeast reconstructions predict the independent oxygen-limited Wiebe culture conditions. The failed chronological forecast and external-growth discrepancies limit biological conclusions despite passed numerical checks. See the [study description](../experiments/yeast_assessment/README.md) for all denominators and source roles.

## Sensing and actuation limits

`sensing-actuation` extends terminal target-set analysis to finite feed resolution, bounded ratio observations and continuous full-state candidate searches. All fixed target problems and selected feasible-set components have distinct denominators. Strict endpoint classifications remain unresolved at numerical boundary scale, and one high-resolution root is incomplete. These are explicit limitations on the finite computational evidence. See the [study description](../experiments/sensing_actuation/README.md).
