# Grid and bounded-observation study

This report recounts the saved finite-tree results. It does not perform new integration or change policy selection.

Primary analysis complete: True; all algebra inclusion checks passed: True.
Primary parallel wall time: 1229.0 s.
Adaptive321 complete: False; completed roots: 5/6; additional wall time: 3615.2 s.

## Fixed-target outcomes

- 41 actions, full: 1/36 inner-band, 5/36 actual-band all-path successes (6 roots).
- 41 actions, open: 0/36 inner-band, 0/36 actual-band all-path successes (6 roots).
- 41 actions, ratio ε=0: 1/36 inner-band, 5/36 actual-band all-path successes (6 roots).
- 41 actions, ratio ε=0.001: 1/36 inner-band, 5/36 actual-band all-path successes (6 roots).
- 41 actions, ratio ε=0.005: 1/36 inner-band, 5/36 actual-band all-path successes (6 roots).
- 41 actions, ratio ε=0.01: 0/36 inner-band, 0/36 actual-band all-path successes (6 roots).
- 41 actions, ratio ε=0.02: 0/36 inner-band, 0/36 actual-band all-path successes (6 roots).
- 81 actions, full: 7/36 inner-band, 11/36 actual-band all-path successes (6 roots).
- 81 actions, open: 0/36 inner-band, 1/36 actual-band all-path successes (6 roots).
- 81 actions, ratio ε=0: 7/36 inner-band, 11/36 actual-band all-path successes (6 roots).
- 81 actions, ratio ε=0.001: 7/36 inner-band, 11/36 actual-band all-path successes (6 roots).
- 81 actions, ratio ε=0.005: 7/36 inner-band, 11/36 actual-band all-path successes (6 roots).
- 81 actions, ratio ε=0.01: 2/36 inner-band, 5/36 actual-band all-path successes (6 roots).
- 81 actions, ratio ε=0.02: 0/36 inner-band, 1/36 actual-band all-path successes (6 roots).
- 161 actions, full: 9/36 inner-band, 12/36 actual-band all-path successes (6 roots).
- 161 actions, open: 0/36 inner-band, 2/36 actual-band all-path successes (6 roots).
- 161 actions, ratio ε=0: 9/36 inner-band, 12/36 actual-band all-path successes (6 roots).
- 161 actions, ratio ε=0.001: 9/36 inner-band, 12/36 actual-band all-path successes (6 roots).
- 161 actions, ratio ε=0.005: 8/36 inner-band, 12/36 actual-band all-path successes (6 roots).
- 161 actions, ratio ε=0.01: 3/36 inner-band, 7/36 actual-band all-path successes (6 roots).
- 161 actions, ratio ε=0.02: 0/36 inner-band, 2/36 actual-band all-path successes (6 roots).
- 321 actions, full: 4/30 inner-band, 7/30 actual-band all-path successes (5 roots).
- 321 actions, open: 0/30 inner-band, 0/30 actual-band all-path successes (5 roots).
- 321 actions, ratio ε=0: 4/30 inner-band, 7/30 actual-band all-path successes (5 roots).
- 321 actions, ratio ε=0.001: 4/30 inner-band, 7/30 actual-band all-path successes (5 roots).
- 321 actions, ratio ε=0.005: 4/30 inner-band, 7/30 actual-band all-path successes (5 roots).
- 321 actions, ratio ε=0.01: 2/30 inner-band, 3/30 actual-band all-path successes (5 roots).
- 321 actions, ratio ε=0.02: 0/30 inner-band, 0/30 actual-band all-path successes (5 roots).

## Finest completed grid per prescribed root

This aggregate uses the explicitly listed finest completed grid for each root. If any adaptive root is unresolved, it is not a complete 321-action panel.

analytic_v0.9_uniform: 321 actions (complete); analytic_v1.3_uniform: 321 actions (complete); analytic_v1.7_uniform: 321 actions (complete); table_v0.9_uniform: 161 actions (unresolved); table_v1.3_uniform: 321 actions (complete); table_v1.7_uniform: 321 actions (complete).

- full: 9/36 inner-band and 12/36 actual-band all-path successes.
- open: 0/36 inner-band and 2/36 actual-band all-path successes.
- ratio ε=0: 9/36 inner-band and 12/36 actual-band all-path successes.
- ratio ε=0.001: 9/36 inner-band and 12/36 actual-band all-path successes.
- ratio ε=0.005: 8/36 inner-band and 12/36 actual-band all-path successes.
- ratio ε=0.01: 4/36 inner-band and 7/36 actual-band all-path successes.
- ratio ε=0.02: 0/36 inner-band and 2/36 actual-band all-path successes.

## Descriptive grid-stability checks

- 41→81: 0/42 nonvacuous passes, 6 empty/empty vacuous passes, 36 failures.
- 81→161: 0/42 nonvacuous passes, 5 empty/empty vacuous passes, 37 failures.
- 161→321: 6/35 nonvacuous passes, 5 empty/empty vacuous passes, 24 failures.

Both relative symmetric-difference length ≤5% and endpoint Hausdorff distance ≤0.001 are required. Passing is descriptive stability, not a convergence proof. Failing is an unresolved grid gap.

## Separate post hoc set-distance diagnostic

- 41→81: 0/32 comparisons nonempty on both grids have target-set Hausdorff distance ≤0.001.
- 81→161: 3/36 comparisons nonempty on both grids have target-set Hausdorff distance ≤0.001.
- 161→321: 9/30 comparisons nonempty on both grids have target-set Hausdorff distance ≤0.001.

This diagnostic uses distance between the target sets themselves, rather than their component endpoints. It does not replace the frozen gates or change adaptive root selection.

## Sensing frontiers

- analytic_v0.9_uniform, 321 actions: target 1.5: ε<0.007607226459; target 2: ε<0.01107814267.
- analytic_v1.3_uniform, 321 actions: none of the six fixed targets has an admitted sensing budget.
- analytic_v1.7_uniform, 321 actions: none of the six fixed targets has an admitted sensing budget.
- table_v0.9_uniform, 161 actions: target 0.75: ε<0.006046098334; target 1: ε<0.01212718227; target 1.25: ε<0.01130980879; target 1.5: ε<0.009036071928; target 2: ε<0.003383869552.
- table_v1.3_uniform, 321 actions: target 1.5: ε<0.005789389588; target 2: ε<0.01162891629.
- table_v1.7_uniform, 321 actions: none of the six fixed targets has an admitted sensing budget.

Finite frontier boundaries are excluded because observation supports are closed. The 1e−7 numerical compatibility pad has already been subtracted. Targets with no feasible nonnegative error allowance are reported as none; a raw zero-clamped display field must not be interpreted as feasibility at zero.

## Interpretation and verification

The frozen study evaluates target feasibility in a finite numerical model with bounded scalar composition measurements. Full-state feedback, ratio-only feedback and common-open plans have explicitly different information. The ratio-only policy accepts measured log ratio and chooses one feed shared by all compatible branches; it has no true scenario-label input.

Independent deployment and source checks are recorded separately in audit/grid_observation_*; a passed algebra check or development integrator gate is not a substitute for those deployment outcomes.

Figures: figures/maturity_grid_resolution.pdf; figures/maturity_observation_resolution.pdf; figures/maturity_sensing_budget.pdf. The sensing-budget matrix uses “—” for no admitted budget and “open” for a successful common-open plan; “open” is categorical, not a finite upper bound.

Independent primary audit: 1053 cases (756 fixed, 297 component points); 32703 logical observation-path episodes. Numerical comparisons passed 2553/2553; optimistic band-label differences 63, conservative differences 37. Overall audit passed=False; source hashes unchanged=True. Counts describe deterministic checks, not independent experimental replicates.

Independent adaptive audit: 444 cases (210 fixed, 234 component points); 13734 logical observation-path episodes. Numerical comparisons passed 1548/1548; optimistic band-label differences 53, conservative differences 29. Overall audit passed=False; source hashes unchanged=True. Counts describe deterministic checks, not independent experimental replicates.
