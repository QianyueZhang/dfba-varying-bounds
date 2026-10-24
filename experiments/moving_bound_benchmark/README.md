# Analytic verification of moving-bound differentiation

This deterministic verification problem isolates the two moving-bound terms in the differentiated finite-barrier KKT equations. It has a closed-form solution, an independently evaluated scalar root, a full equality-constrained KKT derivative integration, and deliberate missing-term diagnostics. No external model matrix or experimental dataset is required.

Run from any directory with the repository's numerical dependencies installed:

```sh
python run_benchmark.py
python render_figures.py
```

Each script resolves paths relative to itself. The first writes numerical records to `results/`; the second renders those saved results to `figures/`. Existing outputs with the same names are replaced. All calculations are deterministic; no random seed is required. Python, platform, NumPy and SciPy versions are recorded in `results/validation.json`. Runtime is measured but is not used to claim a performance advantage.

## Problem and exact solution

Maximise `0.7 v1 + 0.3 v2`, subject to `v1 − v2 = 0` and common strict bounds `l(t) < vi < u(t)` for both fluxes, with

`l(t) = 0.35 + 0.25 sin(0.8 t)` and `u(t) = 1.65 + 0.30 cos(0.6 t + 0.2)` on `0 ≤ t ≤ 12`.

The flux and bound values are dimensionless. Time is a synthetic coordinate expressed in hours; the frequencies therefore have units h⁻¹. The example does not describe an organism or cultivation experiment. The equality matrix `[1, −1]` has full row rank. For all time, `0.1 ≤ l ≤ 0.6` and `1.35 ≤ u ≤ 1.95`, so the interval width is at least 0.75 and a strict interior exists.

After eliminating `v1 = v2 = q`, minimising the negative linear objective with four logarithmic barrier terms gives

`−1 − 2 μ/(q−l) + 2 μ/(u−q) = 0`.

Write `w = u−l` and `k = 2 μ`. A numerically stable closed form is

`q = u − 2 w k / [w + 2 k + sqrt(w² + 4 k²)]`.

The equality multiplier in the convention `−c + Aᵀλ − μ/(v−l) + μ/(u−v) = 0` is `λ = 0.2`. The reference evaluates this expression directly. A separate Brent solve of the scalar first-order condition does not call the closed-form function.

The differentiated implementation solves the full three-by-three KKT linear system at each ODE evaluation. With `DL = μ/(v−l)²` and `DU = μ/(u−v)²`, its flux equation is

`(DL+DU) vdot + Aᵀλdot = DL ldot + DU udot`, with `A vdot = 0`.

Both derivative terms are nonzero over the trajectory. The partial sensitivities of q are `dq/dl = DL/(DL+DU)` and `dq/du = DU/(DL+DU)`, and each is checked by central differences of the independently written closed-form solution while the other bound is held fixed.

## Separate experiments

1. At `μ=0.2`, integrate the complete derivative from the exact finite-barrier initial condition with DOP853 (`rtol=1e−10`, `atol=1e−12`, maximum step 0.2 h). Compare at 601 output times with the exact solution and scalar root. Check equality, stationarity, positive output slacks and multiplier error.
2. Repeat sensitivity central differences for steps `10⁻²` through `10⁻⁶`. The small-step round-off plateau is retained.
3. Deliberately omit the lower-bound term, then separately omit the upper-bound term. These are faulty-formula diagnostics, **not alternative methods or reported failures of the complete formulation**. Each starts from exactly the same consistent initial state. An event terminates a diagnostic at minimum slack `10⁻⁶`, before a bound is reached. No state is clipped. This checks reported trajectory states; it does not certify every internal DOP853 trial stage.
4. Hold μ and all bounds fixed while varying ODE relative tolerance from `10⁻⁵` to `10⁻¹¹`; absolute tolerance is 1% of relative tolerance. The common maximum step is 2 h to allow the tolerance effect to be seen. This differs from the stricter primary-run step cap and is recorded explicitly.
5. Hold the physical bounds unchanged, with **no bound relaxation**, while varying μ over `0.2, 0.05, 0.01, 0.002, 0.0004`. Check the gap to the limiting LP optimum `q=u` against the four-inequality certificate `4 μ`. Separately check the integrated trajectory against the exact finite-μ solution.

The initial local run passed 39 checks. Exact numerical values and timings are in the saved validation record; the main complete-derivative trajectory agreed with the analytic flux to approximately `1.1×10⁻¹⁴` on the output grid. The tolerance sweep is the evidence for integration convergence; the much smaller primary error is specific to this smooth, small problem and its step cap.

## Scope

This example verifies the algebra and implementation of simultaneous lower- and upper-bound motion. It reduces to one independent flux and is intentionally simple enough to admit an analytic reference. It does not establish new sensitivity theory, biological validity, multivariable conditioning, state-feedback dynamics, moving-equality correctness, numerical scalability or superiority over another solver. Existing biological and larger-network studies address separate questions.

The assets are original calculations and diagrams. Their software is author-owned and can use the repository's MIT terms. No third-party matrix, publisher image or thesis document is needed.
