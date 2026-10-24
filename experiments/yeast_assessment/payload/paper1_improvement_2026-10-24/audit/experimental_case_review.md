# Experimental yeast case: independent source and implementation review

The Zampar comparison is useful as an exploratory, source-linked experimental benchmark for the numerical formulation. It does not establish blind external validation, uniquely identified physiology, or accuracy of an uncompleted differentiated trajectory. This review read the stored primary publication XML, extraction code, source CSV, calibration/direct-solver code and retained outcomes; it did not change the original case.

## Experimental observations and provenance

Primary publication: Zampar et al., *Molecular Systems Biology* 9, 651 (2013), DOI `10.1038/msb.2013.11`. The stored primary XML is `research_completion/runs/figure11_source_review_2026-10-21/candidates/zampar2013.xml`. Source workbooks and extraction are in `research_completion/runs/figure11_source_review_2026-10-21/candidates/zampar2013/`.

- The source ZIP SHA256 is `d928c613be23fe67e6af1ec2e75c88390a4a48b7824c349fca736081c5cecaeb`; extracted `zampar2013_observations.csv` SHA256 is `07529687404d471e25363590939322397d1fae08b6c28dede451b8142036ade0`.
- `extract_observations.py` selects `Overview!C:E`, preserving workbook, source row/cell, literal series labels, duplicate times and zeros. C is time in hours relative to experimental glucose exhaustion; D is OD600 or glucose/ethanol in g/L; E is a converted quantity. Regression predictions and reconstructed rates are excluded.
- There are 266 source rows: 138 biomass OD600, 65 glucose, 63 ethanol. The adopted interval is -6.8 to 8.0583333335 h, retaining 256 rows and excluding 10 earlier readings by initialization time.
- Biomass is measured as OD600. The workbook's dry weight is exactly 0.485 times OD and is not an additional independent observation. The primary Methods use 0.486 gDW/L per OD, measured in a separate 10 g/L batch. The adopted model explicitly uses 0.486 for its initial mass and observation operator.
- The primary Methods describe FY4 at 30 degrees C, pH 5, 2 L working volume, nominal 5 g/L glucose, 1200 rpm and 1 vvm air, with three biological cultures. The biomass workbook has four series, while glucose and ethanol have three. Cross-workbook culture identity is unresolved; repeated rows are not automatically independent biological replicates.
- The reviewed experimental cells and Methods do not provide per-observation SD/SE or a numerical LOD/LOQ. There are 16 glucose zeros in the adopted interval. The one zero ethanol observation in the complete source occurs before this interval. Neither zero values nor residual-balancing scales provide an experimental uncertainty model.

## Calibration, derived quantities and information reuse

Authoritative code: `research_completion/scripts/figure11_zampar_2026-10-21/fit_batch.py`; retained outputs: `research_completion/runs/figure11_zampar_2026-10-21/`.

The fit uses only each workbook's Exp1-labelled rows: 13 OD + 16 glucose + 16 ethanol = 45 observations. The other 211 rows (120 OD + 47 glucose + 44 ethanol) are withheld from the parameter objective. The objective sums variable-wise mean squared residuals divided by squared fixed scales [2 OD, 5 g/L, 2 g/L]. Those scales are balancing weights, not standard deviations. The code contains no direct inclusion of withheld values in the residual objective.

The initial glucose 4.84384260862773 g/L and ethanol 0.22730440275683 g/L are the Exp1 `D13` observations at -6.8 h. They are also counted in the 45 calibration scores; their zero residuals arise by construction. Initial OD 0.2028090994103474 is log-interpolated between Exp1 OD(-7.5 h)=0.16 and OD(-6.5 h)=0.2245, giving X0=0.09856522231342883 gDW/L. This uses an observation 0.3 h after the nominal simulation start. It is a declared retrospective initialization, not a strictly causal forecast made at -6.8 h.

Three effective parameters were fitted in the unexpanded LP model: qGmax=11.005414756123924, qOmax=3.919072884652486 and ATP maintenance=5.565032869183548 mmol/gDW/h. The ethanol ceiling is fixed at 10; KG=0.01 g/L, KE=0.1 g/L, repression scale=0.05 g/L and exponent=4 are fixed choices. The biomass growth drain is used without the older 0.56 multiplier. The oxygen capacity is not a measured oxygen-transfer coefficient; the ATP demand is not an independently measured FY4 maintenance parameter. The genome-scale fluxes, growth rates and plotted exchange-rate curves are model outputs.

The 211 withheld rows support an observation-level series assessment. They are not 211 independent cultures. All series are already available on a clock aligned to observed glucose exhaustion, so the comparison conditions on that experimental event and does not independently predict its absolute time from inoculation. Range-normalized RMSE uses the evaluation-set range as a descriptive denominator, not a training feature or uncertainty estimate.

The retained baseline, induction and maintenance variants include their withheld-series scores. This demonstrates exploratory structural investigation; it does not prove that final model choice was blind to withheld outcomes. Although no withheld observation enters the final parameter objective, the archive cannot support a preregistered blind-test claim. A new temporal split is necessarily a retrospective temporal assessment. It must refit using only the training window and disclose the event-aligned clock and initialization information; reusing the existing fit would leak later Exp1 measurements into the forecast.

## Existing numerical evidence and its limits

`direct_run.py` adopts a complete LSODA trajectory with a converged direct finite-barrier KKT solve at every RHS. Warm LP solutions are Newton predictors; the adopted flux is not an LP interpolation. The displayed case uses epsilon=1e-5 and mu=1e-7 with exact dynamic glucose/ethanol uptake lower bounds; only remaining/fixed bounds are expanded. `differentiated_run.py` is incomplete and cannot support a claim that the complete experimental curve was produced by integrated multipliers.

`direct_audit.json` reports 1220 KKT solves, maximum stationarity residual 1.91794e-7, maximum equality residual 8.58278e-11 and minimum slack 1.11717e-7. Its maximum differences from LP0 are [0.0015226654 gDW/L, 0.0023499996 g/L, 0.0013942406 g/L] for [X,G,E]. The stored coarse/fine direct difference is [9.21048e-7, 1.25396e-6, 2.65024e-6] in those units. These are trajectory comparisons, not rigorous global error bounds.

The adopted withheld RMSE is [0.0689936088 OD, 0.0599767567 g/L glucose, 0.1424544966 g/L ethanol]. Numerical differences and these experimental residuals have different meanings. Being numerically smaller than residuals does not validate the biological model, identify parameter uncertainty, or establish new experimental agreement.

The existing direct-minus-LP0 comparison combines bound expansion, barrier effects and integration differences. Opposite-signed expansion and barrier contributions can cancel. Existing `sensitivity.json` already shows LP expansion effects, while `barrier_point_checks.json` checks static states; neither alone separates the full-trajectory signed components at observation times.

## Executable next analysis

Keep the original scripts and results unchanged. The numerical extension is assigned to `paper1_improvement_2026-10-24/numerical/` and uses the same fitted parameters, model and physical bound convention. Freeze source/input hashes and output checks before running two direct trajectories: (epsilon,mu)=(1e-6,1e-8) and (1e-5,1e-8). Compute LP0 and same-epsilon LP trajectories without refitting. At every original observation time, retain the signed expansion term LP_epsilon-LP0, finite-minus-LP_epsilon term, and their exact sum finite-minus-LP0; report maximum absolute and RMS values in observation units as well as concentration units. Retain cancellation diagnostics and all failures. Report the archived adopted-pair coarse/fine comparison separately; it is not an error certificate for the new parameter pairs. A full new integration-refinement campaign would be a separately declared follow-up if these results require it.

The parent agent owns temporal prediction. A defensible additional assessment refits Exp1 using an explicitly frozen training cutoff, keeps later samples out of optimization, and scores all assigned later observations with per-variable counts, physical units and simple baselines. Since the dataset and model have already been explored, describe this as a retrospective temporal holdout, not a new untouched dataset. Avoid fitting additional inactive ethanol capacities or presenting a Hessian-based confidence interval as measurement uncertainty.

All paths above are relative to `/Users/qzhang/Documents/ChatGPT/Thesis`.
