# Numerical and experimental assessment of the yeast model

This study separates three questions: agreement with the supplied cultivation observations, transfer beyond the fitted condition, and numerical differences introduced by bound expansion and finite-barrier regularisation. The package contains Python calculations, attributed numerical observations, protocols, complete outcomes and plotting data. Thesis and manuscript document sources are excluded.

## Inspect the results

From the repository root:

```sh
python experiments/yeast_assessment/run.py verify
python experiments/yeast_assessment/run.py figures
```

The first command checks distributed file identities, frozen scientific inputs and result counts. It retains the failed temporal forecast and the anaerobic growth limitation. A passed package check is not a claim that every scientific experiment succeeded. The second renders the signed numerical decomposition and external oxygen-condition comparison from unchanged result arrays. Both commands require no metabolic-model matrices.

`payload/paper1_improvement_2026-10-24/` contains the numerical, temporal and external-cultivation calculations. Directory names remain fixed because the protocols identify paths and hashes. The `manuscript/data/` and `manuscript/figures/` subdirectories hold numerical plotting assets only; no manuscript text or compiled document is distributed.

- `numerical/results/observation_decomposition.csv` records signed bound-expansion, finite-minus-expanded-LP and total differences at observation times. The signed terms sum to the total; they are not independent error variances.
- `temporal/results.json` retains the incomplete extrapolation, while `temporal/failure_diagnostic.json` assesses its feasibility mechanism and matched capacity/prescribed-uptake calculation.
- `external/results.json` contains 92 primary condition/model/constraint records and all 276 LP calls.
- `external/recalibration/results.json` contains the explicitly retrospective one-condition maintenance assessment, with 20 records and 72 calls. The nominal dilution rate is a process reference, not a separately measured growth series.

The biomass observation is OD600. Published ethanol C-mmol values in the external study are divided by two to obtain compound mmol, with the same conversion applied to reported SEM. The fitted maintenance parameter improves conditional growth agreement, but oxygen-capacity predictions need not equal the observed uptake used to specify the constraint. Zero-growth anaerobic flux ranges are not presented as validation of a growing culture.

## Calculate in a new workspace

The analytic case requires no restricted model inputs:

```sh
python experiments/yeast_assessment/run.py prepare /tmp/yeast-analytic-study --case analytic
python /tmp/yeast-analytic-study/research_completion/depth_review_2026-10-22/moving_bound_benchmark/run_benchmark.py
```

Other cases require exact model files identified in `payload/INPUT_MANIFEST.json`:

```sh
python experiments/yeast_assessment/run.py prepare /tmp/yeast-numerical-study --case numerical-all --source-root /path/to/lawfully-held/model-tree
```

The preparation command refuses an existing destination and preserves supplied results. It records the ordered calculation commands in `RUN_RECORD.json`. Run those commands from the prepared workspace. Available cases include `numerical-lp`, `numerical-tight`, `numerical-small-barrier`, `numerical-all`, `temporal`, `temporal-diagnostic`, `external` and `recalibration`. The temporal calculation may legitimately terminate with its recorded model-feasibility failure; inspect the result and retain the failure.

Use Python 3.11 with `payload/requirements.txt`. The external cached postprocessing wrapper requires SciPy 1.15.1 and retains its recorded equivalence check. Neither package verification nor that implementation check establishes a speed advantage.

## Attribution and rights

Zampar et al. (2013), DOI [10.1038/msb.2013.11](https://doi.org/10.1038/msb.2013.11), supply the batch observations under the recorded CC BY 3.0 terms. The small condition-level Table 1 extraction is attributed to Wiebe et al. (2008), DOI [10.1111/j.1567-1364.2007.00234.x](https://doi.org/10.1111/j.1567-1364.2007.00234.x); no open licence for the complete publisher article is asserted. Publisher files and exact reconstruction matrices are excluded. Author-owned Python code uses MIT; see the root notices and `payload/THIRD_PARTY_NOTICES.md` for source-specific terms.

## Integration-count convention

The current numerical producer reports `accepted_steps = len(sol.t) - 1` and `saved_point_count = len(sol.t)`. The historical result field named `accepted_steps` counted the initial saved point: the tighter-pair and smaller-barrier calculations therefore contain 309 and 299 accepted integration steps, respectively, with 310 and 300 saved points. `payload/paper1_improvement_2026-10-24/numerical/STEP_COUNT_ERRATUM.json` records this metadata correction. Historical numerical outcomes and their original source/protocol identities are preserved; no state, fit, bound, flux, tolerance or scientific conclusion is changed.
