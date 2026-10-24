# Continuous minimax comparison

All 72 assigned method–problem combinations completed: six prescribed uniform-history roots, six fixed targets, two methods, and nine disturbance paths per method–problem. These are mathematical benchmark cases, not independent biological replicates.

The independently deployed common-open plans reached the strict inner band in **1/36** problems and actual acceptance in **2/36**. Causal deployment reached the inner band in **9/36** and actual acceptance in **12/36**. The outer-band counts equal the inner counts. Every formal success requires complete execution, a valid selected plan and physical validity on all nine paths. Applying this gate changed zero raw error-only labels.

All 648 paths passed physical checks. All 864 matched-interval comparisons and 648 accumulated same-feed endpoint comparisons passed their original numerical allowances and all corresponding band labels agreed. The maximum matched log-ratio discrepancy was `1.1365801633189676e-9`; the maximum accumulated discrepancy was `9.676726087093357e-10`. The largest amount error was `0.029221600264286954` times its absolute-plus-relative allowance. No prior failed audit was relabeled by this result.

The common-open inner success occurs for the table-family 0.9 L root, target 2, at error `0.006749581899048174`. Thus zero common-open successes on a finite grid cannot establish continuous-control impossibility. Causal deployment has smaller error by more than the descriptive `2e-7` comparison scale in 12 paired problems; 24 pairs are within that scale. There is no statistical test here, and neither method certifies a global continuous optimum. Planned tree errors and actual observation-based deployment errors remain separate records.

The archive retains all 141,537 evaluated candidates (68,741 open and 72,796 causal). The 142 differential-evolution budget exhaustions among 144 calls are recorded separately from 76 nonconverged SLSQP exits among 360 calls. There were zero optimizer exceptions and zero invalid candidates. All 28 bounded scalar refinements converged; the causal controller made 108 observation-based continuation decisions using a fixed scan and a common seed list, without the true disturbance index.

The shared 41-action grids cost 53,910 model transitions. Open optimization used 824,892 transitions; causal optimization used another 873,552 and depended on the open incumbents. Continuation search used 54,639 transitions. These costs are not a matched-budget or real-time performance claim. The three-worker campaign wall time was 2,338.637 s, excluding precomputed grid construction. Summed solve times were 2,874.537 s for open and 2,921.466 s for causal across concurrent problems; continuation search took a summed 168.537 s. Verification costs are separately retained in `formal_report.json`.

## Records and reproduction

- `formal_report.json`: physically gated, complete aggregation and input hashes.
- `summary.json`: original frozen runner aggregation; its deployed labels are error-only fields, with physical outcomes separately recorded.
- `*_target_*.json`: selected plans, independent deployments, complete path-level comparisons, solver statuses and costs.
- `*_candidates.json.gz`: every evaluated optimization candidate with controls, objective and validity metrics.
- `../../protocols/continuous_protocol.json`: frozen design, seeds, budgets, input and source hashes.
- `../../protocols/continuous_reporting_note.json`: explicit reporting-gate clarification, without changing numerical outcomes.
- `../../manuscript/continuous_comparison_section.tex`: methods and result text.
- `../../figures/continuous_comparison.pdf`: vector figure showing every fixed problem.

For a fresh source-only reproduction, first produce the successful numerical gate and six refined grid41 outputs. In a fresh working directory without active continuous protocols, run `python scripts/prepare_continuous_reproduction.py --reference PATH_TO_ARCHIVED_CONTINUOUS_PROTOCOL`, then `python scripts/continuous_run.py --workers 3`, `python scripts/continuous_report.py`, and `python scripts/continuous_figure.py`. The preparer preserves the numerical sources and scientific inputs, refuses existing active protocols/outcomes, and records fresh run hashes. Numerical outputs and timings are not expected to be byte-identical across machines. Running the frozen runner in the original completed directory resumes existing outcomes; it is not a new numerical reproduction.
