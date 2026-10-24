# Fixed-prediction neural audit

Recompute MSE, RMSE, MAE and R² for the 175-, 350- and 700-training-case networks on the same 200 independent simulator controls. Render prediction/residual plots for the 700-case model. This workflow uses existing public NPZ fixtures and does not retrain a network, deserialize a pickle, rerun a reactor or require model matrices.

From this directory:

```sh
python run_diagnostics.py
```

The script finds the repository root above its directory, or accepts `--repository /path/to/dfba-varying-bounds`. It writes CSV/JSON statistics and SVG/PDF/PNG diagnostics under `results/`, or a directory supplied using `--output`. The default output is regenerated on subsequent runs.

The target is `z=ln(X_SC(tf)/X_KP(tf))`; the decision objective is `J=expm1(z)^2`. Statistics are computed separately for z and J. R² centres the reference on its mean over these 200 audit cases. All responses are dimensionless. Formula-based MSE and R² are checked against scikit-learn and RMSE against the existing run record. Dataset shape, finiteness, alignment, sample uniqueness, disjointness from the 1,000 fitting/selection/test controls, and stored validation-seed selection are checked explicitly.

Inputs already present in the repository:

- `research_completion/runs/surrogate_dataset.npz`
- `research_completion/runs/revision_validation/neural_fresh_audit.npz`
- `research_completion/runs/revision_validation/neural_audit_predictions.npz`
- `research_completion/runs/revision_validation/neural_extension.json`

No new random sampling occurs. The independent control set was originally generated with seed 2731; its arrays are fixed here. Source hashes and software versions accompany the computed results. This workflow checks saved-prediction statistics and their data allocation; without loading/retraining the original networks it does not independently check their weights, architecture or scaling. It assesses simulator–surrogate agreement, not experimental accuracy or performance at an optimiser-selected control. Mechanistic policy verification belongs to the existing `control-neural` workflow.

This author-owned analysis code is covered by the repository MIT Licence. Underlying data and other third-party material retain the terms described in the repository notices.
