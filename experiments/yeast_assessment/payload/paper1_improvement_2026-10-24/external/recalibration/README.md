# Retrospective single-condition maintenance assessment

Only the20.9%oxygen capacity-condition nominal growth reference0.10h⁻¹ was used to fit the existing ATP-maintenance lower bound. The resulting value is1.01606438094mmol gDW⁻¹h⁻¹. All other network and medium settings remain fixed. Original external outcomes were already inspected, so this is retrospective assessment, not blind validation.

- original: positive-oxygen held-out growth MAE0.0472856h⁻¹ across the3conditions.
- single_condition_recalibration: positive-oxygen held-out growth MAE0.00963723h⁻¹ across the3conditions.

Growth agreement improves after one-condition maintenance adjustment; this limits attribution of the original transfer deficit to a general network or formulation failure. Remaining errors and zero anaerobic maximal growth persist. At the3positive-oxygen assessment conditions, predicted ethanol is approximately0,1.5823and2.9414mmol gDW⁻¹h⁻¹ compared with0.1,1.6and2.75. No ethanol data were fit.

The capacity formulation was used for calibration. Applying the same parameter to prescribed glucose/oxygen at20.9%givesgrowth0.09115h⁻¹. This is not a fair independently calibrated empirical comparison between semantics and cannot establish capacity superiority. Other positive-oxygen assessment predictions coincide to numerical precision.

At0%oxygen, maximumgrowth remainszero and ethanolFVA spans0–12.3667. A broad feasible ethanol interval at zero growth is not a successful prediction of the measured growing anaerobic culture.

Every objective solve, feasibility-bracketing call and ethanol FVA solve is recorded in results.json. All successful vectors are retained in flux_vectors.npz; no model repairs or held-out-error parameter selection occurred.

Run in a fresh copy with the completed generated result files absent: `python run_recalibration.py`. It uses the previously verified cached SciPy postprocessing wrapper, checks all frozen input hashes, and leaves source experiments unchanged.
