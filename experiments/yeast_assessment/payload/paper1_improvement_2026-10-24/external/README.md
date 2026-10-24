# Independent steady-state constraint-transfer challenge

This experiment uses the original yeast network, archived growth objective and frozen ATP-maintenance lower bound. It supplies measured glucose/oxygen fluxes from Wiebe et al. (2008), Table 1, as input constraints and compares predicted growth and ethanol secretion with the published outputs. There is no parameter refitting.

Protocol SHA-256: `3a57b04c332a36ce42c812b2266b23a8ccc41577d6f97b9106410d4dc16da445`.
Completed primary LPs: 92; total LP calls including ethanol variability: 276; elapsed computation: 15.2 s.

## archived_medium / capacity

Nominal optima: 5/5; optima across nominal and SEM-corner cases: 23/23. Numerical-validity checks pass for 23 cases.

- 20.9% inlet oxygen: growth 0.0702001 h⁻¹ (reference0.10), ethanol near-optimal range [0, 7.6292e-08] mmol gDW⁻¹ h⁻¹ (reference0). Valid input scenarios 5/5.
- 2.8% inlet oxygen: growth 0.0692602 h⁻¹ (reference0.10), ethanol near-optimal range [0.449496, 0.449496] mmol gDW⁻¹ h⁻¹ (reference0.1). Valid input scenarios 5/5.
- 1% inlet oxygen: growth 0.0492983 h⁻¹ (reference0.10), ethanol near-optimal range [2.26158, 2.26158] mmol gDW⁻¹ h⁻¹ (reference1.6). Valid input scenarios 5/5.
- 0.5% inlet oxygen: growth 0.0395847 h⁻¹ (reference0.10), ethanol near-optimal range [3.61389, 3.61389] mmol gDW⁻¹ h⁻¹ (reference2.75). Valid input scenarios 5/5.
- 0% inlet oxygen: growth 0 h⁻¹ (reference0.10), ethanol near-optimal range [0, 12.3667] mmol gDW⁻¹ h⁻¹ (reference8.35). Valid input scenarios 3/3.

## archived_medium / prescribed

Nominal optima: 5/5; optima across nominal and SEM-corner cases: 23/23. Numerical-validity checks pass for 23 cases.

- 20.9% inlet oxygen: growth 0.0702001 h⁻¹ (reference0.10), ethanol near-optimal range [0, 7.6271e-08] mmol gDW⁻¹ h⁻¹ (reference0). Valid input scenarios 5/5.
- 2.8% inlet oxygen: growth 0.0692602 h⁻¹ (reference0.10), ethanol near-optimal range [0.449496, 0.449496] mmol gDW⁻¹ h⁻¹ (reference0.1). Valid input scenarios 5/5.
- 1% inlet oxygen: growth 0.0492983 h⁻¹ (reference0.10), ethanol near-optimal range [2.26158, 2.26158] mmol gDW⁻¹ h⁻¹ (reference1.6). Valid input scenarios 5/5.
- 0.5% inlet oxygen: growth 0.0395847 h⁻¹ (reference0.10), ethanol near-optimal range [3.61389, 3.61389] mmol gDW⁻¹ h⁻¹ (reference2.75). Valid input scenarios 5/5.
- 0% inlet oxygen: growth 0 h⁻¹ (reference0.10), ethanol near-optimal range [0, 12.3667] mmol gDW⁻¹ h⁻¹ (reference8.35). Valid input scenarios 3/3.

## unlimited_lipid_availability / capacity

Nominal optima: 5/5; optima across nominal and SEM-corner cases: 23/23. Numerical-validity checks pass for 23 cases.

- 20.9% inlet oxygen: growth 0.0715789 h⁻¹ (reference0.10), ethanol near-optimal range [0, 7.95778e-08] mmol gDW⁻¹ h⁻¹ (reference0). Valid input scenarios 5/5.
- 2.8% inlet oxygen: growth 0.0699768 h⁻¹ (reference0.10), ethanol near-optimal range [0.455999, 0.455999] mmol gDW⁻¹ h⁻¹ (reference0.1). Valid input scenarios 5/5.
- 1% inlet oxygen: growth 0.0498084 h⁻¹ (reference0.10), ethanol near-optimal range [2.26621, 2.26621] mmol gDW⁻¹ h⁻¹ (reference1.6). Valid input scenarios 5/5.
- 0.5% inlet oxygen: growth 0.0399943 h⁻¹ (reference0.10), ethanol near-optimal range [3.61761, 3.61761] mmol gDW⁻¹ h⁻¹ (reference2.75). Valid input scenarios 5/5.
- 0% inlet oxygen: growth 0 h⁻¹ (reference0.10), ethanol near-optimal range [0, 12.3667] mmol gDW⁻¹ h⁻¹ (reference8.35). Valid input scenarios 3/3.

## unlimited_lipid_availability / prescribed

Nominal optima: 5/5; optima across nominal and SEM-corner cases: 23/23. Numerical-validity checks pass for 23 cases.

- 20.9% inlet oxygen: growth 0.0715789 h⁻¹ (reference0.10), ethanol near-optimal range [0, 7.95983e-08] mmol gDW⁻¹ h⁻¹ (reference0). Valid input scenarios 5/5.
- 2.8% inlet oxygen: growth 0.0699768 h⁻¹ (reference0.10), ethanol near-optimal range [0.455999, 0.455999] mmol gDW⁻¹ h⁻¹ (reference0.1). Valid input scenarios 5/5.
- 1% inlet oxygen: growth 0.0498084 h⁻¹ (reference0.10), ethanol near-optimal range [2.26621, 2.26621] mmol gDW⁻¹ h⁻¹ (reference1.6). Valid input scenarios 5/5.
- 0.5% inlet oxygen: growth 0.0399943 h⁻¹ (reference0.10), ethanol near-optimal range [3.61761, 3.61761] mmol gDW⁻¹ h⁻¹ (reference2.75). Valid input scenarios 5/5.
- 0% inlet oxygen: growth 0 h⁻¹ (reference0.10), ethanol near-optimal range [0, 12.3667] mmol gDW⁻¹ h⁻¹ (reference8.35). Valid input scenarios 3/3.

## Interpretation limits

- No fit, biological replicate inference, or dynamic kinetic validation.
- Measured realized qG/qO used as inputs; these are not measured uptake capacities.
- Growth reference is nominal chemostat D0.10, not a new fitted growth parameter.
- Input SEM corners and reported output SEM comparisons are descriptive, not statistical acceptance tests.
- Unlimited lipid sensitivity changes potential carbon/energy and is not quantitative medium validation.
- Infeasible constraints and unsuccessful solver outcomes remain in the complete denominator.

## Files and reproduction

`observations.json` records source Table1/body-row/column and all unit conversions. `protocol.json` and its checksum were frozen before solving. `results.json` includes every nominal/corner case, statuses and FVA diagnostics; `results.csv` provides a flat summary. `flux_vectors.npz` retains every successful primary and FVA vector. `summary.json` is descriptive aggregation.

In a fresh copied directory, remove only the generated results files and execute `python run_challenge.py` followed by `python report_challenge.py` with NumPy/SciPy installed. Keep protocol, observations, inputs and numerical script unchanged; their hashes are verified. `prepare_inputs.py` documents original provenance extraction and is not required to rerun the already frozen study.

Source: [Wiebe et al. (2008), Table1](https://doi.org/10.1111/j.1567-1364.2007.00234.x). Source-download metadata and private full-text audit are retained separately. The published means are from CEN.PK113-1A, whereas maintenance was fit to Zampar FY4.
