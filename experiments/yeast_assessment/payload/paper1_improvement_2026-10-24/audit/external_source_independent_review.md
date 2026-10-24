# Independent source and design review: Wiebe 2008 challenge

Reviewed 24 October 2026 by the data-audit subagent. This is a source/code review; it does not certify numerical outcomes before the challenge finishes.

## Source extraction

The retained publisher HTML (`audit/sources/wiebe_2008_article.html`, SHA256 `3c730aeaed3de8f9b77f325017c3c3507f8c0f661701903d5ba623c28bd2b386`) was parsed independently using the Python standard-library HTML parser. All 45 body cells of Table 1 match `external/observations.json` exactly after whitespace normalization. All four HTML table copies are identical. Every carbon-millimole conversion was checked: glucose divided by six, ethanol by two, glycerol by three, with both mean and reported SEM converted. Missing SEM remains missing; `ND` glycerol is not converted into a zero.

The paper identifies *S. cerevisiae* CEN.PK113-1A, glucose-limited chemostats at nominal dilution 0.10 h^-1, pH 5, 30 degrees C and 1.5 vvm. The medium contains 10 g/L glucose, 10 mg/L ergosterol and 420 mg/L Tween-80. The table explicitly reports mean +/- SEM, with different sample counts by variable and repeated measurements from two or four cultivations; these are not 658--4199 independent biological replicates. The observations file preserves these distinctions.

## Frozen design and code

The reviewed protocol describes the appropriate scope: an external steady-state resource-constraint transfer challenge, through the LP limit, with no new fit. It does not validate dynamic Monod kinetics, finite-barrier integration, or measured uptake capacities. The implementation retains five oxygen conditions, including the anaerobic boundary, and reports failures. The primary count is 92: four aerobic conditions each have five input scenarios, the zero-oxygen condition has three; these 23 scenarios are crossed with two constraint semantics and two lipid settings.

The code converts positive observed uptake rates to negative exchange constraints correctly. Capacity semantics use a lower bound and zero upper bound; prescribed semantics set lower equal to upper. Growth and ethanol observations do not enter these constraints. Ethanol variability is evaluated on the near-optimal biomass face, instead of treating a single LP vertex as a unique biological prediction. A numerical acceptance check includes equality and bound residuals, and FVA additionally checks its biomass-face inequality.

## Interpretation limits that must remain explicit

- Oxygen and glucose uptake are measured **inputs** in this challenge. Their agreement cannot be presented as predictive success, particularly in the prescribed case. They are not experimentally measured transport capacities.
- The biomass optimum is compared with the nominal dilution rate. The model does not simulate chemostat biomass concentration or determine a dilution-rate setpoint. The published dilution variation is not a biological confidence interval.
- The inherited effective maintenance parameter was calibrated to FY4 batch data. Transfer to a different strain, growth regime, oxygen level and medium is deliberately unrefitted and may fail.
- Opening ergosterol and oleate bounds to -1000 is an availability sensitivity, not reconstruction of measured medium fluxes. Oleate is an imperfect Tween-80 proxy and can contribute unmeasured carbon/energy. It must not be selected after seeing results and relabeled validated medium.
- Mean +/- SEM corners have no joint confidence interpretation. The zero-ethanol aerobic observation has unknown SEM and detection limit; comparison with that point is descriptive. Tiny numerical distances from zero must not become statistically meaningful rejections.
- No guarantee is made that the archived reconstruction matches a named public release. The copied matrices and reaction mapping identify the tested model.

No source-transcription or carbon-unit errors were found. Numerical statuses, retained interruptions and any runtime amendment require separate outcome review.
