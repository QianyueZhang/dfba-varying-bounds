# Published bacterial composition observations

Source: Gutiérrez Mena, Kumar and Khammash (2022), *Dynamic cybergenetic control of bacterial co-culture composition via optogenetic feedback*, Nature Communications 13, 4808. DOI: https://doi.org/10.1038/s41467-022-32392-z. Article and source observations: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/. Attribution and this licence must accompany reuse of the data.

`data/measured_trajectories.csv` and `data/measured_dose_response.csv` contain extracted observations, with source file/field mappings. `protocol.json` specifies the training/held-out split and metrics; published source filenames sometimes use older supplementary figure numbering, so use the explicit protocol mappings. Digital light intensity is in arbitrary units (0–800); composition is a cell fraction. Neither is a yeast dry-weight fraction or glucose feed rate.

The unmodified source-data archive is `../literature_sources/Kumar2022_source_data.zip`, obtained from:

https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-022-32392-z/MediaObjects/41467_2022_32392_MOESM6_ESM.zip

The original numerical outcomes are retained under `results/`, including every per-run prediction, baseline comparison and sensitivity check. Initial observations initialise each run and are excluded from error scoring. All complete trajectories are retained, with full, through-40-h and after-40-h windows. Training information alone selects the reduced predictors. Both reduced predictors fail the declared macro baseline comparisons; no experimental control-benefit claim is made.

For a fresh primary calculation, copy `protocol.json`, `LOCK.json`, `scripts/run_validation.py`, and the supplied source-data archive into a new study with the same relative layout. Create empty `public_validation/data`, `results`, `figures` and `audit` directories there, and execute `python public_validation/scripts/run_validation.py`. The script refuses a completed output. Install NumPy, SciPy and Matplotlib using the parent study's requirements first.

`secondary_published_model/run_published_model.py` implements a separate fixed-parameter model check. It requires the publisher's supplementary software archive at `literature_sources/Kumar2022_supplementary_software.zip`, obtained from:

https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-022-32392-z/MediaObjects/41467_2022_32392_MOESM5_ESM.zip

That external archive is not bundled or relicensed. Verify its hash against `secondary_published_model/PROTOCOL.json` and follow the publisher's applicable terms. For this optional check, copy the primary study to a separate new workspace, retain its measured observations, copy the secondary `PROTOCOL.json`, current `LOCK.json` and Python source, and run that source there with no previous secondary result. The executed-source hash in the supplied record identifies the original script; the public filename and one source-description phrase were changed as recorded in `SOURCE_MANIFEST.json`. The calculation reads published parameter values and does not fit them against held-out trajectories.

Measured initial internal molecular states are unavailable, so the secondary check assumes the stated ambient steady state. The original authors' parameter-selection overlap with these public records has not been established. Both full-run baseline comparisons fail. The source-data licence does not grant a blanket licence over external software, and the repository's MIT licence does not relicense these observations or that software.

The current secondary `LOCK.json` identifies the public `run_published_model.py` and protocol used for a fresh calculation. `executed_lock.json` preserves the original source/protocol identities attached to supplied results; it must not be substituted for the current lock when running the renamed source. Both identities are recorded in `SOURCE_MANIFEST.json`.
