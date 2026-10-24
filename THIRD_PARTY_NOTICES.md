# Sources and third-party terms

The MIT Licence covers author-owned code and documentation, subject to [TERMS.md](TERMS.md). Scientific inputs and external software retain their distinct rights.

## Experimental observations

Zampar GG et al. (2013), *Temporal system-level organization of the switch from glycolytic to gluconeogenic operation in yeast*, Molecular Systems Biology 9:651, DOI https://doi.org/10.1038/msb.2013.11, Supplementary File 1. The source is available under Creative Commons Attribution 3.0 Unported: https://creativecommons.org/licenses/by/3.0/.

`data/zampar2013/zampar2013_observations.csv` extracts the published `Overview` experimental columns from the biomass, glucose and ethanol workbooks. Time coordinates and reported values are retained. There are 266 raw observations; the analysis interval retains 256, comprising 45 calibration observations and 211 withheld observations. The excluded ten observations precede the representative simulation initial time. Source download links, checksums and extraction details are in `download_manifest.json`; the extraction script can rebuild the CSV from the primary supplement. Workbook labels do not establish matched biological replicate identities across variables.

## Metabolic models

The calculations reference the Edwards reduced network, E. coli core, iJR904, yeast reconstructions for S. cerevisiae and K. phaffii, and iND750/iRC1080. The exact local model matrices are not bundled or relicensed here. The input manifest records content hashes and expected case locations. Reconstruction identity and numerical equivalence must be checked before substituting a model with the same name.

BiGG's published model terms and individual reconstruction publications may have different scope from code-package licences. Consult the primary distribution for each model. A software-tool licence is not automatically a licence for all model content. The local preparation utility copies only exact supplied files and performs no implicit download.

## Python dependencies

NumPy, SciPy, scikit-learn, Matplotlib, Pillow, openpyxl and HiGHS/highspy are installed separately through `requirements.txt` and retain their respective terms. No installed package, toolbox or solver binary is included. MAT-format numerical inputs are loaded directly by SciPy.

## Source records

`data/diauxic_implementation_provenance.json` retains exact hashes of implementation-context records held in the author's private research archive. They are not distributed executable sources or runtime inputs. Their identifiers do not assert equivalence between every earlier result and the specified calculations in this package. The source publications and benchmark adaptations are cited in the study guide; the MIT grant applies only to rights held by the author.

## Published bacterial composition observations

Gutiérrez Mena, J., Kumar, S. and Khammash, M. (2022), *Dynamic cybergenetic control of bacterial co-culture composition via optogenetic feedback*, Nature Communications 13, 4808, DOI https://doi.org/10.1038/s41467-022-32392-z. The article/source-data terms are Creative Commons Attribution 4.0: https://creativecommons.org/licenses/by/4.0/.

The unmodified source-data ZIP and attributed extracted numerical observations are included in `experiments/scenario_feedback/`. Extraction mappings, source URLs and assessment scope are documented in its `public_validation/DATA_AND_METHODS.md`. Preserve the attribution and CC BY 4.0 terms when redistributing observations. The separate supplementary software ZIP is not bundled because a distinct software redistribution licence has not been established; the optional fixed published-model calculation requires users to obtain it under its applicable terms.

## Compiled numerical kernels

The terminal-recovery and scenario-feedback studies specify Numba and llvmlite in their own requirements files in addition to NumPy, SciPy and Matplotlib. These are installed separately under their own terms. Compiled kernels, runtime caches and installed dependencies are not distributed.

## Oxygen-limited yeast observations

Wiebe, M. G. et al. (2008), *Central carbon metabolism of Saccharomyces cerevisiae in anaerobic, oxygen-limited and fully aerobic steady-state conditions and following a shift to anaerobic conditions*, FEMS Yeast Research 8, 140–154. DOI https://doi.org/10.1111/j.1567-1364.2007.00234.x. The yeast-assessment package contains only a small factual Table 1 extraction with source/condition mapping. It contains no article full text or publisher figure; no open-article licence or MIT relicensing of the observations is asserted.

## Sensing and actuation response tables

The supplied process response table and control results are author-generated numerical data. Their distribution does not relicense the metabolic reconstruction from which rates were calculated. Numba and llvmlite are installed separately under their own licences; no compiled kernels or dependency binaries are distributed.
