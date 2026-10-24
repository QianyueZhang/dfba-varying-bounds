# Licence scope

The MIT Licence in `LICENSE`, copyright (c) 2026 Qianyue Zhang, applies to the author-owned Python source code and repository documentation in this release. It covers the original portions of those works and does not replace third-party rights, attribution requirements or licence terms.

The following content retains separate terms:

- `data/zampar2013/zampar2013_observations.csv`, an extraction of published experimental observations, retains the source CC BY 3.0 terms. The source links, attribution and changes are recorded in `THIRD_PARTY_NOTICES.md` and the download manifest. The author-written extraction script is covered by MIT; the observations it extracts are not relicensed.
- Metabolic-model matrices, third-party datasets, third-party source code or excerpts, publication content and external dependencies retain their respective terms. No MIT grant over third-party content is made merely because a script loads it or derives a calculation from it.

Metabolic-model MAT files are not bundled. Their exact hashes and expected local locations are specified in `data/model_inputs.json`; use inputs obtained under terms permitting your intended use. Locally prepared files are ignored by Git and do not acquire a new licence from this release. Supplying numerical results does not transfer rights in an underlying reconstruction or experimental dataset.

The Python dependencies are installed separately under their own licences. This repository distributes Python source and numerical records, with no proprietary solver or installed dependency binaries. The Python workflows use the packages listed in `requirements.txt`.

The constrained-control study distributes author-generated simulation datasets, policy traces and numerical predictor coefficients. These numerical outputs are included in the MIT grant to the extent the author holds rights in them; separately sourced model reconstructions remain excluded and must be obtained under their source terms. No fitted executable serialisation is required: released predictors use JSON numeric arrays.

The adaptive-feedback study likewise distributes author-generated policy traces, response-table calculations, numerical audit records and figures. Author-owned portions are covered by the MIT Licence; this grant does not relicense the underlying metabolic reconstructions or cited publications.

The terminal-recovery and scenario-feedback packages distribute author-generated policy records, target intervals, audit diagnostics and figures under the same scope. The published bacterial observations and source-data archive retain CC BY 4.0 terms, as detailed in `THIRD_PARTY_NOTICES.md`. The external supplementary software archive is not distributed or relicensed. Study-specific dependencies are installed separately from their pinned requirements files.

The yeast-assessment package includes a small attributed factual extraction of Wiebe et al. (2008), Table 1. No open licence over that article is asserted; full text and publisher figures are not distributed. These source observations are not relicensed as author-owned data. Zampar observations retain their existing CC BY 3.0 terms. Author-owned numerical calculations and plotting code retain the MIT scope above.

The sensing/actuation study distributes author-generated response tables, finite-game outcomes, independent amount checks and continuous candidate records. This does not grant rights over the underlying reconstruction; optional acceleration caches can be generated locally.
