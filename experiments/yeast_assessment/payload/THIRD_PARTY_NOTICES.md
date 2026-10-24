# Third-party material and scope of the MIT licence

The root MIT licence applies to author-owned software and documentation. It does not replace third-party rights in data, reconstruction matrices, source publications, dependencies or publisher templates.

**Metabolic-model matrices are excluded.** No `.mat` file is licensed or redistributed by this release. `INPUT_MANIFEST.json` gives exact private/archive source paths, hashes and required mirrored destinations. A user who lawfully holds those files may stage them locally. The archived yeast metadata do not establish a definitive published reconstruction release; the repository does not infer a licence from the species name or a similar published reconstruction. Its no-input analytical example remains executable without those models.

**Zampar cultivation measurements.** Extracted observations are attributed to Zampar et al., *Molecular Systems Biology* 9, 651 (2013), DOI [10.1038/msb.2013.11](https://doi.org/10.1038/msb.2013.11), and its supplementary workbooks. Preserve the source attribution and applicable [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) terms recorded for the supplementary data. The repository's derived predictions and fitting code are separate from those original measurements. No publisher full text or original workbook is included.

**Wiebe condition-level values.** The small factual Table 1 extraction is attributed to Wiebe et al., *FEMS Yeast Research* 8, 140–154 (2008), DOI [10.1111/j.1567-1364.2007.00234.x](https://doi.org/10.1111/j.1567-1364.2007.00234.x). Units, reported SEM and conversion provenance are preserved in `external/observations.json`. No open licence for the complete article is asserted. Publisher HTML, PDFs, figures and the privately archived article are excluded; the paper's bibliographic citation does not grant a right to redistribute them. `prepare_inputs.py` documents the original extraction, but recreating that step requires a separately held source and is not part of the public runner.

**Scientific antecedents.** The reduced-network and core-model examples retain citations to their published sources in the manuscript. The varying-bound study builds on Scott et al. (2018) and identifies the Srisuma dissertation as benchmark lineage. References identify scientific provenance rather than licensing copied source text. No thesis chapters are included in this standalone release.

**Dependencies and typesetting.** NumPy, SciPy, HiGHS/highspy and Matplotlib retain their own licences and are installed separately. The `achemso` class and TeX packages retain their original licences; their source is not included here. JACS class settings specify a typesetting format, not ACS endorsement or acceptance.

Author-written generated results and figure-generation code should be cited with this study and retain the above source-data attribution when source observations are displayed.
