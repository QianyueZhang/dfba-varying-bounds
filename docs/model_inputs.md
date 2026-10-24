# Model and observation inputs

The public package supplies numerical response tables, experimental observations, fitted surrogate coefficients and selected simulation results. It records, but does not distribute, **38 model-input paths representing 30 distinct file contents**. The local preparation command checks exact SHA-256 identities; it neither downloads a reconstruction nor creates the processed model matrices.

## What runs from a clean clone

After installing the pinned requirements, the analytic moving-bound study, fixed-prediction neural diagnostics and table-based constrained-control simulation/training/optimisation/checks/plots use only supplied inputs. The normal short verification suite also has a no-model mode. Direct intracellular checks and the complete culture case sequences require the model files below. The supplied flux tables allow control simulations to run, but do not replace a fresh verification of the underlying intracellular model.

The exact input collection has not been cleared for redistribution in the existing source records. Model names, publication links and a hash manifest are not a complete acquisition route. The release therefore makes no claim that every thesis calculation can be run from the repository alone. A download of a current reconstruction with the same name must be treated as a different input unless its numerical fields, reaction ordering, bounds and case-specific processing are shown to agree.

## Case dependencies

Directory names below are relative to `data/models/`.

| Case | Required model directory or scope |
| --- | --- |
| `edwards` | edwards |
| `diauxic` | diauxic |
| `yeast-experiment` | yeast_experimental |
| `yeast-batch-fedbatch` | yeast_fedbatch |
| `independent-shared` | independent_pools; shared_resource |
| `product-inhibition` | product_inhibition |
| `resource-diversion` | resource_diversion_batch; resource_diversion_fedbatch |
| `biomass-transfer` | biomass_transfer |
| `pyruvate-crossfeeding` | pyruvate_crossfeeding |
| `genome-scale-yeasts` | yeast_control |
| `cstr` | yeast_alga_cstr; cstr |
| `control-neural` | yeast_control for direct checks; table-based stages use supplied data |
| `constrained-control` | None for table-based study; yeast_control for the optional direct command |
| `moving-bound-analytic` | None |
| `neural-audit-statistics` | None |

## Published sources

These verified bibliographic links identify relevant scientific sources. They are **not asserted to be downloads of the exact processed files listed below**, and their inclusion does not establish redistribution rights for those files.

- The core *E. coli* model is described in *Reconstruction and Use of Microbial Metabolic Networks: the Core Escherichia coli Metabolic Model as an Educational Guide*, DOI [10.1128/ecosalplus.10.2.1](https://doi.org/10.1128/ecosalplus.10.2.1). The interaction cases use case-specific `S1_Model.mat` and `S2_Model.mat` matrices as well as the reconstruction file.
- The sequential-substrate model is iJR904, described in *An expanded genome-scale model of Escherichia coli K-12 (iJR904 GSM/GPR)*, DOI [10.1186/gb-2003-4-9-r54](https://doi.org/10.1186/gb-2003-4-9-r54). The two similarly named processed files are distinct: `E-coli_Model.mat` and `E_coli_Model.mat` have different hashes.
- The CSTR yeast reconstruction is iND750, described in *Reconstruction and Validation of Saccharomyces cerevisiae iND750, a Fully Compartmentalized Genome-Scale Metabolic Model*, DOI [10.1101/gr.2250904](https://doi.org/10.1101/gr.2250904).
- The CSTR algal reconstruction is iRC1080, described in *Metabolic network reconstruction of Chlamydomonas offers insight into light-driven algal metabolism*, DOI [10.1038/msb.2011.52](https://doi.org/10.1038/msb.2011.52). Article access or an article licence must not be assumed to establish the terms of every derived model file.
- The yeast benchmark's study source is Srisuma (2018), cited with chapter/page details in the [study guide](studies.md#benchmark-source). The exact `SCerevisiae_Data.mat`, `KPhaffii_Data.mat` and their processed matrices still require source-version and redistribution clarification. The smaller Edwards files likewise have recorded numerical identity but no complete public conversion recipe in this package.

The experimental yeast observations are publicly included under their source CC BY 3.0 terms. They come from Zampar et al. (2013), DOI [10.1038/msb.2013.11](https://doi.org/10.1038/msb.2013.11), [Supplementary File 1](https://pmc.ncbi.nlm.nih.gov/articles/instance/3693829/bin/msb201311-s1.zip). The source ZIP SHA-256 is `d928c613be23fe67e6af1ec2e75c88390a4a48b7824c349fca736081c5cecaeb`. The [download manifest](../data/zampar2013/download_manifest.json) records the publisher alternative, original checksum evidence, extraction columns and allocation; [the extraction script](../data/zampar2013/extract_observations.py) documents the CSV construction.

## Exact model identities

The machine-readable authority is [`data/model_inputs.json`](../data/model_inputs.json). The following list transcribes its paths and SHA-256 hashes; repeated reconstruction hashes are intentional where cases use the same file.

| Path under `data/models/` | SHA-256 |
| --- | --- |
| `yeast_control/KP_Model.mat` | `2c628806275e664f1e2853b5f7434f331c1929bd4a3e07773149a332315cf440` |
| `yeast_control/KPhaffii_Data.mat` | `f53fc287c5e3269472bd07fac93bca5fd0fb387f8537cd0dc0ba8c3e1d390911` |
| `yeast_control/SC_Model.mat` | `6712c7af12964a8e0ba919eee24c49fbbdb1d18392c28a275a3bd9e452d2bc2f` |
| `yeast_control/SCerevisiae_Data.mat` | `e69b562f34cae97d676513d57175f4642559ee629b32bb1a5839177dd943acbd` |
| `biomass_transfer/S1_Model.mat` | `8650c4ab68270960e899d2b3d459c74a598e995e890f1dccba3813456fca4df4` |
| `biomass_transfer/S2_Model.mat` | `5b1c90702a2ee225a4dc9e135dfcb07ef7611fb5f522dd3d82e9d5ef5ece580f` |
| `biomass_transfer/e_coli_core.mat` | `a70628fd2f32341aa877c65b9e992a1caa63deef7ef58bd953d8a4a1957347b2` |
| `independent_pools/S1_Model.mat` | `ab243364097e69c880d4088e50ef5a16efdce545400b53dad1fc013637c52025` |
| `independent_pools/S2_Model.mat` | `824823547dce42ea0001b0432bffdf6ee584fbe057f6d6eac186063a2cc58f14` |
| `independent_pools/e_coli_core.mat` | `a70628fd2f32341aa877c65b9e992a1caa63deef7ef58bd953d8a4a1957347b2` |
| `shared_resource/S1_Model.mat` | `7d68bd24c0a233abd4ea8eaa94acac59708ec25af66bcfa4b53e022e1a66b005` |
| `shared_resource/S2_Model.mat` | `1152fc8d8118a8099288194400b4637cbfc699c04d5a0b888ac775d7268d6708` |
| `shared_resource/e_coli_core.mat` | `a70628fd2f32341aa877c65b9e992a1caa63deef7ef58bd953d8a4a1957347b2` |
| `product_inhibition/S1_Model.mat` | `99aa5036e4aa78e330ecc1bbb39e043e713a781b04788271e11d0b1dc7e0a601` |
| `product_inhibition/S2_Model.mat` | `451d3e487387d68968897be39a3bda339b7ef3a0017bffe0b4608002eb8ab756` |
| `product_inhibition/e_coli_core.mat` | `a70628fd2f32341aa877c65b9e992a1caa63deef7ef58bd953d8a4a1957347b2` |
| `resource_diversion_fedbatch/S1_Model.mat` | `42b5fa7ed110e07ad236670cfbc97262b777d9bf21e214783c009873b86b9eb6` |
| `resource_diversion_fedbatch/S2_Model.mat` | `93089effdaff20844c9f70fdcdd5773cf2c72324d80a797c267305265fa962f7` |
| `resource_diversion_fedbatch/e_coli_core.mat` | `a70628fd2f32341aa877c65b9e992a1caa63deef7ef58bd953d8a4a1957347b2` |
| `resource_diversion_batch/S1_Model.mat` | `37ef07b49bfd5f1be5a893322f7b59b954f8344b5c09cd0edbaf5ca51e7730ff` |
| `resource_diversion_batch/S2_Model.mat` | `e5ad42e7fed6319b3599abe4cce23a6bbe0134987adf176cb427a05c7c83cc89` |
| `resource_diversion_batch/e_coli_core.mat` | `a70628fd2f32341aa877c65b9e992a1caa63deef7ef58bd953d8a4a1957347b2` |
| `pyruvate_crossfeeding/S1_Model.mat` | `c5c819a6dff3d711aba50796b1efef0af05e26d4d824cfbdbad607b05a127fed` |
| `pyruvate_crossfeeding/S2_Model.mat` | `0d100c60107a1efab3b5aa6ba4045f87641b5305950567069c73c3852e113658` |
| `pyruvate_crossfeeding/e_coli_core.mat` | `a70628fd2f32341aa877c65b9e992a1caa63deef7ef58bd953d8a4a1957347b2` |
| `diauxic/E-coli_Model.mat` | `ab51a21da5a7e29eb4cf5a545f987faede7c73b1d1ff0961dbd16b06c3a825fe` |
| `diauxic/E_coli_Model.mat` | `fd7c1da41144290d3fc90345ec2996491585f5f1b32b41fd8381538bb72d5006` |
| `diauxic/iJR904.mat` | `d17be86293d4caafc32b829da4e2d0d76eb45e1bb837e0138327043a83e20c6e` |
| `yeast_experimental/SC_Model.mat` | `6ae5fc5dcc037311ad17a4c39d7c3c464970ac9c5932f79b358bf3ee987d037e` |
| `yeast_experimental/SCerevisiae_Data.mat` | `e69b562f34cae97d676513d57175f4642559ee629b32bb1a5839177dd943acbd` |
| `yeast_fedbatch/SC_Model.mat` | `15cc44f8642498cbe2c2dbc0620ecd7ba75de29382ed379e353df89612beca32` |
| `yeast_fedbatch/SCerevisiae_Data.mat` | `e69b562f34cae97d676513d57175f4642559ee629b32bb1a5839177dd943acbd` |
| `yeast_alga_cstr/RC_Model.mat` | `22e7fb0ec61e1c27a0d09b7e3b441a3ef2bff8687d6934ebf90eb9c30c4a0733` |
| `yeast_alga_cstr/SC_Model.mat` | `b76a7e710232749e39f1c5c2bcf9056eeeb72028c78d032fee06396f658be2a3` |
| `edwards/Ed1.mat` | `8ac6e29276fe806d4e95db48856fccc589344203dc9213be8c45bbd2b96b61f0` |
| `edwards/Edwards_Model.mat` | `f37c89c94080fe5ec3d286d5d216d9c4c750b02ab929705bc947bc99e5970511` |
| `cstr/iND750.mat` | `dd71a80efd38548ba81cc9efbf16f4ad6ce1514801c8821958a3f09b2bb31814` |
| `cstr/iRC1080.mat` | `f9340d5ed404850417de9d1f14ef5ee4ca186441662cf3fafb377ac190c7b9b8` |

## Preparing an authorised local collection

```sh
python tools/prepare_inputs.py --search-root /path/to/local/model/files
python tools/prepare_inputs.py --check
python tools/verify.py --require-models
```

A missing input is a dependency failure, not a numerical result. Keep the input manifest and source-specific terms with any separately supplied collection. Before distributing processed matrices, establish their exact parent reconstruction, applicable terms, processing steps and numerical identity. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and [TERMS.md](../TERMS.md).
