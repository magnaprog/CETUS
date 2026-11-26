# Data

CETUS uses two Titan catalogs. Each has its own geographic splits. The [correction record](../docs/corrections.md#western-titan-coverage) explains how the longitude lookup changed which tiles entered the benchmark.

## Longitude-sector benchmark

[Download the 23,380 processed Titan radar images](https://github.com/magnaprog/CETUS/releases/download/v0.2.0/titan-images.tar.zst). The archive includes one array per tile, its map label in `data/catalog.json` and the five fold assignments under `data/splits/`. Across the five longitude sectors, 23,246 distinct tiles occur in a test set. The [release notice](../release/titan-images/RELEASE_NOTICE.md) gives extraction and verification commands. The [benchmark directory](../experiments/longitude-benchmark/README.md) contains the catalog, protocol, splits and saved test predictions.

## Spatial-block experiment files

The [catalogs](catalogs/), [splits](splits/) and [results](results/) in this directory cover the spatial-block evaluation on 11,371 Titan tiles, transfer from Earth, and comparisons with unlabeled Venus images. Across the five Titan folds, 11,306 distinct tiles occur in a test set. Each fold reserves the same 65 Selk tiles.

### Catalogs

Each tile covers a target square of 45 km in its source projection on a grid of 128 by 128 pixels. The physical area and effective resolution can differ across products. Each entry records a tile ID, file SHA-256, coordinates and pixel summary statistics. The Titan and Earth catalogs also give terrain labels. Join records by tile ID.

| Catalog | Images and labels |
| --- | --- |
| [Titan](catalogs/titan_catalog.json) | 11,371 Cassini SAR/HiSAR tiles. The mosaic stores logarithmically stretched, 8-bit display digital numbers (`hisar_log_dn`). Recovering physical backscatter requires separate calibration data. The map of Lopes et al. supplies six labels in index order: Plains, Dunes, Hummocky, Labyrinths, Lakes and Craters. The map draws partly on the same radar observations. |
| [Earth](catalogs/earth_catalog.json) | 398 tiles at 13 sites. Of 39 Sentinel-1 source products with VV and VH channels, 29 contributed tiles. CETUS took VV (vertical transmit and receive) and calibrated it to linear sigma0. Each tile inherits its site's intended terrain class. Terrain within the footprints needs independent verification. Earth has training examples for five classes, with Hummocky absent. |
| [Venus](catalogs/venus_catalog.json) | 2,945 tiles from a Magellan FMAP radar-brightness display mosaic. The catalog records −1 for missing terrain labels. The right-look global mosaic is the product reference. The exact full mosaic file used to generate these tiles has not been verified. These images enter comparisons of feature distributions. |

### Splits

Each of the 19 split files identifies its catalog by SHA-256. Keep those catalog bytes when checking tile membership.

- **Titan.** The five `titan_spatial_fold_*.json` files assign geographic blocks of roughly 250 km to folds while balancing class and sample counts. Each run trains on three folds, validates on one and tests on one. All folds reserve the same 65 Selk images. Adjacent blocks have no exclusion buffer, so nearby tiles across fold boundaries can remain correlated.
- **Earth transfer.** The [grouped split](splits/earth_transfer_grouped_validation.json) assigns 352 images to training and 46 to validation, holding out entire source products.
- **Earth sites.** Each of the 13 `earth_loso_*.json` files holds out one site. The transfer-recall and feature-distribution analyses that omit sites have separate result files below.

### Results

For seeded classifiers, average five classifier scores within each fold, then report the mean and sample standard deviation of the five fold means. The cosine nearest-neighbor classifier supplies one result per fold. Stored paired two-sided t-tests use the five paired fold means. Overlapping training sets and spatial dependence limit inference from these tests.

| Result file | What it records |
| --- | --- |
| [Probing and Earth transfer](results/probing_cv_results.json) | Linear classifiers and cosine nearest neighbors (k = 20) on DINOv2, DOFA, CROMA and untrained encoder features. The linear classifiers have six outputs. Within-Titan recall covers six classes. Earth-to-Titan recall covers the five with Earth training examples. This file also holds Earth-to-Titan squared maximum mean discrepancy (MMD²), including the check that omits Earth sites one at a time. The untrained encoder's single-precision permutation p-values are unreliable. The A-distance values in `a_distance_appendix` follow the stored protocol. Giving the two domains equal weight in the error rate requires a separate calculation. |
| [Baselines](results/baseline_results.json) | Most frequent training class, random labels drawn at training frequencies, intensity statistics, histograms, texture, a classifier on all three feature groups, and a small supervised CNN. For comparisons with encoder features, pair fold means and retain the specified families for multiple-comparison correction. |
| [Correlation alignment](results/coral_results.json) | Earth-to-Titan transfer after CORAL. The MMD² calculation after alignment uses the Titan features that estimated the transformation. |
| [Spatial-block fine-tuning results](results/finetuning_results.json) | Weighted gradient accumulation was incorrect in these fits. For encoder training, use the [75 longitude-sector fits and checkpoint replays](../experiments/adaptation-controls/README.md). |
| Label efficiency for [CROMA](results/label_efficiency_croma_results.json), [DINOv2](results/label_efficiency_dinov2_results.json), [DOFA](results/label_efficiency_dofa_results.json) and the [untrained encoder](results/label_efficiency_random_init_results.json) | Mean recall over six classes as the number of labeled Titan training images varies. These classifiers use unweighted loss. The primary fit and the training-count comparison with 352 tiles use class weights. |
| [Venus domain gap](results/venus_domain_gap.json) | MMD² among Earth, Venus and Titan, with frequencies, sample sizes, kernel parameters and A-distance. The pools include all 398 Earth tiles and the Titan tiles outside Selk. Earth transfer classification uses the smaller Earth training set. |
| [Selk holdout](results/selk_relative_uncertainty_results.json) | Tile IDs, uncertainty scores and predictions from 15 classifiers for 65 images. Mean entropy is 0.963, mean disagreement is 0.250 and the mean combined score is 0.525. Recomputing entropy needs the individual class-probability vectors. Tile `titan_022929` (6.64° N, 164.83° E) ranks highest among the 65 holdout tiles. The scores rank model uncertainty within Selk. Map ambiguity, gradual terrain boundaries, image content and model error can all contribute. |
| [Result provenance](results/index.json) | Result hashes, folds, seed roles and interval definitions. Here `protocol.ci95` describes a method. In repeated MMD² records, `ci95` gives the 2.5th and 97.5th percentiles across subsamples. |

### Additional experiment records

Recall is stored as a fraction. Multiply recall by 100 for percent, or a difference by 100 for percentage points. Match conditions by model and fold before comparing them.

| Result file | Comparison and scope |
| --- | --- |
| [Training counts](results/constrained_training_results.json) | For a comparison with Earth training, select the 352-image Titan subsets from the uniform or stratified samples and use the Earth-trained scores in the probing file. The fits have equal total training counts. Per-class counts can differ. Both score the same five supported classes on the same Titan test fold. |
| [Display rendering](results/display_rendering_results.json) | Transfer recall for Earth dB inputs and display-DN renderings. Hummocky recall is `null`. Each `mmd` entry is a single estimate from the feature extraction used to score transfer. |
| [Quantization](results/quantization_results.json) | Transfer recall at different numbers of gray levels with the same fitted normalization endpoints. The MMD² estimates here include repeats. |
| [Smoothing](results/smoothing_results.json) | Transfer and Earth-validation recall after smoothing the output grid, including smoothing with display rendering. The file includes texture statistics, a finer smoothing sweep and single MMD estimates. All Titan folds share the same Earth validation split. Output-grid smoothing does not simulate SAR multilooking. |
| [Preprocessing MMD²](results/preprocessing_mmd_results.json) | Use `mmd_repeated` for repeated estimates. For display DN, Earth feature arrays were regenerated separately from the transfer-recall arrays. Matching rendering keys identify conditions. Compare changes within the same bandwidth rule, either re-estimated or fixed. |
| [Earth source-site sensitivity](results/source_site_sensitivity_results.json) | Changes in transfer recall after removing one Earth training site. `NONE` denotes all sites. Records give training counts and identify any lost class. Average over Titan folds for each omission before computing spread across sites. The site jackknife standard error and the SD across Titan folds describe different variation. |

The files in `data/results/` retain fold summaries and selected per-tile records. [Provenance](../release/provenance.json) identifies omitted result fields, catalog hashes and the tile-array format. The [file index](../release/inventory.json) and [checksums](../SHA256SUMS) identify the distributed files. See the [data terms](../licenses/DATA_LICENSES.md) and [third-party notices](../licenses/THIRD_PARTY_NOTICES.md) for attribution, including the VIMS+ISS BSD 3-Clause notice.
