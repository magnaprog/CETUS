# CETUS

CETUS studies how representations learned from Earth images transfer to Cassini
SAR images of Titan. This repository provides catalogs, geographic splits, saved
predictions and software for measuring agreement with an expert terrain map.

Start with the [paper reproduction guide](PAPER_REPRODUCTION.md) to choose an
experiment and the package table below to find its inputs and commands. The
[November saved-results package](release/titan-followup-20261109/README.md)
provides four preserved archives with CPU reconstruction for fitting, coverage,
initialization and epoch-selection analyses. Later companions cover loss
normalization, class ranking and retrieval, the two DOFA input comparisons, and
regional and geographic displays. Both DOFA comparisons use numeric inputs
13.78 and 3.75.

The [display companion](release/crater-retrieval-20261114/README.md) shows
regional AP changes and recall against retrieved tile count for all six terrain
classes, while retaining absolute Craters AP. It also shows individual SGD
classifier seeds, error rates across map mixtures, Titan test locations, Earth
collection sites and nominal image windows, and unlabeled Venus sample centers.

For implementation details, use the [current execution
source](release/current-source-20261109/README.md) for CNN, matched-DINO and
CPU fitting, the [loss-normalization
source](release/loss-source-20261110/README.md), or the [DOFA partial-training
source](release/dofa-partial-source-20261112/README.md). Each states its input
and environment requirements. [SOFTWARE.md](SOFTWARE.md) describes the native
SGD and nearest-neighbor runner with encoder weights unchanged.

The complete current collection of 23,380 processed Titan images is available
in the [image release](https://github.com/magnaprog/CETUS/releases/tag/titan-processed-images-20261109).
Download the archive, [companion notice](https://github.com/magnaprog/CETUS/releases/download/titan-processed-images-20261109/RELEASE_NOTICE.md)
and checksums together. Unauthenticated download, extraction and verification of
all images passed on 9 November 2026; no model ran in that input check.
The [access inventory](release/titan-followup-20261109/DATA_ACCESS.md) distinguishes
these images from the numerical companions' small inspection arrays and from
omitted embeddings, model states and historical Earth/Venus images.

| Package | Contents |
| --- | --- |
| [Current Titan images](https://github.com/magnaprog/CETUS/releases/tag/titan-processed-images-20261109) | All 23,380 processed arrays, the anonymous catalog, five geographic divisions, source notices and an offline input verifier. |
| [November saved results](release/titan-followup-20261109/README.md) | Current Titan fitting study, coverage and map-vote analyses, matched-initialization comparisons, and saved epoch-100 and validation-choice checks. Four archives with separate entrypoints and dependencies. |
| [Complete map census](release/map-census-20261109/README.md) | All 23,380 map grids, SAR validity masks and geographic roles, with CPU reconstruction of class counts. |
| [Craters descriptive analysis](release/crater-analysis-20261110/README.md) | Training-batch availability and map-composition joins for 30 primary CNN and five pretrained DINOv2 prediction records, with saved results and CPU reproduction. No model fitting or new accuracy estimate. |
| [Training loss source](release/loss-source-20261110/README.md) | ResNet18 implementation and synthetic CPU checks for the two loss denominators, retaining the original class weights and image sampling. |
| [Training loss comparison](release/loss-normalization-20261110/README.md) | Saved predictions and compact histories from fifteen additional 100-epoch fits, with CPU reconstruction against the original initialization archive. The [reproduction guide](PAPER_REPRODUCTION.md#training-loss-normalization) gives commands and interpretation limits. |
| [Class ranking](release/class-ranking-20261111/README.md) | Average precision for all six classes and all 60 Craters precision–recall curves from saved initialization predictions, with offline CPU reconstruction. |
| [Ranking and regional displays](release/crater-retrieval-20261114/README.md) | All-class regional AP differences and retrieval curves, absolute Craters AP, regional SGD results, and Titan, Earth and Venus sampling displays, with plotting code and fonts. |
| [DOFA input during partial training](release/dofa-partial-20261112/README.md) | Fifty saved prediction vectors, all class metrics and paired regional differences, with standard-library CPU reconstruction and a separate source package. |
| [DOFA probabilities and histories](release/dofa-scores-20261114/README.md) | Six class probabilities, predictions and fifty-epoch histories for the same fifty fits, with CPU checks of scores, epoch choice and metrics. |
| [DOFA numeric input comparison](release/dofa-band-20261109/README.md) | Five-region comparison of inputs 13.78 and 3.75, with ten prediction records and CPU reconstruction code. |
| [Longitude-sector Titan evaluation](release/longitude-review-20260929/README.md) | 23,380 catalog images, including 140 Craters tiles, and five longitude-sector partitions; DINOv2, DOFA, CROMA and untrained-ViT results with encoder weights unchanged. The test union contains 23,246 images. |
| [Standardized logistic-regression results](release/longitude-auxiliary-20260929/README.md) | Common LBFGS fitting, classical image features and 70 prediction vectors, with separate VIMS display metadata. |
| [Encoder training and training-image replacements](release/longitude-adaptation-controls-20260930/README.md) | Predictions and histories from 75 partial-training fits, training-substitution controls, and CPU reconstruction code. [Reconstruction instructions and acceptance digest](SOFTWARE.md#reconstruct-adaptation-and-training-substitution-results). |
| [Interpretation analyses](release/interpretation-20261001/README.md) | Class differences, regional weighting, distances to reinstated training images, map-component counts and examples from accepted results. |
| [Spatial-block evaluation snapshot](release/v1-review-20260908/README.md) and root `catalogs/`, `splits/`, `results/` | 11,371 Titan images with the western map-sampling omission, 398 Earth images and 2,945 unlabeled Venus images. Original bytes and links are preserved. |

Read [CORRECTIONS.md](CORRECTIONS.md) before interpreting historical results. The
original map sampler omitted valid western terrain. The expanded catalog preserves
original image pixels, adds 12,009 tile IDs and changes 15 existing labels. Keep
each catalog, split and prediction package together; image IDs alone do not
identify a label version. Both population and geographic design changed, so
comparing scores alone cannot separate their effects.

Scores measure agreement with a map informed partly by the same radar
observations. They do not establish independent geological accuracy. Regional
spreads are descriptive, and fitting and initialization comparisons retain their
documented optimization and selection limits. Historical Earth labels still need
expert review. The DOFA comparisons show sensitivity to its numeric input;
the appropriate Cassini convention remains unresolved.

For saved-result checks, follow the November package's [CPU
commands](release/titan-followup-20261109/README.md#reconstruct-saved-results)
using the recorded Python and NumPy versions. These commands reconstruct
arithmetic from supplied records. Image-based runs use the separate image release
and still require the applicable source, weights, states and runtime inputs.
Reproduction of native SGD and nearest-neighbor results with encoder weights
unchanged has a separate producer and verification scope from the matched-DINO
and ResNet18 initialization comparisons.

The current source package documents its input and execution contracts. New
runs bind the downloaded images, separately supplied weights and new plans;
the numerical companions provide the portable saved-result calculations.
The reproduction guide describes the [classifier and interpretation checks](PAPER_REPRODUCTION.md#classifier-and-interpretation-checks).

From the repository root, check file integrity with:

```sh
sha256sum -c SHA256SUMS
```

The September and October packages retain their scientific payloads, metrics
and producer identities. Each versioned package has its own inventory. The
longitude-package guides are sealed with their export-date wording: statements
that software and a tile archive are pending describe that export, while the
source and image-release links above provide the available materials.
The four ZIPs retain
their original bytes, internal inventories and historical review wording; the
wrapper explains their public attribution and completed studies.
[SOFTWARE_EXPORT.json](SOFTWARE_EXPORT.json) preserves the earlier software
export's source history.

[FILES.md](FILES.md) describes the historical root data. Use
[CITATION.cff](CITATION.cff) for citation metadata.
[DATA_LICENSES.md](DATA_LICENSES.md),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the package notices retain
source terms and credits. The [November
notice](release/titan-followup-20261109/NOTICE.md) records authorship; the
[manuscript](https://arxiv.org/abs/2610.07576) describes the scientific methods and results.
