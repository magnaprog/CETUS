# CETUS

How far can visual representations learned from natural images and Earth observations transfer to Cassini SAR images of Titan? CETUS compares pretrained encoders, random initializations and classical image measurements by asking how well classifiers predict terrain classes in the geomorphological map of Lopes et al. The map draws partly on the same radar observations, so the scores measure agreement with expert interpretation. Read regional and class scores with the overall mean, and compare encoders under the stated classifier-fitting procedure.

This repository supplies catalogs, geographic divisions, saved predictions and software. Choose a task before choosing a package:

| Task | Start here | Inputs and scope |
| --- | --- | --- |
| Reconstruct reported scores | [Paper reproduction guide](PAPER_REPRODUCTION.md) and [saved-result commands](release/titan-followup-20261109/README.md#reconstruct-saved-results) | CPU calculations from saved predictions, logits or histories. Each companion specifies its dependencies and which quantities it reconstructs. |
| Replot results or sampling locations | [Ranking and regional displays](release/crater-retrieval-20261114/README.md) | Saved curves, result summaries, geometry and bundled fonts; no new model result. |
| Extract Titan features and fit classifiers | [Native classifier workflow](SOFTWARE.md#install-and-check) | Processed images, matching catalog/splits, upstream encoder weights and the recorded runtime. This route fits classifiers with encoder weights unchanged. |
| Inspect or prepare encoder training | [Current execution source](release/current-source-20261109/README.md), [loss-normalization source](release/loss-source-20261110/README.md), [DOFA partial-training source](release/dofa-partial-source-20261112/README.md) | Implementations and input contracts. Historical plans, model states and other execution inputs have separate access requirements. |

## Choose the study population

The primary Titan evaluation has 23,380 catalog images and 23,246 distinct test images across five longitude sectors. The root `catalogs/`, `splits/` and `results/` describe the earlier 11,371-tile spatial-block evaluation, together with Earth transfer and the unlabeled Venus comparison. [FILES.md](FILES.md) explains those root files; the [reproduction guide](PAPER_REPRODUCTION.md#choosing-an-experiment) maps each comparison to its population.

The original map sampler omitted valid western terrain. The expanded catalog preserves the original image pixels, adds 12,009 tile IDs and changes 15 existing labels. Keep each catalog, split and prediction package together: an image ID alone does not identify its label version. Both population and geographic design changed, so comparing scores across the two evaluations cannot separate their effects. [CORRECTIONS.md](CORRECTIONS.md) gives the correction and the Earth-source limitations.

Regional spreads are descriptive. Fitting and initialization comparisons retain their optimization and selection limits, and historical Earth labels still need expert review. The DOFA comparisons test numeric inputs 13.78 and 3.75; their sensitivity does not resolve the appropriate Cassini convention.

## Obtain the inputs

The complete collection of 23,380 processed Titan images is available in the [image release](https://github.com/magnaprog/CETUS/releases/tag/titan-processed-images-20261109). Download the archive, [companion notice](https://github.com/magnaprog/CETUS/releases/download/titan-processed-images-20261109/RELEASE_NOTICE.md) and checksums together. Unauthenticated download, extraction and verification of all images passed on 9 November 2026; that check ran no model. The [access inventory](release/titan-followup-20261109/DATA_ACCESS.md) separates these images from the numerical companions' small inspection arrays and identifies omitted embeddings, model states and historical Earth/Venus images.

Image-based execution needs the applicable source, weights and runtime as well as the pixels. New runs must identify their actual inputs and plans. The native stochastic-gradient-descent (SGD) and nearest-neighbor reproduction has a separate producer and verification scope from the matched-DINOv2 and ResNet18 initialization comparisons. The [classifier and interpretation checks](PAPER_REPRODUCTION.md#classifier-and-interpretation-checks) explain which calculations were completed.

## Find a result package

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

## Verify files and retain credits

From the repository root, check file integrity with:

```sh
sha256sum -c SHA256SUMS
```

Each dated package has its own inventory and preserves its scientific payload and producer identities. Some sealed guides describe software or images as pending at export; the routes above identify the available source and images. The four numerical ZIPs keep their original inventories and historical review wording, with public attribution in their wrapper. [SOFTWARE_EXPORT.json](SOFTWARE_EXPORT.json) distinguishes current file identities from the original software export. File integrity is separate from checking the scientific interpretation.

Use [CITATION.cff](CITATION.cff) for citation metadata. [DATA_LICENSES.md](DATA_LICENSES.md), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and package notices retain source terms and credits; the [November notice](release/titan-followup-20261109/NOTICE.md) records authorship. The [public preprint, v1 of 6 October 2026](https://arxiv.org/abs/2610.07576v1) describes the study at that date. Later result packages have their own methods and interpretation notes in this repository.
