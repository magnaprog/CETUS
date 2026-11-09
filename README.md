# CETUS

CETUS studies how representations learned from Earth images transfer to Cassini
SAR images of Titan. This repository provides catalogs, geographic splits, saved
predictions and software for measuring agreement with an expert terrain map.

Start with the [paper reproduction guide](PAPER_REPRODUCTION.md) to choose an
experiment. The [November saved-results
package](release/titan-followup-20261109/README.md) adds the fitting, coverage,
initialization and epoch-selection analyses. Its four preserved archives include
CPU reconstruction code. The [current execution source](release/current-source-20261109/README.md)
contains the CNN, matched-DINO and CPU-fitting implementations.
[SOFTWARE.md](SOFTWARE.md) also describes the earlier classifier runner.

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
| [DOFA numeric input comparison](release/dofa-band-20261109/README.md) | Five-region comparison of inputs 13.78 and 3.75, with ten prediction records and CPU reconstruction code. |
| [Expanded Titan evaluation](release/longitude-review-20260929/README.md) | 23,380 catalog images, including 140 Craters tiles, and five geographic folds; DINOv2, DOFA, CROMA and untrained-ViT results with encoder weights unchanged. The test union contains 23,246 images. |
| [Additional classifier results](release/longitude-auxiliary-20260929/README.md) | Common logistic regression, classical image features and 70 prediction vectors, with separate VIMS display metadata. |
| [Encoder training and training-image replacements](release/longitude-adaptation-controls-20260930/README.md) | Predictions and histories from 75 partial-training fits, training-substitution controls, and CPU reconstruction code. |
| [Interpretation analyses](release/interpretation-20261001/README.md) | Class differences, regional weighting, distances to reinstated training images, map-component counts and examples from accepted results. |
| [Original September snapshot](release/v1-review-20260908/README.md) and root `catalogs/`, `splits/`, `results/` | Historical experiments: 11,371 Titan images, 398 Earth images and 2,945 unlabeled Venus images. Original bytes and links are preserved. |

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
expert review. The new DOFA comparison shows sensitivity to its numeric input;
the appropriate Cassini convention remains unresolved.

For saved-result checks, follow the November package's [CPU
commands](release/titan-followup-20261109/README.md#reconstruct-saved-results)
using the recorded Python and NumPy versions. These commands reconstruct
arithmetic from supplied records. Image-based runs use the separate image release
and still require the applicable source, weights, states and runtime inputs.
The earlier reproduction with encoder weights unchanged and the later model
comparisons have separate producers and verification scopes.

The current source package documents its input and execution contracts. New
runs bind the downloaded images, separately supplied weights and new plans;
the numerical companions provide the portable saved-result calculations.
The reproduction guide retains the [earlier checks and metrics](PAPER_REPRODUCTION.md#earlier-results-and-reconstruction).

From the repository root, check file integrity with:

```sh
sha256sum -c SHA256SUMS
```

The five earlier packages retain their original bytes, metrics and export-date
wording. Each versioned package has its own inventory. The four ZIPs retain their
original internal inventories and historical review wording; the wrapper explains
their public attribution and later status.
[SOFTWARE_EXPORT.json](SOFTWARE_EXPORT.json) preserves the earlier software
export's source history.

[FILES.md](FILES.md) describes the historical root data. Use
[CITATION.cff](CITATION.cff) for citation metadata.
[DATA_LICENSES.md](DATA_LICENSES.md),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the package notices retain
source terms and credits. The [November
notice](release/titan-followup-20261109/NOTICE.md) records authorship and AI
assistance.
