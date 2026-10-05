# Reproducing the CETUS paper

This guide locates the software, data and verification records behind the full
CETUS paper and its RPS workshop version. Implementation names, paths and run
hashes are kept here so the papers can concentrate on methods and results.
The details below come from the manuscript at development revision
`5dedb5d762b40e7c331f1bffa597819691e59c96` and the existing experiment records.
They identify earlier runs; this prose revision adds no model training.

Use [SOFTWARE.md](SOFTWARE.md) for runnable installation and reconstruction
commands. The [README](README.md) distinguishes the original and expanded Titan
datasets. File names below remain exact because the software and existing data
use them. Wording changes in the paper leave their meaning and bytes intact.


## Fold numbers in the current papers

The papers and current figures number folds 1 through 5. Saved results, split
files, commands and array indices retain IDs 0 through 4. To locate a paper
fold in these files, subtract one from its displayed number. Image assignments,
metrics, seeds and averaging are unchanged. Earlier figures and execution logs
retain the numbering used when they were produced.

| Paper fold | Saved ID |
| --- | --- |
| 1 | 0 |
| 2 | 1 |
| 3 | 2 |
| 4 | 3 |
| 5 | 4 |

For the expanded Titan evaluation, paper fold 2 contains the single Craters
test tile; paper fold 1 contains the single Craters validation tile. The
original dataset uses different geographic groups, even when fold numbers
coincide. Map-component IDs and seed values are separate identifiers.


## Image examples in the paper

The current full paper retains the original Earth, Titan and Venus example
images. Its Earth display now masks the exact fill value written by SNAP
preprocessing. For each of the five Earth examples, the number of fill pixels
agrees with the catalog's valid fraction. This agreement distinguishes missing
pixels from genuine low backscatter values; values below the fill value remain
visible when valid. Display percentiles use the remaining pixels. The image
arrays, example choices, model inputs and scientific results are unchanged.

Figure inputs, rendering code and checks belong to the development repository.
The paper's plots display folds 1 through 5 as described above. Earlier figures
inside the result packages retain their original appearance and numbering.

## Choosing an experiment

For the primary geographic evaluation, use the expanded catalog and its five
longitude-sector splits. Its catalog has 23,380 tiles, with 23,246 distinct test
tiles across the five folds. Keep the predictions and splits from the same
package together. The original Earth transfer and Venus experiments instead
use the earlier 11,371-tile Titan catalog and a different geographic design.
Their conclusions and model initialization details remain separate.

A baseline is a comparison predictor, such as the most frequent training class
or a linear classifier on existing encoder features. Equalizing class counts,
changing preprocessing and replacing training images are additional experiments.
Calling all of these baselines would lose the distinction between their roles.
Combining confusion counts across regions also differs from averaging regional
F1 scores: the paper uses equal regional weights for its primary comparisons.

## Untrained encoder and other baselines

The paper's **Untrained ViT** is the `random_init` method in the software and
result files. Its encoder receives no training; a classifier learns from its
image features and Titan map labels. For the expanded Titan catalog, the encoder
uses a single input channel and initialization seed 42. The five SGD seeds
repeat classifier training on those same features. They do not repeat encoder
initialization. DINOv2 instead receives three repeated channels with ImageNet
normalization, so this comparison also includes differences in input handling.

The common logistic regression comparison includes intensity statistics,
histograms, texture measurements and their concatenation. These provide the
current classical baselines. The small CNN results belong to the original
catalog; they cannot be combined with the expanded-catalog table. Predicting
the most frequent training class is a separate comparison that ignores images.

The development [baseline followup plan](https://github.com/magnaprog/TitanSAR-dev/blob/revision/full-paper-design-20261002/docs/BASELINE_FOLLOWUP.md)
describes additional experiments using identical encoder architectures and
inputs, repeated encoder initializations, and a CNN trained on the current
geographic splits. These experiments have not been run. The wording and table
formatting revisions add no predictions or training results. Existing method
identifiers and scientific packages retain their bytes. Access to the plan
requires access to the development repository.

## Software versions for the geographic evaluation

| Role | Exact identifier in the original appendix |
| --- | --- |
| Encoder feature calculation for common-classifier results | source `6295af2` |
| Common logistic regression fitting | source `3dd19a2` |
| Classical features and fits | source `54281f0` |
| CETUS revision named for the two data/result packages below | revision `9a06f6c` |

Independent checks covered all 40 fitted estimators, reproduced predicted classes exactly, checked probabilities with absolute tolerance `10^-12`, and checked 120 output hashes. The original SGD and nearest-neighbor checks covered 40 complete run manifests, 100 probe heads, 20 nearest-neighbor vectors, 280 output hashes and five normalization replays from raw pixels. These checks reused saved features rather than repeating pretrained inference.

## Data and result packages

| Repository | Exact relative path | Contents or purpose retained from the original appendix |
| --- | --- | --- |
| CETUS | `release/longitude-review-20260929/` | Catalog, split definitions, original predictions with encoder weights unchanged, fold metrics and source-part counts |
| CETUS | `release/longitude-auxiliary-20260929/` | Common classifier and classical-feature results, saved predicted classes, scientific specifications, and separate VIMS display metadata with corrected coordinates |
| CETUS | `release/longitude-adaptation-controls-20260930/` | Completed partial-training and training-substitution predictions, histories and reconstruction code; its README distinguishes saved-prediction reconstruction from new training and specifies additional inputs |
| CETUS | `release/longitude-review-20260929/evidence/` | Exported label and catalog checks; full execution records remain in the development repository |
| CETUS | `release/v1-review-20260908/DATA_LICENSES.md` | Source-specific terms and required attributions |

The original v1 payload remains unchanged. The three packages must retain their separate population, prediction and producer identities. The manuscript's statement about revision `9a06f6c` refers specifically to the longitude-review and longitude-auxiliary packages, not to every later package.

## Completed training and prediction checks

| Record | Exact SHA256 prefix |
| --- | --- |
| Accepted native fit audit | `c450220d725a` |
| Selected-checkpoint replay audit | `398bf1284aa5` |
| Adaptation analysis binding both audits and the frozen references | `58b2088cb794` |
| SGD training-substitution audit covering 500 new heads | `e175092aa4ca` |
| Common-logistic audit covering 100 new fits and saved-estimator replays | `5bf988d1d22d` |
| Combined display summary retaining fold values, warnings, iteration-cap records and support counts | `0564a6a8a764` |
| Completion of the separate public-source replay of original frozen SGD and nearest-neighbor calculations | `be3f8a8aeb55` |

The adaptation reconstruction checked 165 confusion matrices and 5,670 scalar comparisons. The native experiments above are distinct from the separate public-source replay, which completed 20 feature jobs and 20 analysis jobs, totaling 40 jobs. Do not describe these two execution histories as a single run.

## Development evidence with restricted distribution

| Exact development-repository path | Scope and access qualification |
| --- | --- |
| `audit/spatial_errors_20260929/` | Fixed protocol, all 2,400 spatial-error result rows and independent numerical reconstruction; the rows retain raw/residual coefficients, counts and undefined reasons |
| `audit/EARTH_SITE_REVIEW_2026-09-29.json` | Internal source and geometry record for Earth site review; explicitly absent from the CETUS review package |
| `audit/gpu_execution/archived_mmd/` | Internal stable numerical recalculation of original Random Init arrays; explicitly absent from the CETUS review package |

A link to this description does not make these internal files part of the public package. Maintain the access distinctions. The paper retains the material limitations that prompted these checks: geographic overlap and Earth-label concerns, undefined spatial correlations, invalid original Random Init permutation statistics and unresolved initial model state.

## Exact implementation and image identifiers

* CROMA's mean patch representation passes through `GAP_FFN_s1`. This is its learned SAR feedforward transformation; it must not be confused with the separate contrastive projection.
* The first-ranked Selk tile is `titan_022929`, at 6.64 degrees north and 164.83 degrees east. The paper retains the coordinates and rank. The identifier is a lookup key, not a scientific-priority recommendation.
* The Earth catalog key `wadi_rum` refers to the mislocated Egyptian source. The paper retains its Wadi Rum label, Egypt location and the unresolved Labyrinths assignment. Renaming the key does not validate the terrain label.
* For the original Random Init experiment, the architecture fallback branch and initialized-state hash remain unresolved. Seed 42 is specified in the source; the five classifier seeds do not identify five independently initialized encoders. No missing hash is supplied here.


## Input-field names used by the software

`label_confidence` stores the modal map fraction: the fraction of valid map
samples assigned to the most frequent class in an image. It does not measure
confidence assigned by a geologist. The map grid and valid SAR mask cover
potentially different portions of the tile.

DOFA's final feature normalization is `fc_norm`. The specified model file lacks
`fc_norm.weight` and `fc_norm.bias`, which retain scale one and bias zero.
The separate `norm` parameters in that model file are unused by the evaluated
feature path. This is a description of the tested implementation; pretraining
input conventions remain uncertain. The model uses identifier 13.78 for Titan,
while the upstream Earth example uses 5.405 and Appendix D of the DOFA
preprint, arXiv:2403.15356 version 2, gives SAR identifier 3.75. See SOFTWARE.md for the exact source and model hashes.

## Earth preprocessing and training-count checks

The development record `audit/CONTROL_RESULTS_2026-09-29.json` gives complete
class metrics, differences between corresponding runs and execution checks.
`audit/R7_NUMERICS_2026-09-29.json` checks all 15 bandwidths and 12,000 MMD
resamples from the saved features. These are additional calculations on the
original population and are separate from the preserved September 8 package.
They remain development-repository records; listing them here does not add
those files to this repository. The papers retain the underlying scientific
limits, including Earth label uncertainty and use of Titan training images.

## Package reconstruction and exact software sources

The evaluation software retains scientific code from development revision
`0257a36197c51eda0a31768531ea26eef9887a6e`. Its source hashes are in
[SOFTWARE_EXPORT.json](SOFTWARE_EXPORT.json). The public inference reproduction
used CETUS revision `360eaea532ead477f46906d7ebe0cb5282e8b082` on September 30,
2026. Its 20 feature jobs and 20 analysis jobs reproduced the native expanded
Titan features, classifier probabilities, predictions and saved metrics in the
specified environment. Independent metric calculations allow absolute error
`1e-12`. SOFTWARE.md lists the full completion-record hash and runtime.

The encoder-training package was exported from development revision `3f4a6a4`.
Its `evidence/acceptance.json` has SHA256
`7340fcc5d1fdc52b27ff3c0385d86669df97345897f95efa92bcb795dba5ba03`.
Independent CPU reconstruction reproduced all 51 package files, totaling
10,201,553 bytes, including figures, numerical claims and inventories. The
three input packages retained their bytes. The October interpretation analyses
come from development revision `0297c9a987f4e0ea919b3a40de0ae208103719e6`.

## Citations, data terms and availability

The terrain map is the first edition of [Titan Global Map Shapefiles](https://data.mendeley.com/datasets/f6jrtyfp66/1).
The dataset title and file format retain their original names. Source terms
and credits appear in [DATA_LICENSES.md](DATA_LICENSES.md) and the original
package's [data terms](release/v1-review-20260908/DATA_LICENSES.md).
GSHHG shoreline documentation is at [the provider's site](https://www.soest.hawaii.edu/pwessel/gshhg/).
Bengio and Grandvalet's discussion of cross-validation uncertainty is in
[their JMLR article](https://jmlr.org/papers/v5/grandvalet04a.html).

This repository distributes metadata, predictions and software. Raw radar
products, processed image arrays and pretrained weights require separate
access. Internal verification records named above retain their stated access
limits. Their presence in this guide does not establish geological accuracy,
a universal reproducibility guarantee or access to omitted source data.
