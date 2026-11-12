# Reproducing the CETUS paper

This guide locates the software, data and verification records behind the full
CETUS paper and its RPS workshop version. Run identifiers and data routes were
transcribed from the manuscript at development revision
`5dedb5d762b40e7c331f1bffa597819691e59c96` and the existing experiment records.
The fold-number and image-display notes below reflect later paper revisions;
the earlier run identities remain unchanged. The dated November route below
adds the completed follow-up studies without changing the historical results.

Use [SOFTWARE.md](SOFTWARE.md) for runnable installation and reconstruction
commands. The [README](README.md) distinguishes the original and expanded Titan
datasets. File names and method IDs below match the software and saved data.

## November 2026 follow-up

The [9 November saved-results release](release/titan-followup-20261109/README.md)
contains four unchanged numerical archives. They cover the current 23,380-tile
catalog and 23,246 distinct test tiles. The package's
[provenance](release/titan-followup-20261109/provenance.json) records archive
identities, original scientific producers and the source revision used for
this release. A later documentation or packaging revision does not replace a
scientific producer.

| Archive | Completed work and recipient reconstruction |
| --- | --- |
| Scientific | 315 saved test-prediction vectors: 40 common-LR, 100 SGD, 20 kNN, 75 partial-training and 80 fitting-study vectors. CPU replay checks sixteen fitting settings, twenty contrasts and eleven generated table fragments. It does not refit classifiers or reconstruct gradients. |
| Follow-up | Coverage calculations for 115 saved vectors; map-vote replay for 2,685 sampled grids and sixty SAR inspection packets. The 2,643 partial test tiles differ from the 2,651 partial catalog tiles plus 34 complete inspection tiles used for map votes. |
| Initialization | The completed 90-case study comprises thirty DINOv2 readouts, thirty ResNet18 tuning fits and thirty final fits. Thirty DINO feature jobs supplied the readout inputs. The companion reconstructs sixty native test records, 6,000 validation-history rows from sixty CNN fits, ten learning-rate choices and 180 coverage confusion matrices. |
| Reassessment | Thirty evaluations of saved epoch-100 CNN states, validation-choice checks and thirty-seven SAR/map excerpts. The original validation-selected results remain primary; the epoch-100 comparison involved no retraining or reselection. |

The 8 November records establish completion of the initialization and epoch-100
studies. This public documentation update adds no scientific run or result.
Recipient commands reconstruct saved arithmetic on CPU with Python 3.12.3 and
NumPy 2.4.4. Reports belong outside the extracted packages. They do not
regenerate GPU probabilities, omitted per-epoch prediction vectors or optimizer
updates. Validation-removal checks use supplied outcome records and cannot
independently regenerate every removal score.

The fitting study retains 23 iteration caps and 54 gradient concerns.
Initialization comparisons reuse examined test regions, and the CNN initializations use
separate validation choices of learning rate and epoch. Average seeds within
regions before giving the five regions equal weight; their sample SD is
descriptive. Neither repeated seeds nor the learning curves establish
independent regions, convergence, equal optimization adequacy or a causal
effect of pretraining. Rare Craters support and unsupported-class null summaries
remain part of the reporting rules.

The [access inventory](release/titan-followup-20261109/DATA_ACCESS.md) separates
the supplied inspection excerpts from the full pixel collection, embeddings
and model states, which are not included. The [current execution source](release/current-source-20261109/README.md)
provides the CNN, matched-DINO and CPU-fitting implementations with their
[execution contracts](release/current-source-20261109/CONTRACTS.md). It does not
supply the omitted scientific inputs or historical machine-bound plans. The
earlier classifier runner remains a separate workflow.

## Terrain within each image

The [complete map census](release/map-census-20261109/README.md) supplies all
23,380 saved map grids, their SAR validity masks and five-fold role assignments.
The CPU checker reconstructs class and unknown counts under four coverage rules
for the complete catalog and every training, validation, test and exclusion group.
It uses saved arrays; original SAR decoding and map sampling are separate checks.

With NumPy installed, run from the companion directory:

```sh
python3 -B -m unittest -v test_check_counts
python3 -B check_counts.py --output /absolute/new/path/map-counts.json
```

Craters occurs in 331 images under complete nominal SAR coverage; 140 images
have Craters as their modal label. Of the eligible Craters cells, 21.8% occur
in other modal classes. These counts include overlapping images and describe
the map's spatial mixtures. Existing image labels and model scores are unchanged.
The companion README defines the masks, units and interpretation of each count.

## Craters training and map composition

The [10 November companion](release/crater-analysis-20261110/README.md) preserves
the completed CPU analysis of training-batch availability and modal-Craters test
images. It joins 30 primary, validation-selected CNN records and five pretrained
DINOv2 records to the accepted census. Its reproduction uses the existing
initialization ZIP and public census `counts.json` and `map_cells.npz`; commands
and exact input identities are in the companion README. No model is fitted or
rescored, and per-seed results remain separate.

The batch bound concerns the equal-step average of normalized target-class
cross-entropy coefficients; it does not bound gradients or performance. Map
fractions concern eligible known cells, with empty domains undefined. Repeated
seeds and overlapping images do not provide independent geographic samples.

## Ranking images by terrain class

The [class-ranking companion](release/class-ranking-20261111/README.md) calculates
average precision from the initialization study's saved probabilities. It keeps
each region and fitted model separate, includes all six classes and retains all
60 Craters precision–recall curves. The calculation admits tied scores together
and sums recall increments times precision without interpolation. Seeds are
averaged within each region before giving the five regions equal weight.

Both initialization comparisons improve mean macro average precision. Mean
Craters average precision changes little for DINOv2 and declines for ResNet18.
Region 2 has only one Craters image, whose rank strongly affects its score;
the companion retains all seeds and the differences in the other regions.
This is an exploratory ranking analysis on the same test images. It measures
agreement with the terrain map and does not assess calibrated probabilities
or detection of individual geological craters.

The companion's commands use Python 3.12.3 and NumPy 2.4.4 on CPU. They read the
existing initialization ZIP intact, reconstruct all 360 class scores and their
regional summaries, and compare every saved curve array. No model weights,
image decoding or training are involved.

## Training loss normalization

The [execution source](release/loss-source-20261110/README.md) compares two
denominators for the weighted cross entropy used to train ImageNet ResNet18.
The original divides by the sum of label weights in each batch. The alternative
divides by the batch size times the mean weight across the training images.
Both retain the original class weights and image sampling.

The [numerical companion](release/loss-normalization-20261110/README.md) supplies
saved records from fifteen additional 100-epoch fits and fifteen fresh replays
of the validation-selected states on validation and test images. All fifteen
original and all fifteen additional runs have zero Craters true positives.
Six-class macro F1, averaged over three seeds within each region and then equally
over five regions, changes from 43.84% to 44.38%. Regional macro F1 means rise in
three regions and fall in two. The companion retains all six classes, including
false-positive counts.

The learning rate is inherited from the original loss, and the study reuses the
same five test regions. Seeds repeat training within regions; they do not add
independent geographic samples. The original Craters F1 is zero, so its change
cannot be negative. These results describe sensitivity under the stated settings
and agreement with the SAR-informed map. The original initialization results
remain primary.

Use Python 3.11 or later on Linux; reconstruction needs only the standard
library. From the repository root:

```sh
cd release/loss-normalization-20261110
sha256sum -c SHA256SUMS
python3 -B reconstruct.py \
  --initialization-zip ../titan-followup-20261109/initialization-supplement.zip \
  --output /absolute/new/path/loss-comparison.json
```

Choose a new output filename outside the companion directory. The reader reuses
the unchanged [initialization archive](release/titan-followup-20261109/initialization-supplement.zip)
for the original A runs, labels and geographic divisions; it reads the archive's
data without executing its Python files. It checks saved arithmetic and compact
validation histories without fitting models or rerunning GPU predictions.
The [manifest](release/loss-normalization-20261110/manifest.json) and
[notice](release/loss-normalization-20261110/NOTICE.md) record identities and credits.

The separate source package retains the implementation, study specification and
focused synthetic CPU tests. Its [execution contracts](release/loss-source-20261110/CONTRACTS.md)
describe the original input and model-state dependencies. Model states and full
training inputs are not supplied by the numerical companion.

## DOFA band input comparison

The [DOFA companion](release/dofa-band-20261109/README.md) adds the completed
13.78 versus 3.75 comparison on all five regions. It preserves ten sets of
test predictions and probabilities and reconstructs their arithmetic on CPU.
It is separate from the four earlier numerical packages and does not change
the original Earth or partial-training results.

## DOFA input during partial training

The [partial-training companion](release/dofa-partial-20261112/README.md)
contains fifty saved prediction vectors: two numeric inputs, five seeds and
five geographic regions. It compares inputs 13.78 and 3.75 while retaining
the training settings and identical initial encoder and classifier states
within each pair. The mean macro F1 is 15.68% at 13.78 and 37.52% at 3.75.
The paired difference is 21.84 percentage points, with sample SD 4.24 across
five regional means after averaging seeds. Craters mean F1 at 3.75 is 1.37%.

From the repository root:

~~~sh
cd release/dofa-partial-20261112
python3 -B reconstruct.py --output ../dofa-partial-reconstructed.json
~~~

Use a new output path. The standard-library reader checks all ordered test
labels and predictions, confusion counts, class and macro scores, and paired
aggregates. It does not load models or reproduce training. The result concerns
agreement with map labels on previously examined regions. At 3.75, logistic
regression on unchanged encoder features reaches 38.42% mean macro F1 in the
separate feature comparison; its fitting and epoch choices differ.

The [source package](release/dofa-partial-source-20261112/README.md) contains
the training and replay implementation and focused tests. It supports inspection
and preparation; the full historical source, data, weights, environment and
executable contract remain separate prerequisites. Its documentation contract
does not launch the experiment. The numerical companion above is self-contained.

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

The revised full paper and CVPR supplement show seventeen examples from the
original Earth, Titan and Venus collections. They were chosen from 134
candidates using coverage, contrast and visual inspection. Earth examples
have at least 99.9% valid pixels; Titan and Venus examples have complete
coverage. Titan examples also have modal map fractions of at least 80%.
The Earth Labyrinths image shows Atacama channels, replacing the misnamed
Wadi Rum example. One invalid pixel remains in the Earth Craters display
and is masked in magenta with diagonal hatching. The plots preserve the
underlying arrays and experiment labels; display choices do not change
training inputs or scores. These illustrations are separate from the enlarged
Titan catalog's examples chosen by median modal map fraction.

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
classical baselines. The historical small CNN results belong to the original
catalog; they cannot be combined with the expanded-catalog table. Predicting
the most frequent training class is a separate comparison that ignores images.

The October 2 baseline follow-up plan proposed matching encoder architecture
and inputs, repeating encoder initializations, and training a CNN on the
current geographic splits. The [November release](release/titan-followup-20261109/README.md)
provides the completed follow-up: official DINOv2 ViT-B/14 with five random
states and one pretrained state under matched transforms and C=1 logistic
regression, and ResNet18 trained from random or ImageNet initialization with
three seeds per arm. The CNN comparison retains its separate validation-based
learning-rate choices. These studies do not alter the older `random_init`
definition, the historical CNN, or any of the five earlier release packages.

## Software versions for the geographic evaluation

| Role | Exact identifier in the original appendix |
| --- | --- |
| Encoder feature calculation for common-classifier results | source `6295af2` |
| Common logistic regression fitting | source `3dd19a2` |
| Classical features and fits | source `54281f0` |
| CETUS revision named for the two data/result packages below | revision `9a06f6c` |

Independent checks covered all 40 fitted estimators, reproduced predicted classes exactly, checked probabilities with absolute tolerance `10^-12`, and checked 120 output hashes. The original SGD and nearest-neighbor checks covered 40 complete run manifests, 100 probe heads, 20 nearest-neighbor vectors, 280 output hashes and five normalization replays from raw pixels. These checks reused saved features rather than repeating pretrained inference.

## Data and result packages

| Repository | Exact relative path | Contents or purpose |
| --- | --- | --- |
| CETUS | `release/titan-followup-20261109/` | Four preserved saved-result archives, CPU reconstruction commands, access inventory, public attribution and original producer identities |
| CETUS | `release/longitude-review-20260929/` | Catalog, split definitions, original predictions with encoder weights unchanged, fold metrics and source-part counts |
| CETUS | `release/longitude-auxiliary-20260929/` | Common classifier and classical-feature results, saved predicted classes, scientific specifications, and separate VIMS display metadata with corrected coordinates |
| CETUS | `release/longitude-adaptation-controls-20260930/` | Completed partial-training and training-substitution predictions, histories and reconstruction code; its README distinguishes saved-prediction reconstruction from new training and specifies additional inputs |
| CETUS | `release/interpretation-20261001/` | Class-level changes, alternative regional weighting, geometric exposure, map-part counts and image examples from existing predictions and metadata; its README gives reconstruction commands |
| CETUS | `release/longitude-review-20260929/evidence/` | Exported label and catalog checks; full execution records remain in the development repository |
| CETUS | `release/v1-review-20260908/DATA_LICENSES.md` | Source-specific terms and required attributions |

The original v1 payload remains unchanged. The three longitude packages must retain their separate population, prediction and producer identities. The original appendix's statement about revision `9a06f6c` refers specifically to the longitude-review and longitude-auxiliary packages, not to every later package.

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

These files require access to the development repository and are not included in CETUS. The paper retains the material limitations that prompted these checks: geographic overlap and Earth-label concerns, undefined spatial correlations, invalid original Random Init permutation statistics and unresolved initial model state.

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

The paper's DOFA usage citation links to the
[Sentinel-1 example](https://github.com/zhu-xlab/DOFA/blob/73ff5c0721da322689ae890bed5b4efd78935f47/README.md)
at repository revision `73ff5c0721da322689ae890bed5b4efd78935f47`.
The printed reference uses a short link label; this guide retains the revision
and filename for reproduction. The model-loading example lists the missing
and unused normalization parameters, and the Sentinel-1 example applies channel
standardization and the band identifier 5.405.

The bibliography cites CROMA through DOI
[10.52202/075280-0241](https://doi.org/10.52202/075280-0241) and Mars-Bench
through DOI [10.52202/085713-4098](https://doi.org/10.52202/085713-4098).
Their original proceedings links remain available here for lookup:
[CROMA paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/11822e84689e631615199db3b75cd0e4-Paper-Conference.pdf)
and [Mars-Bench proceedings entry](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b21b8e8823034df7de4a8cc2e56891af-Abstract-Datasets_and_Benchmarks_Track.html).

## Earth preprocessing and training-count checks

The development record `audit/CONTROL_RESULTS_2026-09-29.json` gives complete
class metrics, differences between corresponding runs and execution checks.
`audit/R7_NUMERICS_2026-09-29.json` checks all 15 bandwidths and 12,000 MMD
resamples from the saved features. These are additional calculations on the
original population and are separate from the preserved September 8 package.
They remain development-repository records; listing them here does not add
those files to this repository. The papers retain the underlying scientific
limits, including Earth label uncertainty and use of Titan training images.

## Earlier results and reconstruction

The expanded frozen comparison includes 100 SGD classifiers and twenty nearest
neighbor evaluations. The additional classifier package reconstructs seventy
confusion matrices for 23,246 distinct test tiles. Its VIMS metadata contains
23,379 numerical samples and one unavailable sample at an image edge; these
display samples are not classifier inputs or physical composition measurements.

The interpretation package retains the original weighting comparison: CROMA's
F1 difference after encoder training changes from -0.17 to +0.85 percentage
points when confusion counts are combined across test regions separately for
each training seed. Only 1,136 of 23,246 test centers lie within the original
minimum separation of a reinstated training center. These calculations use
existing predictions; equal regional weights remain primary.

Historical Earth limits also remain: images named Wadi Rum came from Egypt,
some Plains images lie offshore, and some training and validation image areas
overlap. Training-count and preprocessing checks retain those limitations and
assign no new Earth labels.

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

This repository distributes metadata, predictions, software and the November
companions' small SAR/map inspection excerpts. The complete processed Titan
image collection is available through the separate
[image release](https://github.com/magnaprog/CETUS/releases/tag/titan-processed-images-20261109).
Raw radar products, embeddings and pretrained or fitted model weights have
separate access routes. Internal verification records named above
retain their stated access limits. Their presence in this guide does not establish geological accuracy,
a universal reproducibility guarantee or access to omitted source data.
