# CETUS

CETUS studies how representations learned from Earth images transfer to
Cassini SAR images of Titan. This repository contains the tile catalogs,
geographic divisions, predictions and software used to evaluate agreement
with an expert terrain map.

Start with [the paper reproduction guide](PAPER_REPRODUCTION.md) to locate an
experiment's data and results. [SOFTWARE.md](SOFTWARE.md) gives installation
instructions, required inputs and commands. Image arrays, original radar
products, pretrained model weights and embeddings must be supplied separately;
this repository provides no download location or project DOI for those images.

## Data and experiments

The original and expanded Titan datasets belong to different experiments.
Choose the complete package for the result you want to reproduce.

| Package | Contents |
| --- | --- |
| Root `catalogs/`, `splits/` and `results/` | Original sample: 11,371 Titan tiles, 398 Earth tiles and 2,945 Venus tiles without labels. These files retain their original bytes and links. |
| [Original September package](release/v1-review-20260908/README.md) | The 46 files from the September 8 repository, with its index and checksums. Read the subsequent correction notice when interpreting this snapshot. |
| [Expanded Titan dataset](release/longitude-review-20260929/README.md) | 23,380 Titan tiles, including 140 Craters tiles; five geographic folds; linear classifiers and nearest neighbors for DINOv2, DOFA, CROMA and Random Init. Encoder weights stay unchanged. Earth transfer and training of encoder weights belong to separate experiments. |
| [Additional classifier results](release/longitude-auxiliary-20260929/README.md) | Logistic regression and classical image features on the expanded dataset, with 70 prediction vectors. Separate VIMS display metadata uses the same catalog and contributes no classifier input. |
| [Encoder training and training-image replacements](release/longitude-adaptation-controls-20260930/README.md) | Predictions and histories from 75 fits that train part of each encoder, plus SGD and logistic regression experiments that replace some training images. CPU software reconstructs the scores, tables, figures and registered numerical claims. |
| [Interpretation of the Titan results](release/interpretation-20261001/README.md) | Changes by terrain class, alternative weighting of regions, distances to reinstated training images, map-component counts and image examples. These additional analyses use existing predictions and retain equal regional weights for the primary scores. |

Read [CORRECTIONS.md](CORRECTIONS.md) before using the original results. The
original map sampler omitted valid western terrain. The expanded catalog
preserves original image pixels, adds 12,009 tile IDs and changes 15 existing
labels. Use the catalog and geographic split checksums together: an image ID
alone can refer to different labels in the two datasets. Both the sampled
population and the geographic design changed, so their effects on performance
cannot be separated by comparing the resulting scores alone.

Earth labels still need expert review. Images named Wadi Rum came from Egypt;
some Plains images lie offshore, and some training and validation image areas
overlap. The additional experiments examine training counts and preprocessing
while retaining these limits. This repository assigns no new Earth labels.

## Reproduce the results

The expanded Titan comparison includes 100 SGD classifiers and 20 nearest
neighbor evaluations. Its [predictions](release/longitude-review-20260929/results/predictions.json.gz)
and [metrics](release/longitude-review-20260929/results/frozen_results.json)
allow readers to calculate confusion matrices, class scores, regional means
and sample standard deviations. These scores measure agreement with the map.
Geological accuracy requires independent observations.

The additional classifier package applies the same logistic regression settings
to four encoder representations and four classical feature families. It also
includes baselines that always predict the most frequent training class or use
training class frequencies. The predictions reconstruct 70 confusion matrices
for 23,246 distinct test tiles. Its VIMS metadata contains 23,379 numerical
samples and one unavailable sample at an image edge. The values describe a
display composite; physical composition and mission coverage require other data.

The encoder-training package needs both neighboring Titan packages without
changes. It compares complete training configurations and different training
images. Optimization, image content and geography can contribute together to
its score differences. Use the [CPU reconstruction instructions](SOFTWARE.md#reconstruct-adaptation-and-training-substitution-results)
to check its saved predictions. Fine-tuning results in the root directory use
the original dataset; the package contains the corresponding experiments on
the expanded dataset.

The interpretation package shows how regional weighting affects the same
predictions. CROMA's F1 difference after encoder training changes from -0.17 to
+0.85 percentage points when confusion counts are combined across test regions
separately for each training seed. Only 1,136 of 23,246 test centers lie within
the original minimum separation of a reinstated training center. The package
gives the calculations, numerical tolerances and optional figure checks when
image arrays are available.

[SOFTWARE.md](SOFTWARE.md) records the exact source revisions, environments and
verification checks. The public inference run reproduced the native expanded-dataset features,
classifier probabilities, predictions and metrics in 20 feature jobs and 20
analysis jobs. Independent reconstruction also reproduced all 51 files of the
encoder-training package, totaling 10,201,553 bytes. These checks apply to the
specified inputs and runtime. New training of classical classifiers or encoder
weights requires the additional work described in that guide.

## File integrity and use terms

Run `sha256sum -c SHA256SUMS` from the repository root. Each experiment package
also has its own checksum list, checked from that package's directory. At each
location, `file_index.json` lists every tracked file except itself and
`SHA256SUMS`; the checksum list includes the index. Git internals, ignored
environments and local outputs are excluded.

[FILES.md](FILES.md) explains the original root files.
[DATA_LICENSES.md](DATA_LICENSES.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) give the data terms and credits,
including the VIMS+ISS notice. Consult those terms separately from availability:
the processed image arrays are not distributed here.
