# Reproducing the CETUS paper

CETUS asks two related questions: how well representations support terrain classification when classifiers learn from Titan labels, and how an Earth-trained classifier transfers to Titan. The experiments use different populations and geographic divisions. Begin with those distinctions, then select the calculation you want to reproduce.

| Goal | Route | What the output establishes |
| --- | --- | --- |
| Recover a reported numerical comparison | [Saved-result archives](#november-2026-follow-up) and the study-specific companions below | Arithmetic from supplied predictions, logits or histories, with each reader's stated checks and omissions. |
| Inspect terrain composition or recreate a display | [Map census](#terrain-within-each-image), [class ranking](#ranking-images-by-terrain-class) and [sampling displays](#geographic-sampling-displays) | Counts and displays from existing arrays and results. |
| Recompute native Titan features and classifier outputs | [SOFTWARE.md](SOFTWARE.md#install-and-check) | Model execution under specified data, weights and runtime conditions, separate from reading saved predictions. |
| Inspect training implementations | [Execution source](release/current-source-20261109/README.md) and [DOFA partial-training source](release/dofa-partial-source-20261112/README.md) | Code and contracts for preparation; additional execution inputs are needed. |

Use fresh output paths and each package's specified environment. Numerical companions reconstruct saved arithmetic; they do not supply all inputs for model inference or training. The [access inventory](release/titan-followup-20261109/DATA_ACCESS.md) identifies supplied files and omitted inputs. The complete processed Titan image collection has a [separate release](https://github.com/magnaprog/CETUS/releases/tag/titan-processed-images-20261109).

## Choosing an experiment

For the primary geographic evaluation, use the 23,380-tile catalog and its five longitude-sector partitions, with 23,246 distinct test tiles across the five folds. Keep the predictions and splits from the same package together. Earth transfer and the Venus distribution comparison use the 11,371-tile Titan catalog with the western map-sampling omission. Earth transfer is evaluated on its spatial-block test folds. The Venus comparison pools all non-Selk Titan tiles and all Earth source tiles, separately from the foldwise transfer evaluation. Their conclusions and model initialization details remain separate.

The native classifier fitted by stochastic gradient descent (SGD) uses unstandardized encoder features. The common logistic regression (LR) comparison standardizes features using training statistics and fits with limited-memory Broyden–Fletcher–Goldfarb–Shanno (LBFGS). Their scaling, optimization, regularization and training settings differ, so their score difference cannot isolate any one of those choices.

The majority-class predictor and classifiers on classical image measurements provide reference comparisons. Training-count, preprocessing and training-image replacement studies ask additional questions about sensitivity. Their roles and populations determine which scores can be compared. For the primary Titan comparisons, average seed metrics within each region, then give the five regions equal weight. Combining confusion counts across regions answers a different question because it changes the population weighting.

## Fold numbers in the current papers

The papers and current figures number folds 1 through 5. Saved results, split files, commands and array indices retain IDs 0 through 4. To locate a paper fold in these files, subtract one from its displayed number. Image assignments, metrics, seeds and averaging are unchanged. Earlier figures and execution logs retain the numbering used when they were produced.

| Paper fold | Saved ID |
| --- | --- |
| 1 | 0 |
| 2 | 1 |
| 3 | 2 |
| 4 | 3 |
| 5 | 4 |

For the expanded Titan evaluation, paper fold 2 contains the single Craters test tile; paper fold 1 contains the single Craters validation tile. The original dataset uses different geographic groups, even when fold numbers coincide. Map-component IDs and seed values are separate identifiers.


## Untrained encoder and other baselines

The paper's **Untrained ViT** is the `random_init` method in the software and result files. Its encoder receives no training; a classifier learns from its image features and Titan map labels. For the expanded Titan catalog, the encoder uses a single input channel and initialization seed 42. The five SGD seeds repeat classifier training on those same features. They do not repeat encoder initialization. DINOv2 instead receives three repeated channels with ImageNet normalization, so this comparison also includes differences in input handling.

The common logistic regression comparison includes intensity statistics, histograms, texture measurements and their concatenation. These provide the classical baselines. The historical small CNN results belong to the original catalog; they cannot be combined with the expanded-catalog table. Predicting the most frequent training class is a separate comparison that ignores images.

The [initialization study](release/titan-followup-20261109/README.md) compares official DINOv2 ViT-B/14 with five random states and one pretrained state under matched transforms and C=1 logistic regression, and ResNet18 trained from random or ImageNet initialization with three seeds per arm. The CNN comparison retains its separate validation-based learning-rate choices. These studies do not alter the single-state `random_init` definition, the small CNN on the spatial-block partitions, or the scientific results in the September and October packages.

## Input-field names used by the software

`label_confidence` stores the modal map fraction: the fraction of valid map samples assigned to the most frequent class in an image. It does not measure confidence assigned by a geologist. The map grid and valid SAR mask cover potentially different portions of the tile.

DOFA's final feature normalization is `fc_norm`. The specified model file lacks `fc_norm.weight` and `fc_norm.bias`, which retain scale one and bias zero. The separate `norm` parameters in that model file are unused by the evaluated feature path. This is a description of the tested implementation; pretraining input conventions remain uncertain. The model uses identifier 13.78 for Titan, while the upstream Earth example uses 5.405 and Appendix D of the DOFA preprint, arXiv:2403.15356 version 2, gives SAR identifier 3.75. See SOFTWARE.md for the exact source and model hashes.

The paper's DOFA usage citation links to the [Sentinel-1 example](https://github.com/zhu-xlab/DOFA/blob/73ff5c0721da322689ae890bed5b4efd78935f47/README.md) at repository revision `73ff5c0721da322689ae890bed5b4efd78935f47`. The printed reference uses a short link label; this guide retains the revision and filename for reproduction. The model-loading example lists the missing and unused normalization parameters, and the Sentinel-1 example applies channel standardization and the band identifier 5.405.

The bibliography cites CROMA through DOI [10.52202/075280-0241](https://doi.org/10.52202/075280-0241) and Mars-Bench through DOI [10.52202/085713-4098](https://doi.org/10.52202/085713-4098). Their original proceedings links remain available here for lookup: [CROMA paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/11822e84689e631615199db3b75cd0e4-Paper-Conference.pdf) and [Mars-Bench proceedings entry](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b21b8e8823034df7de4a8cc2e56891af-Abstract-Datasets_and_Benchmarks_Track.html).

## November 2026 follow-up

The [9 November saved-results release](release/titan-followup-20261109/README.md) contains four unchanged numerical archives. They cover the current 23,380-tile catalog and 23,246 distinct test tiles. The package's [provenance](release/titan-followup-20261109/provenance.json) records archive identities, original scientific producers and the source revision used for this release. A later documentation or packaging revision does not replace a scientific producer.

| Archive | Completed work and recipient reconstruction |
| --- | --- |
| Scientific | 315 saved test-prediction vectors: 40 common-LR, 100 SGD, 20 kNN, 75 partial-training and 80 fitting-study vectors. CPU replay checks sixteen fitting settings, twenty contrasts and eleven generated table fragments. It does not refit classifiers or reconstruct gradients. |
| Follow-up | Coverage calculations for 115 saved vectors; map-vote replay for 2,685 sampled grids and sixty SAR inspection packets. The 2,643 partial test tiles differ from the 2,651 partial catalog tiles plus 34 complete inspection tiles used for map votes. |
| Initialization | The completed 90-case study comprises thirty DINOv2 readouts, thirty ResNet18 tuning fits and thirty final fits. Thirty DINO feature jobs supplied the readout inputs. The companion reconstructs sixty native test records, 6,000 validation-history rows from sixty CNN fits, ten learning-rate choices and 180 coverage confusion matrices. |
| Reassessment | Thirty evaluations of saved epoch-100 CNN states, validation-choice checks and thirty-seven SAR/map excerpts. The original validation-selected results remain primary; the epoch-100 comparison involved no retraining or reselection. |

The [recipient commands](release/titan-followup-20261109/README.md#reconstruct-saved-results) reconstruct saved arithmetic on CPU with Python 3.12.3 and NumPy 2.4.4. Write reports outside the extracted packages. These calculations do not regenerate GPU probabilities, omitted per-epoch prediction vectors or optimizer updates. Validation-removal checks rely on supplied outcomes because the prediction vectors needed to recalculate every removal score are unavailable.

The fitting study retains 23 iteration caps and 54 gradient concerns. Initialization comparisons reuse examined test regions, and the CNN initializations use separate validation choices of learning rate and epoch. Average seeds within regions before giving the five regions equal weight; their sample SD is descriptive. Neither repeated seeds nor the learning curves establish independent regions, convergence, equal optimization adequacy or a causal effect of pretraining. Rare Craters support and unsupported-class null summaries remain part of the reporting rules.

The [access inventory](release/titan-followup-20261109/DATA_ACCESS.md) separates the supplied inspection excerpts from the full pixel collection, embeddings and model states, which are not included. The [current execution source](release/current-source-20261109/README.md) provides the CNN, matched-DINO and CPU-fitting implementations with their [execution contracts](release/current-source-20261109/CONTRACTS.md). It does not supply the omitted scientific inputs or historical machine-bound plans. The native SGD and nearest-neighbor runner remains a separate workflow.

## Terrain within each image

The [complete map census](release/map-census-20261109/README.md) supplies all 23,380 saved map grids, their SAR validity masks and five-fold role assignments. The CPU checker reconstructs class and unknown counts under four coverage rules for the complete catalog and every training, validation, test and exclusion group. It uses saved arrays; original SAR decoding and map sampling are separate checks.

With NumPy installed, run from the companion directory:

```sh
python3 -B -m unittest -v test_check_counts
python3 -B check_counts.py --output /absolute/new/path/map-counts.json
```

Craters occurs in 331 images under complete nominal SAR coverage; 140 images have Craters as their modal label. Of the eligible Craters cells, 21.8% occur in other modal classes. These counts include overlapping images and describe the map's spatial mixtures. Existing image labels and model scores are unchanged. The companion README defines the masks, units and interpretation of each count.

## Craters training and map composition

The [10 November companion](release/crater-analysis-20261110/README.md) preserves the completed CPU analysis of training-batch availability and modal-Craters test images. It joins 30 primary, validation-selected CNN records and five pretrained DINOv2 records to the accepted census. Its reproduction uses the existing initialization ZIP and public census `counts.json` and `map_cells.npz`; commands and exact input identities are in the companion README. No model is fitted or rescored, and per-seed results remain separate.

The batch bound concerns the equal-step average of normalized target-class cross-entropy coefficients; it does not bound gradients or performance. Map fractions concern eligible known cells, with empty domains undefined. Repeated seeds and overlapping images do not provide independent geographic samples.

For the same map-composition check across all eight primary feature families, run the current [comparison script](scripts/summarize_crater_map_comparison.py) from the repository root:

```sh
python3 -I -B scripts/summarize_crater_map_comparison.py \
  --crater-results release/crater-analysis-20261110/results.json \
  --predictions release/longitude-auxiliary-20260929/results/predictions.json.gz \
  --output-dir /absolute/path/to/new-output-directory
```

The output directory must be new and its parent must exist. Python's standard library is sufficient. The command joins saved predictions to the 39 test images whose eligible known map cells all carry Craters under both map-cell calculations. It writes `results.json` and a TeX table containing regional counts, the total correct among those 39 images, and Craters true and false positives across all 23,246 test images. These pooled counts answer a different question from mean regional F1. The primary DINOv2 extraction is distinct from the initialization study's extraction, and the primary untrained ViT is a separate control. The original companion and its initialization results stay intact.

## Ranking images by terrain class

The [class-ranking companion](release/class-ranking-20261111/README.md) calculates average precision (AP) from the initialization study's saved probabilities. It keeps each region and fitted model separate, includes all six classes and retains all 60 Craters precision–recall curves. The calculation admits tied scores together and sums recall increments times precision without interpolation. Seeds are averaged within each region before giving the five regions equal weight.

Both initialization comparisons improve mean macro average precision. Mean AP rises for Plains, Dunes and Labyrinths in both comparisons. DINOv2's mean Hummocky and Lakes AP fall. Mean Craters AP changes little for DINOv2 and declines for ResNet18. Region 2 has only one Craters image, whose rank strongly affects its score; the companion retains all seeds and the differences in the other regions. This is an exploratory ranking analysis on the same test images. It measures agreement with the terrain map and does not assess calibrated probabilities or detection of individual geological craters.

The companion's commands use Python 3.12.3 and NumPy 2.4.4 on CPU. They read the existing initialization ZIP intact, reconstruct all 360 class scores and their regional summaries, and compare every saved curve array. No model weights, image decoding or training are involved.

The [ranking displays](release/crater-retrieval-20261114/README.md) show all 60 regional class contrasts on a common percentage-point scale. Each DINOv2 point compares one pretrained state's AP with mean AP from five random states; each ResNet18 point averages three paired seed differences. The absolute Craters AP plot and its caption remain available there, with every state or seed shown separately. These displays preserve the different fitting methods and validation choices used in the two model comparisons.

### Displaying retrieved tile counts

The [retrieval companion](release/crater-retrieval-20261114/README.md) renders recall against retrieved test-tile count for all six map classes. Equal scores enter together, with each region and model state shown separately. Every score group has an endpoint, and all six classes select state or seed identity symbols from existing endpoints using the same display-spacing rule. Dense overlap can obscure individual states. The count axis is logarithmic, and its area has no average-precision meaning.

The CPU command reads saved curves and preserves the ranking and model choices. It produces both paper variants with bundled fonts. Retrieved tiles can show overlapping views of a landform, so the count describes images rather than independent geological objects.

### Geographic sampling displays

The same companion plots all 23,246 Titan test centers, the thirteen Earth collection sites and 398 original image windows, and the 2,945 unlabeled Venus centers. Titan colors identify modal map labels. Earth windows use the retained source-window polygons and their geometric centers. These describe nominal sampling geometry; terrain purity and valid SAR coverage throughout a window require separate evidence. Local Earth axes are scaled independently in longitude and latitude, so displayed shapes and areas are not metric comparisons. Venus symbols distinguish complete and partial image coverage. The displays describe sampled images, not independent geological objects or mapped surface area.

The compact coordinate payload supports both paper styles without access to mission archives. Its source and population descriptions accompany the plotting commands in the companion guide. These commands reproduce the companion's versioned sampling diagrams. The later SAR, ISS and relief backgrounds in the 20 November 2026 manuscript revision require the development repository's additional figure inputs and renderers; they are not bundled in this companion.

### Regional SGD variation

The [regional SGD display](release/crater-retrieval-20261114/README.md#geographic-variation-and-map-mixtures) shows all five classifier seeds, their mean and range in each test region. Its second subplot compares error rates in twenty eligible class-region groups at two ranges of modal map fraction. The companion supplies the unchanged saved report and rendering command; the explanation gives the averaging order and which terrain classes contribute. These displays reuse existing results.

## Training loss normalization

The [execution source](release/loss-source-20261110/README.md) compares two denominators for the weighted cross entropy used to train ImageNet ResNet18. The original divides by the sum of label weights in each batch. The alternative divides by the batch size times the mean weight across the training images. Both retain the original class weights and image sampling.

The [numerical companion](release/loss-normalization-20261110/README.md) supplies saved records from fifteen additional 100-epoch fits and fifteen fresh replays of the validation-selected states on validation and test images. All fifteen original and all fifteen additional runs have zero Craters true positives. Six-class macro F1, averaged over three seeds within each region and then equally over five regions, changes from 43.84% to 44.38%. Regional macro F1 means rise in three regions and fall in two. The companion retains all six classes, including false-positive counts.

The study reuses the five test regions and the learning rate selected for the original loss. It therefore compares the loss denominators at that learning rate, rather than comparing separately tuned losses. Seeds repeat training within regions and do not add independent geographic samples. Because the original Craters F1 is zero, a nonnegative difference alone cannot show recovery; the true-positive and false-positive counts are needed to interpret it. The scores measure agreement with the SAR-informed map, and the original initialization results remain primary.

Use Python 3.11 or later on Linux; reconstruction needs only the standard library. From the repository root:

```sh
cd release/loss-normalization-20261110
sha256sum -c SHA256SUMS
python3 -B reconstruct.py \
  --initialization-zip ../titan-followup-20261109/initialization-supplement.zip \
  --output /absolute/new/path/loss-comparison.json
```

Choose a new output filename outside the companion directory. The reader reuses the unchanged [initialization archive](release/titan-followup-20261109/initialization-supplement.zip) for the original A runs, labels and geographic divisions; it reads the archive's data without executing its Python files. It checks saved arithmetic and compact validation histories without fitting models or rerunning GPU predictions. The [manifest](release/loss-normalization-20261110/manifest.json) and [notice](release/loss-normalization-20261110/NOTICE.md) record identities and credits.

The separate source package retains the implementation, study specification and focused synthetic CPU tests. Its [execution contracts](release/loss-source-20261110/CONTRACTS.md) describe the original input and model-state dependencies. Model states and full training inputs are not supplied by the numerical companion.

## DOFA band input comparison

The [DOFA companion](release/dofa-band-20261109/README.md) contains the completed 13.78 versus 3.75 comparison on all five regions. It preserves ten sets of test predictions and probabilities and reconstructs their arithmetic on CPU. It measures input sensitivity with encoder weights unchanged. The fifty-fit partial-training comparison below is a separate experiment; both preserve the Earth results and the three-encoder partial-training comparison.

## DOFA input during partial training

The [partial-training companion](release/dofa-partial-20261112/README.md) contains fifty saved prediction vectors: two numeric inputs, five seeds and five geographic regions. It compares inputs 13.78 and 3.75 while retaining the training settings and identical initial encoder and classifier states within each pair. The mean macro F1 is 15.68% at 13.78 and 37.52% at 3.75. The paired difference is 21.84 percentage points, with sample SD 4.24 across five regional means after averaging seeds. Craters mean F1 at 3.75 is 1.37%.

From the repository root, choose a new output filename in an existing directory outside the checkout. The output and its `.partial` and `.failure.json` sidecars must all be absent.

~~~sh
cd release/dofa-partial-20261112
python3 -B reconstruct.py --output /absolute/external/directory/dofa-partial-reconstructed.json
~~~

Preserved companion recipes use `../` to write outside an extracted package. Inside this checkout, those paths can still write into `release/`; use an external directory to keep later model execution's clean-source check satisfied.

The standard-library reader checks all ordered test labels and predictions, confusion counts, class and macro scores, and paired aggregates. It does not load models or reproduce training. The result concerns agreement with map labels on previously examined regions. At 3.75, logistic regression on unchanged encoder features reaches 38.42% mean macro F1 in the separate feature comparison; its fitting and epoch choices differ.

The [probabilities and histories companion](release/dofa-scores-20261114/README.md) adds the six class probabilities and fifty-epoch histories for these same fits. It checks probability sums, predictions among tied maxima, the earliest maximum of saved validation recall, and the existing test metrics and regional means. The probability and history values were compared directly with all fifty native run records. The recipient check uses the saved values; it does not regenerate probabilities or validation predictions.

From the repository root, with Python 3.11 or later, choose another fresh output filename in an existing directory outside the checkout. Its output and both sidecar paths must be absent.

~~~sh
cd release/dofa-scores-20261114
python3 -B reconstruct.py --output /absolute/external/directory/dofa-scores-checked.json
~~~

The [source package](release/dofa-partial-source-20261112/README.md) contains the training and replay implementation and focused tests. It supports inspection and preparation; the full historical source, data, weights, environment and executable contract remain separate prerequisites. Its documentation contract does not launch the experiment. The numerical companion above is self-contained.

## Image examples in the paper

The earlier full-paper and CVPR galleries contain seventeen examples from the original Earth, Titan and Venus collections, chosen from 134 candidates using coverage, contrast and visual inspection. The 20 November 2026 CVPR supplement retains those choices within an expanded gallery of thirty-four images: ten Earth, twelve Titan from the 11,371-tile catalog, and twelve unlabeled Venus images. Earth examples have at least 99.9% valid pixels; Titan and Venus examples have complete coverage. Titan examples also have modal map fractions of at least 80%. The original Earth Labyrinths choice shows Atacama channels, replacing the misnamed Wadi Rum example. The original Earth Craters choice retains one invalid pixel, masked in magenta with diagonal hatching.

Each added Earth or Titan example is nearest its class's median catalog contrast among eligible images outside the original selection, with tile-ID tie breaking. Earth additions require a different site and exclude Wadi Rum, Arctic tundra and Icelandic sandur because of documented site concerns. The original Arctic tundra assignment remains unverified. The six additional Venus images are nearest six evenly spaced contrast quantiles from 0.1 to 0.9 among fully covered images outside the original selection. Those quantiles use linear interpolation, with tile-ID tie breaking and no repeated selections. Contrast is the standard deviation of Earth sigma0 in dB or Titan/Venus display DN. These illustrations preserve catalog assignments and are not a representative sample or independent geological validation. They remain separate from the enlarged Titan catalog's examples chosen by median modal map fraction.

Figure inputs, rendering code and checks belong to the development repository. Display choices do not change training inputs or scores. The public plotting companion retains its earlier six Venus A–F markers; it does not supply the expanded A–L gallery or the later manuscript backgrounds. The paper's plots display folds 1 through 5 as described above. Earlier figures inside the result packages retain their original appearance and numbering.

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

The author checked native fitting, selected-checkpoint inference, classifier fitting after training-image replacement, and the resulting summaries. The adaptation reconstruction checked 165 confusion matrices and 5,670 scalar comparisons. The training-image replacement checks covered 500 SGD classifiers and 100 common-logistic fits. These checks are distinct from the public-source reproduction's 20 feature jobs and 20 analysis jobs. Their source and runtime conditions are described in [SOFTWARE.md](SOFTWARE.md), and the [adaptation package](release/longitude-adaptation-controls-20260930/README.md) supplies a recipient reconstruction route. A saved-result reader checks its specified arithmetic; it does not repeat those original model executions.

## Development evidence with restricted distribution

Detailed spatial-error checks, Earth-site geometry checks and the numerical recalculation of the original Random Init features are retained in the development repository. They are not distributed here. The spatial record contains 2,400 result rows, including raw and residual coefficients, counts and reasons for undefined values. The relevant scientific limits still apply to the public results: geographic overlap, uncertain Earth labels, undefined spatial correlations, invalid original Random Init permutation statistics and an unresolved initial model state. [CORRECTIONS.md](CORRECTIONS.md) explains the data and model limitations; the public packages expose only the records listed in their inventories.

## Exact implementation and image identifiers

* CROMA's mean patch representation passes through `GAP_FFN_s1`. This is its learned SAR feedforward transformation; it must not be confused with the separate contrastive projection.
* The first-ranked Selk tile is `titan_022929`, at 6.64 degrees north and 164.83 degrees east. The paper retains the coordinates and rank. The identifier is a lookup key, not a scientific-priority recommendation.
* The Earth catalog key `wadi_rum` refers to the mislocated Egyptian source. The paper retains its Wadi Rum label, Egypt location and the unresolved Labyrinths assignment. Renaming the key does not validate the terrain label.
* For the original Random Init experiment, the architecture fallback branch and initialized-state hash remain unresolved. Seed 42 is specified in the source; the five classifier seeds do not identify five independently initialized encoders. No missing hash is supplied here.


## Earth preprocessing and training-count checks

The author checked class metrics, differences between corresponding runs and their execution records, and separately checked 15 bandwidths and 12,000 MMD resamples from saved features. These calculations concern the original population. Their detailed verification records are retained privately; they add no files to the preserved September 8 package. The public [spatial-block file inventory](FILES.md#additional-experiment-records) locates the training-count and preprocessing results. Interpret them with the existing Earth-label uncertainty and the stated use of Titan training images.

<a id="earlier-results-and-reconstruction"></a>

## Classifier and interpretation checks

The expanded frozen comparison includes 100 SGD classifiers and twenty nearest neighbor evaluations. The additional classifier package reconstructs seventy confusion matrices for 23,246 distinct test tiles. Its VIMS metadata contains 23,379 numerical samples and one unavailable sample at an image edge; these display samples are not classifier inputs or physical composition measurements.

The interpretation package retains the original weighting comparison: CROMA's F1 difference after encoder training changes from -0.17 to +0.85 percentage points when confusion counts are combined across test regions separately for each training seed. Only 1,136 of 23,246 test centers lie within the original minimum separation of a reinstated training center. These calculations use existing predictions; equal regional weights remain primary.

Historical Earth limits also remain: images named Wadi Rum came from Egypt, some Plains images lie offshore, and some training and validation image areas overlap. Training-count and preprocessing checks retain those limitations and assign no new Earth labels.

## Package reconstruction and exact software sources

The evaluation software retains scientific code from development revision `0257a36197c51eda0a31768531ea26eef9887a6e`. Its source hashes are in [SOFTWARE_EXPORT.json](SOFTWARE_EXPORT.json). The public inference reproduction used CETUS revision `360eaea532ead477f46906d7ebe0cb5282e8b082` on September 30, 2026. Its 20 feature jobs and 20 analysis jobs reproduced the native expanded Titan features, classifier probabilities, predictions and saved metrics in the specified environment. Independent metric calculations allow absolute error `1e-12`. [file_index.json](file_index.json) retains the completion-record hash; [SOFTWARE.md](SOFTWARE.md) gives the runtime and comparison scope.

The encoder-training package was exported from development revision `3f4a6a4`. Its `evidence/acceptance.json` has SHA256 `7340fcc5d1fdc52b27ff3c0385d86669df97345897f95efa92bcb795dba5ba03`. Independent CPU reconstruction reproduced all 51 package files, totaling 10,201,553 bytes, including figures, numerical claims and inventories. The three input packages retained their bytes. The October interpretation analyses come from development revision `0297c9a987f4e0ea919b3a40de0ae208103719e6`.

The original run identifiers and data routes in this guide were transcribed from development revision `5dedb5d762b40e7c331f1bffa597819691e59c96` and the experiment records. Dated package guides keep their export-time wording and checksums. Later image/source releases and completed reconstruction routes are described here without changing those historical packages or their scientific producer identities.

## Citations, data terms and availability

The terrain map is the first edition of [Titan Global Map Shapefiles](https://data.mendeley.com/datasets/f6jrtyfp66/1). The dataset title and file format retain their original names. Source terms and credits appear in [DATA_LICENSES.md](DATA_LICENSES.md) and the original package's [data terms](release/v1-review-20260908/DATA_LICENSES.md). GSHHG shoreline documentation is at [the provider's site](https://www.soest.hawaii.edu/pwessel/gshhg/). Bengio and Grandvalet's discussion of cross-validation uncertainty is in [their JMLR article](https://jmlr.org/papers/v5/grandvalet04a.html).

This repository distributes metadata, predictions, software and the November companions' small SAR/map inspection excerpts. The complete processed Titan image collection is available through the separate [image release](https://github.com/magnaprog/CETUS/releases/tag/titan-processed-images-20261109). Raw radar products, embeddings and pretrained or fitted model weights have separate access requirements. The [access inventory](release/titan-followup-20261109/DATA_ACCESS.md) states which inputs are supplied. Completed author checks and a recipient's numerical reconstruction have different scopes; neither changes the interpretation of the scores as agreement with the terrain map.
