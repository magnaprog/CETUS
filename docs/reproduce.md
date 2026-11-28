# Reproducing CETUS experiments

Start with a result you can check from the saved labels and predictions. The population, training procedure and averaging rule determine which scores you can compare. The [experiment index](../experiments/README.md) leads to the full calculations. The [training guide](../training/README.md) identifies the software and inputs for a new fit.

## Craters training and map composition

From the repository root, run this comparison with Python's standard library. It creates a fresh output directory and prints the table for eight logistic classifiers.

```sh
cetus_check=$(mktemp -d)
python3 -I -B scripts/summarize_crater_map_comparison.py \
  --crater-results experiments/crater-training/results.json \
  --predictions experiments/regional-predictions/results/predictions.json.gz \
  --output-dir "$cetus_check/craters"
cat "$cetus_check/craters/crater-map-primary-lr.tex"
```

The table separates three counts. The first counts correct predictions among 39 test tiles where every known cell carries Craters, both before and after restricting cells to those with all four corresponding SAR samples valid. The other two pool all 23,246 test tiles. For example, CROMA correctly predicts Craters for 7 of the 132 modal Craters tiles and predicts Craters for 477 of the 23,114 tiles with other modal labels. The neighboring `results.json` supplies the counts by region and feature set.

The [Craters training analysis](../experiments/crater-training/README.md) estimates how often a shuffled training batch can contain a Craters tile. Training counts bound the average normalized Craters coefficient in the loss when every optimizer step has equal weight. Measuring the Craters contribution to the gradient would also require per-example gradients. The same analysis compares predictions from 30 ResNet18 fits and five logistic classifiers on pretrained DINOv2 features with the map cells in each test tile.

## Choosing an experiment

| Evaluation | Images and split |
| --- | --- |
| Within Titan, longitude sectors | The corrected catalog has 23,380 Titan tiles. There are 23,246 distinct test tiles across five folds. Bind predictions to the test IDs in the [longitude-sector splits](../experiments/longitude-benchmark/README.md). |
| Within Titan or transfer from Earth, spatial blocks | The Titan catalog contains 11,371 tiles and has a western map-sampling omission. The five folds test 11,306 distinct tiles and hold out the same 65 Selk tiles. Earth labels identify intended terrain at collection sites. See the [catalogs and splits](../data/README.md) and [coverage correction](corrections.md#western-titan-coverage). |
| Venus feature distributions | The comparison pools all non-Selk tiles in the spatial-block Titan catalog and all Earth source tiles. The 2,945 Venus images have no terrain labels. |

### Geographic region and fold numbers

Region 1 through Region 5 correspond to fold IDs 0 through 4. Region 2 has one Craters test tile, and Region 1 has one Craters validation tile in the longitude-sector evaluation. Spatial-block folds with the same numbers contain different images.

### Earth preprocessing and training-count checks

The [Earth split](../data/splits/earth_transfer_grouped_validation.json) assigns 352 of 398 images to training, including 16 images acquired in Egypt and cataloged as Wadi Rum. Compare the [Earth-trained classifiers](../data/results/probing_cv_results.json) with the [uniformly sampled and stratified Titan training subsets](../data/results/constrained_training_results.json) of 352 tiles. This matches the training total across domains. Per-class counts can differ. The linear classifiers output six scores. Transfer recall covers the five classes represented in Earth training. Earth training pixels set the Earth dB scale, and the full Titan training fold sets the display-intensity limits for both classifiers' Titan test images. The [site-omission and preprocessing records](../data/README.md#additional-experiment-records) show how the site sample, smoothing, quantization and rendering affect the transfer measurements.

## Classifier implementations and source inputs

The native linear classifier fits unstandardized features with SGD. The common logistic regression comparison standardizes features with training statistics and fits with L-BFGS. Scaling, optimization and regularization change together. The longitude-sector benchmark includes 100 SGD classifiers and 20 nearest-neighbor evaluations. The [regional records](../experiments/regional-predictions/README.md) reconstruct 70 auxiliary confusion matrices on the same test tiles. VIMS samples there supply display context.

The [scientific archive](../experiments/encoder-comparisons/scientific-supplement.zip) stores one logistic classifier per encoder and fold for the primary comparison. In `results/scaling/<model>-fold-<id>-standardized-l2/fitted_arrays.npz`, `test_probabilities` gives six probabilities per `test_ids` entry. The file also stores the training means and scales (`reference_mean`, `reference_scale`), coefficients and intercept. The [classical predictions](../experiments/regional-predictions/results/predictions.json.gz) supply decisions for intensity statistics, histograms, texture and all three feature groups together. To reconstruct the confusion matrix, count how often each predicted class occurs for each map label. To rank tiles for one terrain class, use the score for that class from every test tile in the region.

### Untrained encoder and other baselines

On the longitude sectors, `random_init` identifies an untrained ViT with one input channel and encoder seed 42. Five SGD seeds vary the classifier on these same features. DINOv2 receives three repeated channels with ImageNet normalization. To examine pretraining with the input transform held fixed, use the [initialization comparison](../experiments/encoder-comparisons/README.md). It fits the same logistic classifier (`C=1`) to one pretrained DINOv2 and five random states. For ResNet18 it compares random and ImageNet initializations with three seeds per arm and learning rates and epochs selected separately on validation. The small supervised CNN belongs to the spatial-block experiment. The majority baseline predicts the most frequent training class without reading the image.

### Saved-result archives

The [archive guide](../experiments/encoder-comparisons/README.md#reconstruct-saved-results) links four ZIPs for the longitude-sector evaluation and gives a common reconstruction procedure. The 315 scientific prediction vectors cover 40 common logistic fits, 100 SGD fits, 20 nearest-neighbor evaluations, 75 partial-training fits and 80 fitting-study vectors. In the coverage follow-up, 2,643 *test* tiles have partial SAR coverage. The 2,685 inspected grids include 2,651 partial tiles from the *full catalog* and 34 complete tiles.

The initialization archive records validation histories and the selected states. The reassessment evaluates saved states at epoch 100, fixing the epoch while retaining the separate learning-rate choices. Recomputing every effect of removing a validation example requires the per-epoch predictions. For the 80 fitting-study vectors, 23 fits reached the 2,000-iteration limit and 54 had a maximum absolute objective-gradient component above `1e-4` in standardized feature coordinates, including the intercept. The two conditions can overlap.

### Classifier and interpretation checks

Compute metrics for each seed within a region, average seeds, then give the five regions equal weight. The sample standard deviation of the five regional means describes geographic variation. Training sets overlap across regions, and seeds within a region reuse the same test data. Pooling confusion counts changes the weighting. For CROMA, training the last two blocks and classifier with epoch selection by validation macro recall changes mean regional test macro F1 by −0.17 percentage points relative to standardized logistic regression on unchanged features. Pool the confusion counts across regions within each seed before calculating F1 and the difference is +0.85 points. The [interpretation study](../experiments/terrain-interpretation/README.md) also examines training-image replacement. For 1,136 of 23,246 test centers, a reinstated training center lies within the split's minimum separation. Replacement changes both geography and image content.

## Terrain within each image

The [map census](../experiments/map-census/README.md) counts class and unknown cells under four SAR coverage rules. Of 23,380 tiles, 140 have Craters as the most frequent class among known cells. In 331 tiles, at least one Craters cell has four valid SAR samples. Of the 590,472 Craters cells with that coverage, 128,824 (21.8%) lie in tiles with another modal class. Overlapping tiles can count terrain more than once. A cell fraction is undefined when its denominator is empty.

### Titan image examples

The [example sheet](../experiments/terrain-interpretation/examples/repaired_examples.pdf) shows a tile at or nearest the median modal map fraction for each class and a seventh example whose modal class covers less than half the known cells. Match the [seven IDs](../experiments/terrain-interpretation/examples/selected.json) to the [processed Titan images](https://github.com/magnaprog/CETUS/releases/download/v0.2.0/titan-images.tar.zst) to render the source pixels.

### Ranking images by terrain class

Compute [average precision (AP)](../experiments/class-ranking/README.md) within each fit and test region by summing recall increments times precision, without interpolation. Equal scores enter together. For DINOv2 each region compares the pretrained state with the mean over five random states. For ResNet18 it averages three differences paired by seed. Both pretrained models have higher mean macro AP and higher mean AP for Plains, Dunes and Labyrinths than their random comparators. DINOv2 mean AP falls for Hummocky and Lakes. Craters AP changes little for DINOv2 and falls for ResNet18. In Region 2 a single Craters tile determines that class's AP.

### Displaying retrieved tile counts

The [retrieval plots](../experiments/terrain-plots/README.md#reading-the-rankings) show recall against the number of tiles at or above a threshold. Each equal-score group ends at one point. Symbols identify selected endpoints and can overlap. Read AP from the class-score records. Area on this plot depends on the logarithmic tile-count axis. Multiple retrieved tiles may cover one landform.

### Geographic sampling displays

The [sampling maps](../experiments/terrain-plots/README.md#reading-the-maps) show 23,246 Titan test centers, 398 Earth windows at 13 sites, and 2,945 Venus centers. Titan colors denote modal classes. Earth footprints show sampling geometry, with longitude and latitude scaled separately. Measurements of physical area, terrain purity or radar coverage need additional information. The Venus map identifies complete and partial coverage and six examples A–F.

### Regional SGD variation

The [SGD plot](../experiments/terrain-plots/README.md#craters-ap-and-regional-errors) shows all five classifier seeds in each region, their mean and range. Its second panel compares error rates in two ranges of modal map fraction for twenty eligible class–region groups.

## Training loss normalization

The [ResNet18 comparison](../experiments/loss-normalization/README.md) changes the denominator of weighted cross entropy from the sum of the batch's label weights to the batch size times the mean training-set weight. Class weights, image sampling and the original regional learning rate stay fixed. Both sets of fifteen fits have zero Craters true positives. Mean macro F1 over six classes changes from 43.84% to 44.38% after averaging three seeds per region and giving the five regions equal weight. Regional means rise in three regions and fall in two.

## DOFA band input comparison

The [fixed-encoder comparison](../experiments/dofa/frozen-encoder/README.md) changes the numeric band input between 13.78 and 3.75, with one logistic fit per input and region. See the [band conventions](corrections.md#dofa-inputs-and-encoder-training) and [feature pooling and checkpoint details](classifiers.md#what-the-encoders-receive).

### DOFA input during partial training

The [partial-training comparison](../experiments/dofa/partial-training/README.md) changes the same input while training the final two encoder blocks and classifier. For each region and seed, both inputs start from identical encoder and classifier states. Mean macro F1 rises from 15.68% at 13.78 to 37.52% at 3.75. The difference is 21.84 percentage points, with sample SD 4.24 across the five regional means after averaging seeds. Mean Craters F1 at 3.75 is 1.37%. The 38.42% macro F1 for logistic regression at 3.75 comes from fitting only the classifier. For all fifty partial fits, the [probability and history records](../experiments/dofa/probability-scores/README.md) let you check each prediction against its six class probabilities and each selected epoch against the earliest maximum of validation macro recall.

## Input-field names used by the software

`label_confidence` measures the modal class's share of known cells in a tile's 64 by 64 terrain grid, including known cells where SAR pixels are missing. The definitions below explain other fields in the archived source. In the spatial-block population the untrained encoder's initialization hash and architecture fallback are unresolved, and its archived single-precision MMD permutation p-values are unreliable. The [data guide](../data/README.md) identifies those records.

| Object or helper | Operation and interpretation |
| --- | --- |
| Titan tile values | `hisar_log_dn` arrays contain display digital numbers. The description “sigma0 linear” applies only to `linear_sigma0` mode; array shape and float32 storage do not establish calibrated backscatter. |
| Display-DN normalization | `compute_linear_norm_range` takes sample percentiles. Extreme values have less influence than on min/max bounds, but saturation can still determine a percentile bound. A percentile span below `1e-6` returns the declared default range. |
| dB normalization | `compute_norm_range` estimates pixel spread from per-tile means and available variances. The helper does not compare clipping counts. Titan display DN uses the separate percentile routine. |
| MMD estimates | `compute_mmd` returns squared MMD; `compute_mmd_repeated` summarizes those values. Its `ci95` field contains empirical subsampling quantiles, not a parametric confidence interval for population squared MMD. |
| Fallback geographic split | `assign_split` first reduces longitude modulo 360 and checks the Selk holdout. Outside that holdout, west [180, 360) is training; east [0, 180) is validation at absolute latitude at most 30 degrees and test above 30 degrees. Published evaluations use their bound external manifests. |
| Probe fitting and checkpoint selection | `train_linear_probe` uses caller-supplied class weights when present and otherwise unweighted cross-entropy. `train_finetuned_model` runs the declared epoch loop and restores its validation-selected state; validation does not stop the loop early. |
| Earth image patches | `tile_earth_analog` uses `tile_size` for both square-patch dimensions and strides; 128 pixels is the default. |

## Citations, data terms and availability

The terrain map is the first edition of [Titan Global Map Shapefiles](https://data.mendeley.com/datasets/f6jrtyfp66/1). Source terms and credits appear in [DATA_LICENSES.md](../licenses/DATA_LICENSES.md). GSHHG shoreline documentation is at [the provider's site](https://www.soest.hawaii.edu/pwessel/gshhg/). Bengio and Grandvalet's discussion of cross-validation uncertainty is in [their JMLR article](https://jmlr.org/papers/v5/grandvalet04a.html).

The CROMA and Mars-Bench papers are indexed under DOI [10.52202/075280-0241](https://doi.org/10.52202/075280-0241) and [10.52202/085713-4098](https://doi.org/10.52202/085713-4098), respectively. [CROMA paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/11822e84689e631615199db3b75cd0e4-Paper-Conference.pdf) · [Mars-Bench proceedings entry](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b21b8e8823034df7de4a8cc2e56891af-Abstract-Datasets_and_Benchmarks_Track.html)
