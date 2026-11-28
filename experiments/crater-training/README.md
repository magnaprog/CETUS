# Craters in training batches and test predictions

How often can a training batch contain a Craters tile? When a classifier misses a test tile labeled Craters, does its predicted class occur elsewhere in the same map window? The training counts and test predictions address these questions for the five longitude sectors. Of the 23,246 distinct test tiles, 132 have Craters as their most frequent map label.

## Compare logistic classifiers on Craters

From the repository root, run this command with Python’s standard library. It reads the [saved map counts](results.json) and [regional predictions](../regional-predictions/README.md) for eight logistic classifiers.

```sh
cetus_check=$(mktemp -d)
python3 -I -B scripts/summarize_crater_map_comparison.py \
  --crater-results experiments/crater-training/results.json \
  --predictions experiments/regional-predictions/results/predictions.json.gz \
  --output-dir "$cetus_check/craters"
cat "$cetus_check/craters/crater-map-primary-lr.tex"
```

The first three numeric columns count correct Craters predictions on tiles where every known map cell carries Craters. There are 23 such tiles in Region 4 and 16 in Region 5. The same 39 qualify when each cell must also have four valid radar samples. The last two columns count true and false positives across all five test regions. For example, CROMA correctly predicts Craters for 7 of 132 tiles labeled Craters and predicts Craters for 477 of the 23,114 tiles with other labels. The command also writes counts by region and feature set to `results.json` in the output directory.

## Reproduce the results

From the repository root, run the following with Python 3.12.3 and NumPy 2.4.4.

```sh
crater_work=$(mktemp -d)
python3 -B experiments/crater-training/analyze_crater_training.py \
  --initialization-zip experiments/encoder-comparisons/initialization-supplement.zip \
  --map-census experiments/map-census/map_cells.npz \
  --census-metadata experiments/map-census/counts.json \
  --output "$crater_work/results"
cmp experiments/crater-training/results.json "$crater_work/results/results.json"
```

The script verifies the three input hashes before reading the [initialization archive](../encoder-comparisons/README.md) and [map census](../map-census/README.md). It matches test IDs and labels to the saved predictions from 30 ResNet18 fits and five logistic classifiers on pretrained DINOv2 features. Use a fresh output directory for each run. In the recorded environment, `cmp` checks every byte of the regenerated result, including the recorded software versions.

The output `results.json` contains three lists:

| Field | Records |
| --- | --- |
| `training_batch_availability` | 30, one per class and training fold |
| `map_prediction_cohorts` | 70, one per fit and map-counting rule |
| `image_rows` | 1,848, one per test tile, prediction and counting rule |

A training fold draws from outside its held-out test region. The counts for each fold refer to that training set.

## Read the results

For a training set of N images with K members of the target class, a batch of size b contains at least one member with probability `1 − C(N−K,b)/C(N,b)` under uniform shuffling. Sum this probability over the actual batch sizes, including a shorter final batch, to get the expected number of batches containing the class. With M batches, at most `min(K,M)/M` of them can contain it. This bounds the average normalized coefficient of that class in the loss when each step has equal weight. Measuring the actual contribution to the loss or gradient would also require the sampled batches and the losses or gradients of their individual images.

For test images, `baseline` counts known map cells before masking. Under `all_four`, a cell must also have valid radar at all four corresponding positions. Fractions use the total known cells eligible under that rule. An image with none has an undefined fraction. The count `all_known_cells_craters` includes images for which all eligible known cells are Craters. The number of errors among them is this count minus `correct_all_known_cells_craters`.

`errors_with_predicted_class_present` counts errors for which the predicted class appears on the map. To convert it to a rate, divide by the number of errors with at least one eligible known cell. The field `undefined_map_fraction` covers correct and incorrect predictions together. Each seed uses the same test images in its region, so count each tile once per rule when summarizing map composition.

To run the synthetic checks, use `python3 -O -B -m unittest -v test_crater_training` from this directory. The [saved results](results.json) include hashes of the inputs and the analysis script from revision `a8abe8414a3a87ad4677f31d82db50996dd27955`.

Code is covered by the repository [MIT license](../../LICENSE). Preserve the
[data licenses](../../licenses/DATA_LICENSES.md),
[third-party credits](../../licenses/THIRD_PARTY_NOTICES.md),
[initialization notice](../encoder-comparisons/NOTICE.md) and
[map-census attribution](../map-census/NOTICE.md) when using these results.
