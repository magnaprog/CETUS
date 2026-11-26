# Titan longitude benchmark

The catalog contains 23,380 radar tiles. The five longitude folds define which tiles each classifier could use for training, validation and testing. Use the assignment files in `splits/` with `splits/protocol.json`; the catalog alone does not specify a fold. The processed radar arrays and their source terms are distributed separately. [Data guide](../../data/README.md) · [Source licenses](../../licenses/DATA_LICENSES.md) · [Third-party notices](../../licenses/THIRD_PARTY_NOTICES.md)

## Scores and predictions

`results/frozen_results.json` records four feature sets, their normalization, class supports, 100 linear classifier fits and 20 nearest-neighbor evaluations. In `results/predictions.json.gz`, each feature set and fold has five linear classifier predictions and one 20-neighbor prediction per test tile. Both files use the class order in `catalogs/titan_catalog.json`.

Confusion matrices have true map classes on rows and predicted classes on columns. Precision divides true positives by predicted support, recall divides them by true support, and F1 divides twice the true positives by the sum of both supports. Undefined ratios count as zero. Macro scores average the six classes. For the linear classifiers, average the five heads within each fold first. Report the mean and sample SD of the five equally weighted folds. Kappa uses the confusion-matrix marginals. Scores are fractions. Fold SD describes geographic variation across folds with overlapping training sets.

## Map labels and geographic support

Numeric tile ID suffixes index the candidate-grid class counts in `evidence/label_verification.json`. The modal label is the lowest class index with the largest count. Its fraction is rounded to four decimal places. Counts use valid map pixels across the tile footprint, including pixels outside SAR coverage. Lakes includes filled and empty basins in the reference map.

`evidence/crater_support.json` connects tiles to parts of the source polygons. A part can intersect multiple tiles, and fold 1 has just one Craters test tile. Scores measure agreement with the map. `evidence/catalog_comparison.json` documents changes in membership and labels from the earlier spatial-block benchmark. The population and geographic partitions both changed, so score differences cannot isolate either effect. `provenance.json` records source hashes before export removed local paths and superseded inline split assignments.

## Verify

From the repository root, run the bundled verifier with Python and NumPy. It checks the file inventory, fold membership and label counts, then recomputes all 120 confusion matrices and their metrics from saved predictions. Probability checks from the original evaluation are recorded in the results.

```sh
python3 -B experiments/adaptation-controls/code/scripts/export_repaired_release.py \
  --verify-only --output-dir experiments/longitude-benchmark
```
