# Regional classifier predictions

Use these results to compare classifiers on the same five longitude folds. The sibling benchmark supplies the 23,380-tile catalog, fold assignments and linear and nearest-neighbor reference scores. The hashes in `provenance.json` identify the parent files. [Longitude benchmark](../longitude-benchmark/README.md)

`results/predictions.json.gz` stores test IDs and true classes once per fold, with 20 prediction vectors for logistic regression on encoder features, 20 for logistic regression on classical features, five for the majority-class baseline and 25 draws from the training class frequencies. The two reports in `results/` give the corresponding metrics and fitting diagnostics. Protocols and original estimator hashes are in `protocols/` and `evidence/runs.json`.

## Classifiers and scores

The encoder feature sets have 768 columns. The four classical feature sets use columns `0:13`, `13:45`, `45:1865` and `0:1865` in Python slice notation. The first slice, recorded as `raw_intensity_statistics`, summarizes normalized display intensities. Histogram and texture features also use within-image percentiles. These values do not measure calibrated backscatter. Both classifier families use training-set standardization and logistic regression with balanced class weights, L-BFGS, C = 1, seed 0 and a 2,000-iteration limit. The reports retain warnings and iteration counts.

Metrics are fractions computed against the six map classes. Confusion matrices index true class by row and predicted class by column. Macro scores weight all six classes equally, with undefined precision counted as zero. Kappa uses the confusion-matrix marginals. Average the five training-prior draws within each fold, then take the equal-fold mean and sample SD. The SD describes variation among the five geographic partitions.

## VIMS display samples

The VIMS v3 sidecar gives bilinear samples of a display composite at catalog tile centers. Finite zero and signed values can occur under the source masks. One center lacks a valid sample. The raster hashes, affine transform, coordinate reference system and masks identify the sampling inputs. Availability describes numerical sampling, which alone cannot establish observation coverage or composition. The classifiers use SAR features. The sidecar supplies display context and preserves the catalog's earlier VIMS fields. [Data terms](../../licenses/DATA_LICENSES.md) · [Third-party notices](../../licenses/THIRD_PARTY_NOTICES.md)

## Verify

From the repository root, run the bundled verifier with Python and NumPy. It checks the two packages' input hashes, memberships, predictions and scores. The reports also record the original probability checks, estimator replays and VIMS raster comparison.

```sh
python3 -B experiments/adaptation-controls/code/scripts/export_repaired_auxiliary_results.py \
  --verify-only experiments/regional-predictions \
  --parent-package experiments/longitude-benchmark
```
