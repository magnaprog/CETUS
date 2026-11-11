# Class ranking from saved Titan predictions

This companion reproduces within-region, one-vs-rest average precision (AP)
from the initialization study's saved class probabilities. It contains **360
class AP records and all 60 Craters precision–recall curves**, plus the regional
and geographic summaries. Model weights, images and fitting libraries are not
needed. The existing [initialization archive](../titan-followup-20261109/README.md)
supplies the predictions, catalog, five divisions and original selection records.

## Reconstruct

Use an existing **Python 3.12.3 with NumPy 2.4.4** environment. The command runs
offline on CPU. From the CETUS repository root, in one shell:

```sh
set -eu
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset PYTHONPATH
CETUS_AP_PACKAGE="$PWD/release/class-ranking-20261111"
CETUS_AP_INPUT="$PWD/release/titan-followup-20261109/initialization-supplement.zip"
CETUS_AP_WORK="$(mktemp -d "${TMPDIR:-/tmp}/cetus-class-ranking.XXXXXXXX")"
(
  cd "$CETUS_AP_PACKAGE"
  sha256sum -c SHA256SUMS
)
python3.12 -B "$CETUS_AP_PACKAGE/reconstruct.py" \
  --initialization-zip "$CETUS_AP_INPUT" \
  --output "$CETUS_AP_WORK/output"
cat "$CETUS_AP_WORK/output/reconstruction-check.json"
```

The input ZIP must have SHA256
`ada744e9278f4fbac5f5f953b5688132ad0b080b35cff32ed23b7f02c3192486`.
It is read intact; no archived Python code is executed and no files are extracted.
The output directory must be new, with an existing parent outside this package.
Keep the package unchanged, retain failed outputs, and use a fresh directory for
a retry. Use `reconstruct.py`; the copied historical helpers' separate CLIs
require inputs outside this companion.

Success requires `reconstruction-check.json` with status
`reconstruction_verified`: exact equality of all scientific result fields,
360 class rows, 140 regional rows, 70 contrasts and 42 geographic rows, plus
exact keys, dtypes, shapes and values for all 300 arrays in the 60 curves.
Only `execution` metadata is excluded from the result comparison. The analyzer's
`complete.json` records completion of arithmetic; the separate reconstruction
check establishes agreement with the packaged answers. Failed comparisons exit
nonzero and retain a failure record when writable.

## Calculation and interpretation

The sixty records comprise thirty matched DINOv2 readouts and thirty original
ResNet18 final fits. DINOv2 has five random encoder states and one pretrained
state per region; ResNet18 has scratch and pretrained initializations with
paired seeds 0–2. The original CNN loss divides by the batch sum of class weights.
The reader checks the original learning-rate and epoch selections against saved
validation metadata. It does not refit, reselect, calibrate or choose a threshold.

For each class, the saved probability column ranks all original test images
within that region. Descending exact-score ties enter as whole groups. AP sums
each group's recall increase times its cumulative precision, without interpolation.
Scores are not rounded or pooled across regions. `K/N` is the all-tied prevalence
reference; it is not the exact finite-sample expectation of random-ranking AP.
No-positive AP is undefined and remains null; all six classes have positive
test support in these five folds.

Macro AP averages all six class APs within a cell. Seeds are averaged within
each region before equal weighting of the five regions; geographic SD uses
`ddof=1`. CNN contrasts average paired seed differences. DINOv2 compares one
pretrained state with the mean of five random states. Undefined entries propagate
instead of being silently omitted. Saved fold indices 0–4 are Regions 1–5.

The analysis is exploratory and reuses examined test regions. AP measures
ranking agreement with SAR-informed map labels, without establishing independent
geological detection, probability calibration or a deployable decision rule.
Craters test supports are 8, 1, 14, 61 and 48; the singleton region is especially
sensitive to one positive's rank and ties. Seeds do not add independent geographic
tests. The results do not establish convergence or a causal effect of pretraining.

## Files and provenance

- `results.json` and `craters-curves.npz` preserve the accepted outputs exactly.
  The curve index maps each `cell_XX` prefix to its model, fold, initialization
  and seed. Each prefix has `thresholds`, `tp`, `fp`, `precision` and `recall`
  arrays at all observed score groups, without extra display endpoints.
- `scripts/analyze_class_ranking.py` and its two helpers are unchanged from
  analysis revision `0e4def2cb41139f6eeea2de31dd724f6694f0dbc`.
  `reconstruct.py` adds package verification and comparison with saved answers.
- `contract.json` records the portable metric, population and aggregation
  contract. `provenance.json` distinguishes the original model producer,
  accepted analysis, prior independent arithmetic check and this packaging.
  Original execution/status fields in the saved result are historical.
- `manifest.json` and `SHA256SUMS` bind this directory. They are separate from
  the input ZIP's unchanged manifest and checksums.

The small synthetic checks exercise package integrity and rejection of changed
metrics, summaries and curve arrays, without reading the initialization archive:

```sh
cd release/class-ranking-20261111
PYTHONDONTWRITEBYTECODE=1 python3.12 -B -m unittest -v test_reconstruct
```

Code is covered by [LICENSE](LICENSE). Preserve the source-data terms and
attribution in [NOTICE.md](NOTICE.md) and in the initialization archive.
