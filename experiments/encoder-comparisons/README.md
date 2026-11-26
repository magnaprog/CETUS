# Titan encoder comparisons

These four archives let you recalculate classifier scores, check map-label and
radar coverage, and compare pretrained and randomly initialized models. They
contain saved predictions and training records for a catalog of 23,380 Titan
tiles. Across the five geographic folds, 23,246 distinct tiles serve as test
images. The [access inventory](DATA_ACCESS.md) lists the inputs for each
calculation. The [notice](NOTICE.md) gives the code and data credits.

The [image access guide](DATA_ACCESS.md#processed-titan-images) gives the
download and verification instructions for all 23,380 processed Titan tiles
and their five geographic splits.

| Archive | Entries | Compressed bytes | Reconstruction scope |
| --- | ---: | ---: | --- |
| [Scientific](scientific-supplement.zip) | 203 | 49,528,151 | 315 saved test-prediction vectors, including 80 fitting cells; sixteen settings, twenty contrasts and eleven table fragments. |
| [Follow-up](followup-supplement.zip) | 84 | 4,320,495 | Coverage for 115 saved vectors; votes from 2,685 sampled map grids and sixty SAR inspection packets. Coverage reads the scientific ZIP in this release. |
| [Initialization](initialization-supplement.zip) | 19 | 35,858,028 | 90 cells: thirty DINOv2 readouts, thirty ResNet18 tuning fits and thirty final fits. Sixty native test records, 6,000 validation-history rows from sixty CNN fits, ten learning-rate choices and 180 coverage confusions. |
| [Reassessment](reassessment-supplement.zip) | 54 | 32,559,371 | Thirty epoch-100 validation/test records, validation-choice checks and thirty-seven SAR/map excerpts. Requires the unchanged initialization ZIP. |

The four ZIPs total 122,266,045 bytes. Their 360 entries include files shared
by more than one archive. [Provenance](provenance.json) lists the hashes of the original archives and the
ZIPs distributed here. The scientific archive uses `scripts/prepare_tables.py` and
`paper/tables/`. The coverage instructions pin that archive. The numerical
inputs, predictions and table contents retain their original bytes.

For source inspection, the [notes on the archived source](../../docs/classifiers.md#reading-preserved-source-descriptions)
explain the input encoding, feature pooling, normalization, scores and helper
behavior where preserved comments or docstrings are misleading.

## Reconstruct saved results

From the CETUS repository root, use Python 3.12.3 and NumPy 2.4.4. The script
verifies the four ZIPs against [provenance.json](provenance.json), extracts each
into its own directory, and runs the six checks on CPU. Installation uses PyPI.
The checks run offline on the supplied data, including the image excerpts.
Choose a new output directory outside the checkout. A failed run leaves its
files there for inspection.

```sh
set -eu
work=$(mktemp -d)
python3.12 -m venv "$work/venv"
"$work/venv/bin/python" -m pip install --only-binary=:all: numpy==2.4.4
"$work/venv/bin/python" -m pip check
"$work/venv/bin/python" -B experiments/encoder-comparisons/reconstruct.py --output "$work/replay"
```

Read the five JSON reports in `$work/replay/reports/`. If you inspect the
extracted scripts separately, use the initialization archive’s `reconstruct.py`.
Its verifier rejects extra files, including caches and reports, and the
historical helper CLIs require additional inputs. The reassessment checker uses
its own `MANIFEST.json` and reads the initialization ZIP directly.

The scientific report should say `pass`; its vector counts sum to 315, and the separate
table command checks eleven fragments. Coverage must equal the packaged result
after removing only `execution` metadata. The vote checker checks all saved
votes, summaries and sixty inspection packets. Initialization reports
`numeric_reconstruction_pass` with 90 cells, sixty native test vectors, 6,000
history rows, ten learning-rate choices and 180 cohort confusions. Reassessment
reports `supplied_numerical_records_reconstructed` and includes the epoch-100
table check. Inspect warnings and retain command output and runtime versions
with the reports. The script rechecks all four ZIP digests before reporting success.

The commands recalculate metrics from saved results. Repeating training or
generating new probabilities requires the corresponding model inputs and states.
The validation-removal check uses the supplied outcomes. Recomputing every
removal score would require the omitted per-epoch predictions.

## Interpretation and execution limits

The 80-fit study retains 23 iteration caps and 54 gradient concerns. The
initialization study compares five random DINOv2 states with one pretrained
state under matched architecture and inputs. ResNet18 uses three seeds per
initialization and separately selected learning rates. Thirty DINO feature
jobs supplied the readout inputs; they are dependencies of the 90-case analysis.
The epoch-100 comparison evaluated saved states without retraining or
reselection. Original validation-selected results remain primary.

Scores measure agreement with a SAR-informed terrain map. Seeds are averaged
within each region before averaging the five equally weighted regions; the
regional SD is descriptive. The studies reuse examined test regions and do
not establish convergence, equal optimization adequacy, a causal pretraining
effect or independent geological accuracy. Rare Craters support, differences
in class/geographic composition and unsupported-class null summaries remain
part of the interpretation. Coverage differences do not isolate the effect
of missing pixels. DOFA's checkpoint/input convention remains unresolved.

The sixty follow-up inspection packets and thirty-seven reassessment excerpts
are not asserted to be disjoint and do not supply the full pixel collection.
Embeddings, model states, full label rasters and a complete current training
environment are absent. The separate [current source package](../../training/encoder-comparisons/README.md)
provides the execution code and documents these prerequisites. For native SGD
and nearest-neighbor classification with encoder weights unchanged, use
[classifier guide](../../docs/classifiers.md) with the images from the separate release and
upstream weights. See [DATA_ACCESS.md](DATA_ACCESS.md) for the distinction between
recorded author-side checks and independent-recipient access.
