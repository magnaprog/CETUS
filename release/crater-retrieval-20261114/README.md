# Class ranking and regional variation

The [class-ranking package](../class-ranking-20261111/README.md) supplies the
values for three complementary displays: regional average-precision differences
for all six terrain classes, absolute Craters average precision, and Craters
recall against retrieved test-tile count. These plotting commands reuse the
saved scores and rankings.

## Reproduce

Use Python 3.12.3, NumPy 2.4.4, Matplotlib 3.10.9 and Pillow 10.2.0; see
[requirements.txt](requirements.txt). Five bundled Nimbus font faces support
regular 9-point Sans (CVPR) and Roman (full-paper) text. From the CETUS
repository root, run offline on CPU:

```sh
set -eu
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset PYTHONPATH
CETUS_RETRIEVAL_PACKAGE="$PWD/release/crater-retrieval-20261114"
CETUS_RANKING_INPUTS="$PWD/release/class-ranking-20261111"
CETUS_RETRIEVAL_WORK="$(mktemp -d "${TMPDIR:-/tmp}/cetus-crater-retrieval.XXXXXXXX")"
export MPLCONFIGDIR="$CETUS_RETRIEVAL_WORK/matplotlib"
cd "$CETUS_RETRIEVAL_PACKAGE"
sha256sum -c SHA256SUMS
python3.12 -B -m scripts.plot_crater_retrieval \
  --report "$CETUS_RANKING_INPUTS/results.json" \
  --report-sha256 f8d798aec2772253071cece9af84acebe7ba517af5a0c320082b1606b723b03d \
  --curves "$CETUS_RANKING_INPUTS/craters-curves.npz" \
  --curves-sha256 74f8fb2159edf44a7d95c7abb7f3aaa582b74fb2aaf1951222a1fa2cd09c63c4 \
  --output "$CETUS_RETRIEVAL_WORK/render"
cat "$CETUS_RETRIEVAL_WORK/render/complete.json"
```

These input hashes identify CETUS commit
`d48bf00775746c499c14d6e7a7c523e32d2d5e43`. Keep the inputs unchanged and use
a fresh output directory outside this companion, retaining failed attempts.
The AP commands below use the same report and environment. The regional display
uses the accompanying saved SGD report.
Other helper-module commands may depend on omitted reports.

## All six terrain classes

From the companion directory and with the variables above still set:

```sh
python3.12 -B -m scripts.plot_class_contrasts \
  --report "$CETUS_RANKING_INPUTS/results.json" \
  --report-sha256 f8d798aec2772253071cece9af84acebe7ba517af5a0c320082b1606b723b03d \
  --output "$CETUS_RETRIEVAL_WORK/class-contrasts"
```

Each point is a regional difference in mean average precision, in percentage
points. Positive values favor pretraining. DINOv2 compares AP from one pretrained
state with mean AP from five random states. ResNet18 averages three differences
between ImageNet and scratch fits with paired seed IDs and separate validation
choices. Both plots use the same linear scale; they compare different fitting
methods. The five region marks show geographic variation, not uncertainty
intervals. Absolute AP and class counts remain relevant to interpreting a gain.

## Absolute Craters average precision

The earlier [CVPR plot](figures/craters-ap-cvpr.pdf),
[full-paper plot](figures/craters-ap-full.pdf) and
[caption](figures/craters-ap-caption.tex) are retained. Recreate them with:

```sh
python3.12 -B -m scripts.plot_class_ranking \
  --report "$CETUS_RANKING_INPUTS/results.json" \
  --report-sha256 f8d798aec2772253071cece9af84acebe7ba517af5a0c320082b1606b723b03d \
  --output "$CETUS_RETRIEVAL_WORK/absolute-ap"
```

This display retains every encoder state or training seed in each region. The
logarithmic axis shows AP in percent. Horizontal marks give arithmetic means;
dashed lines give class prevalence, the AP obtained when all scores tie. That
line is not the expected AP of a finite random ranking.

## Geographic variation and map mixtures

The accompanying `regional-errors.json` is the unchanged analysis of the saved
SGD classifier predictions. Recreate Figure S7 with:

```sh
python3.12 -B -m scripts.plot_regional_errors \
  --report regional-errors.json \
  --report-sha256 a1cf71116346d44d7239d3736b6580fb9ea1d9c42a17fc1055e2422a6e22dd07 \
  --output "$CETUS_RETRIEVAL_WORK/regional-errors"
```

The left subplot shows five classifier seeds, their mean and range for each
model and test region. Gray marks give macro F1 for the predictor that assigns
the most frequent training class. Seeds describe variation in classifier fitting;
they do not add geographic tests.

The right subplot shows each eligible class-region group's error difference
between modal map fractions below 0.75 and at least 0.90, in percentage points.
Positive values mean more errors in mixed tiles. Region numbers identify the
groups; class labels give their eligible counts. A separate row shows the saved
equal-group mean. The analysis first averages seeds per tile, then tiles within
each bin and class-region group, and finally the 20 eligible groups equally.
Each bin has at least 20 tiles in a group. The groups comprise five each for
Plains, Dunes and Hummocky, three for Labyrinths, two for Lakes and none for
Craters. This association describes the available groups; it does not isolate
a causal effect of map mixtures or estimate an average across all six classes.
The absolute endpoint means remain available in the unchanged report.

Panel A's fixed vertical axis spans 0–50% macro F1; inputs outside that range are
rejected before rendering.

The outputs are `full-regional-errors.pdf`, `cvpr-regional-errors.pdf`, their PNG
copies and rendering records. This command displays the stored seed scores,
within-group error differences and saved equal-group means. It does not rerun
the prediction analysis.
The report retains the source and figure identities of its original analysis;
`manifest.json` identifies the current plotting source and expected displays.

## Reading the retrieval figure

Rows show Regions 1–5; columns show DINOv2 with logistic regression and ResNet18
with full training. Each region retains five random DINOv2 states (42–46) and
one pretrained state (saved seed 42), plus three scratch and three ImageNet
CNN fits (paired seed IDs 0–2). Seeds do not add independent geographic tests.

Small points show every saved whole-score-group endpoint, including zero
recall: `x = TP + FP`, `y = 100 * recall`. Tied scores enter together, without
connectors or interpolation. Larger seed-specific symbols mark only groups
that increase TP; color identifies initialization. Every subplot shares a
logarithmic count axis (1–6,500) and linear recall. Titles give Craters-label
count / test-tile count. Region 2 has one positive, so recall is 0 or 100%.

The display describes agreement with map labels on examined test regions.
Retrieved tiles can overlap and do not constitute an independent crater
inventory. The existing AP results remain separate; area in this log-count
display is not AP.

## Retrieval outputs

Both variants are 6.5 × 7.4 inches:

- `crater-retrieval-cvpr.pdf` and `.png` use Nimbus Sans.
- `crater-retrieval-full.pdf` and `.png` use Nimbus Roman.
- `display.json` maps the saved groups to actual artists and records hashes.
- `complete.json` is written last with output hashes and sizes.

Small marks are rasterized at 180 dpi; larger symbols and PDF text remain
vector. Expected figure hashes and source/input identities are in
[manifest.json](manifest.json). Receipts include runtime versions and local
font locations, so their bytes depend on the installation path.

## Synthetic checks

With pytest 9.0.3 available, run from this companion directory:

```sh
CETUS_RETRIEVAL_TESTS="$(mktemp -d "${TMPDIR:-/tmp}/cetus-retrieval-tests.XXXXXXXX")"
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR="$CETUS_RETRIEVAL_TESTS/matplotlib" \
  python3.12 -B -m pytest tests/test_crater_retrieval_plot.py \
    tests/test_class_ranking_plot.py tests/test_class_contrasts_plot.py \
    tests/test_regional_errors_plot.py -q -p no:cacheprovider --basetemp="$CETUS_RETRIEVAL_TESTS/pytest"
```

Code uses the [MIT license](LICENSE). Preserve the separate font and data
obligations in [NOTICE.md](NOTICE.md).
