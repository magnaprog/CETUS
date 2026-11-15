# Class ranking, regional variation and dataset geography

This companion renders saved ranking results and sampling metadata on CPU. It
includes all six terrain classes, five geographic regions and both initialization
comparisons. The geography displays retain the current Titan test centers, the
original Earth image population and all retained unlabeled Venus centers.

## Reproduce the 40 class and geography assets

Use an existing Python 3.12.3 environment containing NumPy 2.4.4, Matplotlib
3.10.9 and Pillow 10.2.0, as pinned in [requirements.txt](requirements.txt).
Five bundled Nimbus font faces supply 9-point Sans and Roman text. From the
CETUS repository root, run offline:

```sh
set -eu
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
unset PYTHONPATH
CETUS_DISPLAY_PACKAGE="$PWD/release/crater-retrieval-20261114"
CETUS_RANKING_INPUTS="$PWD/release/class-ranking-20261111"
CETUS_DISPLAY_WORK="$(mktemp -d "${TMPDIR:-/tmp}/cetus-displays.XXXXXXXX")"
export MPLCONFIGDIR="$CETUS_DISPLAY_WORK/matplotlib"
cd "$CETUS_DISPLAY_PACKAGE"
sha256sum -c SHA256SUMS

python3.12 -B -m scripts.plot_class_retrieval \
  --report "$CETUS_RANKING_INPUTS/results.json" \
  --report-sha256 f8d798aec2772253071cece9af84acebe7ba517af5a0c320082b1606b723b03d \
  --export data/class-retrieval.json \
  --export-sha256 b03ce1da11d41c6b92b0e75b669f57d7f3db887dff840da6a808ea0d938c9de2 \
  --curves data/class-curves.npz \
  --curves-sha256 f31fab61e09b81555119379d916ddce6c5ec88947c7126b92ea46aa4f0927e1c \
  --output "$CETUS_DISPLAY_WORK/class-retrieval"

for style in cvpr full; do
  python3.12 -B -m scripts.plot_dataset_geography \
    --payload data/dataset-geography.json \
    --payload-sha256 6619ff667273aaa0273bcab1614ef3dff1e6453f1d79309d350da13e5508591d \
    --style "$style" \
    --output "$CETUS_DISPLAY_WORK/geography-$style" \
    --receipt "$CETUS_DISPLAY_WORK/geography-$style-receipt.json"
done
```

These are three plotting calls: one produces all six classes in both font
styles, and one geography call produces each style. Craters is included in the
all-class call. Keep inputs unchanged and use fresh output directories outside
the companion; retain failed attempts separately.

## All-class retrieval

The class command produces `{class}-retrieval-{cvpr,full}.{pdf,png}` for Plains,
Dunes, Hummocky, Labyrinths, Lakes and Craters: twelve PDFs and twelve PNGs.
Rows show Regions 1–5 and columns show DINOv2 with logistic regression and
ResNet18 with full training. All saved states are retained: five random DINOv2
states and one pretrained state per region, plus three scratch and three
ImageNet ResNet18 fits with paired seed IDs.

Each marker represents a complete tied-score group: retrieved count is TP+FP,
and recall is the saved fraction multiplied by 100. All endpoints, including
zero recall, are retained without connectors, interpolation or thinning. The
five common classes use small state-specific markers; Craters also emphasizes
groups that increase true positives. Dense overlapping marks are not all
individually distinguishable. Every subplot uses the same logarithmic count
axis and linear recall axis; titles report positive-label count / test count.

The archive contains 360 curves and 1,800 arrays. The accompanying projection
preserves every accepted metric, curve index and array descriptor while omitting
machine execution metadata and private input paths. It does not recompute AP.
The 60 historical Craters curves and the original numerical companion remain
unchanged. Area in a log-count retrieval display is not AP.

`display.json` records the data-to-artist mapping. `complete.json` is written
last and records hashes for all 24 assets and the display record. Figures are
6.5 × 7.4 inches, with small marks rasterized at 180 dpi. Seeds describe fitting
variation and do not add independent geographic tests. Results measure agreement
with SAR-informed map labels; retrieved tiles may overlap and are not an
independent geological inventory.

## Dataset geography

Each style produces four PDF/PNG pairs:

- `titan-class-locations`: all 23,246 current test centers, separated by the six
  modal map classes, with neutral 72-degree sector boundaries.
- `earth-site-locator`: all 398 original images and the 13-site table, preserving
  352 training and 46 validation assignments.
- `earth-site-footprints`: every retained nominal source-window ring, including
  polygon holes or multiple components; color and line style identify roles.
- `venus-sample-centers`: all 2,945 retained centers, distinguishing 2,526 complete
  and 419 partial images, with the six existing A–F illustration markers.

The Earth population includes the 16 Egyptian images historically named Wadi
Rum. Haughton supplies no images and Earth Hummocky is absent. Site names are
historical intended analogies, not verified terrain throughout each footprint.
Earth uses retained geometric centers and source-window outlines, not displaced
catalog centers or replacement acquisition boxes.

Coordinates are retained without rounding or reprojection. Axis labels can be
rounded for readability. These longitude/latitude diagrams have no physical
scale bar; local Earth extents vary and displayed areas are not comparable.
Titan and Venus points show sampled centers, not surface coverage or independent
geological occurrences. Venus has no terrain labels; A–F do not imply terrain
correspondence. No raw imagery, map sampling, new geometry or model work is
needed by these commands.

The geography renderer checks actual artist coordinates, all IDs/roles/rings,
the site table and example annotations against the payload. Each output
directory ends with `manifest.json`, written after all eight nonempty assets.
The separately located local receipt contains font diagnostics and is not a
public payload. Geography PNGs use 120 dpi; Earth footprint sheets are
6.5 × 8.1 inches.

## Existing AP and regional displays

The original `scripts.plot_class_contrasts`, `scripts.plot_class_ranking`,
`scripts.plot_regional_errors` and `scripts.plot_crater_retrieval` interfaces
remain available. They use the same unchanged inputs and expected figures as
before. The standalone Craters command is an alternative to the corresponding
all-class sheets and is unnecessary for the three-call recipe above.

Regional AP differences and absolute AP answer different questions from curve
shape. The retained [absolute Craters AP figures](figures/craters-ap-cvpr.pdf)
and [caption](figures/craters-ap-caption.tex) remain part of the companion.
The all-tied prevalence reference is not the expected AP of a finite random
ranking.

## Geographic variation and map mixtures

The accompanying `regional-errors.json` is the unchanged analysis of the saved
SGD classifier predictions. Recreate the regional SGD figure with:

```sh
python3.12 -B -m scripts.plot_regional_errors \
  --report regional-errors.json \
  --report-sha256 a1cf71116346d44d7239d3736b6580fb9ea1d9c42a17fc1055e2422a6e22dd07 \
  --output "$CETUS_DISPLAY_WORK/regional-errors"
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

## Integrity and synthetic checks

The package manifest retains every existing `expected_*` section. Its additive
`expected_class_retrieval` section contains 24 asset identities;
`expected_dataset_geography.cvpr` and `.full` each contain eight. Compare PDF/PNG
bytes to those records. Local diagnostic receipts may contain installation
paths and have their own hashes; their bytes are not expected to match a
different installation's receipts.

With pytest 9.0.3, run the six synthetic test files from this directory:

```sh
CETUS_DISPLAY_TESTS="$(mktemp -d "${TMPDIR:-/tmp}/cetus-display-tests.XXXXXXXX")"
TMPDIR="$CETUS_DISPLAY_TESTS" MPLCONFIGDIR="$CETUS_DISPLAY_TESTS/matplotlib" \
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  python3.12 -B -m pytest tests/test_crater_retrieval_plot.py \
    tests/test_class_ranking_plot.py tests/test_class_contrasts_plot.py \
    tests/test_regional_errors_plot.py tests/test_class_retrieval_plot.py \
    tests/test_dataset_geography_plot.py -q -p no:cacheprovider \
    --basetemp="$CETUS_DISPLAY_TESTS/pytest"
```

These checks use synthetic records. They do not reconstruct the original
science or establish actual figure readability. Preserve the [MIT license](LICENSE),
font terms and data attribution in [NOTICE.md](NOTICE.md).
