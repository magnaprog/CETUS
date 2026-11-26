# Class rankings, regional errors and sampling locations

These CPU scripts plot the saved terrain rankings for six classes in five Titan
regions. They also show regional classification errors and the locations of
the Titan, Earth and Venus samples. The sampling diagrams use coordinates and
window outlines. For the image galleries and radar backgrounds, see the
[paper reproduction guide](../../docs/reproduce.md).

## Plot the six classes and four sampling maps

From the repository root, use Python 3.12.3 with the packages in
[requirements.txt](requirements.txt). The commands below write 24 ranking and
16 geography files to a temporary directory. Both `sans` (Nimbus Sans) and
`roman` (Nimbus Roman) plot the same observations.

```sh
set -eu
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
unset PYTHONPATH
CETUS_DISPLAY_PACKAGE="$PWD/experiments/terrain-plots"
CETUS_RANKING_INPUTS="$PWD/experiments/class-ranking"
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

for style in sans roman; do
  python3.12 -B -m scripts.plot_dataset_geography \
    --payload data/dataset-geography.json \
    --payload-sha256 6619ff667273aaa0273bcab1614ef3dff1e6453f1d79309d350da13e5508591d \
    --style "$style" \
    --output "$CETUS_DISPLAY_WORK/geography-$style" \
    --receipt "$CETUS_DISPLAY_WORK/geography-$style-receipt.json"
done
```

### Reading the rankings

Each row is a test region. The two columns compare five random DINOv2 states
with one pretrained state using logistic regression, and three ResNet18 models
trained from scratch with three initialized from ImageNet using the same
training seeds. The pretrained DINOv2 state appears once per region.

Each point includes every tile at or above one distinct class-score threshold.
Its horizontal position is the number of selected tiles, including false
positives. Its height is the percentage of all tiles with the target map label
that were selected. The region headings give target-label and total test counts.
The dashed line shows the expected recall under random ordering. For example,
with one positive tile the observed recall can be only 0% or 100%.

All 360 curves retain every score threshold. Larger symbols identify states or
seeds at existing points using a common 36-point spacing rule. Open ResNet18
symbols indicate training from scratch and filled symbols indicate ImageNet
initialization. Exact ties can place symbols on top of one another. The area on
the logarithmic count axis does not measure average precision (AP).
Test tiles can overlap, and all seeds in a region share the same test images.

### Reading the maps

- `titan-class-locations` shows 23,246 test centers by modal map class. Gray
  points show all test centers in each panel. Lines mark the 72-degree sectors.
- `earth-site-locator` shows 398 images at 13 sites, including 352 training and
  46 validation images. The coastline comes from Natural Earth at 1:110m.
- `earth-site-footprints` shows the source-window outlines, including overlaps,
  holes and gaps in radar coverage within a window. Independent longitude and
  latitude scales can distort shapes. Site counts include overlapping images.
- `venus-sample-centers` shows 2,945 unlabeled images. Of these, 2,526 have
  complete radar coverage and 419 have partial coverage. Letters A–F identify
  six examples.

The Earth set includes 16 Egyptian images cataloged as Wadi Rum. Haughton
contributed no images. The Earth site labels describe intended analogies to
Titan terrain. Their geographic extent within each window was not verified.
The coordinates and window geometry are preserved in the payload. The plotted
centers describe sampling density.

## Craters AP and regional errors

The Craters AP figure shows every encoder state or training seed in each
region. The dashed reference gives AP when all scores tie. The precomputed
PDFs share a [caption](figures/craters-ap-caption.tex) and are available in
[Nimbus Sans](figures/craters-ap-sans.pdf) and [Nimbus Roman](figures/craters-ap-roman.pdf).
The [display workflow](../../.github/workflows/terrain-plots.yml) gives the
commands for absolute AP, regional AP contrasts and standalone Craters curves.

To plot the regional SGD results, run this in the shell and package directory
used above.

```sh
python3.12 -B -m scripts.plot_regional_errors \
  --report regional-errors.json \
  --report-sha256 a1cf71116346d44d7239d3736b6580fb9ea1d9c42a17fc1055e2422a6e22dd07 \
  --output "$CETUS_DISPLAY_WORK/regional-errors"
```

The left panel gives macro F1 for five classifier seeds and shows their mean
and range within each region. The constant predictor assigns the most common
training class. The right panel compares error rates for tiles below 75%
modal map fraction and tiles at or above 90%. A positive difference means a
higher error rate in the lower-fraction group.

Each class and region contributes if both bins have at least 20 tiles. Five
regions qualify for each of Plains, Dunes and Hummocky, three for Labyrinths,
two for Lakes and none for Craters. The mean weights the 20 eligible groups
equally after averaging first over seeds per tile, then over tiles within
each bin. These comparisons describe agreement with the map labels.

## Verify the outputs

The [manifest](manifest.json) records the SHA-256 and size of each of the 56
expected PDFs and PNGs. The workflow compares the generated files with these
records. Local diagnostic JSON files also record the paths used for each run.
The commands read the archived scores and metadata and require no model weights.

With pytest 9.0.3, run the six test modules from this directory.

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

See [NOTICE.md](NOTICE.md) for the data and font credits and [LICENSE](LICENSE)
for the plotting code.
