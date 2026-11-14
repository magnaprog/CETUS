# Craters retrieval from saved rankings

Render all 60 saved series from the existing
[class-ranking package](../class-ranking-20261111/README.md), using its data
in place. The display shows recall of Craters map labels against retrieved
test-tile count.

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
Other CLI actions in the copied helper modules are unsupported here and may
require omitted reports.

## Reading the figure

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

## Outputs

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
    -q -p no:cacheprovider --basetemp="$CETUS_RETRIEVAL_TESTS/pytest"
```

Code uses the [MIT license](LICENSE). Preserve the separate font and data
obligations in [NOTICE.md](NOTICE.md).
