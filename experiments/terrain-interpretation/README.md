# Interpreting the longitude benchmark

These analyses use saved Titan predictions to examine three questions. How does pooling test tiles across folds affect the apparent benefit of fine-tuning? Where does replacing training tiles bring training and test centers closer together? How many mapped Craters components contribute to the test scores? The primary scores give the five folds equal weight. All comparisons measure agreement with the expert map.

## Reading the results

- **Pooling.** In `class-changes/`, each contrast subtracts logistic regression from fine-tuning within the same fold. Class F1 is computed separately for each of the five fine-tuning seeds before averaging. The CSV reports all five fold contrasts, their mean and sample SD. Scores are percentages and differences are percentage points. In the alternative pooled calculation in `interpretation/`, sum confusion counts across folds within each seed before computing the scores. Pool logistic regression counts once. CROMA's macro F1 difference changes from −0.17 to +0.85 percentage points. With pooling, the differences are +4.55 for DINOv2 and −21.47 for DOFA. These contrasts compare the full fitting recipes.
- **Geographic exposure.** Of 23,246 test centers, 1,136 are closer than 108639.61030678928 m to a reinstated training center, using the split's Titan radius of 2574700 m. The exposure tables give class supports and errors for these centers, the remaining centers and the full test set across four models, five folds and five replacement draws. Positive replacement-minus-baseline error differences mean more errors after replacement. Training geography and composition both change.
- **Map support.** The 132 modal Craters test tiles intersect 17 source polygon parts. Parts 58 and 53 intersect 52 and 30 tiles, respectively. The raster intersection rule includes boundary contact. Part counts describe map support and cannot establish how many distinct physical craters the tiles contain. The validation calculation varies one prediction at a time while holding the others constant. It cannot determine which model a new round of validation would select.

In `interpretation/`, `analysis.json` and nine CSVs give scores and error differences as fractions and distances in metres. The filename prefixes are `m1_` for aggregation, `m2_` for exposure and errors, and `m3_` for Craters support and validation sensitivity. Four TeX tables display scores as percentages and differences in percentage points. `examples/selected.json` identifies the radar tiles used in the example figure.

## Reconstruct the results

Run these commands from the repository root with Python 3.12.3. The verifier reads three sibling packages and checks their input digests. [Longitude benchmark](../longitude-benchmark/README.md) · [Regional predictions](../regional-predictions/README.md) · [Adaptation controls](../adaptation-controls/README.md)

```sh
EXPERIMENTS="$PWD/experiments"
PACKAGE="$EXPERIMENTS/terrain-interpretation"
WORK=$(mktemp -d)
python3.12 -m venv "$WORK/venv"
. "$WORK/venv/bin/activate"
python -m pip install --only-binary=:all: --require-hashes -r "$PACKAGE/code/requirements.txt"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR="$WORK/tmp" MPLCONFIGDIR="$WORK/matplotlib-cache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
python -B "$PACKAGE/code/verify_reconstruction.py" --experiments-root "$EXPERIMENTS" \
  --output "$WORK/numerical"
```

The verifier recomputes the JSON and CSV values, all 90 class-by-fold contrasts, the TeX tables and the example selection. It also redraws the class figure. It requires exact counts, identities and exposure flags. Absolute tolerances are 1e-12 in score units and 1e-6 m for nearest-center distances. On success it writes `verification.json` and exits with status zero. Use a new external directory for each run. Reconstruction needs no network connection after installation.

## Reproduce the example figure

From all test tiles, the selection takes the tile nearest each class's median modal map fraction, breaking decimal-distance ties by tile ID. If all six fractions reach one-half, it also takes the tile nearest the median of the subset below one-half. The selection uses metadata alone.

For the image panels, set `TILES` to the extracted `data/tiles` directory or a directory containing the selected `TILE_ID.npy` files. The renderer verifies the NPY hashes, 128 × 128 float32 arrays and SAR-valid fractions. It masks nonfinite and nonpositive values and stretches each panel independently between its 2nd and 98th percentiles. Displayed brightness cannot be compared across panels. The modal map fraction describes a different support from the SAR-valid fraction. [Processed images and source notice](https://github.com/magnaprog/CETUS/releases/tag/v0.2.0)

```sh
python -B "$PACKAGE/code/verify_reconstruction.py" --experiments-root "$EXPERIMENTS" \
  --tiles-dir "$TILES" --exact-figures --output "$WORK/figures"
```

The exact comparison checks the PDF and PNG bytes for both figures. It requires the pinned environment, including NumPy 2.4.4 and Matplotlib 3.10.9. Fonts and system libraries can affect rendering. Omit `--exact-figures` to check the numerical reconstruction and render the selected examples without requiring byte-identical images.

## Source and tests

The analysis comes from producer commit `0297c9a987f4e0ea919b3a40de0ae208103719e6`. The source manifest records the producer and public hashes. The public analysis reads inputs by their current experiment paths. A helper uses a generic host prefix in its privacy check. The class-change JSON replaces one local input path with a path relative to `experiments/`; the transformation receipt identifies both versions. All other scientific artifacts retain their producer bytes. [Source manifest](source_export.json) · [Transformation receipt](transformation_receipt.json) · [Code license](code/LICENSE)

To run the focused tests with the same environment and external work directory:

```sh
python -m pip install pytest==9.0.3
(cd "$PACKAGE/code" && python -B -m pytest -q -p no:cacheprovider \
  --basetemp "$WORK/tests" tests/test_repaired_interpretation.py \
  tests/test_repaired_class_changes.py tests/test_repaired_examples.py \
  tests/test_interpretation_package.py)
```
