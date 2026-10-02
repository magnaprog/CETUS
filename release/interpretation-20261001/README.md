# Descriptive interpretation of the repaired Titan evaluation

This package contains the 1 October 2026 CPU analyses of existing predictions
and metadata. The scientific producer is development commit
`0297c9a987f4e0ea919b3a40de0ae208103719e6`. No new model fitting or inference is
included. Equal-fold scores remain the primary results; pooled scores and local
exposure comparisons are post hoc descriptive analyses.

Use the complete, unchanged sibling packages
[`longitude-review-20260929`](../longitude-review-20260929/README.md),
[`longitude-auxiliary-20260929`](../longitude-auxiliary-20260929/README.md), and
[`longitude-adaptation-controls-20260930`](../longitude-adaptation-controls-20260930/README.md).
The exported analysis pins the catalog, splits, saved predictions, membership,
and supporting evidence by SHA256. Examples additionally pin the public split
protocol. Historical releases retain their own populations and results.

| Location | Contents and units |
|---|---|
| `class-changes/` | Three-panel PDF/PNG, JSON and CSV. Compute each FT seed's class F1, average five seeds within a fold, then subtract the same-fold common LR score. Retain all five contrasts, their equal-fold mean and sample SD. Scores are percent; differences and SD are percentage points. The six-class average reproduces the registered macro contrast. |
| `interpretation/` | `analysis.json`, nine CSVs and four TeX fragments. Precision, recall and F1 are fractions in JSON/CSV and percent in TeX; error-rate differences are fractions in JSON/CSV and percentage points in TeX. Distances are metres. |
| `examples/` | Deterministic `selected.json`, example PDF/PNG and rendering manifest. Tile identifiers, NPY hashes, modal map fractions, SAR-valid fractions, display limits, source product and selection rule are explicit. Raw arrays are not included. |
| `code/` | The three producer scripts, their required helpers, three focused producer test files and two imported fixture modules. A packaging verifier supports reconstruction without a development checkout or Git metadata. |

The nine interpretation CSVs are `m1_scores`, `m1_differences`,
`primary_lr_confusion`, `m2_exposure`, `m2_support`, `m2_errors`, `m3_parts`,
`m3_support`, and `m3_validation`. The TeX fragments are `m1_aggregation`,
`m2_exposure`, `m2_errorcontrasts`, and `m3_support`.

Pooling sums fold confusion counts separately within each FT seed, computes
precision/recall/F1, then averages seed scores. LR counts are pooled once.
This alternative changes CROMA's FT−LR macro F1 from −0.17 to +0.85 percentage
points. DINOv2's pooled difference is +4.55 points and DOFA's is −21.47 points.
These comparisons retain the full fitting recipes and do not isolate a causal
effect of adaptation.

Exposure means a test center is strictly closer than the stored split threshold
of 108639.61030678928 m to any reinstated training center, on the split's sphere
of radius 2574700 m. There are 1,136 exposed test centers among 23,246. Error
tables preserve all four models, five folds, five replacement draws and their
means, and exposed/remaining/full subsets with six class supports. Positive
replacement-minus-baseline error differences mean more errors. Replacing
training membership changes geography and composition together; this is not a
causal leakage estimate.

Map-part counts are distinct modal-Craters test tiles associated with zero-based
parts of the original source MultiPolygon. Associations use the existing
conservative source-raster-pixel intersection evidence, including boundary-only
contact. They are neither area estimates nor counts of independent physical
craters. The 132 test tiles intersect 17 parts; parts 58 and 53 cover 52 and 30
tiles. Single-tile validation leverage holds all other predictions fixed and
does not establish which checkpoint would be selected after a changed label or
prediction. All classification scores measure agreement with the expert map;
independent geological accuracy and geographic independence remain unmeasured.

## Reconstruct without image arrays

Python, NumPy and Matplotlib suffice. The recorded figure runtime is Python
3.12.3, NumPy 2.4.4 and Matplotlib 3.10.9. The wheel lock in
[`code/requirements.txt`](code/requirements.txt) reuses the tested CPU display
runtime. Installing it requires network access or a separately supplied wheel
directory; reconstruction itself is offline. It needs no PyTorch, rasterio,
encoder weights or tile arrays.

Set paths for the local checkout and a new working directory outside all release
packages. Create the environment and caches there:

```sh
PACKAGE=/absolute/path/to/CETUS/release/interpretation-20261001
RELEASE=$(dirname "$PACKAGE")
WORK=/absolute/path/to/external-interpretation-work
mkdir -p "$WORK/tmp" "$WORK/matplotlib-cache"
python3.12 -m venv "$WORK/venv"
. "$WORK/venv/bin/activate"
python -m pip install --only-binary=:all: --require-hashes -r "$PACKAGE/code/requirements.txt"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR="$WORK/tmp" MPLCONFIGDIR="$WORK/matplotlib-cache"
cd "$PACKAGE/code"
python -B verify_reconstruction.py --release-root "$RELEASE" --output "$WORK/numerical"
```

Success requires exit zero and final `status: pass`; the detailed result is
`$WORK/numerical/verification.json`. Output directories must be new and disjoint
from the package and inputs. Failed attempts remain available; use a fresh name
for a retry.

The verifier checks export/input hashes, reconstructs every interpretation
JSON/CSV value from predictions, checks all 90 class/fold contrasts and their
summaries, regenerates the class figure and TeX fragments, and reproduces the
example selection from public metadata. Integer counts, identities, ordering
and exposure flags must match exactly. Floating comparisons use absolute
tolerance `1e-12` in the recorded score units, except nearest-center distances,
which allow `1e-6` m. Runtime fields are recorded rather than required to equal
the producer for numerical comparison. Passing in another runtime demonstrates
that particular reconstruction; it does not promise compatibility with every
future dependency version. Figures are not compared bytewise in this mode.

The packaging verifier calls the exported functions directly. The original
interpretation CLI is also retained unchanged, but its private execution receipt
requires Git metadata. Use the packaging entry point for an unpacked export.
Other historical auditor/exporter entry points retained in helper modules are
outside this reconstruction workflow.

## Rerender the image examples

Selection uses only the public catalog and exact test union: choose the tile
nearest each class's median recorded modal map fraction, with decimal arithmetic
and tile-ID tie breaking. If none of the six selections is below one-half,
add the median-nearest tile from that mixed subset. It does not inspect image
appearance. The metadata selection is fully reproducible without arrays.

To rerender, supply only the selected `TILE_ID.npy` files in an external
directory. Each must match the NPY SHA256 in `examples/selected.json`, shape
128 × 128, float32 dtype, and recorded SAR-valid fraction. No download endpoint
or array archive is supplied here. Do not place arrays inside this package.

```sh
TILES=/absolute/path/to/supplied/selected-arrays
python -B verify_reconstruction.py --release-root "$RELEASE" \
  --tiles-dir "$TILES" --output "$WORK/with-examples"
```

The producer image renderer checks selection and file hashes again, masks
nonfinite/nonpositive DN, and applies independent per-tile 2nd–98th percentile
display stretches. This does not recover calibrated radar backscatter. The
modal map fraction and valid SAR fraction use separate supports. Brightness
cannot be compared quantitatively across these panels, and the selections do
not establish representative geology.

For the optional strict comparison, use Python 3.12.3 and every version in the
wheel lock, then add `--exact-figures` and choose another fresh output directory:

```sh
python -B verify_reconstruction.py --release-root "$RELEASE" \
  --tiles-dir "$TILES" --exact-figures --output "$WORK/exact-figures"
```

This mode checks both PDF/PNG pairs byte-for-byte and refuses other recorded
package versions. Fonts, rendering libraries and platform details can still
affect bytes; a numerical reconstruction is the scientific comparison outside
the verified display runtime. The example figure cannot be rerendered without
the separately supplied matching arrays.

## Source identity and public transformation

[`source_export.json`](source_export.json) records the full producer commit,
each exported source path and original/public SHA256, generated packaging code,
and all scientific payload hashes. Producer code and tests are copied directly
from that Git object. One helper's two machine-specific host denylist entries
are replaced by their generic shared prefix, as in the preceding release;
its numerical routines are unchanged. The interpretation test file is retained
because the producer CLI includes it in source hashing. The additional fixture
modules support the class-change tests. Generated package verification code has
its own identity and is not attributed to the scientific producer commit.

[`transformation_receipt.json`](transformation_receipt.json) records the only
scientific-payload transformation: `/input/path` in
`class-changes/class_changes.json` becomes
`longitude-adaptation-controls-20260930/results/adaptation.json`, relative to the
supplied CETUS release directory. Original and public artifact hashes and a hash
of the replaced path value are retained. Every other field and value is
unchanged. All other scientific payloads preserve producer bytes. Local paths,
SSH details, private execution receipts and raw arrays are excluded. The
outer CETUS release inventories establish the publication's file coverage;
this source manifest is not an independent endorsement of scientific claims.

Optional focused tests require pytest (tested with 9.0.3). Keep its installation
and temporary output external:

```sh
python -m pip install pytest==9.0.3
python -B -m pytest -q -p no:cacheprovider --basetemp "$WORK/tests" \
  tests/test_repaired_interpretation.py tests/test_repaired_class_changes.py \
  tests/test_repaired_examples.py tests/test_interpretation_package.py
```

These tests exercise synthetic arithmetic, coordinates, selection, masking,
rendering and packaging comparisons. They do not train models or establish
independent geological labels. Source license: [`code/LICENSE`](code/LICENSE).
