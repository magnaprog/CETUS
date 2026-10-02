# Accepted adaptation and training-substitution evidence

This package reconstructs saved-prediction metrics, nested summaries, two figures
and a finite manuscript claims index with Python, NumPy and Matplotlib. Native
checkpoint, head and estimator acceptance comes from the bound audits. The code
provided here runs CPU reconstruction of saved predictions.

Obtain the complete three sibling directories from the trusted CETUS revision
identified in [the release instructions](../../README.md):
`longitude-review-20260929`, `longitude-auxiliary-20260929`, and
`longitude-adaptation-controls-20260930`. Their exact inventories are verified.
The outer instructions also supply the independently accepted SHA256 of this
package's `evidence/acceptance.json`. Use that published value in the commands
below. A locally calculated digest checks bytes; release acceptance is external.

Create an external working directory and virtual environment using the exact
Python patch version recorded in `code/SOURCE_EXPORT.json`:

```sh
PACKAGE=/absolute/release/longitude-adaptation-controls-20260930
WORK=/absolute/cetus-reconstruction
mkdir -p "$WORK"
python3 -m venv "$WORK/venv"
. "$WORK/venv/bin/activate"
```

Install the eleven pinned PyPI wheels with their exact SHA256 hashes:

```sh
python -m pip install --only-binary=:all: --require-hashes -r "$PACKAGE/code/requirements.txt"
python -m pip check
```

Install once while packages are accessible; reconstruction can then run offline.
An authenticated wheel directory can be supplied with `--no-index --find-links`.
This lock describes the tested PyPI reconstruction environment. The original
accepted display environment included four Ubuntu package builds. A separate
Linux x86_64, glibc 2.39, Python 3.12.3 PyPI trial reproduced all sixteen display
artifacts and the adaptation checksum sidecar byte for byte. That comparison
covers the recorded runtime and font settings. It does not establish byte
identity on arbitrary platforms or a complete operating-system reconstruction.
The release instructions identify the independently accepted runtime evidence.
These headless figures require no TeX installation.

Keep the environment, caches, logs and outputs outside all three input packages.
Choose writable temporary storage outside them. Then verify saved evidence:

```sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$WORK/matplotlib-cache" TMPDIR="$WORK/tmp"
mkdir -p "$MPLCONFIGDIR" "$TMPDIR"
ACCEPTANCE_SHA=VALUE_FROM_TRUSTED_OUTER_RELEASE_INSTRUCTIONS
cd "$PACKAGE/code"
python -B -m scripts.verify_repaired_adaptation_release \
  --package "$PACKAGE" \
  --frozen-package "$PACKAGE/../longitude-review-20260929" \
  --auxiliary-package "$PACKAGE/../longitude-auxiliary-20260929" \
  --acceptance-sha256 "$ACCEPTANCE_SHA"
```

Verification reconstructs numeric tables in temporary storage. Success requires
process exit zero and `status: pass`. To rebuild the complete sibling package,
repeat that command with `--output "$WORK/rebuilt-package"`. Its parent must
exist; both `rebuilt-package` and `rebuilt-package.receipt.json` must be absent.
The output must be disjoint from inputs and source. Rebuild includes both display
directories and a complete receipt. A retained hidden pending hardlink beside
the receipt records successful publication. Preserve failed evidence and choose
a fresh output name for a retry. PDFs, PNGs, CSVs and TeX fragments are under
`display/adaptation` and `display/substitution`.

`display/claims.json` binds the final manuscript identities, Main's accepted
numeric-checker receipt, generated assets, numeric sources and registered
occurrences. Public verification checks this finite index and its values.
Manuscript coverage comes from Main's independent check of the manuscript bytes.
The index covers registered printed numerals and fixed artifact values. Main
separately checks English-word quantities, inherited values, inequality predicates,
external audit bookkeeping and the independent public frozen replay. The index
makes no claim to those registrations.
The package carries the identities of those manuscripts. The trusted acceptance
digest belongs to the outer release instructions, preserving an acyclic build.

Five adaptation seeds are averaged within each fold. Matching SGD seeds are
paired; common LR has one reference fit per model and fold. Substitution averages
five SGD heads per draw, then five draws; LR fits once per draw. Means weight five
folds equally and SD uses ddof=1. Contrasts describe complete recipes or changes
to training membership. All six map classes, signed differences and warned or
capped fits are retained. Raw scores are fractions; displayed rates are percent
and changes are percentage points. Display sources already contain these units.
Fold spread is descriptive. Labels measure agreement with the expert map.
Independent geological accuracy remains unmeasured. Histories cover all fifty
epochs. Validation vectors come from checkpoint replay.

The preserved native exporter records provenance and requires private accepted
inputs. Optional tests use synthetic fixtures; pytest is separate from the CPU
runtime. Copy code to an external testing directory before installing pytest or
running tests, so test caches preserve the sealed input inventories.
