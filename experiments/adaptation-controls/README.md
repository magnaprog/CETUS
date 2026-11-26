# Adaptation and training-substitution controls

Recompute the published scores from saved predictions to compare partial fine-tuning with the linear classifier baselines, or to measure how predictions change when training tiles are replaced. Both analyses use the same five geographic test folds. The original audits record the checks on trained models and estimators. This CPU replay starts with their predictions.

## Reconstruct scores and figures

From the repository root, set up Python 3.12.3 and the pinned NumPy and Matplotlib environment. Keep the work directory outside the checkout. All three sibling packages must be present. [Longitude benchmark](../longitude-benchmark/README.md) · [Regional predictions](../regional-predictions/README.md)

```sh
PACKAGE="$PWD/experiments/adaptation-controls"
WORK=$(mktemp -d)
python3.12 -m venv "$WORK/venv"
. "$WORK/venv/bin/activate"
python -m pip install --only-binary=:all: --require-hashes -r "$PACKAGE/code/requirements.txt"
python -m pip check
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR="$WORK/tmp" MPLCONFIGDIR="$WORK/matplotlib-cache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
```

Set `ACCEPTANCE_SHA` to the SHA-256 published in the classifier guide. The verifier checks that value before it reads the saved results. [Acceptance digest](../../docs/classifiers.md#reconstruct-adaptation-and-training-substitution-results)

```sh
cd "$PACKAGE/code"
python -B -m scripts.verify_repaired_adaptation_release \
  --package "$PACKAGE" \
  --frozen-package "$PACKAGE/../longitude-benchmark" \
  --auxiliary-package "$PACKAGE/../regional-predictions" \
  --acceptance-sha256 "$ACCEPTANCE_SHA" \
  --output "$WORK/rebuilt-package"
```

A successful run exits with status zero and prints a JSON record with `"status": "pass"`. The output contains regenerated tables, figures and an inventory. The receipt is beside it. Omit `--output` to verify the inputs and reconstruct the numerical tables in temporary storage. Choose a fresh output path for each rebuild. Reconstruction runs offline after installation. To install offline, supply the eleven matching wheels using `--no-index --find-links`. The PyPI environment on Linux x86_64 with glibc 2.39 reproduced all sixteen display artifacts and the checksum sidecar from the original environment, which used four Ubuntu package builds. Figure bytes can depend on the runtime and fonts.

## Interpret the comparisons

For adaptation, average five fine-tuning seeds within each fold. Compare corresponding SGD seeds or use the single logistic regression fit for each feature set and fold. For training substitution, average five SGD heads per draw, then five replacement draws. Logistic regression uses one fit per draw. Weight all five folds equally and use their sample SD to describe geographic variation. The comparisons measure differences between whole training recipes or training memberships. All scores refer to agreement with the six expert map classes.

Scores in the result files are fractions. The figures report percentages and differences in percentage points. The diagnostics retain signed differences, fitting warnings and iteration caps. Training histories cover all 50 epochs. The validation vectors come from replaying the selected model weights.

`display/claims.json` links numerical statements to the measurements used in the paper. The verifier checks 540 entries, including their units, calculations and locations in the original registration. Signed adaptation differences in macro F1 also appear in the figure data. The index excludes numbers written as words, inequality statements and external audit records.

The native exporters in `code/scripts/` preserve the original production procedure and its inputs. The public reconstruction command above uses the packaged results. To run the synthetic tests, use an external copy of `code/` and install pytest there.
