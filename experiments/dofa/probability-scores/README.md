# DOFA saved probabilities and training histories

This companion contains saved records for fifty DOFA fits:
five geographic folds, five seeds and numeric inputs 13.78 and 3.75. It retains
all six class-probability columns, ordered test IDs/labels, supplied predictions
and fifty epochs of scalar history per fit. The [prediction-only reader](../partial-training/README.md)
uses the same fifty fits.

Use Python 3.11 or later on a POSIX system supporting `O_NOFOLLOW` and hard links.
No third-party package, network, model or GPU is needed. Run from
`experiments/dofa/probability-scores/`.

```sh
replay_dir="$(mktemp -d)"
python3 -B reconstruct.py --output "$replay_dir/dofa-scores.json"
```

Choose a new output path outside this package with an existing parent directory.
Failed attempts retain diagnostic and partial files; use a new name to retry.
For a deliberately marked synthetic fixture package, append `--synthetic`.
Scientific mode rejects marked fixtures. The output identifies synthetic fixtures.

`records.json.gz` contains five ordered truth lists and fifty cells. Each cell
holds predictions, probabilities, five 50-entry histories, selected-epoch and
update-budget metadata, and the reported best validation macro recall.
`expected.json` supplies independent fixed-truth, metric, tie-count and selection
references. `provenance.json` attributes source hashes, sizes and field routes.
The manifest binds eight substantive files; SHA256SUMS binds those and the
manifest. The checksums identify the distributed files.

The reader checks finite six-column probabilities in [0,1], lossless binary32
values and row sums within 1e-6 of one. A prediction must name an exact maximum;
any tied maximum is allowed. Supplied values and predictions are not changed.
History checks require fifty epochs with positive integer optimizer updates,
valid skipped-update budgets and the earliest exact maximum of saved validation
macro recall. Scalar comparisons use absolute tolerance 1e-12 and zero relative
tolerance; epoch ordering and integer identities remain exact. Validation
predictions/confusions are unavailable, so these checks verify policy against
stored scalars, not the computation of validation recall.

Existing confusion counts, class precision/recall/F1, macro F1 and macro recall
are reconstructed from saved predictions. Undefined ratios are zero. Differences
are 3.75 minus 13.78; five seeds are averaged within each region, then five
regions are equally weighted. Sample SD describes those five geographic values.
All scores remain fractions. Repeated seeds do not add geographic tests, and
these exploratory results measure agreement with map labels.

This reader does not regenerate probabilities or softmax, verify calibration,
compute AP, choose thresholds, fit models, replay states or authenticate original
execution. For example, interchanging two nonmaximal probabilities can preserve the
predicted class and row sum. Checking each probability against its producer
requires the original outputs. The [training source](../../../training/dofa-finetuning/README.md)
describes the fits.

Run the synthetic tests with:

```sh
python3 -B -m unittest -v test_reconstruct.py
```

The tests use invented inputs and an independent rational metric oracle.
`DOFA_COMPANION_TEST_ROOT` optionally retains their temporary directories;
leave it unset for normal cleanup.
