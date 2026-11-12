# DOFA partial-training numerical companion

This companion contains accepted saved predictions. Its tests use invented data.
This package reconstructs metrics from fifty saved class-prediction
vectors: five geographic folds, five seeds, and numeric inputs 13.78 and 3.75.
It requires no model, GPU, network access or third-party Python package.

Use Python 3.11 or later on a POSIX system supporting `O_NOFOLLOW` and hard links.
From this directory, run:

```sh
python3 -B reconstruct.py --output ../reconstructed.json
```

The output must be a new path outside the package, with an existing parent
directory. The reader checks the inventory, hashes, ordered test IDs/labels,
exact confusion counts and independently supplied metric references. It emits
`dofa-partial-reconstruction-v1`; this is numerical reconstruction, not an
execution-acceptance certificate. Failed attempts retain diagnostic and any
partial bytes. Choose a new output name before retrying.

`records.json.gz` contains five ordered truth lists and fifty prediction lists.
`expected.json` supplies independently exported fixed-test digests, counts and
metrics. `provenance.json` attributes source hashes, sizes and field locations;
selected epochs are attributed metadata only. The other files are this README,
LICENSE, NOTICE.md, reconstruct.py, test_reconstruct.py, manifest.json and
SHA256SUMS. The manifest binds eight substantive files; SHA256SUMS binds those
eight plus the manifest. Checksums establish integrity, not authenticity.

Each cell has six-class precision, recall and F1, macro F1 and macro recall
(`balanced_accuracy`). Undefined ratios are zero. All twenty components remain
fractions. Differences are 3.75 minus 13.78 within each fold/seed pair. Five
seeds are averaged within each fold, then the five geographic means receive
equal weight. `sample_std` describes those five geographic values, including
paired differences. Seed repetitions do not create additional regions. Counts
and identities must match exactly; metric comparisons use absolute tolerance
1e-12 and zero relative tolerance. Percent/percentage-point display conversion
is outside this reader.

The comparison retains the same architecture and original weighted loss. It
was motivated by inspection of earlier test results and reuses those regions;
pre-execution registration does not remove that exploratory scope. Metrics
measure agreement with map labels. Numerical-input sensitivity does not resolve
the physical convention for Cassini or explain historical driver effects.
The three-encoder comparison of seventy-five partial fits and the twenty-matrix,
ten-fit comparison with unchanged encoder weights remain separate studies.

This package does not regenerate probabilities, check argmax/ties, compute AP,
verify epoch selection, fit models, reproduce training or replay checkpoints.
It does not render manuscript tables or figures. Execution-source availability
is separate; older source bundles are not the new study's scientific producer.

Run the tiny synthetic tests with normal temporary-directory cleanup:

```sh
python3 -B -m unittest -v test_reconstruct.py
```

For a deliberately marked fixture package, append `--synthetic` to the
reconstruction command. Scientific mode refuses marked fixtures. Synthetic
mode requires the exact marker in every input container and labels both its
JSON result and console output. `DOFA_COMPANION_TEST_ROOT` optionally retains
test directories for inspection; leave it unset for automatic cleanup. Tests
use an independent rational-arithmetic oracle, not this reader's output as
their expected metrics. No synthetic output should be published as a result.
