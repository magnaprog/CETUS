# Loss-normalization execution source

This code compares two denominators for class-weighted cross-entropy when
fine-tuning ImageNet ResNet18 on Titan SAR. The original loss divides the weighted
batch loss sum by the sum of the batch's label weights. The comparison divides
the same sum by the actual batch size times the mean label weight over the
training population. Both retain the original float32 class weights. The
[study protocol](docs/LOSS_NORMALIZATION_COMPARISON.md) defines rounding, paired
settings and validation-based checkpoint selection.

Run from this package root in a separate environment, so `titansar`, `scripts`
and `tests` resolve here. Use Python 3.12.3 and compatible Torch/torchvision builds;
[requirements.txt](requirements.txt) records the local dependency versions used
for the focused CPU checks. It is not a complete installation lock. Keep the older
CETUS runner off this environment's Python module search path.

Prepare a declared job with:

```sh
python -B -m scripts.run_model_followup --spec "$JOB_SPEC"
```

Preparation checks runtime, inputs and bindings without constructing a model.
Inspect `ready` and `missing_prerequisites`; invalid or unavailable control
artifacts can also raise an error. Adding `--execute` performs the declared job.
[CONTRACTS.md](CONTRACTS.md) describes the specification and execution checks.

The comparison requires the complete historical A control bundle, including its
initial, selected and final model states, original inputs and matching runtime/GPU
bindings. Those states remain server-only under the author's retention decision.
The public numerical companion cannot replace that bundle, so this package does
not provide turnkey public retraining or checkpoint replay.

For the synthetic CPU checks, use a fresh external copy with its own local Git
commit. Hide CUDA, block network access and set TMPDIR to a directory outside
that copy. Run only these two modules:

```sh
python -B -m pytest tests/test_population_loss.py tests/test_loss_comparison.py \
  -q -ra -p no:cacheprovider --basetemp "$TMPDIR/pytest"
```

Keep reports and caches outside the copy. The two modules generate their own
tiny inputs and import helpers from `tests/test_model_followup.py`; do not collect
that helper's tests, some of which need additional fixtures.

[SOURCE_EXPORT.json](SOURCE_EXPORT.json) records source provenance and contents;
[file_index.json](file_index.json) and `SHA256SUMS` provide the complete inventory.
Check `sha256sum -c SHA256SUMS` from this directory. The dated
`release/current-source-20261109/` export remains separate and unchanged.

Author-owned material is by Kevin Lee under [MIT](LICENSE). Retain the full
[CROMA MIT notice](licenses/CROMA-LICENSE), [data credits](DATA_LICENSES.md)
and [third-party notices](THIRD_PARTY_NOTICES.md). Dependencies and upstream weights
retain their own terms. The study measures agreement with a SAR-informed map and
retains its test-reuse and optimization qualifications.
