# Loss normalization in ResNet18 training

The two training procedures sum the same class-weighted cross-entropy for each batch of Titan images. One divides by the sum of the batch's label weights. The other divides by the actual batch size times the mean training-set weight. Both retain the original float32 class weights. The [study protocol](docs/LOSS_NORMALIZATION_COMPARISON.md) gives the rounding rules, region and seed matching, and validation criterion.

The [saved results](../../experiments/loss-normalization/README.md) let you check all fifteen comparisons on CPU. To repeat the training, the runner also needs the corresponding control fit's initial, selected and final model states. Those states are held separately. The [execution contracts](CONTRACTS.md) describe the required source, data and GPU bindings.

## Prepare a job

Run the [model runner](scripts/run_model_followup.py) from this directory in an environment separate from the root CETUS library. A new job verifies its `cetus_sar` modules against the twelve source hashes in its specification. It also checks the original control against the eleven source paths and hashes recorded when that control ran.

```bash
python -B -m scripts.run_model_followup --help
python -B -m scripts.run_model_followup --spec "$JOB_SPEC"
```

Preparation reports `ready` and `missing_prerequisites`. Invalid control records can raise an error. Add `--execute` to run the declared job after its prerequisites pass.

## Check the implementation

The [loss comparison](scripts/loss_comparison.py) verifies the control fit and its original eleven source hashes. The [source manifest](SOURCE_EXPORT.json) identifies those producers and the twelve runtime files for the new jobs. From this directory, run `sha256sum -c SHA256SUMS` to check the package.

The CPU tests create their own images and states. Use a disposable, clean Git copy of the package, and set `TMPDIR` to a fresh absolute directory outside it. The [training test workflow](../../.github/workflows/training.yml) sets up this environment and blocks network access during these tests.

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -B -m pytest tests -q -p no:cacheprovider --basetemp "$TMPDIR/pytest"
```

The [requirements](requirements.txt) record versions observed in the CPU checks. Kevin Lee's code uses the [MIT license](LICENSE). See the [CROMA license](licenses/CROMA-LICENSE), [data credits](../../licenses/DATA_LICENSES.md) and [third-party notices](../../licenses/THIRD_PARTY_NOTICES.md) for the other source terms.
