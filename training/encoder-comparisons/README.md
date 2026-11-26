# Encoder training and classifier fitting

This package runs the DINOv2 and ResNet18 initialization comparisons and the 80-fit classifier study on Titan. To check the reported scores from existing predictions, start with the [saved results](../../experiments/encoder-comparisons/README.md). To prepare a new image job, obtain the [processed Titan tiles](../../experiments/encoder-comparisons/DATA_ACCESS.md#processed-titan-images) and the model assets listed in the [execution contracts](CONTRACTS.md#model-assets).

| Task | Code | Inputs and instructions |
| --- | --- | --- |
| Train ResNet18, extract DINOv2 features or fit a logistic classifier to them | [Model runner](scripts/run_model_followup.py) | The [job guide](docs/MODEL_FOLLOWUP.md) describes the specification, initialization, folds and validation rule. |
| Compare input scaling and regularization across 80 logistic fits | [Fitting worker](scripts/run_fitting_sensitivity.py) and [launcher](scripts/launch_fitting_sensitivity.py) | The [fitting study](docs/FITTING_SENSITIVITY.md) identifies the protocol and feature caches each fit needs. |
| Check coverage across the original model grid | [Grid analysis](scripts/analyze_model_followup.py) and [coverage analysis](scripts/analyze_model_followup_coverage.py) | The historical analysis authenticates its original reports, catalog and states. The [training input guide](docs/DATA_ACCESS.md) lists the files needed to repeat inference. |

## Prepare a job

Run from this directory in a separate Python environment. The root CETUS package also provides a module named `cetus_sar`. The training runner checks where its modules came from and binds their file hashes to the job.

```bash
python -B -m scripts.run_model_followup --help
python -B -m scripts.run_model_followup --spec "$JOB_SPEC"
```

Preparation reads and verifies the inputs. Inspect `ready` and `missing_prerequisites` in the report before adding `--execute`. New runs need a clean Git revision, local model assets and a specification for the actual runtime. For the public catalog, set `inputs.binding_contract` to `anonymous-titan-v1`. The runner checks the ordered catalog and fold hashes before it reads the images.

## Verify the source

Run `sha256sum -c SHA256SUMS` here. The [source manifest](SOURCE_EXPORT.json) identifies the training and fitting producers and the origin of each file. The [CPU test instructions](CONTRACTS.md#validation) cover all seven test modules and their fixtures. Three integration cases use locally supplied model assets.

During ResNet18 training, each optimizer step divides the sum of weighted losses by the sum of the label weights in that batch. The logged epoch loss uses the total weight across the epoch. See the [loss-normalization source](../loss-normalization/README.md) for the experiment that changes the training denominator.

Kevin Lee's code is available under the [MIT license](LICENSE). The [CROMA license](licenses/CROMA-LICENSE), [data credits](../../licenses/DATA_LICENSES.md) and [third-party notices](../../licenses/THIRD_PARTY_NOTICES.md) identify the other rights holders and source terms.
