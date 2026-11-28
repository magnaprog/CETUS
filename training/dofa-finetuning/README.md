# DOFA fine-tuning with two band inputs

This study fine-tuned DOFA with numeric band inputs of 13.78 and 3.75. Each of the five Titan regions and five training seeds contributes one fit at each input. The [study protocol](docs/DOFA_PARTIAL_BAND_PLAN.md) specifies how the two runs share initial weights and how validation selects an epoch. The [band review](docs/DOFA_INPUT_REVIEW.md) explains the origin of the two values.

| Task | Start here |
| --- | --- |
| Recompute metrics for the 50 completed fits | [Saved predictions](../../experiments/dofa/partial-training/README.md) |
| Check probabilities and validation histories | [Scores and histories](../../experiments/dofa/probability-scores/README.md) |
| Inspect training and selected-state replay | [Runner](scripts/run_dofa_partial_band.py), [reference training code](reference-training/) and [execution contracts](CONTRACTS.md) |

## Check the source

Run `sha256sum -c SHA256SUMS` in this directory. The [source manifest](SOURCE_EXPORT.json) records each file's origin. The [training contract view](contracts/training-contract-view.json) lists the original input, state and source hashes. Executing that contract requires the full historical checkout, model assets, data and machine bindings. The [inventory](contracts/historical-source-inventory.json) identifies the 12 original Python files supplied here and the hashes of the other 49.

The [reference dataset loader](reference-training/titansar/data/dataset.py#L405-L413) scales the Titan mosaic’s display digital numbers without converting them to calibrated radar backscatter. The [DINOv2 wrapper](reference-training/titansar/models/foundation_models.py#L150-L166) copies the image into three channels, then normalizes each copy with its own ImageNet mean and standard deviation. The [CROMA wrapper](reference-training/titansar/models/foundation_models.py#L364-L372) feeds two identical copies to its VV and VH inputs. The copies supply no independent polarization measurement, and the evaluation does not establish which learned features respond to the shared signal.

Run the two sets of CPU tests from their respective import roots. The [current runner](scripts/run_dofa_partial_band.py) and [original training module](reference-training/scripts/run_finetuning.py) both import modules named `scripts`. The reference tree keeps the original `titansar` module paths because the DOFA runner checks their Git blobs and imported files before training.

```bash
python -B -m pytest tests -q -p no:cacheprovider
(cd reference-training && python -B -m pytest tests -q -p no:cacheprovider)
```

The tests use synthetic inputs. Actual training and checkpoint replay require the [external dependencies](CONTRACTS.md#runtime-and-external-assets).

Kevin Lee's code uses the [MIT license](LICENSE). See the [CROMA license](licenses/CROMA-LICENSE), [data credits](../../licenses/DATA_LICENSES.md) and [third-party notices](../../licenses/THIRD_PARTY_NOTICES.md) for the other source terms.
