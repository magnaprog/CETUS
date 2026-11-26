<a name="titan-data-and-saved-results"></a>

# Inputs for encoder training

## Anonymous image inputs

Extract and verify the [processed Titan images](../../../experiments/encoder-comparisons/DATA_ACCESS.md#processed-titan-images). Keep `data/catalog.json`, `data/splits/` and `data/tiles/` together. For a ResNet18 training job or a DINOv2 extraction, set `inputs.binding_contract` to `anonymous-titan-v1` in the job specification. Supply absolute paths and SHA-256 digests for the catalog and the split for the chosen fold. With no selector, the runner expects the original study catalog and splits.

From the training package directory, run `python -m scripts.run_model_followup --spec JOB.json` and inspect `ready` and `missing_prerequisites`. Preparation checks the tile hashes, ordered labels, split counts and training normalization before model execution. The [model guide](MODEL_FOLLOWUP.md) includes example specifications.

## Weights and runtime

ResNet18 uses torchvision. For an ImageNet start, supply the ImageNet-1K V1 weights. DINOv2 requires the pinned local ViT-B/14 source and either the pretrained weights or the random state for the chosen seed. Its C=1 logistic classifier needs the complete extraction output, including `initial_state.pt`. Each job records its source and runtime. See the [asset hashes](../CONTRACTS.md#model-assets) and [runtime requirements](../CONTRACTS.md#runtime-and-third-party-dependencies).

<a name="additional-numerical-companion"></a>

## Supported reconstruction

The [saved-result commands](../../../experiments/encoder-comparisons/README.md#reconstruct-saved-results) reconstruct metrics from archived predictions on CPU. To repeat inference from images, the original ResNet18 fits require their trained states. The pretrained DINOv2 fits also require their fitted scalers and classifiers. These fitted states are absent from the public archives. The [execution records](../CONTRACTS.md#historical-execution-records) describe the checks made with the retained states.

<a name="primary-source-records"></a>

The experiment guide lists the [archive contents](../../../experiments/encoder-comparisons/DATA_ACCESS.md#original-scientific-archive), [later companions](../../../experiments/encoder-comparisons/DATA_ACCESS.md#later-companions) and [image and map credits](../../../experiments/encoder-comparisons/DATA_ACCESS.md#source-records-and-credits).
