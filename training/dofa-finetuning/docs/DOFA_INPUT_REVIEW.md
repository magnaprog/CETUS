# DOFA input review

This dated review concerns the earlier frozen-feature comparison. Its reported
scores are not outcomes of the 50-fit partial-training comparison. External
links retain their original targets; this source export did not revisit them.

The [9 November comparison](../CONTRACTS.md#earlier-studies) now tests
13.78 and 3.75 across all five geographic regions with the retained weights.
Mean macro F1 is 35.91% and 38.42%, respectively. The comparison measures
sensitivity to this numeric input; it does not establish a physical Cassini
band mapping or the exact pretraining history. Earlier partial-training
results still use 13.78. The source investigation below records the evidence
available before this comparison and explains the remaining interpretation.

The retained results use the upstream version 1 implementation at
`73ff5c0721da322689ae890bed5b4efd78935f47` and checkpoint SHA256
`4720985e42b918ac0307009eb06121a3435d9bbce6fd95446f84824a538165b1`.
On 7 November 2026, streaming the existing local checkpoint confirmed that
digest and its 447,669,164-byte size. Static inspection found 173 tensor
entries consistent with the earlier loading audit. No inference was run.

The upstream filename alone is insufficient to identify these weights.
Hugging Face history shows a different file under the same name in March
2024, followed by the retained file in August 2024. The model called DOFAv2
has a separate weight file; it is also distinct from revision 2 of the paper.

## What the sources establish

| Evidence | Finding | Limit |
| --- | --- | --- |
| [Pinned Hub loader](https://github.com/zhu-xlab/DOFA/blob/73ff5c0721da322689ae890bed5b4efd78935f47/hubconf.py) | Loads the version 1 encoder with strict loading disabled. | The checkpoint contains no complete training configuration. |
| [Pretraining band file](https://github.com/zhu-xlab/DOFA/blob/73ff5c0721da322689ae890bed5b4efd78935f47/pretraining/datasets/waves.json), [engine](https://github.com/zhu-xlab/DOFA/blob/73ff5c0721da322689ae890bed5b4efd78935f47/pretraining/engine_pretrain.py), [paper Appendix D](https://arxiv.org/html/2403.15356v2) | Use 3.75 to distinguish Sentinel-1 inputs. | This is a modality identifier; it does not establish a Cassini input or prove the exact recipe used for the retained weights. |
| [Pinned notebook](https://github.com/zhu-xlab/DOFA/blob/73ff5c0721da322689ae890bed5b4efd78935f47/demo.ipynb) | Uses the same weight filename with 3.75. | Its saved output does not bind the weight digest or runtime. |
| [README change](https://github.com/zhu-xlab/DOFA/commit/90bcba896d8eb6c77fc4abce3e53bd2113c6487b) | Changes 3.75 to 5.405 and adds the Hub loader in February 2025. | The change leaves the conflicting pretraining and notebook values. |
| [TorchGeo discussion](https://github.com/torchgeo/torchgeo/issues/3931) and [documentation change](https://github.com/torchgeo/torchgeo/pull/3942) | Distinguish the version 1 placeholder 3.75 from version 2's 5.405. | TorchGeo uses converted weights; their exact relation to CETUS's file was not established. |

The source evidence favors 3.75 as the documented version 1 Sentinel-1
training identifier. The contrary README remains relevant. Neither source
establishes the appropriate identifier or image normalization for Cassini.
The dynamic embedding code performs no frequency-to-wavelength conversion.

The existing CETUS path uses one channel, training-derived DN scaling, input
13.78, patch averaging and `fc_norm`. Its scale and bias remain one and zero
because the checkpoint lacks those entries. Separate `norm` weights are
unused in this path. The upstream README lists the same missing and unused
keys. This alone establishes neither a loading defect nor an explanation
for DOFA's lower scores after partial training.

## Before another DOFA experiment

Obtain a checkpoint-specific account of the band identifiers, source image
units, channel order, normalization, dynamic-kernel arrangement and feature
pooling. A useful maintainer question would identify the full digest above
and ask why the README differs from the retained notebook and pretraining
files. No message has been sent.

Then define a Cassini input mapping from that account and the available
display-DN product. The mapping must be justified before inspecting new test
scores. Copying Sentinel-1 channel constants or trying several identifiers
and keeping the best Titan result would not resolve the training history.
Any changed setting needs new features and, for partial-training claims,
new training. Keep the original results with their original settings.

This unresolved DOFA question does not prevent the separately specified
DINOv2 comparison, which has now completed as part of the
[initialization study](../CONTRACTS.md#earlier-studies).
It does prevent presenting a new DOFA setting as a verified
repair. The cited sources describe the evidence available for this dated
review; the underlying investigation records are not included in this export.

The 8 November source review found a full-training weight file in the upstream
release history, but only its metadata were inspected. Establishing an exact
tensor export relationship and the executed band configuration would be needed
before using it to explain the retained encoder. A filename or a generic training
argument list cannot supply that relationship. No replacement DOFA input or
correction run is established by this review.
