# Source and execution contracts

## Source layout

The root `scripts/` and `tests/` contain the current runner and tests from
`3a91cc7`. The runner uses `run_dofa_band_sensitivity.py` for file handling,
runtime inspection and supervision; that helper's older pilot CLI is a separate
workflow. `historical-ed2eb3e/` contains the original `ed2eb3e` training modules,
including `scripts.run_finetuning` and `scripts.summarize_review_frozen`.
[SOURCE_EXPORT.json](SOURCE_EXPORT.json) records full revisions, paths and hashes.

The historical modules belong in a separate Python root. They include eight
scientific files and four package initializers; CROMA is included because the
original protocol requires its hash, although these fits do not use that branch.
Historical `scripts/` is a namespace package with no initializer. The empty
initializer in the current `scripts/` is a packaging addition. These are source
trees without an installer. Six historical dependencies differ from the older
`current-source-20261109` export, so that export is not an interchangeable source.

## Execution limits

The original source check requires the exact historical Git commit and all
**61** tracked Python files under `scripts/` and `titansar/`. This package supplies
**12** of them; the [inventory](contracts/historical-source-inventory.json) lists
the other 49 as omitted. A source archive or a new local commit cannot satisfy
that original check.

The CLI exposes `check-plan`, `fit` and `replay`; `_fit-worker` and `_replay-worker`
are supervised internal modes. Even `check-plan` needs the original prerequisites:
a clean declared consumer, the historical checkout, original catalog and splits,
complete feature/analysis directories and manifests, pretrained weights, pinned
upstream code, and interpreter, runtime, device and lock bindings. The supplied
contract view is not executable.

Preflight checks complete file inventories, cache arrays and sidecars, all five
reference heads, normalization and input identities. These references establish
provenance; each current fit starts with a fresh head. The specified 50-fit
design pairs both numeric inputs at each fold and seed, using the same initial
encoder/head, runtime and physical GPU. Each fit has a separate selected-state
replay. The two full-budget pilot fits belong to the 50-fit design; both fits
and their replays need separate acceptance before the remaining 48 fits.
The unchanged [recipe](docs/DOFA_PARTIAL_BAND_PLAN.md) gives training and selection details.

The runner supervises one fit or replay. The three-queue dispatcher, outer
invocation records, storage preflight, collection and independent acceptance
tools are separate and omitted. Images, catalog/cache/reference payloads,
pretrained weights, selected states and current result payloads are also absent.
Replaying original checkpoints requires those states. Saved-prediction
reconstruction is separate from training and checkpoint selection.

## Public inputs

The runner requires the original catalog and protocol identities recorded in
the contract view. The [public processed images](https://github.com/magnaprog/CETUS/releases/tag/titan-processed-images-20261109)
use a different anonymous catalog and matching splits. Their identities and the
older followup runner's `cvpr2027-anonymous-v1` selector are documented in its
[input contracts](https://github.com/magnaprog/CETUS/blob/main/release/current-source-20261109/CONTRACTS.md).
This DOFA runner has no such selector; renaming files or pseudonymizing paths
does not satisfy its original seals.

A recipient run therefore needs a new source/input contract and validated
mappings: ordered IDs or an explicit bijection, labels, image hashes, role
membership and training-derived normalization. Changing IDs or order can alter
normalization sampling and loader order.

## Runtime and external assets

[requirements.txt](requirements.txt) lists the helper's required package versions;
it is not an installation lock or a verified available environment. The consumer
also requires Python 3.12.3 and checks imported files, mapped native libraries,
CUDA/cuDNN, physical devices, deterministic/precision settings and threads.
A different CPU test environment does not establish those runtime bindings.

The contract view identifies the pinned DOFA source files and the
`DOFA_ViT_base_e100.pth` checkpoint; neither source nor weights are included.
The pinned repository's [software license](https://github.com/zhu-xlab/DOFA/blob/73ff5c0721da322689ae890bed5b4efd78935f47/LICENSE)
is MIT, copyright 2024 Zhitong Xiong. The externally hosted checkpoint's terms
and the full dependency closure remain unestablished.

## Metadata and verification

[The contract view](contracts/candidate02.public.json) has a documentation-only
schema and status. Its rules retain scientific policy, ordered cells and fold
counts, device-index pairing, deadlines and source/input/initial-state hashes.
Host paths become logical resources; physical UUIDs and lock locations are
omitted. Historical bindings move to the relative-path inventory, and class
order comes from the historical protocol. The view has its own digest; its
original-contract hash identifies the omitted executable document. It cannot
restore that document's machine bindings, approval or scientific acceptance.

`SOURCE_EXPORT.json` records file origins and the two guides' exact replacements.
`file_index.json` covers all files except itself and SHA256SUMS; SHA256SUMS covers
every other file. Verification is static. The documentation revision checks links,
inventories and changed bytes; source checks are inherited from the unchanged
export. Tests have not been run in this layout, and packaging imported no candidate
module or model. Current tests use mocks and inert subprocesses; helper tests
also need NumPy, and historical tests need NumPy/Torch. Their GPU identifiers are
synthetic fixtures. Future testing needs separate current/historical import roots,
checked module origins and external outputs, with the historical namespace intact.

## Earlier studies

The supplied [historical protocol](historical-ed2eb3e/audit/REPAIRED_FINETUNING_PROTOCOL_2026-09-29.json)
describes 75 fits across three models. The current consumer reuses its reference
checks for the distinct paired 50-fit design. The [9 November DOFA companion](https://github.com/magnaprog/CETUS/tree/main/release/dofa-band-20261109)
covers frozen features and classifier fitting; the [initialization companion](https://github.com/magnaprog/CETUS/tree/main/release/titan-followup-20261109)
and [older source export](https://github.com/magnaprog/CETUS/tree/main/release/current-source-20261109)
also retain separate identities. Scores in the dated input review belong to the
earlier frozen-feature comparison, not the current partial-training study.

The current comparison reuses previously studied test regions. Fold variation
is descriptive, and seed repeats do not add independent regions. Scores measure
agreement with a SAR-informed terrain map rather than independent geological
accuracy. The physical Cassini interpretation of DOFA's numeric inputs remains
unresolved; these comparisons do not establish a causal explanation.

The training guide's original protocol-prose link leads to the exact JSON
protocol above. The input review's two audit links lead to this section.
Underlying investigation records and the earlier pilot/full
frozen-feature guides (`DOFA_BAND_SENSITIVITY.md` and `DOFA_FULL_STUDY.md`) remain
outside this package. Other external links are references, not a fresh access
check; only the pinned software license has the new bounded source verification.
