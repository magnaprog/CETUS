# CETUS frozen evaluation software

This source checkout runs frozen Titan terrain evaluation with DINOv2, DOFA,
CROMA and the fixed random ViT control. Scientific source files are unchanged
from development commit 0257a36197c51eda0a31768531ea26eef9887a6e. SOFTWARE_EXPORT.json lists their
SHA256 hashes and distinguishes generated packaging and tests.

This export covers feature extraction, five seeded linear probes per fold,
and cosine k-nearest-neighbor classification. Use the five released contiguous
Titan folds. Earth transfer, classical baselines, adaptation, preprocessing,
private acceptance auditors and their CLIs are outside this software release.
Some shared modules retain those older functions to preserve source identity;
only the Titan workflow below is supported here.

## Install and check

After this source bundle is integrated into CETUS:

```sh
git clone https://github.com/magnaprog/CETUS.git
cd CETUS
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[encoders,test]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m pytest -q
python -m scripts.run_probing --help
git status --porcelain
```

Install a PyTorch build appropriate for the machine before editable installation
if its default build is unsuitable. CPU checks can install `.[test]` without
the optional encoder dependencies. Python 3.12 is the exercised CPU environment;
dependency lower bounds are compatibility declarations, not a tested matrix.
The CPU tests create their own small tiles and caches. They test normalization,
cache integrity, random-control identity and Titan-only routing. The synthetic
smoke test trains real linear probes and runs k-NN over a tiny deterministic
encoder. It does not validate pretrained-model inference or paper scores.

The completed repaired frozen runs used the production environment below.
These versions were read from all twenty extraction and twenty analysis
manifests, whose hashes match the accepted frozen audit. Those runs used source
commit `6295af2bc35c8d8a2e47fc982b196809245f4550`; a new public run records its
own source revision. The public bundle's CPU tests used the separate environment
in the last column. These records describe observed environments, not a wheel
lock or a claim that dependency lower bounds reproduce the same numbers.

| Component | Completed production runs | Public bundle CPU tests |
| --- | --- | --- |
| Python | 3.12.3 | 3.12.3 |
| PyTorch | 2.12.1+cu126 | 2.11.0 |
| torchvision | 0.27.1+cu126 | Optional encoder path not exercised |
| timm | 1.0.27 | Architecture construction replaced by a test fixture |
| NumPy | 2.4.4 | 2.4.4 |
| SciPy | 1.17.1 | 1.17.1 |
| scikit-learn | 1.9.0 | 1.8.0 |
| huggingface-hub | 1.19.0 | External asset loading not exercised |
| CUDA runtime | 12.6 | CUDA hidden |
| cuDNN version returned by PyTorch | 91002 | GPU path not exercised |
| Extraction GPU | NVIDIA GeForce RTX 3080 Ti | None |

Production analysis ran on CPU with CUDA hidden. The later supplemental
environment snapshot records einops 0.8.2 and NVIDIA driver 580.173.02; those
two versions were not captured in the individual frozen launch manifests.
The CPU test tools were pytest 9.0.3 and setuptools 81.0.0. Editable installation
was checked in a temporary environment using the existing system packages,
with `--no-deps --no-build-isolation`; fresh resolution of all optional encoder
dependencies was not tested as part of this bundle validation.

Run from the repository root with `python -m scripts.run_probing`. Editable
installation is intentional: the manifest reads this checkout's Git revision
and refuses a dirty worktree. A wheel or source archive without Git metadata
does not provide this execution contract. Commit reviewed source changes before
scientific execution. Keep data, weights, environment logs and results outside
the checkout. Never bypass the clean-source check to label a run reproducible.

## Supply the data and weights

The separate public data package is `release/longitude-review-20260929` in
CETUS. Copy its `catalogs/titan_catalog.json` unchanged to a data directory
outside the checkout. Put the separately supplied arrays in a directory named
`tiles` beside that catalog, as `tiles/TILE_ID.npy`. There is no tile-directory
CLI override. Each tile must match the product, tile ID and NPY file hash in the
catalog's `tile_sha256` field. Verify that inventory before extraction; the
runner binds normalization samples but does not hash every raw tile itself.
There is no tile archive URL or project DOI supplied by this software bundle.

Use `splits/titan_contiguous_fold_0.json` through
`splits/titan_contiguous_fold_4.json` from that same public package. Each split
binds the sanitized catalog bytes. Do not edit catalog paths inside JSON, mix
historical splits with the repaired catalog, or substitute private caches. The
sanitized catalog has a different byte hash from the original scientific input;
new caches and manifests must identify the public input and the public source
commit honestly. Existing sealed records remain unchanged.

Supply trusted upstream weights separately. The wrappers check these hashes:

| Encoder | Required checkpoint SHA256 |
| --- | --- |
| DINOv2 | `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73` |
| DOFA | `4720985e42b918ac0307009eb06121a3435d9bbce6fd95446f84824a538165b1` |
| CROMA | `0238d814b53108f3574bf1ea240e38a0a6edd46173816d9a6962070561893b63` |

DINOv2 uses Torch Hub repository `facebookresearch/dinov2` at
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`. DOFA uses `zhu-xlab/DOFA` at `73ff5c0721da322689ae890bed5b4efd78935f47`. Their checkpoint
names are `dinov2_vitb14_pretrain.pth` and `DOFA_ViT_base_e100.pth` under
`torch.hub.get_dir()/checkpoints`. Offline use requires the matching Torch Hub
source checkouts as well as weights in the cache expected by Torch Hub. The
wrappers can access the network when assets are missing; no fully offline
asset installation command is provided here. CROMA reads
`~/.cache/croma/CROMA_base.pt`. Do not pass untrusted checkpoints to PyTorch.
The random control needs timm, has encoder seed 42, and records the actual
state digest plus timm version. It does not download pretrained weights.
Always use `--require_real_weights`; this still allows the intentional random
control but rejects placeholder substitutes for pretrained encoders.

The released Titan arrays contain HiSAR display DN, declared as `hisar_log_dn`
in the catalog. The dataset fits a range on sampled training pixels, scales
values into [0, 1], and clips values outside that range. A calibration from
these display values to sigma0 or dB is unavailable in this release.

The wrappers use different input and feature operations:

| Encoder | Input after dataset normalization | Tile feature |
| --- | --- | --- |
| DINOv2 | Repeat the channel three times, resize to 224 by 224, then apply channel means [0.485, 0.456, 0.406] and standard deviations [0.229, 0.224, 0.225] | CLS token |
| DOFA | Keep one channel and resize to 224 by 224 | Mean patch features followed by `fc_norm` |
| CROMA | Repeat the channel twice and resize to 120 by 120 | Mean patch features followed by the learned `GAP_FFN_s1` projection |
| Random Init | Keep one channel and resize to 224 by 224 | CLS token |

DINOv2's repeated channels differ after their channel-specific normalization.
CROMA's two channels duplicate the same SAR display values. DOFA applies only resizing after dataset normalization and receives
the recorded Cassini identifier 13.78, extending
the numeric convention in its pinned upstream README. Its checkpoint's
pretraining identifier convention remains unresolved, as described in the
wrapper. The README's Earth example also applies channel standardization;
the applicability of those channel constants to Titan and their use in the
checkpoint's pretraining remain unverified. This export preserves the recorded Titan
input recipe.

The pinned DOFA factory uses mean patch pooling followed by LayerNorm through
`fc_norm`. A separate CPU inspection of the specified checkpoint and factory
verified that `fc_norm.weight` and `fc_norm.bias` are absent from the checkpoint
and retain their constant initial values of one and zero. LayerNorm still
normalizes the pooled features. The checkpoint's `norm` entries do not load
into `fc_norm`, and the factory's classification head is unused by feature
extraction. This inspection covered model construction and checkpoint loading.

Random Init uses one fixed encoder realization with seed 42 across all folds.
The five probe seeds vary the linear heads. Its score therefore describes this
single untrained encoder baseline. A full public rerun and comparison with the
accepted native features and predictions remains pending.

## Run one model and fold

Set the following shell paths to real directories on your machine. MODEL is
one of `dinov2`, `dofa`, `croma`, `random_init`; FOLD is 0 through 4. Repeat this
pair for all twenty model/fold combinations. Each output directory must be new.

```sh
PUBLIC_DATA="$PWD/release/longitude-review-20260929"
TITAN_DATA=/absolute/path/to/supplied/titan
RESULTS=/absolute/path/to/new/results
MODEL=dinov2
FOLD=0
GPU_ID=0
CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -m scripts.run_probing --within_titan_only --require_real_weights \
  --titan_catalog "$TITAN_DATA/titan_catalog.json" \
  --titan_split_manifest "$PUBLIC_DATA/splits/titan_contiguous_fold_${FOLD}.json" \
  --models "$MODEL" --device cuda --batch_size 32 --seeds 0 1 2 3 4 \
  --features_only --output_dir "$RESULTS/features/$MODEL/fold-$FOLD"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.run_probing \
  --within_titan_only --require_real_weights \
  --titan_catalog "$TITAN_DATA/titan_catalog.json" \
  --titan_split_manifest "$PUBLIC_DATA/splits/titan_contiguous_fold_${FOLD}.json" \
  --models "$MODEL" --device cpu --batch_size 32 --seeds 0 1 2 3 4 \
  --cached_features_dir "$RESULTS/features/$MODEL/fold-$FOLD/features" \
  --output_dir "$RESULTS/analysis/$MODEL/fold-$FOLD"
```

These settings match the completed frozen plan. Set `GPU_ID` to the physical
GPU selected for extraction. With that single device in `CUDA_VISIBLE_DEVICES`,
PyTorch exposes it as logical device 0, so `--device cuda` addresses the selected
GPU; do not substitute its physical index as a logical CUDA index. CPU analysis
hides all CUDA devices. The thread counts and cuBLAS workspace setting preserve
the recorded numerical settings; they do not guarantee bitwise agreement across
different hardware, drivers or library versions.

Cache analysis still requires the training tiles to recompute the exact
normalization record. Both modes bind catalog, split, actual benchmark track,
normalization, encoder revision and weights/state identity. The selected
training pixel digest and membership digest have explicit formats in the
normalization record. Held-out pixels do not fit this normalization.

Preserve both run manifests, all cache arrays and sidecars, predictions and
`probing_results.json`, the input inventory, and `python -m pip freeze` output
outside the checkout. Accept a job only when its manifest says `complete`,
every output hash matches, the requested model is present, and the recorded
encoder revision and weights/state digest match the intended encoder. Require
the five distinct probe seeds 0 through 4, the intended fold and exact test IDs.
The legacy permissive mode can emit placeholders or mark partial parallel
results complete. Keep strict weights enabled in both commands and inspect
`weights_source`: pretrained encoders must say `pretrained`, and only the
intentional random control may say `random_init`.
Failed jobs can leave a `started` manifest; retain
that evidence and rerun into a new directory. Probe outputs include tile IDs,
true labels, six class probabilities and confusion counts. k-NN outputs
include IDs, labels and predictions. `macro_accuracy` is mean recall over
classes present in that test fold, not macro F1. The six classes in index order
are plains, dunes, hummocky, labyrinths, lakes, craters.

This narrow bundle does not include the private aggregate acceptance script.
For probe metrics, reconstruct confusion counts for each seed, average the
five head results within each fold, then report the equal-fold mean and sample
standard deviation. For kNN, compute each fold's metric from its single
deterministic prediction vector, then report the same fold summary.
Fold standard deviations are descriptive; overlapping training populations
do not support an independence claim or an inferential confidence interval.
One repaired test fold has only one Crater tile. Neither these folds nor the
software release prove a causal effect of spatial buffering.

## Remaining public protocol work

Classical baseline and adaptation CLIs require new public protocols that bind
the sanitized catalog, split bytes, supplied tile hashes, public source hashes,
normalization, and newly produced reference cache/result/manifest hashes.
Adaptation also needs the same-fold five-seed frozen reference and exact
matched-head seed. Preserve the original private protocol seals and describe
any new run as a public-input rerun. This export does not claim those CLIs are
portable or those experiments have been repeated with the public bytes.

Project code uses the root MIT license. The vendored CROMA implementation has
its upstream MIT notice in `licenses/CROMA-LICENSE`. Upstream source and weights
remain subject to their own terms; this package contains no pretrained weights.
