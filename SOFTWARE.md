# Running the Titan classifiers

This software calculates DINOv2, DOFA, CROMA and Random Init image features,
then fits classifiers with the encoder weights unchanged. Scientific code
comes from development revision `0257a36197c51eda0a31768531ea26eef9887a6e`.
SOFTWARE_EXPORT.json gives the source hashes and identifies the packaging and
tests. [PAPER_REPRODUCTION.md](PAPER_REPRODUCTION.md) maps paper results to
their data packages and verification records.

The manifest's `files` entries bind the current checkout. Its builder hash
and development commit identify the original export. The `documentation_updates`
records identify changes to the guide, citation and author metadata, including their
previous and current hashes. Scientific source hashes remain unchanged. The
original generated guide and manifest remain in CETUS commit
`360eaea532ead477f46906d7ebe0cb5282e8b082`.

The runner calculates image features, trains five linear classifiers per fold
with different initialization seeds, and classifies by cosine nearest neighbors.
Use the five contiguous Titan folds supplied here. Earth transfer, classical
classifiers, encoder training, additional preprocessing and private verification
commands require the development repository. Some imported files retain functions
for those experiments so their original bytes remain reproducible. The commands
below exercise the Titan workflow.

## Install and check

From a source checkout:

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
The CPU tests create small image arrays and caches. They check normalization,
cache integrity, the random encoder's weights and routing to the Titan workflow.
A small synthetic example trains real linear classifiers and runs nearest
neighbors with a simple encoder. Checking pretrained inference or the paper's
scores requires the separately specified model inputs and result files.

The expanded Titan experiments used the production environment below. These
versions appear in all twenty feature-calculation and twenty analysis manifests,
whose hashes agree with the experiment checks. The runs used source revision
`6295af2bc35c8d8a2e47fc982b196809245f4550`; a new run must identify its own source.
The last column gives the separate CPU test environment. These are the versions
actually used. Reproducing numbers under other allowed dependency versions
requires another comparison.

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
The random baseline needs timm and uses encoder seed 42. Each run includes the
actual weight digest and timm version. It generates its weights locally.
Always use `--require_real_weights`: the deliberate random baseline is allowed,
while placeholder substitutes for pretrained encoders are rejected.

The Titan arrays contain HiSAR display DN, identified by `hisar_log_dn` in the
catalog. Training pixels determine the brightness range used to scale values
into [0, 1], with clipping outside that range. Converting these display values
to sigma0 or dB requires radiometric information unavailable in the package.

The wrappers use different input and feature operations:

| Encoder | Input after dataset normalization | Tile feature |
| --- | --- | --- |
| DINOv2 | Repeat the channel three times, resize to 224 by 224, then apply channel means [0.485, 0.456, 0.406] and standard deviations [0.229, 0.224, 0.225] | CLS token |
| DOFA | Keep one channel and resize to 224 by 224 | Mean patch features followed by `fc_norm` |
| CROMA | Repeat the channel twice and resize to 120 by 120 | Mean patch features followed by the learned `GAP_FFN_s1` feedforward network |
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

Random Init uses the same encoder weights from seed 42 in every fold. The five
classifier seeds change only the linear classifier initialization. Its score
therefore concerns one random encoder, without measuring variation among
independently generated random encoders.

The public inference reproduction used CETUS revision
`360eaea532ead477f46906d7ebe0cb5282e8b082` and compared all twenty encoder and
fold combinations with native revision `6295af2bc35c8d8a2e47fc982b196809245f4550`.
Twenty extraction jobs and twenty CPU analysis jobs passed. All sixty feature
arrays, their ID and label arrays, one hundred probe heads and twenty kNN
prediction vectors matched the native records under the declared comparisons.
Feature arrays, probe probabilities, predictions and saved metrics matched
exactly. Independently recomputed metrics used absolute tolerance `1e-12`.
The completion report SHA256 is
`be3f8a8aeb55ff019cb7978bf8ef038904bca5ca403cc1c1b6221bb549d6349f`;
its monitor execution exited zero. This verifies the recorded source, data,
weights and runtime combination. It does not promise identical results on
other hardware or library versions.

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

These settings match the completed inference experiment. Set `GPU_ID` to the
physical GPU used for image features. When `CUDA_VISIBLE_DEVICES` contains that
single device, PyTorch numbers it as logical device 0, which `--device cuda`
addresses. A different physical index must still use logical index 0 in this
case. CPU analysis hides CUDA devices. Thread counts and the cuBLAS workspace
setting reproduce the numerical configuration; agreement on other hardware,
drivers or library versions still needs verification.

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

The private verification script is outside this package. To calculate the
linear classifier metrics, reconstruct each seed's confusion counts, average
five classifier scores per fold, then calculate the mean and sample standard
deviation of the five folds. Nearest neighbors supplies one prediction vector
per fold; calculate its metric and the same regional mean and SD. These SDs
describe variation among the five regions, whose training sets overlap. One
fold contains only one Craters tile. Establishing geographic independence,
inferential uncertainty or the effect of spatial separation needs further
experimental evidence.

## Reconstruct adaptation and training substitution results

The sibling `release/longitude-adaptation-controls-20260930` provides CPU
verification and reconstruction from saved predictions. Keep it beside the
unchanged `longitude-review-20260929` and `longitude-auxiliary-20260929`
packages. This workflow requires Python, NumPy and Matplotlib, with eleven
pinned PyPI wheels. It does not load encoder weights or train models.

Use Python 3.12.3 in an external virtual environment. The tested runtime was
Linux x86_64 with glibc 2.39, NumPy 2.4.4 and Matplotlib 3.10.9. The exact wheel
hashes are in the package's `code/requirements.txt`. A separate display trial
reproduced all sixteen accepted display artifacts and the adaptation checksum
sidecar byte for byte. The subsequent standalone public verifier and full
rebuild also exited zero in that isolated PyPI environment. All 51 package
files, totaling 10,201,553 bytes, matched the native export exactly, including
the displays, finite claims index and inventories. All three input packages
remained unchanged. The full-rebuild comparison record has SHA256
`a61d995a1cd6e0a8fd0db0e4b003b82b9190528f43d44e50d73a7e2e134bc40c`.
These checks apply to the recorded runtime and do not establish identical
output on other platforms. This runtime is separate from frozen GPU inference.

Run these commands from the CETUS root. Set `WORK` to a new external directory.
The `ACCEPTANCE_SHA` below is the SHA256 of the package's
`evidence/acceptance.json`, also listed in PAPER_REPRODUCTION.md. It identifies
the native export used for the completed independent package reconstruction.

```sh
PACKAGE="$PWD/release/longitude-adaptation-controls-20260930"
WORK=/absolute/path/to/new/cetus-reconstruction
ACCEPTANCE_SHA=7340fcc5d1fdc52b27ff3c0385d86669df97345897f95efa92bcb795dba5ba03
mkdir -p "$WORK"
python3.12 -m venv "$WORK/venv"
. "$WORK/venv/bin/activate"
python -c 'import sys; assert sys.version_info[:3] == (3, 12, 3)'
python -m pip install --only-binary=:all: --require-hashes -r "$PACKAGE/code/requirements.txt"
python -m pip check
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR="$WORK/matplotlib-cache" TMPDIR="$WORK/tmp"
mkdir -p "$MPLCONFIGDIR" "$TMPDIR"
cd "$PACKAGE/code"
python -B -m scripts.verify_repaired_adaptation_release \
  --package "$PACKAGE" \
  --frozen-package "$PACKAGE/../longitude-review-20260929" \
  --auxiliary-package "$PACKAGE/../longitude-auxiliary-20260929" \
  --acceptance-sha256 "$ACCEPTANCE_SHA"
```

Success requires exit zero and `status: pass`. Verification reconstructs the
numeric tables. To rebuild both figures and the complete sibling package,
repeat the verifier command with `--output "$WORK/rebuilt-package"`.
Both that output directory and `rebuilt-package.receipt.json` must be absent.
Keep all caches and outputs outside the three input packages. Preserve failed
outputs and select a fresh destination for a retry. Installation can use an
authenticated wheelhouse with `--no-index --find-links`; reconstruction can run
offline after installation. The package README gives the full input and receipt
contract. These figures do not require TeX.

The package also preserves development tests. Running every test in its
`code/tests/` directory gives one known failure:
`test_helper_projection_changes_only_denylist_constants` expects the original
private helper's hash, while the package contains the public helper. The public
verifier above checks the exported helper and reconstructs all 51 package files.
The October 5 review reproduced those files exactly; the other 152 adaptation
tests passed. The preserved test is not part of the root CPU test command.

The CPU workflow checks synthetic inference and cache handling. The inventory
workflow also reconstructs this package from saved predictions in the specified
separate environment. GPU inference and encoder training remain outside these
CI jobs. The numerical claims index checks the passages registered in it;
other manuscript statements require their own review.

## Preparing additional training runs

Running classical classifiers or training encoder weights with the public
inputs requires new experiment specifications. They must identify the sanitized
catalog, geographic splits, image hashes, software sources, normalization and
newly produced classifier results and feature caches. Encoder training also
needs the five classifier seeds from the same fold and the corresponding
classifier initialization. Preserve the original private specifications and
identify new runs by their actual public inputs. This package supplies the
completed predictions; execution of these additional training commands with
the public files remains future work.

Project code uses the root MIT license. The vendored CROMA implementation has
its upstream MIT notice in `licenses/CROMA-LICENSE`. Upstream source and weights
remain subject to their own terms; this package contains no pretrained weights.
