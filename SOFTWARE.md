# CETUS software and reconstruction

Choose the calculation before installing model dependencies. Reconstructing a score from saved predictions needs much less input than extracting image features or training an encoder. The [paper reproduction guide](PAPER_REPRODUCTION.md#choosing-an-experiment) identifies the population and comparison behind each result.

| Task | Instructions | Required inputs |
| --- | --- | --- |
| Reconstruct fitting, coverage, initialization or epoch-selection results | [Saved-result commands](release/titan-followup-20261109/README.md#reconstruct-saved-results) | The specified numerical archives, Python 3.12.3 and NumPy 2.4.4; write outputs outside the packages. |
| Reconstruct partial training and training-image replacements | [CPU reconstruction below](#reconstruct-adaptation-and-training-substitution-results) | Three unchanged sibling packages and the separate pinned CPU environment. |
| Extract image features and fit the native classifiers | [Install](#install-and-check), [supply inputs](#supply-the-data-and-weights), then [run one model and fold](#run-one-model-and-fold) | Processed Titan images, catalog/splits, upstream weights/source caches and a compatible recorded runtime. |
| Inspect or prepare additional training | [Training prerequisites](#preparing-additional-training-runs) and [DOFA source](#dofa-partial-training) | Implementations and contracts plus the separate data, state, environment and run specifications they require. |

## Frozen Titan classifier runner

The root software extracts features from DINOv2, DOFA, CROMA and an untrained ViT with encoder weights unchanged. It then fits five linear classifiers per fold with stochastic gradient descent (SGD), and evaluates cosine nearest neighbors. Its scientific source is development revision `0257a36197c51eda0a31768531ea26eef9887a6e`. The standardized logistic-regression, matched-DINOv2 and CNN comparisons have separate implementations and inputs. The scientific archive's four reference source files are for inspection and must not replace this runner.

The [current execution source](release/current-source-20261109/README.md) exports CNN, matched-DINO and CPU-fitting code from development revision `9089214`; its [contracts](release/current-source-20261109/CONTRACTS.md) identify additional images, weights, plans and supervision. Keep it in its own environment because both source packages import as `titansar`. [SOFTWARE_EXPORT.json](SOFTWARE_EXPORT.json) distinguishes the root export's original source hashes from current file hashes. Editorial updates do not identify new scientific runs.

The commands below cover the five contiguous Titan folds. Historical commands for Earth transfer, classical classifiers, encoder training, additional preprocessing and private verification depend on the development repository. Imported library functions for those experiments do not supply a complete execution route; the separate source packages identify their own prerequisites.

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

Install a PyTorch build appropriate for the machine before editable installation if its default build is unsuitable. CPU checks can install `.[test]` without the optional encoder dependencies. Python 3.12 is the exercised CPU environment; dependency lower bounds are compatibility declarations, not a tested matrix. The CPU tests create small image arrays and caches. They check normalization, cache integrity, the random encoder's weights and routing to the Titan workflow. A small synthetic example trains real linear classifiers and runs nearest neighbors with a simple encoder. Checking pretrained inference or the paper's scores requires the separately specified model inputs and result files.

The expanded Titan experiments used the production environment below. These versions appear in all twenty feature-calculation and twenty analysis manifests, whose hashes agree with the experiment checks. The runs used source revision `6295af2bc35c8d8a2e47fc982b196809245f4550`; a new run must identify its own source. The last column gives the separate CPU test environment. Reproducing numbers under other allowed dependency versions requires another comparison.

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

Production analysis ran on CPU with CUDA hidden. The supplemental environment snapshot records einops 0.8.2 and NVIDIA driver 580.173.02; those two versions were not captured in the individual frozen launch manifests. The CPU test tools were pytest 9.0.3 and setuptools 81.0.0. Editable installation was checked in a temporary environment using the existing system packages, with `--no-deps --no-build-isolation`; fresh resolution of all optional encoder dependencies was not tested as part of this bundle validation.

Run from the repository root with `python -m scripts.run_probing`. Editable installation is intentional: the manifest reads this checkout's Git revision and refuses a dirty worktree. A wheel or source archive without Git metadata does not provide this execution contract. Commit reviewed source changes before scientific execution. Keep data, weights, environment logs and results outside the checkout. Never bypass the clean-source check to label a run reproducible.

## Supply the data and weights

Download the [current Titan image release](https://github.com/magnaprog/CETUS/releases/tag/titan-processed-images-20261109), including its [companion notice](https://github.com/magnaprog/CETUS/releases/download/titan-processed-images-20261109/RELEASE_NOTICE.md) and `SHA256SUMS.release`. The archive `anonymous-titan-images.tar.zst` contains 23,380 arrays and is 1,312,330,584 bytes, SHA256 `83db434978996511bf487add0c0d28b95d6b197a867463570b0fe0b9e7ddabcb`. Follow the notice to verify the download, extract outside the checkout and run the bundled standard-library verifier. Public download and input verification passed on 9 November 2026, without model execution or normalization calculation.

Set `TITAN_DATA` below to the extracted `data` directory. It contains `catalog.json`, `tiles/TILE_ID.npy` and `splits/titan_contiguous_fold_F.json`. The catalog and five divisions exactly match those in `release/longitude-review-20260929`; no renaming or JSON rewrite is needed. Each tile must match its catalog `tile_sha256`. The bundled verifier checks every array; the runner binds normalization samples but does not hash every raw tile itself. There is no tile-directory CLI override.

Use the matching split for fold 0 through 4. Each split binds the sanitized catalog bytes. Do not edit catalog paths inside JSON, mix spatial-block splits with the longitude-sector catalog, or substitute private caches. The sanitized catalog has a different byte hash from the original scientific input; new caches and manifests must identify the public input and the public source commit honestly. Existing sealed records remain unchanged.

Supply trusted upstream weights separately. The wrappers check these hashes:

| Encoder | Required checkpoint SHA256 |
| --- | --- |
| DINOv2 | `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73` |
| DOFA | `4720985e42b918ac0307009eb06121a3435d9bbce6fd95446f84824a538165b1` |
| CROMA | `0238d814b53108f3574bf1ea240e38a0a6edd46173816d9a6962070561893b63` |

DINOv2 uses Torch Hub repository `facebookresearch/dinov2` at `7764ea0f912e53c92e82eb78a2a1631e92725fc8`. DOFA uses `zhu-xlab/DOFA` at `73ff5c0721da322689ae890bed5b4efd78935f47`. Their checkpoint names are `dinov2_vitb14_pretrain.pth` and `DOFA_ViT_base_e100.pth` under `torch.hub.get_dir()/checkpoints`. Offline use requires the matching Torch Hub source checkouts as well as weights in the cache expected by Torch Hub. The wrappers can access the network when assets are missing; no fully offline asset installation command is provided here. CROMA reads `~/.cache/croma/CROMA_base.pt`. Do not pass untrusted checkpoints to PyTorch. The random baseline uses timm to generate its encoder weights locally with seed 42. Each run records the actual weight digest and timm version. Always use `--require_real_weights`: the deliberate random baseline is allowed, while placeholder substitutes for pretrained encoders are rejected.

The Titan arrays contain display digital numbers from the SAR/HiSAR mosaic, identified by `hisar_log_dn` in the catalog. Training pixels determine the brightness range used to scale values into [0, 1], with clipping outside that range. Converting these display values to sigma0 or dB requires radiometric information unavailable in the package.

The wrappers use different input and feature operations:

| Encoder | Input after dataset normalization | Tile feature |
| --- | --- | --- |
| DINOv2 | Repeat the channel three times, resize to 224 by 224, then apply channel means [0.485, 0.456, 0.406] and standard deviations [0.229, 0.224, 0.225] | CLS token |
| DOFA | Keep one channel and resize to 224 by 224 | Mean patch features followed by `fc_norm` |
| CROMA | Repeat the channel twice and resize to 120 by 120 | Mean patch features followed by the learned `GAP_FFN_s1` feedforward network |
| Random Init | Keep one channel and resize to 224 by 224 | CLS token |

DINOv2's repeated channels differ after their channel-specific normalization. CROMA's two channels duplicate the same SAR display values. DOFA applies only resizing after dataset normalization and receives the recorded Cassini identifier 13.78, extending the numeric convention in its pinned upstream README. Its checkpoint's pretraining identifier convention remains unresolved, as described in the wrapper. The README's Earth example also applies channel standardization; the applicability of those channel constants to Titan and their use in the checkpoint's pretraining remain unverified. This export preserves the recorded Titan input recipe.

The pinned DOFA factory uses mean patch pooling followed by LayerNorm through `fc_norm`. A separate CPU inspection of the specified checkpoint and factory verified that `fc_norm.weight` and `fc_norm.bias` are absent from the checkpoint and retain their constant initial values of one and zero. LayerNorm still normalizes the pooled features. The checkpoint's `norm` entries do not load into `fc_norm`, and the factory's classification head is unused by feature extraction. This inspection covered model construction and checkpoint loading.

Random Init uses the same encoder weights from seed 42 in every fold. The five classifier seeds vary the linear classifier's initialization and minibatch order while retaining those encoder features. Its score therefore concerns one random encoder and does not measure variation among independently initialized encoders.

The public inference reproduction used CETUS revision `360eaea532ead477f46906d7ebe0cb5282e8b082` and compared all twenty encoder and fold combinations with native revision `6295af2bc35c8d8a2e47fc982b196809245f4550`. Twenty extraction jobs and twenty CPU analysis jobs passed. All sixty feature arrays, their ID and label arrays, one hundred probe heads and twenty kNN prediction vectors matched the native records under the declared comparisons. Feature arrays, probe probabilities, predictions and saved metrics matched exactly. Independently recomputed metrics used absolute tolerance `1e-12`. The monitor exited zero; [file_index.json](file_index.json) retains the completion record's identity. That comparison verifies the recorded source, data, weights and runtime combination. Agreement on other hardware or library versions needs its own check.

## Run one model and fold

Set the following shell paths to real directories on your machine. MODEL is one of `dinov2`, `dofa`, `croma`, `random_init`; FOLD is 0 through 4. Repeat this pair for all twenty model/fold combinations. Each output directory must be new.

```sh
TITAN_DATA=/absolute/path/to/extracted/titan-images/data
RESULTS=/absolute/path/to/new/results
MODEL=dinov2
FOLD=0
GPU_ID=0
CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -m scripts.run_probing --within_titan_only --require_real_weights \
  --titan_catalog "$TITAN_DATA/catalog.json" \
  --titan_split_manifest "$TITAN_DATA/splits/titan_contiguous_fold_${FOLD}.json" \
  --models "$MODEL" --device cuda --batch_size 32 --seeds 0 1 2 3 4 \
  --features_only --output_dir "$RESULTS/features/$MODEL/fold-$FOLD"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.run_probing \
  --within_titan_only --require_real_weights \
  --titan_catalog "$TITAN_DATA/catalog.json" \
  --titan_split_manifest "$TITAN_DATA/splits/titan_contiguous_fold_${FOLD}.json" \
  --models "$MODEL" --device cpu --batch_size 32 --seeds 0 1 2 3 4 \
  --cached_features_dir "$RESULTS/features/$MODEL/fold-$FOLD/features" \
  --output_dir "$RESULTS/analysis/$MODEL/fold-$FOLD"
```

These settings match the completed inference experiment. Set `GPU_ID` to the physical GPU used for image features. When `CUDA_VISIBLE_DEVICES` contains that single device, PyTorch numbers it as logical device 0, which `--device cuda` addresses. A different physical index must still use logical index 0 in this case. CPU analysis hides CUDA devices. Thread counts and the cuBLAS workspace setting reproduce the numerical configuration; agreement on other hardware, drivers or library versions still needs verification.

Cache analysis still requires the training tiles to recompute the exact normalization record. Both modes bind catalog, split, actual benchmark track, normalization, encoder revision and weights/state identity. The selected training pixel digest and membership digest have explicit formats in the normalization record. Held-out pixels do not fit this normalization.

Preserve both run manifests, all cache arrays and sidecars, predictions and `probing_results.json`, the input inventory, and `python -m pip freeze` output outside the checkout. Accept a job only when its manifest says `complete`, every output hash matches, the requested model is present, and the recorded encoder revision and weights/state digest match the intended encoder. Require the five distinct probe seeds 0 through 4, the intended fold and exact test IDs. The legacy permissive mode can emit placeholders or mark partial parallel results complete. Keep strict weights enabled in both commands and inspect `weights_source`: pretrained encoders must say `pretrained`, and only the intentional random control may say `random_init`. Failed jobs can leave a `started` manifest; retain that evidence and rerun into a new directory. Probe outputs include tile IDs, true labels, six class probabilities and confusion counts. k-NN outputs include IDs, labels and predictions. `macro_accuracy` is mean recall over classes present in that test fold, not macro F1. The six classes in index order are plains, dunes, hummocky, labyrinths, lakes, craters.

The private verification script is outside this package. To calculate the linear classifier metrics, reconstruct each seed's confusion counts, average five classifier scores per fold, then calculate the mean and sample standard deviation of the five folds. Nearest neighbors supplies one prediction vector per fold; calculate its metric and the same regional mean and SD. These SDs describe variation among the five regions, whose training sets overlap. One fold contains only one Craters tile. Establishing geographic independence, inferential uncertainty or the effect of spatial separation needs further experimental evidence.

## Reconstruct adaptation and training substitution results

The sibling `release/longitude-adaptation-controls-20260930` provides CPU verification and reconstruction from saved predictions. Keep it beside the unchanged `longitude-review-20260929` and `longitude-auxiliary-20260929` packages. This workflow requires Python, NumPy and Matplotlib, with eleven pinned PyPI wheels. It does not load encoder weights or train models.

Use Python 3.12.3 in an external virtual environment. The tested runtime was Linux x86_64 with glibc 2.39, NumPy 2.4.4 and Matplotlib 3.10.9. The exact wheel hashes are in the package's `code/requirements.txt`. A separate display trial reproduced all sixteen accepted display artifacts and the adaptation checksum sidecar byte for byte. The subsequent standalone public verifier and full rebuild also exited zero in that isolated PyPI environment. All 51 package files, totaling 10,201,553 bytes, matched the native export exactly, including the displays, finite claims index and inventories. All three input packages remained unchanged. The full-rebuild comparison record has SHA256 `a61d995a1cd6e0a8fd0db0e4b003b82b9190528f43d44e50d73a7e2e134bc40c`. These checks apply to the recorded runtime and do not establish identical output on other platforms. This runtime is separate from frozen GPU inference.

Run these commands from the CETUS root. Set `WORK` to a new external directory. The `ACCEPTANCE_SHA` below is the SHA256 of the package's `evidence/acceptance.json`, also listed in PAPER_REPRODUCTION.md. It identifies the native export used for the completed independent package reconstruction.

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

Success requires exit zero and `status: pass`. Verification reconstructs the numeric tables. To rebuild both figures and the complete sibling package, repeat the verifier command with `--output "$WORK/rebuilt-package"`. Both that output directory and `rebuilt-package.receipt.json` must be absent. Keep all caches and outputs outside the three input packages. Preserve failed outputs and select a fresh destination for a retry. Installation can use an authenticated wheelhouse with `--no-index --find-links`; reconstruction can run offline after installation. The package README gives the full input and receipt contract. These figures do not require TeX.

The package also preserves development tests. Running every test in its `code/tests/` directory gives one known failure: `test_helper_projection_changes_only_denylist_constants` expects the original private helper's hash, while the package contains the public helper. The public verifier above checks the exported helper and reconstructs all 51 package files. The October 5 review reproduced those files exactly; the other 152 adaptation tests passed. The preserved test is not part of the root CPU test command.

The CPU workflow checks synthetic inference and cache handling. The inventory workflow also reconstructs this package from saved predictions in the specified separate environment. GPU inference and encoder training remain outside these CI jobs. The numerical claims index checks the passages registered in it; other manuscript statements require their own review.

## Preparing additional training runs

Running classical classifiers or training encoder weights with the public inputs requires new experiment specifications. They must identify the sanitized catalog, geographic splits, image hashes, software sources, normalization and newly produced classifier results and feature caches. Encoder training also needs the five classifier seeds from the same fold and the corresponding classifier initialization. Preserve the original private specifications and identify new runs by their actual public inputs. This package supplies the completed predictions. The November model comparisons have already run with their original inputs and producers; publishing their saved results does not make those training commands portable. Execution with the public files still needs the separate source, input and environment work described above.

Project code uses the root MIT license. The vendored CROMA implementation has its upstream MIT notice in `licenses/CROMA-LICENSE`. Upstream source and weights remain subject to their own terms; this package contains no pretrained weights.

## DOFA partial training

The [DOFA source package](release/dofa-partial-source-20261112/README.md) contains the runner and supporting modules for the fifty-fit numeric-input comparison. It preserves the original implementation without presenting the source subset as a complete runnable environment. Its contracts describe the additional inputs and execution checks. The separate [numerical companion](release/dofa-partial-20261112/README.md) reconstructs scores from saved predictions using only the Python standard library.

## Reading preserved source descriptions

The table below gives the current interpretation of comments and docstrings in
[current-source](release/current-source-20261109/README.md),
[loss-source](release/loss-source-20261110/README.md), the
[historical DOFA modules](release/dofa-partial-source-20261112/README.md), and
`reference/titansar/` inside the
[scientific archive](release/titan-followup-20261109/README.md). It supersedes the
specific descriptions listed in the last column. These source snapshots retain
their producer bytes; the root runner's editable descriptions are corrected
in place.

| Object or helper | Operation and interpretation | Historical wording superseded |
| --- | --- | --- |
| DOFA numeric input | The wrapper passes the recorded numeric band identifier. Its checkpoint-specific pretraining convention remains unresolved, so the comparison does not validate physical wavelength conditioning. | `defaults.py`, `DOFAWrapper` and cache messages that label the input GHz or call the Titan comparison a band-extrapolation test; current-source, loss-source, historical DOFA and archived reference copies. |
| Encoder features | DINOv2 and Random Init use CLS features. The accepted DOFA factory uses mean patch features followed by `fc_norm`; CROMA uses mean patch features followed by `GAP_FFN_s1`. The base wrapper delegates to each encoder. | Historical and archived base-wrapper and DOFA descriptions that call every output a CLS token. |
| Repeated input channels | DINOv2 uses means [0.485, 0.456, 0.406] and standard deviations [0.229, 0.224, 0.225], so its repeated channels differ after normalization. CROMA receives two copies of the same SAR image, without an independently measured second polarization. The wrapper does not establish which learned features remain usable. | Historical and archived DINOv2 claims of identical normalization constants and CROMA claims that all learned cross-polarization features are unusable. |
| Titan tile values | `hisar_log_dn` arrays contain display digital numbers. The description “sigma0 linear” applies only to `linear_sigma0` mode; array shape and float32 storage do not establish calibrated backscatter. | The unconditional “sigma0 linear” file-format bullet in the historical and archived dataset docstrings. |
| Display-DN normalization | `compute_linear_norm_range` takes sample percentiles. Extreme values have less influence than on min/max bounds, but saturation can still determine a percentile bound. A percentile span below `1e-6` returns the declared default range. | Historical and archived claims that percentile bounds prevent saturated pixels from setting the scale. |
| dB normalization | `compute_norm_range` estimates pixel spread from per-tile means and available variances. The helper does not compare clipping counts. Titan display DN uses the separate percentile routine. | The historical and archived dB helper's claim that far fewer pixels clip at 0 or 1. |
| MMD estimates | `compute_mmd` returns squared MMD; `compute_mmd_repeated` summarizes those values. Its `ci95` field contains empirical subsampling quantiles, not a parametric confidence interval for population squared MMD. | Current-source repeated-estimator descriptions that call the returned quantity MMD. |
| Fallback geographic split | `assign_split` first reduces longitude modulo 360 and checks the Selk holdout. Outside that holdout, west [180, 360) is training; east [0, 180) is validation at absolute latitude at most 30 degrees and test above 30 degrees. Published evaluations use their bound external manifests. | Closed longitude intervals in the historical and archived fallback docstrings. |
| Probe fitting and checkpoint selection | `train_linear_probe` uses caller-supplied class weights when present and otherwise unweighted cross-entropy. `train_finetuned_model` runs the declared epoch loop and restores its validation-selected state; validation does not stop the loop early. | Historical probing descriptions that promise inverse-frequency weights unconditionally or describe checkpoint selection as early stopping. |
| Legacy Earth patches | `tile_earth_analog` uses `tile_size` for both square-patch dimensions and strides; 128 pixels is the default. | The current-source helper's fixed “128×128” description. |

## Source identity and maintenance

The root export's builder and development commit identify its original source. `SOFTWARE_EXPORT.json` binds current files and retains the original `source_sha256` values and additive documentation history. Earlier comment/docstring corrections preserved executable ASTs after docstrings were excluded; a separate cleanup removed only the unused `Optional`, `np` and `TitanSARConfig` imports. Original source, guide and manifest bytes remain in CETUS commit `360eaea532ead477f46906d7ebe0cb5282e8b082`. The current k-nearest-neighbor helper additionally rejects nonpositive or noninteger `k` and empty training arrays before computing predictions; valid-input calculations are unchanged. The root `scripts/summarize_crater_map_comparison.py` reconstructs the expanded Craters table from two bundled result files, with their input hashes checked before calculation. Its command and output definitions are in [PAPER_REPRODUCTION.md](PAPER_REPRODUCTION.md#craters-training-and-map-composition).

Keep frozen producers and package inventories together. A later documentation commit is not the producer of historical results, and a full-source or full-function AST check can include docstrings. Historical root catalogs, splits and results retain their paths and September snapshot copies. Repeated bytes can serve separate reconstruction routes, tests or interfaces; they are not a reason to delete a package.
