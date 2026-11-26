# Fitting classifiers on Titan images

Use the native runner to keep an encoder fixed, fit five linear classifiers by stochastic gradient descent (SGD), and evaluate cosine nearest neighbors on each regional split. Choose DINOv2, DOFA, CROMA or an untrained ViT. For standardized logistic regression, CNN training or DOFA fine-tuning, see the [training guide](../training/README.md). To reconstruct a score from existing predictions, see the [experiment index](../experiments/README.md).

## Install and check

To install the native runner, clone the repository and run the remaining commands from its root.

```sh
git clone --branch v0.2.0 https://github.com/magnaprog/CETUS.git
cd CETUS
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[encoders,test]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m pytest -q
python -m scripts.run_probing --help
git status --porcelain
```

Choose a PyTorch build appropriate for your hardware. For CPU tests alone, install `.[test]`. The suite checks normalization, cache integrity, the untrained encoder's weights and routing to the Titan workflow. A synthetic example fits linear classifiers and runs nearest neighbors. The `cetus-sar` 0.2.0 wheel installs the `cetus_sar` Python package. Run the command-line workflow from a clean Git checkout because its manifest records the revision and rejects uncommitted changes. Keep images, weights and outputs outside the checkout.

## Supply the data and weights

Download the [processed Titan image archive](https://github.com/magnaprog/CETUS/releases/download/v0.2.0/titan-images.tar.zst), the [release notice](https://github.com/magnaprog/CETUS/releases/download/v0.2.0/RELEASE_NOTICE.md) and the [release checksums](https://github.com/magnaprog/CETUS/releases/download/v0.2.0/SHA256SUMS.release). The archive contains 23,380 arrays in 1,312,278,780 bytes. Its SHA-256 is `3ea1c147af5dd6daf663da2d8bc1bdf1c822faf0d6b6764b77ce2a3361392aad`. Follow the notice to verify the download against the release checksums, extract outside the checkout and run the bundled verifier.

Set `TITAN_DATA` to the extracted `data` directory. It contains `catalog.json`, `tiles/TILE_ID.npy` and `splits/titan_contiguous_fold_F.json` for folds 0 through 4. The catalog and splits match the [longitude-sector benchmark](../experiments/longitude-benchmark/README.md). Each split binds the exact catalog bytes. The public catalog omits local paths and has a different byte hash from the original input. Preserve its contents and use its hash in new cache and run records.

The archive verifier checks every tile against its catalog hash. The runner checks the samples used for normalization and locates tile files relative to the catalog. To supply another tile directory, prepare it in that layout.

Supply trusted upstream weights separately. The wrappers check the following hashes.

| Encoder | Required checkpoint SHA-256 |
| --- | --- |
| DINOv2 | `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73` |
| DOFA | `4720985e42b918ac0307009eb06121a3435d9bbce6fd95446f84824a538165b1` |
| CROMA | `0238d814b53108f3574bf1ea240e38a0a6edd46173816d9a6962070561893b63` |

DINOv2 uses Torch Hub repository `facebookresearch/dinov2` at `7764ea0f912e53c92e82eb78a2a1631e92725fc8`. DOFA uses `zhu-xlab/DOFA` at `73ff5c0721da322689ae890bed5b4efd78935f47`. Their checkpoint names are `dinov2_vitb14_pretrain.pth` and `DOFA_ViT_base_e100.pth` under `torch.hub.get_dir()/checkpoints`. Offline use requires the matching Torch Hub source checkouts as well as weights in the cache expected by Torch Hub. The wrappers can fetch missing assets over the network. CROMA reads `~/.cache/croma/CROMA_base.pt`. Do not pass untrusted checkpoints to PyTorch. The random baseline uses timm to generate its encoder weights locally with seed 42. Each run records the actual weight digest and timm version.

## Run one model and fold

Set the following shell paths to real directories on your machine. Set `MODEL` to `dinov2`, `dofa`, `croma` or `random_init` and `FOLD` to an integer from 0 through 4. Run extraction and analysis once for each encoder and fold you need. Each output directory must be new.

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

These commands expose one physical GPU to PyTorch as logical device 0 during extraction and hide CUDA during analysis. Thread counts and the cuBLAS workspace size match the recorded numerical configuration.

Cached analysis still reads training tiles to reconstruct the normalization record. Both run modes bind the catalog, split, benchmark track, encoder revision, weight digest, normalization samples and training membership. The cache loader also checks tile IDs and integer labels against the requested split and catalog. In both commands, `--require_real_weights` rejects placeholder encoders. Check `weights_source` for `pretrained` on DINOv2, DOFA and CROMA, or `random_init` on the intentional untrained baseline.

Retain both run manifests, cache arrays and sidecars, predictions, `probing_results.json`, the input inventory and the output of `python -m pip freeze`. For a completed fit, verify the manifest status `complete`, every output hash, the intended encoder revision and weight digest, fold membership, test IDs and all five distinct classifier seeds. A failed run can leave a `started` manifest. Preserve it and retry in a new directory. The permissive runner can emit placeholders or mark partial parallel results complete.

Linear-classifier outputs include tile IDs, labels, six probabilities and confusion counts. Nearest-neighbor outputs include IDs, labels and decisions. In index order the classes are Plains, Dunes, Hummocky, Labyrinths, Lakes and Craters. The field `macro_accuracy` gives mean recall over the classes present in the test fold. To reproduce a regional summary, compute each seed's metric from its confusion counts, then average the five scores within the fold. Nearest neighbors contributes one score per fold. Give the five folds equal weight and report their sample standard deviation. Their training sets overlap, and Region 2 has just one Craters test tile.

## What the encoders receive

The Titan arrays contain display digital numbers from the SAR/HiSAR mosaic, identified by `hisar_log_dn` in the catalog. For each fold, the runner samples up to 512 finite positive pixels per tile from up to 200 training tiles. The first and 99th percentiles of the pooled sample set the brightness limits for scaling into [0, 1], with clipping outside that range. Converting these display values to sigma0 or dB requires radiometric information unavailable in the package.

Each encoder transforms the normalized tile before extracting features.

| Encoder | Input after dataset normalization | Tile feature |
| --- | --- | --- |
| DINOv2 | Repeat the channel three times, resize to 224 by 224, then apply channel means [0.485, 0.456, 0.406] and standard deviations [0.229, 0.224, 0.225] | CLS token |
| DOFA | Keep one channel and resize to 224 by 224 | Mean patch features followed by `fc_norm` |
| CROMA | Repeat the channel twice and resize to 120 by 120 | Mean patch features followed by the learned `GAP_FFN_s1` feedforward network |
| Random Init | Keep one channel and resize to 224 by 224 | CLS token |

DINOv2's repeated channels differ after their channel-specific normalization. CROMA's two channels duplicate the same SAR display values. DOFA applies only resizing after dataset normalization and receives the recorded Cassini identifier 13.78, extending the numeric convention in its pinned upstream README. Its checkpoint's pretraining identifier convention remains unresolved, as described in the wrapper. The upstream Earth example also standardizes channels. Its constants have not been verified for Titan or for this checkpoint’s pretraining.

The pinned DOFA factory uses mean patch pooling followed by LayerNorm through `fc_norm`. A separate CPU inspection of the specified checkpoint and factory verified that `fc_norm.weight` and `fc_norm.bias` are absent from the checkpoint and retain their constant initial values of one and zero. LayerNorm still normalizes the pooled features. The checkpoint's `norm` entries do not load into `fc_norm`, and the factory's classification head is unused by feature extraction. This inspection covered model construction and checkpoint loading.

Random Init uses the same encoder weights from seed 42 in every fold. The five seeds vary classifier initialization and minibatch order on that one encoder’s features.

An inference check repeated the production runs for all twenty encoder–fold combinations. Sixty feature arrays and their ID and label arrays, one hundred probe heads, and twenty nearest-neighbor prediction vectors passed their checks. Features, probe probabilities, predictions and saved metrics matched exactly. Independently recomputed metrics used absolute tolerance `1e-12`. The [file index](../release/inventory.json) identifies the completion record. See [the current nearest-neighbor rule](#source-identity-and-maintenance) before comparing a new run with those results.

<a id="source-identity-and-maintenance"></a>

## Cosine nearest neighbors

The current k-nearest-neighbor helper checks positive integer `k`, nonempty training arrays, aligned labels, matching feature widths and finite features. It accepts integer and float16/32/64 feature arrays. Each nonzero row is copied to float64, divided by its largest absolute component and then divided by its Euclidean norm, without an added epsilon. This computes cosine similarities while avoiding overflow or underflow of the original row norm. Genuine zero rows have zero similarity to every row. For each query, the helper orders training rows by descending similarity and uses up to `k` neighbors. The most frequent label among them wins. When labels have equal maximum counts, the smallest class index wins.

The float64 working arrays and similarity batches require more memory than float16/32 arithmetic. Rounding can change close neighbor rankings, and integers beyond float64's exact range can lose precision during conversion. Boolean, complex, object and wider floating-point feature arrays are rejected. Tests cover synthetic scale changes, extreme finite values, zeros and an independent cosine oracle; they do not establish identical predictions for every formerly accepted input.

The reported nearest-neighbor scores used `norm + 1e-8` to normalize each feature row in the [production environment](#recorded-environment). Whether the current rule gives identical predictions on those feature arrays has not been tested.

## Recorded environment

Twenty feature-extraction jobs and twenty CPU analysis jobs ran with the production versions below. The last column describes a separate CPU test environment. Dependency lower bounds in the package describe allowed versions. Numerical agreement under other versions needs its own check.

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

The production analysis hid CUDA. A separate environment snapshot records einops 0.8.2 and NVIDIA driver 580.173.02. The CPU tests used pytest 9.0.3 and setuptools 81.0.0. The editable installation check reused system packages with `--no-deps --no-build-isolation`. It did not test fresh resolution of every optional encoder dependency.

## Reading preserved source descriptions

The [field and helper definitions](reproduce.md#input-field-names-used-by-the-software) explain how the archived implementations handle geographic splits, display intensities, model selection and MMD. The [software inventory](../release/software-sources.json) records the source files.

## Reconstruct adaptation and training substitution results

The [adaptation verifier](../experiments/adaptation-controls/README.md) checks its acceptance record before reconstructing tables or figures. Pass the following digest with `--acceptance-sha256 "$ACCEPTANCE_SHA"`.

```sh
ACCEPTANCE_SHA=7734c1be30d721f6f0e4a40de58b3c08ba47ab67b94073fe9950748ac60b6711
```

## Preparing additional training runs

The [training guide](../training/README.md) identifies the implementation and model assets for each fitting experiment. For checkout integrity and maintenance, see the [release guide](../release/README.md). The vendored CROMA implementation carries its [upstream MIT notice](../licenses/CROMA-LICENSE).
