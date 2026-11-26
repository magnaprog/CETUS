# Execution and access contracts

This package uses the CETUS training API. The [source manifest](SOURCE_EXPORT.json) lists the files from producer `9089214a` and the SHA-256 of each file in this release. New job specifications bind the release source. The analysis scripts still check the source and input records from the original fits.

## Model specifications and inputs

`python -m scripts.run_model_followup --spec JOB.json` prepares one job.
Preparation reads and hashes inputs but creates no model or output directory.
Check `ready` and `missing_prerequisites`: exit zero can accompany `ready=false`.
Adding `--execute` performs the declared computation. There is no `--prepare`,
grid-launch, resume or retained-state inference mode.

The JSON envelope contains `specification` and `specification_sha256`, using
the canonical sorted compact JSON digest in model_followup_support.py, with
nonfinite values rejected. File bindings contain absolute `path` and full
`sha256`. Real jobs use `schema_version: model-followup-v1`,
`purpose: titan_followup`, and kind `resnet18`, `dino_features` or `dino_c1`.
Bind fold 0–4, device, fresh external output, actual clean source and runtime,
and the job-specific model/recipe/input fields described in the model guide.
The eleven `SOURCE_FILES` hashes authenticate the existing runner contract;
the separate SOURCE_EXPORT.json also covers initializers, CROMA and every
other exported file. For a new job, use the paths and hashes from this copy.

For image jobs, the original catalog SHA256 is
`2c77251230c47596f33d13eafd91ff7d0c927c8f8275451d814fad875df893ec`.
Its five exact split fixtures are included under
`fixtures/original-splits/`. Their hashes are in file_index.json
and model_followup_support.py. Their historical catalog-path field is metadata,
not an instruction to create a file under `/tmp`.

The anonymous catalog in the sibling scientific ZIP instead has SHA256
`8152226f876e0cb5fad105e0b00a37ef5b62d156d7f6eb3a85b77c549a402967`.
Select it explicitly with `inputs.binding_contract = "anonymous-titan-v1"`
and use its matching five anonymous divisions. All six hashes are already
pinned in model_followup_support.py. Omitting the selector keeps the original
contract; renaming files or substituting one catalog for the other cannot
satisfy the guards.

Both image routes need the original `tiles/TILE_ID.npy` arrays beside the
catalog: float32, shape `(128,128)`, matching active tile hashes. The current
catalog has 23,380 ordered rows and class order plains, dunes, hummocky,
labyrinths, lakes, craters. Train/val/test totals, six-class counts, assignments,
benchmark track and training-only normalization must agree. No fallback or
relabeling of real data as `synthetic_cpu_test` is supported. Metadata in the
scientific ZIP does not include these pixels. The [image access guide](../../experiments/encoder-comparisons/DATA_ACCESS.md#processed-titan-images) gives the download, source terms and verification steps for all 23,380 arrays. Use the extracted `data/catalog.json`, `data/splits/titan_contiguous_fold_F.json` and `data/tiles/` together. The anonymous selector applies to these image jobs. Historical execution plans authenticate their original inputs.

Final CNN jobs require all three complete same-fold/initialization tuning
bundles with the same source/runtime/input identity, including states and
epoch records. The fixed recipe is full-model FP32 AdamW for 100 epochs at
batch 32, with candidate LRs 1e-4, 1e-3 and 1e-2; earliest maximal unrounded
validation macro F1 selects the epoch, and lower LR breaks rate ties. Tuning
uses seed 100; final jobs use seeds 0/1/2. New 9089214 jobs cannot silently
reuse tuning specifications bound to 1f3fc8fe. DINO C=1 readout authenticates
the complete extraction output inventory, including `initial_state.pt`.
Feature-only transport is not implemented.

## Model assets

| Required local asset | Fixed identity and scope |
| --- | --- |
| Official DINOv2 source | Revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`; 157-file Python inventory digest `929c0b25f4287511f065c8696667b39d2b9499c68f284bcab30b3e27099a488f`. Imported official modules must resolve inside the supplied tree. |
| DINO pretrained state | `dinov2_vitb14_pretrain.pth`, SHA256 `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`. |
| DINO random states | Seeds 42–46, each reused across folds; record file and tensor-state digests and verify reload equality. `load_matched_dino` supplies the builder, but there is no standalone state-preparation CLI in this export. |
| ResNet18 ImageNet state | `resnet18-f37072fd.pth`, SHA256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`; exact installed torchvision source/version and official enum URL are also bound. |

Matched DINO uses official ViT-B/14 with constructor size 518. Both model
families use the full normalized tile resized to 224 and the recorded channel
normalization. Keep `XFORMERS_DISABLED=1`. The older random ViT comparator is
distinct. The source does not fetch weights in these current paths. Supply
authentic local bytes under their upstream terms. No weight availability was
checked during export, and no model states are included.

## CPU fitting plan

The launcher supports `--plan`, `--plan-sha256`, `--workers` (1–4), and either
selected `--cells` or `--check-all`. It uses the included worker script. The
worker checks its actual clean Git head, all `source_sha256` entries, exact
Python/package versions and `runtime_identity` before fitting. A public run
needs a newly sealed plan for its actual checkout and environment. No new plan
builder is included in this pass.

| Plan field | Required content |
| --- | --- |
| `producer_commit`, `source_sha256` | Actual execution revision and source hashes. The original plan bound both fitting scripts, both fitting tests and docs/FITTING_SENSITIVITY.md. |
| `python_version`, `package_versions`, `runtime_identity` | Actual interpreter, package versions, module/callable/native-library paths and hashes. Historical machine paths are not transferable. |
| `accepted_protocol`, `accepted_protocol_sha256` | Bound protocol containing `specification.caches` and `specification.folds`. The original protocol is omitted and embeds private paths. |
| `feature_root` | Model/fold-N cache directories with exact `run_manifest.json` and all protocol-listed files. |
| `output_root`, `cells` | Fresh external outputs and exactly 80 distinct cells: four encoders × five folds × raw/standardized × none/L2. |
| Original design metadata | `schema_version`, `design_path`, `design_sha256`, `limits`. Retain separately identified provenance when preparing a new plan. |

The original four representation IDs are `dinov2`, `dofa`, `croma`,
`random_init`. The worker reads
`features/MODEL/titan_{train,test}_{feats,ids,labels}.npy`, checks finite
768-dimensional float32 features, unique/disjoint IDs, labels and counts,
and hashes **every** protocol-listed cache member plus the manifest. Retained
Selk files and probing_results.json are also bound; six train/test arrays
alone are insufficient. The later matched-DINO features have a different
layout and identity and cannot substitute for these older caches.

The original accepted protocol SHA256 is
`48515a7583e99c0cfd9fb179600cdfeda7dab484f424b4bcbdb912570632f9ca`;
the original fitting plan SHA256 is
`8cce9a4491a341648cb0f77bad3688d1a2e38235177e52156836027042840aae`.
These are historical identities, not accessible files supplied here. Editing
private paths creates different bytes and requires a distinct, verified
protocol/plan. Preserve the originals, scientific row/cache identities and
rejection guards. No public route to the complete original caches is supplied.

## Historical validators and supervision

The full-grid analyzer still pins the original catalog and five original
split hashes. It cannot validate a newly generated anonymous-input grid
unchanged, even though the one-job runner supports that input contract.
Historical complete-output checks hash retained state files at bound paths.
The coverage analyzer additionally requires original report/acceptance hashes
and the recorded analyzer source path. Redacted portable records have different
hashes and do not replace those originals.

The private grid launcher, GPU process supervisor, mapped-native-library
checker, job/dependency receipt builders, retained-state inference harness
and three historical CPU analysis scripts are outside this package. The
one-job CUDA runner checks the supplied dependency receipt's hash; that alone
does not reproduce the supervisor's before/after loaded-library checks.
No private harness or replacement runtime API is included. These limitations
do not change the separately supplied numerical-companion consumers.

## Runtime and third-party dependencies

The scientific model records describe Python 3.12.3, Torch 2.12.1+cu126,
torchvision 0.27.1+cu126 and NumPy 2.4.4. The dated guide separately records
CPU checks with Torch 2.11.0+cu130, torchvision 0.26.0+cu130 and sklearn 1.8.0.
The CPU fitting study used Python 3.12.3, NumPy 2.4.4, SciPy 1.17.1,
scikit-learn 1.9.0, joblib 1.5.3, threadpoolctl 3.6.0 and narwhals 2.28.0.
New runs must bind their actual imported runtime. Generic lower bounds such
as Torch >=2.0 are insufficient for the per-operation FP32 precision API
used here; fitting also calls version-sensitive sklearn/SciPy internals.

The selected static closure uses numpy, torch, torchvision, scipy, sklearn,
threadpoolctl, timm, einops, skimage and pywt, plus pytest for tests. Some
imports belong to deferred legacy branches. The full package also preserves
deferred pyproj, rasterio and ESA SNAP `snappy` paths; Python's compression
package `python-snappy` is not ESA SNAP. The extra preprocessing paths are
outside the supported current model/fitting scope. No all-path installation
claim or new dependency lock accompanies the source export.

Author-owned files retain their MIT license. The vendored CROMA implementation
requires the included Anthony Fuller MIT notice. The two data notices retain
map, VIMS/ISS and Copernicus credits. Official DINO source, torchvision,
dependencies and weights remain separately supplied under their own terms.

## Historical execution records

The guides link here where original private audit records were omitted. Those
records covered the completed 90-cell study, coverage and three author-side
retained-model checks in Region 1. Twelve additional cases in Regions 2–5
bring the inference coverage to fifteen cases across all five regions, with
scratch ResNet18, ImageNet ResNet18 and pretrained DINOv2 in each region.
The added cases reproduce 112,062 native validation/test prediction rows
exactly; no fitting or new accuracy estimate occurred. These author-side
checks are separate from the public image download/input verification and do
not establish independent-recipient inference or access to retained model
states. They are not new execution validation of this source export. Original
producer identities are retained in SOURCE_EXPORT.json. Public saved records, numerical
commands and qualifications are in the sibling [saved-results wrapper](../../experiments/encoder-comparisons/README.md)
and its [provenance](../../experiments/encoder-comparisons/provenance.json). The wrapper is
not a byte-identical copy of the omitted private audits.

## Omitted display generator

The original display program formatted the accepted comparison report. To reproduce the underlying scores from saved predictions, use the [public reconstruction commands](../../experiments/encoder-comparisons/README.md#reconstruct-saved-results).

## Guide destination map

The [model guide](docs/MODEL_FOLLOWUP.md) describes a single training or feature-extraction job. The [fitting guide](docs/FITTING_SENSITIVITY.md) covers the 80 CPU fits. The [training input guide](docs/DATA_ACCESS.md) explains how to use the public tiles and which model states each job needs.

The [numerical companions](../../experiments/encoder-comparisons/README.md#reconstruct-saved-results) reproduce the saved arithmetic. The historical coverage validator also authenticates the original scientific archive against its recorded execution and acceptance hashes.

## Validation

The [CPU workflow](../../.github/workflows/training.yml) checks imports and runs seven encoder test modules. Three integration cases require local model assets.

For a new validation, first check SHA256SUMS and file_index.json in the source
package. Use a fresh external temporary copy with an isolated environment;
install its dependencies, including `typing_extensions`, within that environment.
Verify that `cetus_sar`, `scripts` and `tests` import from the training copy. Set
`PYTHONDONTWRITEBYTECODE=1`, `PYTHONNOUSERSITE=1`, `CUDA_VISIBLE_DEVICES=''`,
`XFORMERS_DISABLED=1`, and numerical thread counts to one. Clear inherited
PYTHONPATH. For the synthetic suite, unset both `CETUS_RUN_MODEL_INTEGRATION` and its older alias `TITANSAR_RUN_MODEL_INTEGRATION`.

Five checked-in historical split JSONs are already included. Additionally,
supply the sibling `scientific-supplement.zip` in the temporary copy
at `fixtures/scientific-supplement.zip`: 49,528,151 bytes, SHA256
`5169c9ec1a2e7ac8c5f78499064d2399b196225bb04423ac9a78795314a92dcc`.
The metadata tests read archive members directly; no full extraction or image
bundle is required. This fixture is external to the 60-file package inventory.

The temporary copy must have a real Git HEAD even for synthetic tests. After
supplying the fixture, run these commands **only inside that external copy**:

```sh
git init --initial-branch=synthetic-tests
git add -- .
git -c user.name='CETUS synthetic tests' \
  -c user.email='cetus-synthetic-tests@example.invalid' \
  -c commit.gpgsign=false commit \
  -m 'Synthetic CPU test fixture; not a scientific producer'
test -z "$(git status --porcelain=v1)"
```

Record this local test commit separately; it does not replace the exported-from
revision or any scientific producer. Keep outputs and caches outside the copy.
When checking its payload file set, exclude only its top-level `.git/` metadata
and allow the declared external ZIP in addition to the 60 package files.

From that prepared copy, use the chosen environment's Python:

```sh
python -B -m pytest \
  tests/test_model_followup.py tests/test_anonymous_followup_inputs.py \
  tests/test_model_followup_integration.py tests/test_analyze_model_followup.py \
  tests/test_model_followup_coverage.py tests/test_fitting_sensitivity.py \
  tests/test_fitting_launcher.py -q -ra \
  --basetemp=/absolute/path/to/new/external/test-tmp \
  -o cache_dir=/absolute/path/to/new/external/pytest-cache
```

Default integration skips must be reported. The remaining tests use synthetic
small CPU networks/classifiers, metadata and process-control fixtures. They
do not perform Titan scientific fits. Optional actual-model tests need a
separate explicit opt-in and authentic local DINO/ResNet assets; do not infer
their success from skips. Keep test results outside the source package and
separate from the already completed four numerical-consumer checks.
