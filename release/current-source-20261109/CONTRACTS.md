# Execution and access contracts

This document describes the source exported from
`9089214a5a6134a088eed058b02df6e021fe80dc`. It supplies no sealed execution
plan, private access route or additional runtime API. The dated guides retain
their examples and historical evidence; the qualifications below govern what
this public package actually contains.

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
other exported file. The existing source tuple and all guards are unchanged.

For image jobs, the original catalog SHA256 is
`2c77251230c47596f33d13eafd91ff7d0c927c8f8275451d814fad875df893ec`.
Its five exact split fixtures are included under
`audit/gpu_execution/repaired_splits/`. Their hashes are in file_index.json
and model_followup_support.py. Their historical catalog-path field is metadata,
not an instruction to create a file under `/tmp`.

The anonymous catalog in the sibling scientific ZIP instead has SHA256
`8152226f876e0cb5fad105e0b00a37ef5b62d156d7f6eb3a85b77c549a402967`.
Select it explicitly with `inputs.binding_contract = "cvpr2027-anonymous-v1"`
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
scientific ZIP cannot supply the missing pixels. There is no established
public processed-image endpoint in this export.

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
retained-model checks. Their existence does not establish independent access
to images/states or new validation of this source package. Original producer
identities are retained in SOURCE_EXPORT.json. Public saved records, numerical
commands and qualifications are in the sibling [saved-results wrapper](../titan-followup-20261109/README.md)
and its [provenance](../titan-followup-20261109/provenance.json). The wrapper is
not a byte-identical copy of the omitted private audits.

## Omitted display generator

`scripts/prepare_model_followup_displays.py` belongs to the upstream paper
display workflow and is not needed by the exported execution import closure.
It is omitted. The dated model guide retains its historical command for
context; that command is unavailable in this package. It formats already
accepted results and supplies no model execution. Use the numerical companions'
own documented outputs; no replacement display command is implied here.

## Guide destination map

Paths in the first column below are **upstream historical paths**, not files
promised in this package. SOURCE_EXPORT.json records each exact original link
target, projected target and both complete-guide hashes. The access guide also
corrects the count to sixty CNN validation histories with 6,000 epoch records.
Other guide content, including examples, scientific settings, limitations and
disclosures, is unchanged. The fitting guide contains no links and has no byte
changes.

| Upstream historical path | Public destination and omission |
| --- | --- |
| `docs/CVPR2027_GPU_EXECUTION.md`, `docs/CVPR2027_FOLLOWUP.md` | [Historical execution records](#historical-execution-records); the private execution guide and comparison plan are omitted. |
| `audit/CVPR2027_INITIALIZATION_2026-11-08.md`, `audit/cvpr2027/followup-results.json`, `audit/CVPR2027_REPRODUCTION_2026-11-08.md` | [Historical execution records](#historical-execution-records); raw review, receipt and inference audit are omitted. |
| `scripts/prepare_model_followup_displays.py` | [Omitted display generator](#omitted-display-generator); historical command retained but not available. |
| `paper/cvpr2027/{scientific,followup,initialization}-supplement.zip` | Byte-preserved archives under the sibling `titan-followup-20261109/` package; links point directly to those files. |
| `paper/cvpr2027/README.md#initialization-results-and-reconstruction`, `paper/cvpr2027/README.md#epoch-100-and-validation-checks` | [Public reconstruction instructions](../titan-followup-20261109/README.md#reconstruct-saved-results); the upstream paper README is omitted. |

Included runner/support/coverage links and cross-links among the three guides
still resolve to the actual included files. External primary-product links
are retained as upstream references; they are not processed-input endpoints
and were not visited or availability-tested during this export.

## Validation

Local checks on 9 November verified exact project import origins. The
seven-module run recorded 218 passed, six Git-setup failures, three actual-model
skips and 15 passed subtests. After a clean synthetic fixture commit, the
affected test_model_followup.py module passed all 56 cases. The other six
modules retain 168 passes, three skips and 15 passed subtests from the earlier
run. These are separate runs, not one complete passing suite; detailed setup
history and warnings remain in retained private receipts.

The actual imported runtime was Python 3.12.3, Torch 2.15.0+cpu,
torchvision 0.29.2+cpu, NumPy 2.4.4, SciPy 1.17.1 and scikit-learn 1.9.0;
pytest was 9.2.0. Imported library versions agree with distribution metadata.
SOURCE_EXPORT.json binds the receipts. The subsequent
[hosted run](https://github.com/magnaprog/CETUS/actions/runs/42614128106)
installed a fresh CPU environment and passed all seven modules: 224 tests
and 15 subtests, with three actual-model skips. This is a separate result
from the local attempts above; the Python sources are identical.

For a new validation, first check SHA256SUMS and file_index.json in the source
package. Use a fresh external temporary copy with an isolated environment;
install its dependencies, including `typing_extensions`, within that environment.
Verify that `titansar`, `scripts` and `tests` import from that copy, not CETUS's
older root runner or another installed distribution. Set
`PYTHONDONTWRITEBYTECODE=1`, `PYTHONNOUSERSITE=1`, `CUDA_VISIBLE_DEVICES=''`,
`XFORMERS_DISABLED=1`, and numerical thread counts to one. Clear inherited
PYTHONPATH and leave `TITANSAR_RUN_MODEL_INTEGRATION` unset.

Five checked-in historical split JSONs are already included. Additionally,
supply the unchanged sibling `scientific-supplement.zip` in the temporary copy
at `paper/cvpr2027/scientific-supplement.zip`: 49,621,594 bytes, SHA256
`67b6aa65be5bab9b7cbe83df90aafcf7b479117291562471652f59ed7f182a1d`.
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
Retain all package hashes and source guards. The original no-Git temporary-copy
assumption was incorrect; no Python source change was required.

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
