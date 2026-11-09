# Model follow-up: one-job preparation and execution

The authorized comparison completed on 8 November 2026. See the
[execution record](../CONTRACTS.md#historical-execution-records) and
[result review](../CONTRACTS.md#historical-execution-records). The commands
below document how a new producer must prepare its own inputs and environment;
they are not a request to rerun the completed study.

The [runner](../scripts/run_model_followup.py) prepares or executes one
`resnet18`, `dino_features`, or `dino_c1` job. The default is read-only
preparation. It checks local bindings and reads input arrays; it does not
construct a model, fit a classifier, download assets, or create the job output
directory. `ready=true` means those preparation checks passed. It does not
establish model compatibility, GPU availability, convergence, or accuracy.
Check `ready` and `missing_prerequisites`: a preparation report with missing
assets can still exit zero. Invalid bindings raise an error.

The examples below are instructions for a later prepared producer. Paths under
`/srv/cetus` and `/opt/cetus` are explicit example locations to replace with
local locations. Asset locations and authentication receipts belong to the
private model-preparation handoff; they are not distributed with this guide.
Use a clean committed checkout, keep protocols and outputs outside it, and
capture the actual imported runtime on that producer. Do not copy a previous
machine's runtime lock or label real inputs as `synthetic_cpu_test`.

```bash
cd /srv/cetus/source/TitanSAR
export PYTHON=/opt/cetus/env/bin/python
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 XFORMERS_DISABLED=1 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
# After creating the specification below:
"$PYTHON" -m scripts.run_model_followup \
  --spec /srv/cetus/protocols/resnet18-scratch-fold0-lr1e-4.json
```

There is no `--prepare` flag. `--execute` opts into the declared job and is
reserved for the separately authorized execution phase. This is not a grid
launcher and has no automatic retry, resume, or budget extension.

**Specification and original inputs.** A job file contains `specification`
and its canonical `specification_sha256`, calculated by
`model_followup_support.digest`. A file binding is an existing absolute `path`
and its full `sha256`, calculated by `binding`.

| Field | Contract |
| --- | --- |
| `schema_version`, `purpose`, `kind` | `model-followup-v1`; `titan_followup` for real inputs; one of the three job kinds above. |
| `fold`, `device`, `output_dir` | Saved fold ID 0–4; `cpu` or explicit `cuda:N`; fresh external attempt directory that does not overlap inputs, assets, or source. |
| `source`, `runtime` | Current clean Git commit and `SOURCE_FILES` hashes; actual imported module/callable/native-library identities and execution settings from `runtime_identity()`. |
| `inputs` | Image jobs bind catalog, split, benchmark track, per-role counts, and training normalization digest. C=1 jobs instead bind a completed feature producer. |
| `inputs.counts` | Each of `train`, `val`, `test` has `total` and six `class_counts`; every truth class must occur in every role. |
| `model`, `recipe` | Local model/source/state contract and the job-specific recipe described below. |

The original current catalog must have SHA256
`2c77251230c47596f33d13eafd91ff7d0c927c8f8275451d814fad875df893ec`,
23,380 entries, and class order **plains, dunes, hummocky, labyrinths, lakes,
craters**. Put its original float32 `(128,128)` arrays in adjacent
`tiles/{tile_id}.npy` files. Each active tile hash is checked. Use
`audit/gpu_execution/repaired_splits/titan_contiguous_fold_{f}.json` for fold
`f`; the five exact hashes are pinned in
[model_followup_support.py](../scripts/model_followup_support.py).
`policy.test_fold`, catalog binding, assignments, counts, and ordered IDs must
agree. `buffer_excluded` and `selk_holdout` remain outside the three active roles.
The anonymous catalog has a separate, explicit input route. For the catalog
and divisions in `scientific-supplement.zip`, set
`inputs.binding_contract = "cvpr2027-anonymous-v1"`, bind that catalog and its
matching fold file, and place the original arrays beside the catalog in
`tiles/{tile_id}.npy`. The runner checks the exact anonymous catalog and fold
hashes before its existing image, count, role and training-normalization checks.
Omitting the field retains the original input requirements above; it does not
accept anonymous metadata implicitly. Completed feature records carry the
same input identity into C=1 evaluation.

CPU input validation has exercised all five anonymous folds with the actual
23,380 image arrays. The retained values, ordered role IDs, labels, image
hashes and training normalization agree with the original inputs. This checks
input compatibility; it does not repeat feature extraction, inference or
training. Model assets, a clean execution producer and its actual runtime
remain necessary. See [data access](CVPR2027_DATA_ACCESS.md) for package scope
and availability.

The inherited dataset derives DN normalization from training tiles only and
requires complete, nonfallback provenance shared by all three roles. Both
architectures use the full normalized tile, bilinear resize to 224 with
`align_corners=False`, `antialias=False`, three repeated channels, and ImageNet
mean/std. There is no ImageNet center crop or new missing-pixel treatment.

**Create one complete specification.** With the environment above, this example
creates a fold-0 scratch ResNet tuning specification for preparation on CPU.
It reads/hashes the original arrays to bind normalization, but constructs no
model. The source lock checks actual project imports; the runtime lock captures
the interpreter that will run the CLI. Keep environment/settings identical for
capture and use. Call `configure_runtime()` once in a fresh process, before
model work or runtime capture; the CLI does this itself. It sets both Torch
thread pools to one, strict deterministic algorithms, deterministic cuDNN with
benchmarking disabled, and IEEE FP32 for CUDA matmul and cuDNN convolution.
It uses only the per-operation `fp32_precision` API available in Torch 2.11/2.12;
do not mix it with legacy precision setters. See the pinned
[Torch 2.12 precision contract](https://docs.pytorch.org/docs/2.12/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices).

```bash
"$PYTHON" - <<'PY'
import importlib
import json
from pathlib import Path
import torch
import torchvision
from scripts import model_followup_support as s
from titansar.reproducibility import git_state, sha256_file

s.configure_runtime()
git = git_state(s.ROOT)
assert git["clean"], "Use a clean committed producer"
source = {"commit": git["commit"], "files": {
    name: sha256_file(s.ROOT / name) for name in s.SOURCE_FILES}}
s.verify_source_imports(source["files"])
runtime = s.runtime_identity()
fold, seed = 0, 100
catalog_path = Path("/srv/cetus/inputs/titan/catalog.json")
split_path = s.ROOT / f"audit/gpu_execution/repaired_splits/titan_contiguous_fold_{fold}.json"
catalog = json.loads(catalog_path.read_text())
split = json.loads(split_path.read_text())
counts = {}
for role in s.ROLES:
    labels = [row["label"] for row in catalog["tiles"]
              if split["assignments"][row["tile_id"]] == role]
    assert set(labels) == set(range(6)), f"Missing truth class in {role}"
    counts[role] = {"total": len(labels),
                    "class_counts": [labels.count(i) for i in range(6)]}
resnet_source = Path(importlib.import_module("torchvision.models.resnet").__file__)
spec = {
    "schema_version": "model-followup-v1", "purpose": "titan_followup",
    "kind": "resnet18", "fold": fold, "device": "cpu",
    "output_dir": "/srv/cetus/runs/resnet18-scratch-fold0-lr1e-4-attempt01",
    "source": source, "runtime": runtime,
    "inputs": {"catalog": s.binding(catalog_path), "split": s.binding(split_path),
               "benchmark_track": catalog["benchmark_track"], "counts": counts},
    "phase": "tuning", "seed": seed, "loader_seed": 10000 + seed,
    "model": {"initialization": "scratch", "backbone_seed": seed,
              "head_seed": 20000 + seed, "torchvision_version": torchvision.__version__,
              "resnet_source_sha256": sha256_file(resnet_source)},
    "recipe": {**s.RECIPE, "lr": 1e-4},
}
spec["inputs"]["normalization_sha256"] = s.validate_inputs(spec)["normalization_sha256"]
job_path = Path("/srv/cetus/protocols/resnet18-scratch-fold0-lr1e-4.json")
# Insert a job variation below here, before publishing, if needed.
job_path.parent.mkdir(parents=True, exist_ok=True)
s.publish_json(job_path, {"specification": spec,
                          "specification_sha256": s.digest(spec)})
print(job_path)
PY
```

Publication never overwrites a prior protocol. Use a new protocol filename and
attempt directory for a changed job. Capture CUDA job locks on the intended
clean execution producer, with its reviewed device/environment settings,
before sealing; the CPU example above grants no Titan execution authorization.

**ResNet initialization and selection.** For ImageNet initialization, change
`model.initialization` to `pretrained`, add
`weights="ResNet18_Weights.IMAGENET1K_V1"` and
`checkpoint=s.binding("/srv/cetus/assets/resnet18-f37072fd.pth")`, and give the
job distinct protocol/output paths. The fixed official authority is
[resnet18-f37072fd.pth](https://download.pytorch.org/models/resnet18-f37072fd.pth),
full SHA256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
The implementation verifies the installed enum URL and this fixed hash; an
arbitrary caller-declared checksum is insufficient. Provision weights explicitly
in advance; runtime fetching is absent. Both arms reset the six-class head
after backbone initialization/loading and use a separate loader generator.

The fixed recipe is full-model FP32 AdamW, 100 epochs, batch 32, weight decay
0.01, betas `(0.9,0.999)`, epsilon `1e-8`, and cosine scheduling, without AMP
or augmentation. Use all three LRs `1e-4`, `1e-3`, `1e-2` with tuning seed 100
for each fold and initialization. Final seeds are 0, 1, 2; backbone seed is
`s`, head seed `20000+s`, loader seed `10000+s`. Thirty tuning plus thirty final
fits are the finite budget; no convergence claim or automatic extension follows.

Each fit selects the earliest epoch attaining the maximum **unrounded**
six-class validation macro F1 after optimizer updates. Final LR selection needs
all three complete same-fold/same-arm tuning bundles, with matching source,
runtime, inputs, and other recipe/model settings. LR ties choose the lower LR.
Do not pool folds for selection. Tuning does not score test images; final fits
replay the selected checkpoint on validation before scoring test images.

For example, insert this variation at the marker in the scratch script above
after the three tuning jobs have completed. The runner independently verifies
the inventories, candidate completeness, scores, and selection; merely listing
three paths is insufficient.

```python
paths = [Path(f"/srv/cetus/runs/resnet18-scratch-fold0-lr{lr}-attempt01/result.json")
         for lr in ("1e-4", "1e-3", "1e-2")]
results = [json.loads(path.read_text()) for path in paths]
chosen = max(results, key=lambda r: (
    r["history"][r["selected_epoch"] - 1]["val_macro_f1"],
    -r["specification"]["recipe"]["lr"]))
seed = 0
spec.update(phase="final", seed=seed, loader_seed=10000 + seed,
            tuning_results=[s.binding(path) for path in paths],
            output_dir="/srv/cetus/runs/resnet18-scratch-fold0-final-seed0-attempt01")
spec["model"].update(backbone_seed=seed, head_seed=20000 + seed)
spec["recipe"]["lr"] = chosen["specification"]["recipe"]["lr"]
job_path = Path("/srv/cetus/protocols/resnet18-scratch-fold0-final-seed0.json")
```

**DINO assets and state preparation.** Both arms use the official ViT-B/14
source at `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, constructor image size
518, positional tensor `(1,1370,768)`, and normalized 768-dimensional CLS
output. Input images still use the common 224 transform. Keep
`XFORMERS_DISABLED=1`; imported official source and attention backend are checked
during construction. The older random ViT comparator is a separate model.

Before creating extraction jobs, prepare a local **model dictionary** JSON,
for example `/srv/cetus/protocols/dino-model.json`, with these fields:

| Field | Value or derivation |
| --- | --- |
| `initialization`, `encoder_seed` | `random` with seed 42–46, or `pretrained` with seed 42. |
| `source_dir`, `source_revision` | `/srv/cetus/assets/dinov2-source` and the exact revision above. |
| `source_files` | `{str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*.py"))}` with `root=Path(source_dir)`. The full 157-file inventory must match the pinned digest in `followup.py`. |
| `checkpoint` | `binding` of the prepared plain state dictionary: `/srv/cetus/assets/dinov2-random-seed42.pt` or `/srv/cetus/assets/dinov2_vitb14_pretrain.pth`. |
| `initial_state_sha256` | Canonical tensor-state digest from `titansar.models.followup.state_digest`, separate from the checkpoint file SHA. |

The pretrained file's fixed SHA256 is
`0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`.
The private handoff retains source/weight authentication receipts. A separate
authorized state-preparation step must construct the pinned official random
model for each seed, save its complete state, record both file and tensor-state
digests, and check reload equality. `load_matched_dino` supports that preparation
without a supplied random checkpoint; extraction specifications require the
prepared checkpoint and digest. Also bind the pretrained tensor-state digest.
Reuse each random state across all five folds. Test state artifacts do not
automatically become scientific producers. Default preparation does not create
these states or prove their seeded equality; construction checks that equality.

To prepare one extraction, insert this variation at the same marker in the
first script, using the completed model dictionary:

```python
spec = {k: spec[k] for k in ("schema_version", "purpose", "fold", "device",
                            "source", "runtime", "inputs")}
spec.update(kind="dino_features", recipe={"batch_size": 32},
            model=json.loads(Path("/srv/cetus/protocols/dino-model.json").read_text()),
            output_dir="/srv/cetus/runs/dino-random42-fold0-attempt01")
job_path = Path("/srv/cetus/protocols/dino-random42-fold0.json")
```

Name paths to match the chosen arm/seed. Extraction writes ordered train, val,
and test feature arrays with labels, IDs, raw-input and normalization identities,
encoder/source/transform identities, and `initial_state.pt`. All 25 random and
five pretrained extraction jobs and their scientific readouts completed in
the authorized study. Preserve their existing outputs when preparing a new run.

**CPU C=1 readout.** For each newly completed extraction bundle, create a
separate readout envelope with this standalone script, using the same hidden-CUDA
environment and a clean producer. It reads producer metadata to make the spec;
the CLI verifies the complete bundle before fitting. Keep the whole producer
directory, including its state artifact, available on the execution host.
Feature-only transport is not implemented.

```bash
"$PYTHON" - <<'PY'
import json
from pathlib import Path
import torch
from scripts import model_followup_support as s
from titansar.reproducibility import git_state, sha256_file
s.configure_runtime()
git = git_state(s.ROOT)
assert git["clean"], "Use a clean committed producer"
source = {"commit": git["commit"], "files": {
    name: sha256_file(s.ROOT / name) for name in s.SOURCE_FILES}}
s.verify_source_imports(source["files"])
path = Path("/srv/cetus/runs/dino-random42-fold0-attempt01/result.json")
producer = json.loads(path.read_text())
spec = {"schema_version": "model-followup-v1", "purpose": "titan_followup",
        "kind": "dino_c1", "fold": 0, "device": "cpu",
        "output_dir": "/srv/cetus/runs/dino-random42-fold0-c1-attempt01",
        "features": s.binding(path), "feature_identity": producer["identity"],
        "source": source, "runtime": s.runtime_identity()}
out = Path("/srv/cetus/protocols/dino-random42-fold0-c1.json")
out.parent.mkdir(parents=True, exist_ok=True)
s.publish_json(out, {"specification": spec, "specification_sha256": s.digest(spec)})
PY
"$PYTHON" -m scripts.run_model_followup \
  --spec /srv/cetus/protocols/dino-random42-fold0-c1.json
# Later, with execution authorized, append --execute to run this CPU fit.
```

The existing common readout fits StandardScaler on training features only,
then balanced multinomial L2 logistic regression with C=1, LBFGS, tolerance
`1e-4`, maximum 2,000 iterations, and seed 0. There is no C search. Outputs
include scaler/coefficient state, resolved arguments, warnings, iteration counts,
and native validation/test logits, probabilities, and predictions. Predictions
come from native logit decisions; rounded probabilities never choose a class.
The runner does not claim convergence from a returned estimator.

**Outputs, failures, and CPU evidence.** A complete job publishes `result.json`
last, with its output inventory. ResNet additionally retains initial, selected,
and final states; per-epoch validation predictions/confusions; losses using the
global class-weight denominator; and selected-checkpoint replay. Retain failed
attempts and partial staging files. Guarded failures attempt to write
`failed.json`; secondary receipt IO failures are attached to the original
exception rather than masking it. Retry with a fresh directory.

The retained CPU evidence before the parent's full suite is scoped as follows:

| Record | Outcome and limit |
| --- | --- |
| `unit-01` | 21 passed, 2 failed: overly strict interpolation-roundoff oracle and whole-batch versus batch-summed float32 loss oracle. Both original failures remain retained. |
| Corrected numerical oracles | Exact inherited-wrapper tensor equality is checked separately. The analytic interpolation reference uses float64 with a bound of `8 * float32_epsilon / channel_std`; weighted loss uses independent per-example float64 log-sum-exp and a bound of `8 * float32_epsilon * max_example_loss`. Input physics and class weighting did not change. |
| `focused-02` | 26 passed, 1 failed, 6 expected xFormers warnings. Actual official DINO state/seed/reload checks and one pretrained synthetic CPU forward passed. Authentic ResNet preload exposed 20 absent legacy BatchNorm counters. |
| Legacy checkpoint compatibility | Source review confirmed Torch supplies zero for those counters under strict loading. The adapter records/checks them; all other backbone tensors must match the authenticated checkpoint. No missing learned parameter or running moment is accepted. |
| `focused-03` | 27 passed, 1 deliberately deselected, 6 existing NumPy/Torch conversion warnings. Units and actual ResNet step/replay/preload passed. The earlier DINO check was not repeated; its test and relevant functions remained unchanged. No missing-asset skips occurred in the enabled runs. |
| CLI preparation | Synthetic envelope returned `ready=true`, no model checks or accuracy, and created no job output directory. Tiny-network extraction-to-C=1 tests cover pipeline behavior separately from actual-model integration. |

The parent's `full-tests-01` retained one C=1 replay-test failure: 2,658 passed,
12 skipped, 592 warnings and 29 passed subtests in 989.72 seconds. The manual
oracle omitted native float32 scaler-statistic casts present in **both** sklearn
1.8 and 1.9. Its transformed-array error was the same in both environments;
the retained 1.8 float64 coefficients produced a logit error of `3.275e-15`,
while the 1.9 float32 coefficients exposed `1.669e-6`. This was not a change in
scaler arithmetic. No-fit restoration of serialized `StandardScaler` parameters
and state reproduced saved logits and predictions exactly in both environments.
The corrected test independently checks training mean, variance, scale and
sample count, prohibits refitting during replay, and retains `rtol=0, atol=1e-12`
for its matrix assertion. One targeted test passed in each environment; the
1.9 run emitted one unrelated dateutil deprecation warning. Production code
is unchanged. The parent owns the subsequent full-suite outcome.

The exercised local imports were Python 3.12.3, Torch `2.11.0+cu130`,
torchvision `0.26.0+cu130`, NumPy `2.4.4`, and sklearn `1.8.0`; the private
environment receipt records their actual locations. These are evidence of that
environment, not a substitute for capturing the later producer's imports.
No GPU or Titan model execution occurred in these checks. The independent
six-file review found no required cleanup; the parent owns the full-suite result.

Actual-model tests default to explicit skips. For a separately authorized
offline CPU integration run, use the hidden-CUDA environment above and local
assets; `TITANSAR_DINO_SOURCE_SPEC` is the pretrained **model dictionary**, not
a job envelope. There are no implicit downloads or substitute models.

```bash
export TITANSAR_RUN_MODEL_INTEGRATION=1
export TITANSAR_RESNET18_CHECKPOINT=/srv/cetus/assets/resnet18-f37072fd.pth
export TITANSAR_DINO_SOURCE_SPEC=/srv/cetus/protocols/dino-pretrained-model.json
"$PYTHON" -m pytest tests/test_model_followup.py tests/test_model_followup_integration.py \
  -q -ra --basetemp=/srv/cetus/checks/model-01/tmp \
  -o cache_dir=/srv/cetus/checks/model-01/cache
```

**Authorized GPU phase.** The first complete 100-epoch tuning cells and original
DINO extraction cells can establish feasibility while remaining within the
[reviewed comparison plan](../CONTRACTS.md#historical-execution-records). There is no shortened recipe
or separate pilot purpose. Keep the same source, runtime, dependency receipt,
inputs and selection rules throughout the comparison. A complete finite recipe
does not establish convergence.

For each CUDA job, add the following fields before sealing its specification.
The parent wrapper supplies one common, sealed receipt of the actual resolved
CUDA-bearing Torch, cuBLAS/cuDNN and driver dependencies. Place that receipt
outside the job-output tree, for example in `/srv/cetus/environment/`.

```python
spec["device"] = "cuda:0"
spec["execution"] = {
    "gpu_uuid": "GPU-<full canonical UUID from the verified lane inventory>",
    "cuda_dependencies": s.binding("/srv/cetus/environment/cuda-dependencies.json"),
}
```

Export that full UUID as `CUDA_VISIBLE_DEVICES` before starting each fresh
process. The runner requires exactly one visible GPU and checks its actual UUID
before model work. UUIDs stay outside the shared runtime identity, so independent
lanes can supply tuning candidates. The dependency receipt must match across
candidates. The runner verifies the small receipt's hash before/after work and
links it from the GPU execution record; the parent wrapper owns actual loaded
library-path/byte checks before and after execution and keeps those reports
outside job directories. Receipt binding alone is not library verification.

CUDA preparation requires `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before process
startup, both TF32 override variables unset or zero, IEEE matmul/convolution,
no CUDA autocast, strict determinism and at most two threads per Torch pool.
Runtime identity binds those settings, backend source files and override values.
CPU jobs do not require CUDA settings or a dependency receipt. Preparation only
checks host configuration and files; device count, UUID and context checks occur
in execution. GPU records include name, memory, CUDA/cuDNN and peak
allocated/reserved bytes, with failed measurements reported when unavailable.
The current preparation changes have no new scientific model results; the
parent owns subsequent execution and acceptance records.

## Coverage analysis after the complete comparison

The [coverage consumer](../scripts/analyze_model_followup_coverage.py) uses
the complete, accepted report from `analyze_model_followup`. Successful
supervision and saved-state replay must be checked before supplying that
report. This command checks saved predictions and cohort arithmetic; it does
not establish those earlier checks or execute a model.

Its plan contains `schema_version: model-followup-coverage-v1`,
`purpose: titan_followup`, and four absolute path/SHA256 bindings:
`followup_report`, `coverage_report`, `coverage_package`, and
`coverage_acceptance`. The last three identify the existing historical
coverage calculation, its scientific ZIP, and its independent acceptance
record. Their required identities are in the consumer's `HISTORICAL`
constant. The complete report's original plan, source, catalog, geographic
divisions and saved results must remain readable.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PYTHON" -B -m scripts.analyze_model_followup_coverage \
  --plan /srv/cetus/analysis/coverage-plan.json \
  --plan-sha256 EXACT_PLAN_SHA256 \
  --output /srv/cetus/analysis/new-coverage-result.json
```

The consumer analyzes all 60 test prediction vectors: 30 DINO readouts and 30
final CNN fits. Its report contains their input bindings and cohort metrics.
The 30 CNN tuning fits contribute no test predictions. The
consumer joins images by ID and checks labels against the original catalog.
It uses the existing complete/partial SAR cohorts, with 20,603 and 2,643 test
images respectively, and checks that their confusion counts sum to the
23,246-image test union. It opens only the catalog in the historical ZIP;
it does not recalculate the earlier 115 prediction vectors or map votes.

Means first average seeds within each region, then give equal weight to the
five regions. If a cohort lacks a truth class in any region, its six-class
regional summary remains undefined. Per-class counts, false positives and
defined ratios remain available. Changing the evaluated image population
also changes its terrain and geographic composition, so this comparison
cannot isolate the effect of missing SAR pixels.

## Tables and figure from accepted results

The [display generator](../CONTRACTS.md#omitted-display-generator) formats
the already accepted full comparison report. It copies existing metrics,
regional summaries and validation choices into seven TeX table fragments and
one figure. It performs no fitting, checkpoint selection or new aggregation.

```bash
"$PYTHON" -B -m scripts.prepare_model_followup_displays \
  --report /srv/cetus/analysis/accepted-comparison.json \
  --report-sha256 EXACT_REPORT_SHA256 \
  --output /srv/cetus/displays/new-initialization-comparison
```

Use a fresh output directory outside the source checkout. The main table
compares initialization within DINOv2 and within ResNet18. The figure shows
all five regional differences in macro F1. The supplement tables give
regional and seed scores, class precision/recall/F1, chosen learning rates
and final checkpoint epochs. Sample SD across five regions describes their
spread; it is not a confidence interval. The two model families use different
fitting methods, so the main table does not highlight a cross-family winner.

The private `display-values.json` maps numeric entries to their unrounded
input values, and `manifest.json` identifies the output and source bytes.
Only the TeX and figure files belong in the manuscript. An anonymous
reproduction package needs its own projection of the private records. A
synthetic-purpose report produces visibly labelled previews. A failed export
may leave a partial directory, which must be retained separately and never
mistaken for a complete output. The actual displays passed the numerical and visual checks recorded in the
[initialization review](../CONTRACTS.md#historical-execution-records).
