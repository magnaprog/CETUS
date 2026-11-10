# Execution contracts

The CLI accepts `--spec` and optional `--execute` only. Its JSON envelope has
`specification` and `specification_sha256`; the latter is the canonical sorted,
compact JSON digest implemented in `scripts/model_followup_support.py`.
Preparation imports runtime libraries and reads/hashes the declared prerequisites.
It can return `ready=false` or raise; this package supplies no ready specification.

For the loss comparison, the specification uses `model-followup-v1`,
`purpose=titan_followup`, `kind=resnet18`, `phase=final`, pretrained initialization
and learning rate 0.0001. `loss_comparison` has exactly three keys:

* `schema_version`: `population-weight-v1`;
* `control`: the historical A result's file binding (`path`, `sha256`);
* `original_source`: the original commit and exact eleven-file source hash map
  pinned in `scripts/loss_comparison.py`.

The historical A producer is `1f3fc8fe0e831777e3188d70caf2c5450b656333`.
The loss-study producer is `11033ff9f6fb21db81b3cc91b18bff4f9a58d02b`.
A future public packaging commit and a synthetic-test fixture commit have separate
roles and do not replace either producer. New scientific execution must bind the
actual clean Git revision and all twelve runtime `SOURCE_FILES` entries. The
package manifest additionally covers the complete source/test/notice payload.
A Git-less extraction does not satisfy the execution guard.

The B specification has no `tuning_results`: it reuses A's authenticated
learning-rate selection receipt. Preserve the protocol's 100 epochs, batch size
32 including the last partial batch, full-model FP32 AdamW/cosine recipe, and
seeds s, 20000+s and 10000+s. Selection is the earliest maximum recorded unrounded
six-class validation macro F1. No extra epoch-100 test endpoint is introduced.

The comparison authenticates every artifact in the original A output inventory,
including `initial_state.pt`, `selected.pt`, `final.pt`, preparation/start records
and 100 epoch records. It verifies the original initial tensors before creating
the training optimizer. Public saved-result records and checkpoint hashes alone
cannot satisfy these checks. The original states remain server-only and are
omitted from this package.

Except for source, output directory, tuning-results and comparison fields, the
paired original/new specifications must match. This includes inputs, runtime,
recipe and GPU identity. Real CUDA jobs require the declared physical GPU exposed
as `cuda:0`, the dependency receipt, deterministic IEEE FP32 settings and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. The external supervisor's checks and fresh
selected-state replay harness are not supplied by this one-job runner.

Ordinary image jobs support `inputs.binding_contract=cvpr2027-anonymous-v1` with
the matching anonymous catalog/divisions. That selector does not convert the
historical loss comparison to a new machine or catalog: changing only B breaks
the exact A pairing contract. File bindings for actual execution use absolute
caller-local paths, but no historical host paths or sealed private plans are
published here. A redacted projection is not an original sealed record.

The source-tree imports require NumPy, Torch, torchvision, SciPy and scikit-learn.
The intact deferred branches also reference timm, einops, scikit-image and
PyWavelets. The retained CROMA file and its MIT notice preserve that local branch
closure; they do not expand the supported ResNet comparison into a CROMA study.
Official DINO source and all upstream weights remain separately provisioned.
`requirements.txt` lists observed local distribution versions for this scoped
CPU validation, not a complete installation lock or an all-branch execution claim.
Runtime seals also bind imported implementation/native-library bytes, not only
version strings.

The two focused synthetic modules use tiny generated inputs and temporary states.
They import helpers from `tests/test_model_followup.py`; do not collect that helper
module or run the full tests directory here. No split fixtures or scientific ZIPs
are included. Use a disposable local Git fixture, keep its reports outside the
copy, and make the pytest basetemp a descendant of the same TMPDIR.

This package has no numerical companion, accepted B results, model weights,
mission files, image arrays, launch plans or scientific inference records. Public
saved-result reconstruction and scientific acceptance are separate work. The
dated protocol retains prospective wording; copying it does not certify a run.
