# Prospective paired DOFA partial-training comparison

This guide records the recipe and execution interfaces at source revision
`3a91cc7f19ab6c3f91b9015f5e9e09b97364aaf0`. Its public projection changes
documentation only. It is not launch authorization or a report of completed
fits. The [source contracts](../CONTRACTS.md#execution-limits) explain the
omitted execution prerequisites and the distinction from the earlier studies.
Synthetic tests do not establish a scientific result.

The comparison contains **50 current-runtime fits**: five existing geographic
folds, seeds 0–4, and numeric inputs 13.78 and 3.75. The first two cells are the
fold-0, seed-0 pair, each with the complete 50-epoch budget. They are included in
the 50 cells. Both pilot fits and selected-state replays require external
acceptance before the remaining 48 cells. There is no score-based pilot choice,
short pilot followed by a restart, or automatic retry.

The number supplied to DOFA conditions its dynamic patch embedding even when
that embedding's learned parameters are frozen. The alternative 3.75 follows
the documented upstream version-1 Sentinel-1 identifier. Its use for Cassini's
single display-DN channel tests numeric-input sensitivity. It does not establish
a physical Cassini mapping or a repaired pretraining convention; see
[the retained source review](DOFA_INPUT_REVIEW.md). Both arms run under the
current environment so the historical/current driver difference is not an arm
difference. Reused test geography still makes this an exploratory comparison.

## Fixed computation

The consumer calls helpers from the unchanged historical producer
`ed2eb3ecd4722c22c0aff8b5370ac9a4b23e4376`. Its
[original protocol](../historical-ed2eb3e/audit/REPAIRED_FINETUNING_PROTOCOL_2026-09-29.json) remains
sealed at SHA256
`dd280f9b830acd8fba84a0603504c12ee39941b40fb89912a3e264002c8f44cb`.
`prepare_repaired_run` still validates the original 13.78 reference arrays,
source, membership and normalization. This is a provenance anchor, not the
current control arm or a newly fitted classifier. It supplies no initial head.
The new result nests this historical provenance separately from its actual band.

The same source supplies `repaired_training_metadata`, `train_finetuned_model`,
`repaired_history_metadata`, `evaluate_finetuned`, prediction checks and
checkpoint saving. Neither those helpers nor their module globals are patched.
The consumer imports the complete checked historical package/script inventory
in a fresh process, rejects shadowed imports and records the actual origins.
Its small process/file/runtime helper is the unchanged, byte-pinned
`run_dofa_band_sensitivity.py`; that helper's model loader and pilot runner are
not invoked.

| Item | Both current arms |
| --- | --- |
| Encoder | Pinned DOFA v1 `73ff5c0721da322689ae890bed5b4efd78935f47`, original factory defaults |
| Weights | Original pretrained `DOFA_ViT_base_e100.pth`, SHA256 `4720985e42b918ac0307009eb06121a3435d9bbce6fd95446f84824a538165b1` |
| Images | Original one-channel DN scaling/zero fill, training-only normalization, CPU bilinear 224×224 resize with align_corners=False |
| Pooling | Mean of patch tokens, then frozen `fc_norm` with its accepted initial scale/bias |
| Trainable state | Blocks 10 and 11, 14,175,744 parameters; fresh six-output head, 4,614 parameters |
| Initialization | Fresh pretrained encoder each cell; seed reset after upstream import/loading; identical head and full encoder state within each pair |
| Data order | Historical shared job-seeded generator; train shuffles, validation/test do not; no new iterator before fitting |
| Budget | All 50 epochs; microbatch 8, accumulation 4, two workers; retain partial batches/windows |
| Optimization | AdamW, backbone LR 1e-4, head LR 1e-3, weight decay 0.01, original defaults, 50-epoch cosine schedule |
| Objective | Original training-frequency weights and effective-batch weighted-mean cross entropy |
| Precision | Original CUDA AMP/GradScaler in training and validation; original test evaluation outside autocast |
| Selection | Earliest highest validation macro recall among epochs with successful updates; no early stopping |
| Test | Once after selection per fit, plus separately recorded saved-state replay |
| Device | One predeclared physical GPU for both bands in a fold/seed pair |

No channel duplication/standardization, pooling change, mask input, loss change,
extra trained parameter, LR search, seed selection or compensation for skipped
AMP updates is part of this comparison. Counts and initial head hashes are
fixed in the consumer. The actual one-element band list is passed directly to
the pinned upstream forward by a consumer-local subclass. A `TypeError` is an
error, never a retry without a band. Aggregate call counts and an ordered digest
cover every training, validation, test and replay call without retaining a
large per-batch trace.

## Original executable contract (not supplied)

`scripts/run_dofa_partial_band.py` does not manufacture or approve its own
contract. The execution operator creates its JSON **after clean source freeze**.
It must have schema `dofa-partial-band-v1`, status `reviewed_for_execution`, and
the exact `POLICY` value from the reviewed consumer. Every binding is
`{"path": "/absolute/path", "sha256": "64 lowercase hex", "bytes": 123}`.
Paths, hardware identifiers and actual payload inventories stay in that private
contract rather than this conceptual guide.

Required top-level fields are:

- `consumer`: clean checkout path/commit and exact script binding; `helper`:
  binding of the unchanged helper, SHA256
  `998d877d4af302f29b2146755da198b27f934f2319091c6193e86c3e5fccc8c6`.
- `historical_source`: path, original commit and binding of **every tracked
  Python file under `titansar/` and `scripts/`** at that commit. Working bytes
  must match Git blobs, including all eight original sealed scientific files.
- `protocol`, `catalog`, `weights`: bindings of the original protocol, original
  23,380-tile catalog and pretrained checkpoint. Catalog SHA256 is
  `2c77251230c47596f33d13eafd91ff7d0c927c8f8275451d814fad875df893ec`.
- `upstream`: pinned commit, directory and bindings for `hubconf.py`,
  `dofa_v1.py` and `wave_dynamic_layer.py`; their bytes and directory must agree.
- `python_entrypoint` and `python`: the original virtual-environment entrypoint
  and resolved binary binding. Do not invoke the resolved system binary instead.
- `initial_feature_state`: the expected authenticated initial state digest in
  the unchanged helper's `feature_state` format, including 170 entries and the
  explicitly excluded unused head. The current loader also records the full
  encoder digest, including that head, for equality within each pair.
- `head_state_sha256`: the five exact historical head hashes in seed order,
  matching `HEADS` in the consumer.
- `tiles`: binding of a JSON with schema `dofa-partial-tiles-v1`,
  `catalog_sha256`, and `files` mapping **every catalog tile ID** to its raw file
  binding. Paths must be the catalog's actual sibling `tiles/{tile_id}.npy`;
  expected hashes must equal the catalog. No new copy of the tiles is needed.
- `folds`: five ordered records with `fold`, `[train,val,test]` counts,
  `split`, `reference`, `analysis_manifest`, `feature_manifest` bindings and
  `features_dir`. Original split and reference hashes come from the sealed
  protocol, not new or anonymous catalog projections.
- `devices`: three distinct UUID/existing-lock-path records; `output_root`:
  existing private result parent outside all protected input/source trees.
- `cells`: exactly 50 ordered records with `cell`, `fold`, `seed`, numeric
  `band`, device index, `output` and `replay_output`. Cell index is
  `2*(5*fold+seed)+arm`, where arms 0/1 are 13.78/3.75. Each pair shares a device.
  All 100 attempt destinations are distinct direct children of `output_root`.
- `fit_timeout_seconds`: reviewed bound 2400–7200; `replay_timeout_seconds`:
  reviewed bound 60–1800. Timeouts retain failures and do not trigger retries.

This binds all 50 cells to their expected input bytes before execution.
Admission checks raw-image hashes for the selected fold's complete train,
validation and test membership, then runs the unchanged original CPU preflight.
Post-work metadata checks detect changes without a second full raw-payload
hash sweep. This is ordinary immutable-input operation, not an OS-enforced
read-only snapshot; the execution operator must exclude concurrent input writers.

Package/runtime checks retain Python 3.12.3, Torch 2.12.1+cu126, torchvision
0.27.1+cu126, timm 1.0.27, NumPy 2.4.4 and the unchanged helper's other package
pins. Loaded CUDA 12.6/cuDNN 91002, actual module/native files, physical GPU,
driver, precision/attention settings, deterministic flags and thread settings
are captured. Runtime equality within each band pair is required. No claim of
historical binary-environment equality is made. Actual training feasibility
and complete loaded dependency compatibility still require real preflight.

## Commands and pilot barrier

Commands require an exact externally reviewed contract SHA256. These are
prospective interfaces, not instructions to run this study now:

```text
<original-venv>/bin/python -B scripts/run_dofa_partial_band.py check-plan \
  --contract <reviewed-plan.json> --contract-sha256 <reviewed-sha256>
<original-venv>/bin/python -B scripts/run_dofa_partial_band.py fit \
  --contract <reviewed-plan.json> --contract-sha256 <reviewed-sha256> --cell 0
<original-venv>/bin/python -B scripts/run_dofa_partial_band.py replay \
  --contract <reviewed-plan.json> --contract-sha256 <reviewed-sha256> --cell 0
```

`check-plan` imports no scientific packages and checks frozen source and bound
reference records. It does not perform raw-image/model/runtime preflight.
Repeat fit/replay for cell 1. A second-band fit requires its current first-band
fit to have completed successfully; both pilots are full-budget cells. The
execution operator's acceptance file must have schema `dofa-partial-pilot-acceptance-v1`,
status `accepted`, the exact contract binding, `cells: [0,1]`,
`selection_by_test_score: false`, independent `checker`/`report` file bindings,
and `outputs` containing each cell's exact `fit` result and `replay` bindings.
The consumer requires complete process/worker outputs for all four operations.
An arbitrary JSON flag does not establish independent review; that review and
receipt creation are the execution operator's responsibility.

For cells 2–49, both fit and replay also require:

```text
--pilot-acceptance <reviewed-pilot-acceptance.json> \
--pilot-acceptance-sha256 <reviewed-pilot-acceptance-sha256>
```

Execution scheduling must account for every planned cell, stop new dispatch after
failure, preserve active bounded workers and terminal/not-reached dispositions,
and retain the outer controller exit. The consumer supervises one cell under
an exclusive GPU lock and a deadline; its fresh child inherits the lock. It
does not implement a dispatcher, create locks or pick only passing devices.
`_fit-worker`/`_replay-worker` are internal supervised modes, not direct launch
interfaces. Outer execution receipts must cover failures before an attempt
directory exists, supervisor SIGKILL and host loss; an in-process writer cannot
guarantee terminal evidence in those cases.

## Retention, checks and interpretation

Each fit retains its selected model/head state, 50-epoch histories, selection,
test IDs/labels/probabilities/predictions/confusion and metrics, source/input
bindings, initial state/runtime, final runtime, bounded band-call evidence,
logs and started/terminal/failure records. No final-epoch, per-epoch, optimizer
or resume checkpoints are saved. A separate fresh replay restores the selected
state with strict loading and checks exact test-record equality using the same
numeric band and the original test precision. This consumer check is not
independent scientific acceptance; separate output review is still required.

If the historical metadata checker rejects a returned training history, the
consumer retains it as `returned-history-unvalidated.txt` before propagating
the original exception. This diagnostic may contain nonfinite values as text;
it is never accepted scientific JSON. Diagnostic-write failures are recorded
without replacing the original rejection. Accepted finite/history/update and
selection checks remain unchanged, and rejection publishes no successful
result or checkpoint.

Selected model states remain outside this source release. Runtime and storage
requirements depend on the supplied environment and inputs. Source inspection
does not establish execution feasibility or a completion time.

Focused synthetic checks can run with no model imports or GPU use:

```text
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES= \
python3 -B -m pytest tests/test_dofa_partial_band.py -q -p no:cacheprovider \
  --basetemp=<fresh-external-test-directory>
```

These checks exercise admission rejection, bands/precision/call budgets,
runtime/source drift, both-pilot gates, immutable output/failure handling and
inert subprocess termination. They do not establish encoder compatibility,
numerical equality, GPU memory use or completion. Clean source freeze, exact
private input plan, source review and real preflight precede scientific launch.

Report per-seed paired differences, average the five seeds within each fold,
then equal-weight means and sample SD over five folds. Test performance does
not choose the input, seed, hyperparameters or checkpoint. The same geographic
test regions have been studied before; seeds are not independent regions.
Rare-class validation support, SAR-informed map labels and the unresolved
physical convention remain separate interpretation limits.
