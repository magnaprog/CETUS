# Current Titan execution source, 9 November 2026

This versioned package makes the authored CNN, matched-DINO and 80-fit CPU
execution code available for inspection and preparation of new runs. It exports
development revision `9089214a5a6134a088eed058b02df6e021fe80dc`, including its
anonymous-input contract and output-directory protection. It contains all 23
`titansar` files, nine selected scripts, eight test files, five split fixtures,
source metadata, three guides and the required notices. The earlier CETUS root
runner has a separate source identity and remains unchanged.

Complete imagery, feature caches, weights, trained states and historical
machine-bound plans are absent. Source availability does not establish a
turnkey historical run or independent-recipient model reproduction. The four
[numerical companions](../titan-followup-20261109/README.md) provide the separate
portable saved-result workflow. Their [access inventory](../titan-followup-20261109/DATA_ACCESS.md)
states which inputs are supplied. No public endpoint for the complete processed
image collection is established by this export.

| Entry surface | Included implementation | Execution prerequisite or limit |
| --- | --- | --- |
| One-job preparation/execution | [run_model_followup.py](scripts/run_model_followup.py), support code, ResNet18 and official-source matched-DINO builders, training/evaluation and C=1 readout | Authenticated local inputs/assets and a new clean-source/runtime specification; preparation is the default and `--execute` runs the declared job. |
| Existing 80-fit study | [run_fitting_sensitivity.py](scripts/run_fitting_sensitivity.py) and [launch_fitting_sensitivity.py](scripts/launch_fitting_sensitivity.py) | A separately prepared plan, accepted protocol, complete original-format feature caches and matching runtime. The private plan builder is not included. |
| Historical model-grid and coverage analysis | [analyze_model_followup.py](scripts/analyze_model_followup.py), [analyze_model_followup_coverage.py](scripts/analyze_model_followup_coverage.py), [analyze_sar_coverage.py](scripts/analyze_sar_coverage.py) | Original catalog/split and receipt identities; the full-grid analyzer does not accept the anonymous catalog unchanged. Use the numerical companions for portable saved-result arithmetic. |
| Remaining package and helpers | All `titansar` modules, plus run_finetuning.py and summarize_review_frozen.py | Preserved import closure and historical facilities. This export does not claim to support every legacy CLI or preprocessing path. |

## Read the contracts first

[CONTRACTS.md](CONTRACTS.md) identifies model assets, source/runtime seals,
input identities, omitted supervision and test fixtures. The dated source guides
are [model execution](docs/MODEL_FOLLOWUP.md), [CPU fitting](docs/FITTING_SENSITIVITY.md)
and [artifact access](docs/CVPR2027_DATA_ACCESS.md). Scientific settings,
example code and historical outcomes retain the upstream wording. Links in
two guides point to existing public files or qualified sections of CONTRACTS.md.
The access guide also corrects the count to sixty CNN validation histories
containing 6,000 epoch records and records verified public access to the saved
numerical ZIP. Full imagery and model weights remain separate requirements.
The fitting guide is byte-identical.
[The destination map](CONTRACTS.md#guide-destination-map) identifies
every omitted document or script behind those links.

Statements in the dated guides about review status, earlier preparation phases
and author-side checks describe those records. They are not new validation of
this package. Their `/srv/cetus` and `/opt/cetus` locations are examples to
replace. The five historical split files retain the exact generic
`/tmp/cetus-repaired-titan-catalog.json` provenance field, which is not a public
download route. The original guide bytes remain identified by their upstream
commit and `source_sha256` in SOURCE_EXPORT.json.

## Source and environment identity

The original 90-cell scientific producer is
`1f3fc8fe0e831777e3188d70caf2c5450b656333`; the original 80-fit producer is
`56b3356a937021b91152de40a203ccc871db4c59`. The two CPU execution scripts retain
those original producer bytes. The 9089214 maintenance export does not replace
either scientific producer or the later saved epoch-100 evaluation producer.

Use a separate environment and run from this package root. Both this package
and the older frozen runner use the `titansar` name. `pyproject.toml` installs
only `titansar*`; selected scripts are source-tree modules. The empty
`scripts/__init__.py` is a packaging addition to avoid namespace collisions;
it is not a source file from 9089214. New jobs must bind the actual clean public
execution commit, imported files and runtime. Git can discover the enclosing
CETUS checkout from this directory. A directory extracted without Git does
not satisfy the existing execution guard. Do not bypass it.

The preserved dependency lower bounds are historical metadata, not a tested
runtime lock. The model guide records Python 3.12.3, Torch 2.11.0+cu130 and
torchvision 0.26.0+cu130 for earlier local CPU checks; original scientific
model execution used Torch 2.12.1+cu126 and torchvision 0.27.1+cu126. NumPy was
2.4.4. The CPU fitting study used scikit-learn 1.9.0 and SciPy 1.17.1.
[Runtime details](CONTRACTS.md#runtime-and-third-party-dependencies) distinguish
these environments and the deferred preprocessing dependencies. The local CPU
checks used an existing environment. Hosted CI installed dependencies in a
fresh CPU environment and ran the synthetic tests; it did not load the actual
model assets.

## Integrity and validation

The package has 60 files. [SOURCE_EXPORT.json](SOURCE_EXPORT.json) records each
upstream/current hash, the exact guide projections, notices, generated files
and scientific producer identities. [file_index.json](file_index.json) covers
every file except itself and SHA256SUMS; SHA256SUMS covers every other file,
including the index and source manifest. From this directory:

```sh
sha256sum -c SHA256SUMS
```

Source, inventory and link inspection are separate from runtime validation.
The [focused validation instructions](CONTRACTS.md#validation) preserve the
seven test modules and their real metadata fixtures. Supply the unchanged
scientific ZIP in a fresh temporary test copy at the path those tests expect;
it is not duplicated in these 60 files. Commit that external copy and fixture
locally before testing: synthetic tests also read Git HEAD. This test commit
is separate from scientific producer identities. Default actual-model integration tests
skip unless explicitly enabled with local assets. Synthetic tests exercise
small CPU networks and classifiers; they do not repeat the scientific fits.

The [hosted CPU run](https://github.com/magnaprog/CETUS/actions/runs/42614128106)
on 9 November passed 224 tests and 15 subtests across all seven modules.
Three tests that require actual model assets were skipped. All 40 upstream
Python files remained unchanged. [Validation scope and runtime](CONTRACTS.md#validation)
describe the earlier local attempts and the test setup. These checks use
synthetic CPU models and classifiers; the numerical companions provide the
separate reconstruction of scientific results.

## Attribution

Author-owned code is by Kevin Lee under [MIT](LICENSE). Preserve the vendored
[CROMA MIT notice](licenses/CROMA-LICENSE), [data terms](DATA_LICENSES.md) and
[third-party notices](THIRD_PARTY_NOTICES.md). DINO source, torchvision and
weights are separately provisioned under their own terms. These notices do
not establish access to omitted inputs.

The experiments measure agreement with a SAR-informed terrain map. Regional
spreads are descriptive. Preserve the documented test reuse, sparse classes,
optimization limits and unresolved DOFA input convention; source publication
does not establish independent geological accuracy or causal pretraining effects.
