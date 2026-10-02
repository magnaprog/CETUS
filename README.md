# CETUS

Cross-domain Evaluation of Earth-to-Titan Transfer Using SAR. This private
repository contains scientific metadata, results and the frozen Titan evaluation
source. [SOFTWARE.md](SOFTWARE.md) describes installation, required assets and
commands. Image tiles, raw source products, encoder weights and embeddings are
supplied separately. There is no public tile-archive endpoint or project archive DOI.

The software preserves scientific source from development commit `0257a36`.
Its CPU tests cover normalization, cache checks, model identity and a small
synthetic evaluation. The forty-job frozen replay through immutable checkout
`360eaea` completed on September 30, 2026. Its twenty extraction and twenty
analysis jobs reproduced the native feature arrays, probe probabilities,
predictions and saved metrics exactly in the recorded environment.
[SOFTWARE.md](SOFTWARE.md) gives the verification identity and scope.
The [export manifest](SOFTWARE_EXPORT.json) records source hashes. This source
package supports frozen Titan evaluation. The new sibling package includes
CPU reconstruction code; classical and adaptation training commands remain in
the development repository.

The versioned packages separate historical v1 from the repaired Titan population:

| Contents | Status and population |
| --- | --- |
| Root `catalogs/`, `splits/`, and `results/` | Historical v1: 11,371 Titan tiles, 398 Earth tiles, and 2,945 unlabeled Venus tiles. Original scientific payloads remain unchanged for existing links. Results describe the retained v1 population and its known limitations. |
| [Original review bundle](release/v1-review-20260908/README.md) | All 46 files from the September 8 repository tree, preserved byte-for-byte with their original index and checksums. Historical descriptions in that snapshot are superseded by the correction notice. |
| [Repaired Titan package](release/longitude-review-20260929/README.md) | 23,380 Titan tiles, including 140 Craters; five contiguous geographic folds; verified frozen DINOv2, DOFA, CROMA, and RandomInit probe/kNN results with predictions and counts. Earth transfer and repaired fine-tuning are not included. |
| [Additional repaired results](release/longitude-auxiliary-20260929/README.md) | Common-classifier and classical v2 results, all 70 prediction vectors, and corrected VIMS v3 display metadata. The package uses the repaired parent catalog and folds. VIMS values remain separate from classifier inputs. |
| [Adaptation and training substitution](release/longitude-adaptation-controls-20260930/README.md) | Saved predictions, histories, checkpoint-replay evidence and summaries for 75 repaired adaptation fits, with SGD and common LR training-substitution controls. CPU code reconstructs metrics, tables, figures and a finite manuscript claims index. |
| [Interpretation analyses](release/interpretation-20261001/README.md) | Post hoc class contrasts, aggregation sensitivity, boundary exposure, map-part support and deterministic input examples. Separate CPU reconstruction uses the accepted sibling predictions and preserves equal-fold primary results. |

Read [CORRECTIONS.md](CORRECTIONS.md) before interpreting the historical
results. The v1 longitude sampler omitted valid western terrain. The repair
retains original pixels, adds 12,009 tile IDs, and changes 15 retained labels.
Catalog and split hashes identify the correct track; joining by tile ID alone
across versions can mix labels. Both population and partition design change in
the repaired evaluation, so score differences cannot isolate either change.

Earth source labels still require expert review. Wadi Rum tiles came from
Egypt; some Plains footprints are offshore, and Earth training/validation
footprints overlap. Historical source-cap and preprocessing controls remain
sensitivity analyses. This package assigns no new Earth labels.

The repaired evaluation includes all 100 probe heads and 20 kNN evaluations.
Its [prediction arrays](release/longitude-review-20260929/results/predictions.json.gz)
and [metrics](release/longitude-review-20260929/results/frozen_results.json)
allow reconstruction of confusion matrices, per-class scores, fold means, and
sample standard deviations. Reconstruction from saved predictions does not
establish independent geological truth. The separately verified forty-job
replay repeated frozen inference and probe fitting. Fine-tuning results at the
root belong to historical v1; completed repaired adaptation results belong to
the new sibling package.

The additional package reports precision, recall and F1 for four encoder and
four classical feature families under one fixed logistic classifier. It also
includes majority and class-prior controls. Its predictions reconstruct all
70 confusion matrices across 23,246 distinct test tiles. The corrected VIMS
sidecar has 23,379 available numerical samples and one unavailable edge sample;
these values describe a display composite and do not measure composition or
mission coverage. The package README explains verification and provenance.

The adaptation package requires both repaired sibling packages unchanged.
Its comparisons describe complete training recipes and changes to training
membership. They do not isolate unfreezing or spatial leakage as causes.
[SOFTWARE.md](SOFTWARE.md#reconstruct-adaptation-and-training-substitution-results)
gives the CPU verification command and tested runtime.

The native export of the adaptation package completed from development commit
`3f4a6a4`. The SHA256 of its `evidence/acceptance.json` is
`7340fcc5d1fdc52b27ff3c0385d86669df97345897f95efa92bcb795dba5ba03`. Use this externally supplied identity
with the CPU verifier. Independent CPU verification and reconstruction completed
with exit zero in the isolated PyPI environment described in SOFTWARE.md.
All 51 package files, totaling 10,201,553 bytes, matched the native export
exactly, and all three input packages remained unchanged. This check includes
the displays, finite claims index and inventories. The frozen GPU replay used
`360eaea`, a separate source identity.

The October interpretation package comes from development source `0297c9a`.
It reports how class errors and weighting affect the description of the same
accepted runs. CROMA's adaptation contrast changes from -0.17 to +0.85 F1
points under a specified pooled estimator, while only 1,136 of 23,246 test
centers meet the boundary-exposure rule. These are descriptive comparisons.
The package README provides CPU reconstruction, numeric tolerances and an
optional exact figure check with separately supplied tile arrays. Earlier
release files and scientific producer identities remain unchanged.

[FILES.md](FILES.md) explains the root historical files. Source terms remain in
[DATA_LICENSES.md](DATA_LICENSES.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), including the VIMS+ISS notice.
Those terms do not imply that a processed image archive has been published.

Run `sha256sum -c SHA256SUMS` from the repository root to check the entire
versioned payload. Ignored environments and local outputs are excluded. Each versioned package also has its own `SHA256SUMS`, checked
from that package's directory. At each level, `file_index.json` covers all
payload files except itself and `SHA256SUMS`; the checksum list also covers
`file_index.json`. Git internals are excluded. The root index distinguishes
the historical root catalogs from the repaired catalog.
