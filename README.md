# CETUS

Cross-domain Evaluation of Earth-to-Titan Transfer Using SAR. This private
repository contains scientific metadata, results and the frozen Titan evaluation
source. [SOFTWARE.md](SOFTWARE.md) describes installation, required assets and
commands. Image tiles, raw source products, encoder weights and embeddings are
supplied separately. There is no public tile-archive endpoint or project archive DOI.

The software preserves scientific source from development commit `a9c17ad`.
Its CPU tests cover normalization, cache checks, model identity and a small
synthetic evaluation. Full inference through this checkout remains a release
acceptance task. The [export manifest](SOFTWARE_EXPORT.json) records source
hashes. This source package currently supports frozen Titan evaluation;
classical baselines and adaptation need separate public protocols.

The September 29 correction separates two dataset identities:

| Contents | Status and population |
| --- | --- |
| Root `catalogs/`, `splits/`, and `results/` | Historical v1: 11,371 Titan tiles, 398 Earth tiles, and 2,945 unlabeled Venus tiles. Original scientific payloads remain unchanged for existing links. Results describe the retained v1 population and its known limitations. |
| [Original review bundle](release/v1-review-20260908/README.md) | All 46 files from the September 8 repository tree, preserved byte-for-byte with their original index and checksums. Historical descriptions in that snapshot are superseded by the correction notice. |
| [Repaired Titan package](release/longitude-review-20260929/README.md) | 23,380 Titan tiles, including 140 Craters; five contiguous geographic folds; verified frozen DINOv2, DOFA, CROMA, and RandomInit probe/kNN results with predictions and counts. Earth transfer and repaired fine-tuning are not included. |

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
sample standard deviations. These checks do not rerun model inference or
establish independent geological truth. Fine-tuning results at the root belong
to historical v1; no completed repaired fine-tuning experiment is claimed here.

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
