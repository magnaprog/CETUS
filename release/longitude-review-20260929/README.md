# Repaired Titan frozen evaluation

This data-only package contains the 23,380-tile repaired Titan catalog and a
verified frozen-encoder evaluation. It contains no image tiles, code, encoder
weights, embeddings, or fine-tuning results. The repository is private at
preparation; no public tile archive endpoint or project DOI is assigned here.

Use `catalogs/titan_catalog.json`, the five `splits/titan_contiguous_fold_*.json`
manifests, and `splits/protocol.json` together. Assignments in the manifests
define membership. Inline legacy splits were removed from the exported catalog.
The catalog's SHA-256 changes because private paths and legacy split fields
were removed; all other selected scientific values and tile order are retained.
Source file hashes in `provenance.json` identify the inputs before sanitization.

`results/frozen_results.json` retains all four model identities, 100 probe-head
confusion matrices, 20 kNN confusion matrices, normalization records, class
support, per-class metrics, and fold summaries. `results/predictions.json.gz`
contains the corresponding test IDs, true labels, five probe prediction vectors
and one kNN vector for every model/fold. Probabilities and feature arrays are
not included; their original validation is recorded, not repeated here.

Reconstruct a 6 by 6 confusion matrix with true classes on rows and predicted
classes on columns. For each class, precision is TP / predicted support,
recall is TP / true support, and F1 is 2 TP / (predicted + true support).
Undefined ratios are zero. Average each metric over all six classes. Average
the five probe heads within each fold, then report the equal-weight mean and
sample standard deviation (ddof=1) of the five fold values. kNN has one vector
per fold and k=20. Cohen's kappa uses the confusion-matrix marginals. Saved
metrics are fractions, not percentages. Fold SD is descriptive variation,
not a confidence interval; overlapping training sets are not independent trials.

`evidence/catalog_comparison.json` records membership and label changes from
v1. `evidence/label_verification.json` retains the original candidate-grid class
vote counts: numeric tile-ID suffixes index that array; the modal class is the
smallest class index attaining the maximum, and its fraction is rounded to four
decimals. These votes include valid map pixels without a SAR-validity mask.
`evidence/crater_support.json` records source polygon parts, sample support,
and split roles. Parts do not establish independent physical craters. Fold 1
has only one test Craters tile. Labels reproduce an expert map, not independent
geological truth. Lakes includes filled and empty basins under that ontology.

The repaired population and geographic partitions both differ from v1; score
changes cannot isolate either change. Earth transfer remains outside this
evaluation and requires expert source-label curation. No new Earth labels are
assigned here. Software export and a usable tile archive remain pending.
Source-product terms are in the repository's DATA_LICENSES.md and
THIRD_PARTY_NOTICES.md; software licensing does not replace those terms.

Run `sha256sum -c SHA256SUMS` from this directory. `file_index.json` covers every
payload except itself and SHA256SUMS; SHA256SUMS also covers file_index.json.
There are no timestamps or checkout paths in generated package metadata.
