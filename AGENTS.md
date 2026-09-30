# CETUS release instructions

Read [README.md](README.md), [SOFTWARE.md](SOFTWARE.md) and the README of the
specific release package before editing or running anything. The active review
branch is `revision/repaired-titan-2026`, in
[draft PR 1](https://github.com/magnaprog/CETUS/pull/1). Check current Git state;
`1cc34ca7c6a1d54f01852ce6b54a848e97017d54` is the accepted snapshot before the
30 September memory update, rather than a permanent branch tip.

## Release boundaries

Root historical catalogs and `release/v1-review-20260908` retain the original
population. Repaired Titan evidence lives in `release/longitude-review-20260929`.
Common classifier, classical and VIMS records live in
`release/longitude-auxiliary-20260929`. Adaptation and training substitution
predictions, metrics and reconstruction code live in
`release/longitude-adaptation-controls-20260930`. Keep each package's catalog,
split, protocol and source identities together. Preserve historical bytes.

Public frozen inference at source `360eaea532ead477f46906d7ebe0cb5282e8b082`
completed 20 feature and 20 analysis jobs. The 51-file adaptation/control
package was independently rebuilt from saved predictions in the recorded CPU
runtime. It does not rerun model training. New public adaptation and classical
execution require the additional protocols described in SOFTWARE.md.
The release supplies neither tile arrays nor pretrained weights.

Scores measure agreement with an expert geomorphological map informed by SAR.
Five geographic folds support descriptive spreads. They do not establish
independent geological accuracy, a causal leakage effect or the mechanism of
DOFA's decline. Titan and Venus use display DN; calibration cannot be inferred
from their scaling. Preserve the DOFA identifier qualification and distinguish
common logistic regression from native SGD fitting recipes.

## Editing and validation

Keep machine paths, SSH details, private raw data and private agent memories out
of this release. Development and both paper sources belong in TitanSAR-dev.
Do not change repository visibility, Pages, tags or DOI metadata as an incidental
part of a documentation update.

Every tracked file must appear in `file_index.json` except that index and
`SHA256SUMS`. Every tracked file except `SHA256SUMS` must appear in the checksum
file. Rebuild both inventories after edits and validate the exact file sets,
hashes and `SOFTWARE_EXPORT.json` source records using
[the inventory workflow](.github/workflows/release-inventory.yml).
Its saved-prediction reconstruction uses the pinned runtime from SOFTWARE.md.
The frozen CPU tests cover synthetic inference and cache behavior; they do not
run pretrained GPU inference or adaptation training.

Write clear prose while preserving technical terms, identifiers, citations,
units, mathematical notation and necessary code punctuation. Review claims
against the actual package and its receipts. Record new material decisions in
the relevant release documentation and local project memory.
