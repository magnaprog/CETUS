# CETUS release instructions

Read [the paper reproduction guide](PAPER_REPRODUCTION.md) for the methods,
software identities and data routes. The scientific packages retain their
original bytes.

Maintainers should read [README.md](README.md), [SOFTWARE.md](SOFTWARE.md) and
the relevant package README before editing or running commands. Check current
Git and PR state before integration; start subsequent work from updated `main`.
The September/October release and its earlier documentation revisions retain
their history. Do not resume an old review branch from a dated instruction.

The [November saved-results package](release/titan-followup-20261109/README.md)
contains four immutable archives. Read its access inventory, notice and
provenance before changing the public wrapper. The 90-case initialization
study and thirty saved epoch-100 evaluations are complete. Later document
commits do not replace those scientific producers. The separate
[current source export](release/current-source-20261109/README.md) retains
its own source manifest and execution contracts. Keep it separate from the
earlier root runner and from the portable saved-result consumers. Source
availability does not supply the omitted scientific inputs or private supervision.

The October interpretation package is `release/interpretation-20261001`.
Its README and source export identify CPU producer `0297c9a` and distinguish
post hoc analyses from the equal-fold primary results. Development handoffs
retain their own scientific producer identities and are not public release
instructions. Keep machine-specific continuation paths outside this repository.

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
The November archives supply sixty SAR inspection packets and thirty-seven
SAR/map excerpts; these counts are not a claim of disjoint samples. Full
catalog pixels, embeddings and model states are not included. The earlier
frozen runner still requires separately supplied tiles and weights.

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
Use each package's documented saved-result entrypoints with the pinned runtime
from SOFTWARE.md. Keep extracted inputs unchanged and put reports, environments
and caches outside them. Check archive hashes against the wrapper provenance;
the reassessment archive uses MANIFEST.json rather than SHA256SUMS. The
initialization verifier rejects extra files. Retain the older reconstruction
checks and their documented scope, including the historical private-helper-hash
test failure.
The frozen CPU tests cover synthetic inference and cache behavior; they do not
run pretrained GPU inference or adaptation training.

Write clear prose while preserving technical terms, identifiers, citations,
units, mathematical notation and necessary code punctuation. Review claims
against the actual package and its receipts. Preserve full licenses, source
credits, copyright holders and truthful AI-assistance disclosures. Anonymous
review wording inside the four ZIPs describes their original export; explain
later status and author attribution in the public wrapper without rewriting
the archives. Record material decisions in the relevant release documentation.
