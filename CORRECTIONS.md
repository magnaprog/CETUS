# Scientific corrections, September 29, 2026

The root catalogs, splits and result files retain the historical v1 payloads.
The exact earlier review package is preserved under
[release/v1-review-20260908/](release/v1-review-20260908/README.md).
Its valid checksums establish file identity, not scientific correctness.
This correction does not relabel historical Earth data or replace v1 results.

**Titan catalog and evaluation.** A longitude-wrap error treated part of the
0 to 360 degree label raster as nodata, omitting valid western terrain. The repaired
track, `TitanSAR-HiSAR-IMG-longitude-review-20260929`, contains 23,380 tiles.
All original 11,371 IDs remain with unchanged pixels and centers; 12,009 IDs
are added. Fifteen retained class labels and 1,727 retained modal fractions
change. The repaired class counts, in order Plains, Dunes, Hummocky, Labyrinths,
Lakes, Craters, are `[15433, 4050, 2764, 498, 495, 140]`.

The [catalog comparison](release/longitude-review-20260929/evidence/catalog_comparison.json)
records exact added IDs, transitions, and fraction changes. The
[label verification](release/longitude-review-20260929/evidence/label_verification.json)
contains candidate-grid map counts and the saved-catalog checks. Tile IDs index
the original enumeration; class voting is modal over valid map samples, with
the lowest class index breaking ties. Votes are not masked to SAR-valid pixels,
and there is no minimum modal fraction. Expert-map agreement is the target;
these labels do not provide independent geological truth. Under the source
ontology, Lakes includes filled and empty lakes and seas, and Hummocky includes
mountains. See [Lopes et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7271969/)
and the [source map dataset](https://data.mendeley.com/datasets/f6jrtyfp66/1).

The [new protocol](release/longitude-review-20260929/splits/protocol.json) uses
five contiguous longitude sectors and footprint gaps between active roles.
It reserves 65 Selk tiles and a further 69 Selk-buffer tiles, leaving a test
union of 23,246 tiles. Four frozen encoders supply five probe-head seeds per
fold and one kNN evaluation per fold. Results retain all six classes. The
[Crater support evidence](release/longitude-review-20260929/evidence/crater_support.json)
records source polygon parts and roles; parts do not establish independent
physical craters. Fold 1 has a single Craters test tile. Fold variation is
descriptive, and neither the partition design nor artifact verification proves
spatial independence. Differences from v1 combine population and protocol changes.

**Earth source limitations.** Sixteen historical training tiles cataloged as
`wadi_rum` were acquired in Egypt, not the named Jordanian site. Their Labyrinths
assignment remains unvalidated. Wadi Rum is withheld from future default
acquisition pending a verified boundary and defensible landform annotation;
no replacement bounding box or inferred label is supplied here. Shoreline
checks place 42 Arctic Plains footprints entirely offshore, with ambiguous
surface state requiring acquisition-specific interpretation. Two Iceland
overlays find the examined footprint mostly ocean. Ocean location alone does
not distinguish bare water from sea ice and does not justify a replacement
landform label. Twenty-one validation footprints overlap training footprints.
Corrected Earth transfer needs expert annotation and strict source-site
isolation. Source-cap controls keep their historical class definitions.

Future Earth preparation also corrects a half-source-pixel center offset and
records the source window. Existing catalog coordinates and arrays are retained
as historical outputs. That correction does not explain the large offshore
locations, change terrain labels, or repair old split overlap.

**Model conditioning and incomplete experiments.** The original v1 probe
extraction used DOFA's numeric SAR identifier 13.78 for both domains. Earth
uses 5.405 under the convention in the authors' [repository example](https://github.com/zhu-xlab/DOFA/blob/73ff5c0721da322689ae890bed5b4efd78935f47/README.md). Their [paper's version 2, Appendix D](https://arxiv.org/html/2403.15356v2#A4) instead gives 3.75 for Sentinel-1. The released checkpoint contains no identifier history, so that pretraining detail remains unresolved. Independently generated preprocessing control arrays need separate provenance checks. The repaired within-Titan
evaluation uses 13.78 and records encoder revisions, checkpoint hashes and
normalization. It does not correct Earth transfer. Repaired fine-tuning is
excluded from this package; completion requires separate validation of all
75 planned fits before a subsequent release can claim those results.

**Historical figure index.** A separate earlier software-side v1 artifact
index retained the figure-manifest digest
`95086c224d114bbadc3b2ebd4466693350d45269618e82ef91b313c9a66e92e6`
after the distributed manifest changed to
`792a253b8149f2392f7b441210f4c83d9cb12638339b302828fbe4f3686e7f7c`.
This stale nested entry is distinct from the valid checksum chain of the
46-file review bundle preserved here. Frozen bytes are not rewritten.

This branch prepares a private data-only correction. It does not publish image
tiles, provide runnable software, register a DOI, or change repository visibility.
