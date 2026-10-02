# Additional repaired Titan results and display metadata

This metadata package depends on the unchanged sibling
`release/longitude-review-20260929`. Parent hashes in `provenance.json` bind its
23,380 tile catalog, geographic protocol, five splits and frozen results.
The producer catalog has a different byte hash because the parent export
removed private paths and legacy inline splits. Tile values and order agree.
Both catalogs and queued replay inputs retain their original bytes.

The two `results` audit reports are exact accepted bytes. Common fits use
source 3dd19a2 and classical v2 uses 54281f0. Protocol extracts retain each
source and runtime identity, original protocol digests and selected scientific
fields. The original sealed protocols remain the authority for execution.
`evidence/runs.json` omits private paths, hosts, commands and logs. Its output
hashes also identify the estimator files held with the accepted run artifacts.

`results/predictions.json.gz` shares test IDs and labels once per fold. It holds
20 common classifier vectors, 20 classical vectors, five majority vectors and
25 training-prior draws. Common features have 768 columns. Classical feature
slices are 0:13, 13:45, 45:1865 and 0:1865. Both fitted families use training-only
StandardScaler and balanced LogisticRegression, LBFGS, C=1, max_iter=2000,
seed 0 and the fixed protocol settings. Warnings and iteration caps are retained
in the audits.
The raw_intensity_statistics key describes normalized display values;
histogram and texture additionally use the existing within-image percentile
transform. Physical backscatter calibration remains outside this track.

For a 6 by 6 confusion matrix, true classes index rows and predictions index
columns. Precision is TP/predicted support, recall is TP/true support, and F1
is 2 TP/(predicted support+true support). Undefined precision is zero. Average
over all six classes, then report the equal-fold mean and sample SD, ddof=1.
Average the five prior draws within a fold first. Kappa uses the confusion
marginals. Metrics are fractions. Fold SD describes variation across geographic
partitions. All predictions refer to expert map labels. Independent geological
accuracy remains unmeasured.

VIMS metadata v3 uses source a1d5652. The sidecar and checker report retain
exact accepted bytes and their original producer catalog hash. Tile IDs and
centers link them to the parent catalog. Values are bilinear samples of a
display composite. Calibrated spectral interpretation remains outside their
scope. Finite zero and signed values are allowed under the declared source
masks. Availability records successful numeric sampling. Observation coverage
and physical composition require separate evidence. The one unavailable sample
remains null. Source CRS, affine, masks and raster hashes are retained separately.
Classifier inputs are SAR features and reference-map labels. This sidecar
supplies ancillary display context. Existing catalog VIMS fields keep their
historical values.

The verifier comes from the TitanSAR development source:
`scripts/export_repaired_auxiliary_results.py`, with its helper
`scripts/export_repaired_release.py`. Their hashes appear in `provenance.json`.
Use a development checkout containing both files, with Python and NumPy
installed. Run this command from that checkout's root:

```bash
python3 -m scripts.export_repaired_auxiliary_results --verify-only PACKAGE --parent-package PARENT
```

PACKAGE is this auxiliary directory. PARENT is the existing repaired package
at `release/longitude-review-20260929`. Use absolute paths for both directories.
These two metadata directories supply the verifier's data inputs. The development
checkout supplies the program; the CETUS data package supplies the metadata.

Verification covers indexes, parent identities, memberships and prediction
arithmetic. The bundled audits record earlier accepted probability checks,
estimator replays and VIMS raster reconstruction. Fine-tuning and adaptation
results remain outside this release.
