# File guide

Join catalogs, splits, and Selk records by tile ID. Macro recall averages five random seeds within each spatial
fold. The paper reports the sample standard deviation across the five
fold means. Paired two-sided t-tests use those five fold means as
uncertainty units. The `ci95` field in `results/final_results.json` is
the matching 95% interval on the same five fold estimates.

## Catalogs

Every sample is 128 by 128 pixels covering 45 km by 45 km. Each catalog
row keeps a tile ID, the SHA-256 of the tile file, coordinates, a label,
and summary statistics of the pixel values. Local paths were removed.

- `catalogs/titan_catalog.json`: 11,371 labeled Titan tiles from the
  USGS Cassini HiSAR global mosaic, which stores 8 bit display digital
  numbers after logarithmic stretching (`hisar_log_dn`). Classes follow
  Lopes et al. 2020: Plains, Dunes, Hummocky, Labyrinths, Lakes, Craters.
  The scores measure how closely model predictions match that map. The
  labels are expert interpretations drawn partly from the same radar
  swaths.
- `catalogs/earth_catalog.json`: 398 labeled Earth tiles from 29
  Sentinel-1 products across 13 morphological analog sites covering five
  of the six classes, processed by calibration to linear sigma0. Earth
  to Titan scores use those five classes and omit Hummocky.
- `catalogs/venus_catalog.json`: 2,945 unlabeled tiles from the Magellan
  FMAP right look global mosaic, used only to compare distributions of
  learned representations. Intensity is display digital numbers. Every
  label is -1.

## Splits

Nineteen split files. Each records the SHA-256 of the matching
catalog and maps every tile ID to train, val, test, or selk_holdout.

- `splits/titan_spatial_fold_0.json` through
  `splits/titan_spatial_fold_4.json`: five Titan spatial folds. A
  deterministic process assigns each roughly 250 km region to one fold
  while balancing class and sample counts. Adjacent regions share a
  boundary. We set aside 65 tiles from Selk, so they are held out of
  every fold. Each run uses three folds for training, one
  for validation, and one for testing.
- `splits/earth_transfer_grouped_validation.json`: Earth source set for
  transfer. Of the 398 tiles, 352 are used for transfer training.
  Earth validation holds out whole source products.
- `splits/earth_loso_*.json`: 13 files named after the morphological
  analog sites. Each holds out one site. A separate MMD check holds out
  one Earth source site at a time.

## Results

Fold summaries keep means, sample standard deviations, and values for
each fold. Prediction arrays for each tile were dropped so each file
stays under 8 MB. The Selk file still has scores for all 65 tiles.

- `results/probing_cv_results.json`: DINOv2, DOFA, CROMA, and Random
  Init. We keep encoder weights unchanged. Linear classifier and cosine
  k-nearest neighbors (k = 20). Within Titan uses six classes. Earth to
  Titan uses five. Earth to Titan MMD^2 is stored here.
- `results/baseline_results.json`: most common class, random predictions
  sampled from training class frequencies, intensity statistics,
  histograms, texture descriptors, their combination, and a small
  supervised CNN.
- `results/coral_results.json`: Correlation alignment (CORAL) after
  training on Earth and testing on Titan. MMD^2 after alignment is
  measured on the Titan features used to estimate the transformation.
- `results/finetuning_results.json`: encoder weights unchanged versus
  fine-tuning the final two transformer blocks on Titan. Fine-tuning
  measures adaptation when Titan labels are available.
- `results/label_efficiency_croma_results.json`,
  `results/label_efficiency_dinov2_results.json`,
  `results/label_efficiency_dofa_results.json`,
  `results/label_efficiency_random_init_results.json`: macro recall
  across six classes versus the number of labeled Titan training tiles.
  Split by model so each file stays under 8 MB. These linear
  classifiers use unweighted loss; the primary experiment uses class
  weights. The appendix comparison that uses 352 Titan training tiles
  uses class weights on those subsets.
- `results/venus_domain_gap.json`: MMD^2 among Earth, Venus, and Titan.
  Venus is unlabeled. Numeric fields are MMD^2 even if a summary string
  says MMD.
- `results/selk_relative_uncertainty_results.json`: 65 Selk tiles. Mean
  entropy is 0.963, mean classifier disagreement is 0.250, and the
  combined score has a mean of 0.525. These scores are relative to the
  Selk evaluation set and the 15 classifiers. A high score can reflect
  ambiguity in the expert map, gradual boundaries between terrain
  classes, image contents, or model error.
- `results/final_results.json`: SHA-256 values for the result files,
  and the protocol over five folds.

## Other files

- `README.md`: contents list, and the `SHA256SUMS` check.
- `DATA_LICENSES.md` and `THIRD_PARTY_NOTICES.md`: source terms,
  including the VIMS+ISS BSD 3-Clause notice.
- `file_index.json` and `SHA256SUMS`: file list and hashes.
- `provenance.json`: catalog hashes and which result keys were dropped.
