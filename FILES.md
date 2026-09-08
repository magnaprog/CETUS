# File guide

Join catalogs, splits, and Selk records by tile ID. For the primary
classifier results, scores are averaged over five random seeds within
each spatial fold. The paper reports the mean and sample standard
deviation across the five fold means. Paired two-sided t-tests use the
five paired fold means. In `results/final_results.json`, `protocol.ci95`
stores a text description of the confidence-interval method. For repeated
MMD estimates, `ci95` instead gives the 2.5th and 97.5th percentiles of the
resampled values, describing variation across those samples.

## Catalogs

Every sample is 128 by 128 pixels covering 45 km by 45 km. Each catalog
row keeps a tile ID, the SHA-256 of the tile file, coordinates, a label,
and summary statistics of the pixel values. Local paths were removed.

- `catalogs/titan_catalog.json`: 11,371 labeled Titan tiles from the
  USGS Cassini HiSAR global mosaic, which stores uncalibrated 8-bit
  display digital numbers after logarithmic stretching (`hisar_log_dn`).
  Classes follow Lopes et al. 2020: Plains, Dunes, Hummocky, Labyrinths,
  Lakes, Craters.
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
  while balancing class and sample counts. Adjacent regions have no
  exclusion buffer, so nearby tiles across fold boundaries may remain
  correlated. We set aside 65 tiles from Selk, so they are held out of
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
- `results/coral_results.json`: correlation alignment (CORAL) evaluated
  for transfer from Earth to Titan. MMD^2 after alignment is
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
- `results/venus_domain_gap.json`: squared maximum mean discrepancy
  (MMD^2) among Earth, Venus, and Titan. Venus is unlabeled. The MMD
  statistics report MMD^2 even where summary strings say MMD. Other
  fields contain frequencies, sample counts, kernel parameters, and
  A-distance. Its comparison pool includes all Earth tiles and differs
  from the Earth training pool used for transfer classification.
- `results/selk_relative_uncertainty_results.json`: 65 Selk tiles. Mean
  entropy is 0.963, mean classifier disagreement is 0.250, and the
  combined score has a mean of 0.525. These scores describe uncertainty
  and disagreement in terrain predictions, relative to the Selk
  evaluation set and the 15 classifiers. Hazard assessment and
  operational planning are outside their scope. A high score can reflect
  ambiguity in the expert map, gradual boundaries between terrain
  classes, image contents, or model error.
- `results/final_results.json`: SHA-256 values for the result files,
  the number of folds, the role of random seeds, and the confidence-interval
  method.

## Additional experiment records

These files contain existing measurements for the paper's training and
preprocessing comparisons. Recall values are fractions; multiply by 100
for percentages. A difference between recall fractions must also be
multiplied by 100 to obtain percentage points. Match records across
files using model, fold, and experimental condition.

- `results/constrained_training_results.json`: within-Titan training
  with uniform or stratified sampling at several tile counts, including
  352. Records retain the training-count settings and the five seed
  scores. The reported recall covers the five Earth-supported classes.
  This is separate from the unweighted six-class label-efficiency curves.
- `results/display_rendering_results.json`: transfer recall for the
  original Earth preprocessing and the display-DN renderings. Undefined
  Hummocky recall is stored as `null`. Retained `mmd` values use a single
  estimate; the paper's repeated MMD estimates are in
  `preprocessing_mmd_results.json`.
- `results/quantization_results.json`: transfer recall after changing
  the number of gray levels while retaining the original normalization
  endpoints. Its `mmd` values use repeated estimation.
- `results/smoothing_results.json`: transfer recall and Earth validation
  recall for output-grid smoothing conditions, including smoothing
  combined with display rendering. Earth validation uses a fixed split;
  repeated entries across Titan folds are not independent measurements.
  The file also includes image texture statistics and the finer smoothing
  sweep. Retained `mmd` values are single estimates. Smoothing on the
  output grid is not a simulation of multilooking.
- `results/preprocessing_mmd_results.json`: MMD^2 measurements for the
  preprocessing comparisons. Use `mmd_repeated` for the paper's repeated
  estimates. A separate set of records compares re-estimated bandwidths
  with a fixed bandwidth. Keep these estimators separate when computing
  changes from the baseline.
- `results/source_site_sensitivity_results.json`: transfer recall after
  omitting one Earth training site at a time. `NONE` identifies the full
  source pool. Records include training counts and whether a class was
  lost. Average over Titan folds for each omitted site before calculating
  the spread across sites. This describes sensitivity to source-site
  omission. A jackknife standard error and the standard deviation across
  Titan folds are different statistics.

### Finding evidence for the paper

- Main classification scores and per-class recall: `probing_cv_results.json`.
- Comparisons with image-statistics classifiers and the CNN:
  `probing_cv_results.json` and `baseline_results.json`. Pair fold means
  after averaging seeds within each fold; use the paper's stated test
  families for multiple-comparison correction.
- Constrained-training comparison: `constrained_training_results.json`,
  with Earth transfer scores from `probing_cv_results.json`.
- Rendering, bit-depth, and smoothing comparisons: their corresponding
  files above, with repeated and fixed-bandwidth MMD^2 records in
  `preprocessing_mmd_results.json` where applicable.
- Sensitivity of transfer recall to Earth source composition:
  `source_site_sensitivity_results.json`. The MMD site-omission analysis
  is separate and remains in `probing_cv_results.json`.
- Alignment, fine-tuning, learning curves, Venus, and Selk: the result
  files described in the preceding section.

Filenames in this subsection are relative to `results/`. Source filenames
inside the added JSON files identify original measurement files that are
external to this package.

## Other files

- `README.md`: contents list, and the `SHA256SUMS` check.
- `DATA_LICENSES.md` and `THIRD_PARTY_NOTICES.md`: source terms,
  including the VIMS+ISS BSD 3-Clause notice.
- `file_index.json` and `SHA256SUMS`: file list and hashes.
- `provenance.json`: catalog hashes and which result keys were dropped.
