# Classifier fitting study

This exploratory study asks how feature standardization and L2 regularization
change classification of Titan terrain with the accepted DINOv2, DOFA, CROMA
and untrained transformer features. It uses the five existing geographic folds.
Their test results were already examined when this study was designed.

The design has 80 fits: four encoders, five folds, raw or standardized features,
and C equal to 1 or infinity. Every classifier is newly fitted in the same CPU
environment. Earlier classifiers remain separate comparisons. Folds use IDs
0 through 4 in files and numbers 1 through 5 in the paper.

## Fitting settings

The environment uses Python 3.12.3, NumPy 2.4.4, SciPy 1.17.1 and
scikit-learn 1.9.0. Each process uses one numerical thread; at most four
processes run together. The estimator is `LogisticRegression` with LBFGS,
balanced class weights, an intercept, `l1_ratio=0`, `max_iter=2000`,
`tol=1e-4`, `random_state=0` and `warm_start=False`.

The saved features and fitting matrices retain float32 precision. StandardScaler
fits unweighted means and population variances using training rows alone.
Its float32 transform and constant-column handling remain unchanged. No
validation search, feature removal or test-dependent refit is part of this study.

For class weights q and their sum S, the mathematical loss is the weighted mean
multinomial negative log likelihood plus `||W||² / (2 C S)`. The intercept has
no penalty. C equal to infinity removes the penalty. Standardizing features
changes the L2 penalty's geometry as well as numerical conditioning. With
infinite C, the affine classifier family is the same in exact arithmetic;
finite computations can still return different predictions. A finite
unpenalized maximum likelihood estimate need not exist.

The earlier PyTorch classifier divides by the class weight sum separately in
each minibatch. Its initialization, updates and arithmetic also differ.
This study therefore cannot isolate the optimizer's contribution to the
earlier change in encoder rankings.

## Numerical checks

Two calculations accompany each fit. `mathematical_float64` independently
evaluates the smooth loss and gradient after promoting the saved values to
float64. `solver_oracle` reuses the pinned library's loss at the saved
parameters in the fitting dtype, including rounded weights and their native
sum. SciPy forwards trial parameters in the dtype of the initial vector in
this configuration. The two calculations can disagree through rounding.
The library replay is not an independent implementation or a record of why
the optimizer stopped.

Both gradients are also expressed in coordinates defined by the training
scaler. For raw features, the slope gradient becomes
`(G_W - g_b * mean) / scale`; the intercept gradient stays the same.
Each calculation retains its own penalty. These transformed residuals are
screening quantities and do not certify optimality. Residuals above `1e-4`,
warnings and iteration limits remain visible in the results. They never
trigger a change in settings or an automatic retry.

Predictions are reconstructed with the original input and coefficient dtypes.
Both training and test predictions must agree with the estimator. Probability
rows must be finite, lie between zero and one, and sum to one within `1e-6`
for float32 or `1e-12` for float64. Differences from float64 calculations,
including changed argmax decisions, are retained separately.

## Running and accepting results

The private JSON plan identifies the clean producer commit, source files,
actual imported implementations, numerical libraries, accepted input
specification, feature directory, output directory and all 80 cells. Its
SHA256 is supplied separately on the command line. These records are necessary
for reruns and belong with the code and research records, not in paper prose.

```bash
python scripts/launch_fitting_sensitivity.py --plan PLAN.json \
  --plan-sha256 SHA256 --workers 1 --cells dinov2-fold-0-raw-none
```

Check this first run's execution and serialized predictions before launching
the remaining declared cells with at most four workers. This check concerns
implementation, not whether the scientific score is favorable. Never rerun
into an existing output directory. Preserve unsuccessful attempts and the
reason for any later attempt.

```bash
python scripts/launch_fitting_sensitivity.py --plan PLAN.json \
  --plan-sha256 SHA256 --check-all
```

The completion check requires all 80 cells, process exits, matching plan and
source identities, complete results and the expected array hashes. A failed
or missing cell cannot become a zero score or disappear from a five-fold mean.
Execution checks alone do not establish numerical correctness: a separate
reader must reconstruct predictions from serialized coefficients, scaler
values and authenticated features, then recompute confusion matrices and
class metrics before any result enters the manuscript.

Compare all five regional differences and equal-fold means with sample
standard deviations. Keep class precision, recall and F1, confusion counts
and predicted class proportions. Compare scaling at each C, regularization
at each scaling setting, and their difference of differences within each
encoder and fold. Encoders are not independent geographic replicates.

The study measures agreement with the expert geomorphological map. It does
not establish independent geological accuracy, the causal effect of pretraining,
the best regularization setting or canonical DOFA performance. The accepted
abstract and earlier papers remain unchanged.
