# Class ranking from saved Titan predictions

Reconstruct average precision (AP) for each of six terrain classes in five Titan test regions. The [encoder comparison archive](../encoder-comparisons/README.md) supplies the original test probabilities, labels and splits. This calculation uses CPU and requires no model weights or image files.

## Reconstruct

From the repository root, run the following with Python 3.12.3 and NumPy 2.4.4. The reconstruction checks both versions and the hashes of its inputs.

```sh
set -eu
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset PYTHONPATH
ranking_work=$(mktemp -d)
python3.12 -B experiments/class-ranking/reconstruct.py \
  --initialization-zip experiments/encoder-comparisons/initialization-supplement.zip \
  --output "$ranking_work/output"
cat "$ranking_work/output/reconstruction-check.json"
```

Use a new output directory for each run. A successful `reconstruction-check.json` reports `reconstruction_verified`. The check compares the 360 class AP records, 140 regional summaries, 70 contrasts and 42 geographic summaries with [results.json](results.json). It also compares the names, shapes, dtypes and values of all 300 arrays in the 60 Craters curves. Execution metadata can differ between machines.

## What the scores measure

Within each region, DINOv2 uses logistic regression on features from five randomly initialized encoders and one pretrained encoder. ResNet18 trains from scratch or from ImageNet initialization at three seeds each. The archive records the learning rates and epochs chosen on validation data. The reconstruction reads the corresponding test scores for each fit.

For each class, sort all test images in the region by decreasing class probability. All tiles with the same score enter together. AP sums the gain in recall at each distinct threshold times the precision there. When all scores tie, AP equals the fraction of images with that map label. A class with no positives would have undefined AP. All six classes occur in every test region, although the five regions have only 8, 1, 14, 61 and 48 Craters tiles, respectively.

Macro AP weights the six classes equally within a fit. Results average the seeds within each region, then give each region equal weight. The geographic SD is the sample SD over five regions. ResNet18 differences compare the two initializations at the same seed. DINOv2 compares its one pretrained state with the mean over five random states. This exploratory analysis reuses previously examined test regions and measures agreement with the map labels.

## Files

- [results.json](results.json) contains the AP values and summaries. Its curve index maps each prefix in `craters-curves.npz` to the model, region, initialization and seed. Each prefix has five arrays for score thresholds, true positives, false positives, precision and recall.
- [contract.json](contract.json) defines the metrics and aggregation. [provenance.json](provenance.json) identifies the source of the saved predictions and the three original analysis modules in `scripts/`. Use `reconstruct.py` to run them together with the packaged contract.
- [Terrain plots](../terrain-plots/README.md) draws the class rankings, including the threshold curves for all six classes.

To test the comparison routines on synthetic inputs, run `python3.12 -B -m unittest -v test_reconstruct` from this directory. See [LICENSE](LICENSE) for the code and [NOTICE.md](NOTICE.md) for the data and model credits.
