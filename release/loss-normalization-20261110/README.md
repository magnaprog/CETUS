# Loss normalization in Titan terrain classification

This companion compares two ways to divide the weighted cross entropy in
ResNet18 training. Each comparison uses the same training images, initial
weights and training settings. The original runs divide by the sum of class
weights in each batch. The additional runs divide by batch size times the mean
class weight across the training images.

The files contain the fifteen additional runs, covering five geographic regions
and three random seeds. The original fifteen runs, map labels and geographic
divisions come from `initialization-supplement.zip`, available in the same
CETUS release. Its SHA256 is
`ada744e9278f4fbac5f5f953b5688132ad0b080b35cff32ed23b7f02c3192486`.
Keep that ZIP intact. The reader opens its data directly without executing its
Python files.

## Reproduce the comparison

Use Python 3.11 or later on Linux. The reader uses the Python standard library.
From the unpacked companion directory:

```sh
python3 -B reconstruct.py \
  --initialization-zip /path/to/initialization-supplement.zip \
  --output /path/outside/this/directory/comparison.json
```

Choose a new output filename. The reader checks file identities, all fifteen
comparisons and the independently calculated answers before writing the result.
Keep the unpacked directory unchanged; put outputs beside it. The `-B` option
prevents Python from adding cache files to that directory.

Each run contributes its selected test predictions and the validation scores
from all one hundred epochs. Selection uses the earliest epoch with the highest
unrounded validation macro F1. Integer class counts reproduce precision,
recall, F1 and macro F1. Logits reproduce the saved predicted classes. Floating
point metric comparisons allow an absolute difference of `8 * 2**-53`, with zero
relative tolerance; this allowance does not merge validation scores during epoch
selection.

The output retains all six terrain classes for each run. Differences in Craters
F1 and macro F1 are averaged across the three seeds in each region, then equally
across the five regions. Multiply differences by one hundred to express them in
percentage points. Internal fold indices zero through four correspond to
Regions 1 through 5 in the paper.

This comparison measures sensitivity to the loss denominator under the stated
training settings. The learning rate comes from the original comparison, and
the same test regions have been examined in earlier analyses. The three seeds
represent repeated training in each region. They do not provide fifteen
independent geographic tests.

## Contents

`records.json.gz` contains predictions, logits and compact validation histories.
`expected.json` contains answers from the independent numerical check.
`provenance.json`, `manifest.json` and `SHA256SUMS` identify the inputs and files.
`reconstruct.py` performs the CPU calculation. `test_reconstruct.py` uses
invented examples to test missing runs, class counts, selection and file checks:

```sh
python3 -B -m unittest -v test_reconstruct
```

The compact histories retain loss numerators and denominators, so their stored
ratios can be checked. Repeating training or GPU prediction from model weights
is a separate operation. This companion omits images, model weights, per-batch
training records and the complete validation prediction vectors at every epoch.
The accompanying execution-source package contains the training code and its
original input requirements.
