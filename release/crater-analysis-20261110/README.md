# Craters descriptive analysis

`analyze_crater_training.py` reproduces training-batch availability and
joins saved predictions for modal-Craters test images to accepted map-cell
counts. It uses Python 3.12 and NumPy (tested environment: NumPy 2.4.4).

Run from this directory using the existing
[initialization companion](../titan-followup-20261109/README.md) and
[map census](../map-census-20261109/README.md):

```sh
python3 -B analyze_crater_training.py \
  --initialization-zip ../titan-followup-20261109/initialization-supplement.zip \
  --map-census ../map-census-20261109/map_cells.npz \
  --census-metadata ../map-census-20261109/counts.json \
  --output /absolute/path/to/external-work/crater-analysis-new
```

Use an existing parent directory outside the CETUS checkout;
`crater-analysis-new` must not exist. The script checks the following SHA256
identities before decoding the consumed bytes. There is no
hash override. Output is `results.json` in the new directory, published by an
atomic rename after encoding. Existing outputs are never reused; an interrupted
write may retain its temporary file in the reserved directory.

| Input | SHA256 |
| --- | --- |
| `initialization-supplement.zip` | `ada744e9278f4fbac5f5f953b5688132ad0b080b35cff32ed23b7f02c3192486` |
| `map_cells.npz` | `d6511e2ffb01cd6fc5f5e78b68f4d9ae7fe7ecf6176fb67aea4f0e756359ab96` |
| Census `counts.json` | `302f12413c0a92cf0b8152383bf30765adac57d05719ff2609852efdab037126` |

The public `counts.json` supplies `array_conventions`, including the ordered
class, mode and role axes. Its `map_cells_npz_sha256` binds the saved arrays;
`original_results_sha256` records historical lineage. The original census
result file is not an input to this script.

The ZIP's `data/records.json.gz` supplies 30 primary, validation-selected CNN
final-phase cells and five pretrained DINOv2 cells. Both store predictions at
`cell["test"]`. The packaged catalog and five split files establish ordered test
IDs and labels. Every catalog label and fold role is checked against the census;
the six training counts are checked against `fold_counts`. The exact cell set,
class order, prediction joins, batch size and repeated CNN weights are checked.
No model execution, score recalculation, raw-image loading or map resampling is
performed. The accepted census's existing count and mask verification is reused.

The three result lists are `training_batch_availability` (30 class/fold rows),
`map_prediction_cohorts` (70 model/fold/seed/mode rows) and `image_rows` (1,848
image/prediction/mode rows). Repeated predictions remain separate. Input receipts
contain logical names, byte sizes and hashes; they contain no machine paths.

For training size N, class count K and actual batch sizes b (including the last
partial batch), the expected number of batches containing that class is the sum
of `1 - C(N-K,b)/C(N,b)` under a uniform random permutation. Impossible no-class
subsets contribute zero to that ratio. Integer combinations and rational
arithmetic precede float conversion. With M batches, `min(K,M)/M` is the upper
bound on the equal-step average of normalized target-class cross-entropy
coefficients, stored as `loss_coefficient_mass_upper_bound`. It is not a logged
epoch-loss share, numerical loss share, gradient bound or performance bound.
Class absence from a batch does not imply zero classifier gradient or no learning.
Expected counts do not replay the historical batch order.

`baseline` uses known map cells before SAR masking; `all_four` requires all four
nominal SAR samples of each map cell to be valid. Fractions divide by the sum of
six known-class counts. Unknown map cells remain separate. Zero known counts
produce null fractions and do not enter the all-known-cells-Craters endpoint.
Endpoint errors equal `all_known_cells_craters - correct_all_known_cells_craters`;
an endpoint error rate is undefined when its endpoint count is zero. This
endpoint concerns only that mode's eligible known cells and implies neither
physical purity nor, for `all_four`, full-footprint coarse-map purity.

`errors_with_predicted_class_present` is a count. A later error-conditioned rate
must use errors with `known_cells > 0` and separately retain empty-domain errors;
the all-cohort `undefined_map_fraction` cannot be subtracted from all errors.
No seed or fold aggregation is performed. A later aggregate should use equal-seed
means within model/fold/mode with a stated undefined-rate policy, or specify its
alternative in advance. Map composition must be counted once per tile/mode when
summarized across predictions. Repeated seeds and overlapping images do not
provide independent geographic samples or independent landform counts.

Run the synthetic checks without opening scientific inputs:

```sh
python3 -O -B -m unittest -v test_crater_training
```

The included `results.json` preserves the accepted CPU reproduction exactly.
Its SHA256 is
`c0f1d60faaedc9b083be115e1468e97f25f40c676d0e9a25651076dec1759716`.
The unchanged script has SHA256
`984020ced2828392d0793734924922001a950400cac636f313b6421460b41736`,
from analysis source revision
`a8abe8414a3a87ad4677f31d82db50996dd27955`. This identifies the descriptive
analysis implementation; the input companions retain their model and census
producer histories. Runtime versions, input identities and class order are
recorded in the result. Packaging these files adds no scientific execution.
The test file changes only its import to load the colocated script.

Code is covered by the repository [MIT license](../../LICENSE). Preserve the
[data licenses](../../DATA_LICENSES.md),
[third-party credits](../../THIRD_PARTY_NOTICES.md),
[initialization notice](../titan-followup-20261109/NOTICE.md) and
[map-census attribution](../map-census-20261109/NOTICE.md) when using these results.
