# Titan terrain map census

For each of the 23,380 catalog images, `map_cells.npz` stores a 64 × 64 terrain-label grid, a packed 128 × 128 radar-validity mask and its assignment in each of five folds. The checker recounts those arrays and compares the totals with [counts.json](counts.json).

## Map composition

The modal class is the most frequent known class in the unmasked grid. For the last two columns, a cell counts only if radar is valid at all four corresponding sample positions (the `all_four` rule). An image containing several classes contributes to more than one row of the “Images containing class” column.

| Class | Modal-class images | Images containing class | Map cells with all four SAR samples valid |
| --- | ---: | ---: | ---: |
| Plains | 15,433 | 20,144 | 59,731,857 |
| Dunes | 4,050 | 6,927 | 16,442,040 |
| Hummocky | 2,764 | 10,974 | 13,646,770 |
| Labyrinths | 498 | 1,193 | 2,129,298 |
| Lakes | 495 | 1,851 | 2,467,255 |
| Craters | 140 | 331 | 590,472 |

Of those 590,472 Craters cells, 128,824 (21.8%) occur in images with another
modal class. Test region 2 (saved fold 1) has eleven images containing Craters
and one modal-Craters image. Some tiles overlap, so the totals include repeated samples of the same terrain. The same image also appears in more than one training fold.

Before SAR masking, 4,443 of 95,764,480 map cells have unknown labels, across 15 images. The largest count in one image is 1,277. Unknown labels (code 255) are counted separately, even if the corresponding radar samples are valid.

## Run the check

From the repository root, run the following with Python and NumPy. The reconstruction was tested with Python 3.12.3 and NumPy 2.4.4.

```sh
census_work=$(mktemp -d)
python3 -B experiments/map-census/check_counts.py --output "$census_work/recount.json"
```

The checker verifies both saved files against their SHA-256 hashes. It reconstructs per-image counts, modal labels and the catalog and fold summaries using the stored map grids, masks and split roles. The output gives the reconstructed summaries and the verification status. Use a fresh output filename, or omit `--output` to print the report. To run the synthetic tests, use `python3 -B -m unittest -v test_check_counts` from this directory.

To inspect one grid and mask in a Python session started from this directory:

```python
from pathlib import Path
import numpy as np
from check_counts import bound_inputs, load_arrays

payloads = bound_inputs(Path('.'))
arrays = load_arrays(payloads['map_cells.npz'])
i = 0  # choose a row, or locate it by arrays['tile_ids']
grid = arrays['map_grids'][i]
mask = np.unpackbits(arrays['sar_valid_packed'][i], bitorder='little').reshape(128, 128)
print(arrays['tile_ids'][i], grid.shape, mask.shape, arrays['roles'][i])
```

## Arrays and count units

Every array's first axis follows `tile_ids`. IDs need not be consecutive.
There are eight arrays, all stored in C order:

| Array | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `tile_ids` | `(23380,)` | `<U12` | Unique `titan_XXXXXX` catalog identifiers |
| `map_grids` | `(23380,64,64)` | `uint8` | Class 0–5; 255 means unknown |
| `sar_valid_packed` | `(23380,2048)` | `uint8` | 16,384 mask bits per image, little bit order, C row order |
| `roles` | `(23380,5)` | `uint8` | One role in each saved fold 0–4 |
| `modal_labels` | `(23380,)` | `uint8` | Most frequent known baseline class; lowest code breaks ties |
| `counts` | `(23380,4,6)` | `<i8` | Stored class counts, axes image/mode/class |
| `unknown` | `(23380,4)` | `<i8` | Stored unknown counts, axes image/mode |
| `weight_histograms` | `(23380,5)` | `<i8` | Map-cell counts with 0, 1, 2, 3 or 4 valid SAR samples |

Class codes 0–5 are plains, dunes, hummocky, labyrinths, lakes and craters.
Role codes 0–4 are train, val, test, buffer_excluded and selk_holdout.
`roles[i,f]` is image `i`'s role in fold `f`; paper region numbers are `f + 1`.
The packed mask records the producer's finite-and-positive SAR validity before
normalization. For map cell `(y,x)`, its nominal weight is the sum of mask
entries `(2y,2x)`, `(2y,2x+1)`, `(2y+1,2x)` and `(2y+1,2x+1)`.

| Mode index/name | Contribution per map cell | Unit |
| --- | --- | --- |
| 0 / `baseline` | 1 | Map cells, without SAR masking |
| 1 / `weighted` | Number of valid SAR samples, 0–4 | Integer SAR-valid samples |
| 2 / `all_four` | 1 if weight is 4, otherwise 0 | Map cells |
| 3 / `any_valid` | 1 if weight is positive, otherwise 0 | Map cells |

These same rules apply separately to unknown cells. Weighted totals have four
times the unit scale of a completely covered map-cell total; division by four
gives effective map cells, without establishing physical area. Each histogram
sums to 4,096. Known plus unknown counts equal each mode's coverage denominator.

`counts.json` retains the complete accepted payload: conventions, input hashes,
historical reproduction/verification statements and aggregates. `counts.catalog`
and the 25 entries under `counts.folds[*].roles` give 26 groups and 104 mode
summaries. Each includes image counts, modal-class image counts, class totals,
known/unknown totals, images containing each class, and a 6 × 6 matrix with
**modal class as rows and dense class as columns**. Presence means positive
contribution under that mode; weighted presence still counts images, not samples.

## Data and scope

The original analysis sampled the terrain map at nearest neighbors on a Titan sphere, using rounded catalog centers and nominal windows 45 km on a side. The link between map and radar assumes approximate grid alignment. The packed arrays occupy 2.28 MB on disk and about 150 MB when decoded.

The original checker also read the SAR images and verified the catalog labels and fractions for all 23,380 images. This package reconstructs the counts from the saved arrays. The census metadata identifies the original inputs and verification record. See [NOTICE.md](NOTICE.md) for the map and SAR credits and [LICENSE](LICENSE) for the code.
