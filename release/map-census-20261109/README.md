# CETUS saved map census

This companion contains all 23,380 saved 64 × 64 terrain grids, their
packed 128 × 128 SAR-validity masks and assignments in the same five geographic
folds used by CETUS. It lets you inspect the saved samples and reconstruct the
aggregate census with Python and NumPy on CPU. The two data files preserve the census outputs. The check uses these saved
arrays without loading a raster, SAR images or a model.

## Map composition

The primary table uses `all_four`: a nominal map cell counts when all four
corresponding SAR samples are valid. The modal class is the most frequent known
class in the unmasked map grid. A mixed image can contain several classes.

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
and one modal-Craters image. These are observations about map mixtures, with
overlapping images retained. The totals count repeated samples. Unique surface area, independent landform
counts, global terrain prevalence and causes of classification error call for
separate evidence. Geographic folds and all original role assignments stay
the same; summing roles across folds counts many images repeatedly.

Before SAR masking, 4,443 of 95,764,480 map cells are unknown, in fifteen images;
the maximum in one image is 1,277. Unknown map labels (`255`) are separate from
missing SAR. A valid SAR sample does not provide a terrain class at an unknown
map location. Unknowns have their own totals and are excluded from known-class
totals and modal-label denominators.

## Run the check

Use Python 3 with NumPy (`python3 -m pip install numpy` if needed). Preparation
was exercised with Python 3.12.3 and NumPy 2.4.4. No GIS or model library is used.
From this directory:

```sh
python3 -B -m unittest -v test_check_counts
python3 -B check_counts.py --output /absolute/path/to/external-work/recount.json
```

Use an existing parent directory outside the CETUS checkout and a new report
filename. Tests use synthetic data only. The recount writes a JSON report containing
all reconstructed summaries. `--output` refuses to overwrite an existing file;
use a fresh filename for another invocation. Failures and interruptions are
retained in that report when the destination is writable, and exit nonzero.
Without `--output`, the report goes to standard output. `--data-dir DIRECTORY`
can locate another copy of these same two accepted files. There is no CLI option
to substitute different expected hashes. The checker reports the first mismatch.

Both accepted SHA256 values are checked before either file is decoded. NumPy
loads the authenticated in-memory NPZ bytes with `allow_pickle=False`. Checks
cover exact array names, shapes and dtypes, 23,380 unique valid IDs, class and
role codes, per-image class/unknown counts and weight histograms, modal labels,
and exact JSON values/types for every catalog/fold-role summary. Complete ID
membership is bound by the accepted NPZ hash; no external catalog join occurs.
Stored `counts`, `unknown` and `weight_histograms` arrays are comparison targets.
Reconstruction uses `map_grids`, `sar_valid_packed` and `roles`; modal labels
are recomputed from the baseline grid counts and compared to the saved labels.

To inspect one grid and mask after checking both hashes:

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

Scientific producer: `1c3b2881820c5673906f45a000cb80575ed8672a`.
Accepted file SHA256 values:

```text
302f12413c0a92cf0b8152383bf30765adac57d05719ff2609852efdab037126  counts.json
d6511e2ffb01cd6fc5f5e78b68f4d9ae7fe7ecf6176fb67aea4f0e756359ab96  map_cells.npz
```

The raw NPZ is 2,279,451 bytes (2.28 MB compressed). Its decoded arrays occupy
about 150 MB, and checking needs additional working memory. The original census
sampled a raster made from the Lopes et al. map, using accepted rounded catalog
centers, the Titan sphere, 45 km square image areas and nearest-neighbor sampling.
The SAR-to-map correspondence is the existing nominal 2 × 2 convention.
Sampling spacing and mask agreement do not measure physical registration.

The original accepted study reproduced catalog labels and four-decimal fractions
for all 23,380 images and 2,685 earlier grids/weights. Its separate checker
reported 346,924 checks with zero differences, including reconstruction of
original SAR masks. Those are retained historical results. **This companion
checks saved arrays only.** It does not resample the original raster, read or
re-verify original SAR, rejoin external catalogs/splits, replay the earlier
2,685 grids, or re-evaluate catalog fractions absent from this package.
It supplies no independent geological labels, contact accuracy or model score.
Hashes bind the two accepted files, not arbitrary rewritten manifests.

Author-owned code is under the included [MIT license](LICENSE). Map and SAR
credits and source terms are in [NOTICE.md](NOTICE.md). Figure reproduction
is separate from this count check.
