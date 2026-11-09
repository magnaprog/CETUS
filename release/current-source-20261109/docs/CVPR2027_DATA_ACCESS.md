# CVPR 2027 artifact access

Updated 9 November 2026. The current evaluation uses **23,380 Titan tiles**
and five geographic folds with **23,246 unique test tiles**. Historical v1
results use 11,371 Titan tiles, 398 Earth tiles and 2,945 unlabeled Venus
tiles; their catalogs, splits and results are separate.

The inventory below describes the original 203-file saved-results ZIP.
The later [84-file companion](../../titan-followup-20261109/followup-supplement.zip)
adds coverage results, sampled map grids and integer weights, and sixty
original SAR inspection arrays with pixel masks. Its independent NumPy checker
reproduces the saved vote calculations. It does not supply the full image
collection, map raster, feature arrays or encoder weights. The
[follow-up receipt](../CONTRACTS.md#historical-execution-records) gives its exact
identity and scope. The original ZIP remains unchanged.

The [19-file initialization companion](../../titan-followup-20261109/initialization-supplement.zip)
adds sixty native test records, sixty CNN validation histories with 6,000 epoch records, validation
choices and numerical targets for the DINOv2 and ResNet18 comparisons. Its CPU
checker reconstructs the 90 cells and 180 coverage confusion matrices. Its
catalog and divisions retain the original anonymous ZIP bytes. It contains no
model states, image features, full imagery or per-epoch prediction vectors;
it cannot repeat training or regenerate probabilities. See the
[initialization review](../CONTRACTS.md#historical-execution-records) and the
[paper README](../../titan-followup-20261109/README.md#reconstruct-saved-results)
for package identity, commands and numerical qualifications. This companion
has its own inventory; the table below remains the original ZIP's inventory.

The [anonymous saved-results ZIP](../../titan-followup-20261109/scientific-supplement.zip)
is publicly available in this repository; an unauthenticated download was
verified on 9 November 2026. It contains 203 files, totals 49,621,594 bytes,
and has SHA256
`67b6aa65be5bab9b7cbe83df90aafcf7b479117291562471652f59ed7f182a1d`.
It supplies saved numerical results, with no full imagery or model weights.
There is no verified public processed-tile endpoint or registered project
archive DOI. A checksum identifies bytes; it does not supply omitted artifacts.

Paths in the inventory below are relative to the **extracted ZIP**, not this
repository. The ZIP's `README.md`, `manifest.json` and `SHA256SUMS` define its
scope and integrity records.

| Artifact | Included and accessible to a package recipient | Missing or separate prerequisite |
| --- | --- | --- |
| Catalog | `data/catalog.json`: current sanitized metadata, labels, tile IDs, tile-file hashes and scalar coverage summaries. | Image arrays; historical catalogs are not included. Sanitized catalog bytes differ from the original producer input. |
| Splits | `data/splits/protocol.json` and `titan_contiguous_fold_0.json` through `titan_contiguous_fold_4.json` in that directory. | No substitution of historical splits or edited catalog bytes. The five folds bind the packaged catalog. |
| Original mission/source products | Source references and notices only. | No original mission files, SAR mosaics, map shapefiles or VIMS/ISS mosaic. The Titan and Venus inputs are processed display mosaics, not raw radar measurements. |
| Arrays | All 80 fitting-study `results/scaling/*/fitted_arrays.npz` files and associated numerical records; these include saved predictions, probabilities and linear coefficients. | No 128-by-128 image tiles, image-derived feature matrices or complete image-to-result inputs. Fitting arrays are not embeddings or encoder checkpoints. |
| Masks and map raster | Catalog `valid_fraction`, labels and label fractions; reference loading code. | No pixelwise validity masks or either label raster. A scalar fraction cannot recover a mask. Zero after normalization can represent fill or valid low DN; map votes were not intersected with SAR validity. |
| Features | Reference feature adapters and configuration under `reference/titansar/`. | No embeddings, classical feature caches or their complete generation pipeline. Reference files are for inspection, not an installable inference package. |
| Weights | Recorded identities in `reference/checkpoint-identities.json`; upstream code references in `reference/README.md`. | No pretrained encoder weights or trained encoder checkpoints. Upstream availability of the exact recorded bytes was not tested for this inventory. |
| Training | Saved partial-training predictions and fitting-study outputs. | No complete training/retraining package. The separate CETUS software export supports frozen Titan extraction/probing/kNN, not the entire classical, adaptation or preprocessing workflow. |
| Evaluation | `scripts/replay_metrics.py`, `scripts/prepare_cvpr_tables.py`, saved predictions and eleven table inputs. | No fresh image inference, feature extraction, refitting, probability/gradient reconstruction, or historical Earth/Venus/CNN evaluation. |
| Environment | `requirements.txt` pins NumPy 2.4.4; the replay README records Python 3.12.3. Table checking uses the standard library. | No Python runtime, wheels, container or full training environment. Separate recorded production/CPU-test environments are not a complete lock or a cross-platform equivalence guarantee. |

The catalog SHA256 is
`8152226f876e0cb5fad105e0b00a37ef5b62d156d7f6eb3a85b77c549a402967`.
Preserve its ordering and the matching splits when consuming saved predictions.

## Additional numerical companion

The [reassessment companion](../../titan-followup-20261109/README.md#reconstruct-saved-results)
adds thirty epoch-100 validation/test records, validation-choice comparisons,
and thirty-seven inspected SAR/map excerpts. Its 54-file ZIP requires the
unchanged initialization companion. It reconstructs saved arithmetic and map
votes, without model execution. Single-image removal counts use supplied
outcome records; complete per-epoch prediction vectors are omitted. This adds
small image excerpts, not a public full-image collection or model checkpoints.

## Anonymous image inputs

The follow-up runner now accepts the anonymous catalog and its five divisions
through the explicit `cvpr2027-anonymous-v1` input contract. All five folds have
passed CPU input validation with the original 23,380 float32 image arrays,
including their hashes, ordered labels, role counts and training normalization.
[The model guide](MODEL_FOLLOWUP.md) specifies the field and directory layout.
This input check did not construct or execute a model. The existing numerical
ZIPs still omit the full imagery and model states; their contents and scope
are unchanged. No public processed-image download is established by this check.

A subsequent [execution check](../CONTRACTS.md#historical-execution-records)
uses a fresh extraction of that bundle on the author's server. Three retained
models reproduce their validation and test outputs exactly: ResNet18 from
scratch, ResNet18 from ImageNet and pretrained DINOv2 with its saved scaler and
linear classifier. The cases cover Region 1 and retain their original GPU
devices. DINOv2 also reproduces its training features without fitting.
Independent recipients still need access to the full image bundle, usable
source, environment and retained states.

## Supported reconstruction

After obtaining and inspecting the ZIP, use an external environment with the
recorded Python/NumPy versions. From the extracted package root, its documented
entrypoints are:

```sh
python3 -B scripts/replay_metrics.py --output ../saved-metric-replay.json
python3 -B scripts/prepare_cvpr_tables.py --output paper/cvpr2027/tables --check
```

The first writes a new report outside the package; the second checks existing
tables. Previously recorded extracted-package checks passed for 315 saved
test-prediction vectors, including 80 fitting cells, sixteen settings, twenty
contrasts and eleven tables. No replay was executed for this document.
These checks reconstruct saved-result metrics, not the experiments that
produced the predictions. They do not independently calculate every
supplementary claim or establish convergence.

Image-to-result reproduction additionally requires the exact tile arrays
matching every catalog `tile_sha256`, matching catalog/split versions, complete
producer source, encoder source and weight bytes, preprocessing and feature
definitions, fold-specific normalization, and an appropriate recorded software
environment. The separate frozen runner expects `tiles/TILE_ID.npy` beside
its catalog; even cached-feature operation needs training tiles to reconstruct
normalization. New runs must bind their actual inputs and source rather than
inherit the identities of earlier runs. The anonymous reference code alone
does not fulfill these requirements. DOFA's checkpoint/input convention remains
unresolved in this package.

Rebuilding labels additionally needs the original map geometry, rasterization
and sampling settings. A separate reconstruction differed from the retained
raster at seven cells; the anonymous package retains the accepted catalog and
does not include either raster. Exact raster regeneration and SAR/map
registration are not established by the saved-result replay.

## Primary source records

These identify upstream products, not a processed benchmark download:

- Current Titan imagery: [USGS Cassini SAR HiSAR Global Mosaic 351 m, through T104](https://astrogeology.usgs.gov/search/map/titan_cassini_sar_hisar_global_mosaic_351m).
- Current label source: [Titan Global Map Shapefiles, version 1](https://data.mendeley.com/datasets/f6jrtyfp66/1), DOI `10.17632/f6jrtyfp66.1`.
- Ancillary catalog display samples: [Caltech VIMS/ISS mosaic](https://data.caltech.edu/records/8q9an-yt176), DOI `10.22002/D1.1173`; these are not classifier features.
- Historical Venus imagery: [USGS Magellan FMAP right-look global mosaic 75 m](https://astrogeology.usgs.gov/search/map/venus_magellan_sar_fmap_right_look_global_mosaic_75m).
- Historical Earth product family: [Copernicus Sentinel-1 product documentation](https://sentiwiki.copernicus.eu/web/s1-products). This does not identify or deliver the exact retained GRD product inventory, processed Earth arrays or terrain-correction inputs.

The Mendeley, Caltech and Copernicus records were read directly. USGS product
metadata were available through indexed primary pages; direct page retrieval
returned HTTP 403. No source payload was downloaded or checked against retained
source hashes for this inventory. Preserve the package's `DATA_LICENSES.md`
and `THIRD_PARTY_NOTICES.md`; source permissions do not establish publication of
derived artifacts.

The recorded storage decision retains original Sentinel ZIPs and the Venus
mosaic on the GPU server and requires new checkpoints to remain there.
Older local checkpoint archives are historical preservation copies, not public
distribution. This access inventory neither changes that decision nor supplies
a route to private retained assets. Exact processed images, masks/rasters,
features, weights, a complete training release and its environment remain
separate access or reproduction requirements.
