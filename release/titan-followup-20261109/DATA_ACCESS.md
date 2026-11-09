# Data and software access

Updated 9 November 2026. The four ZIPs in this directory supply saved results
for 23,380 Titan catalog tiles and five geographic folds with 23,246 unique
test tiles. Historical v1 results use 11,371 Titan tiles, 398 Earth tiles and
2,945 unlabeled Venus tiles; their catalogs, splits and results remain separate.

The archives and their exact identities are listed in [README.md](README.md)
and [provenance.json](provenance.json). They contain small inspection arrays,
but **the full processed pixel collection is not included**. There is no
verified public processed-tile endpoint or registered project archive DOI
supplied here. A checksum identifies bytes; it does not provide access to
omitted files. Upstream product links below are not benchmark-image downloads.

## Original scientific archive

This table describes only the 203-entry `scientific-supplement.zip`. Paths are
relative to its extracted root. Later companions have separate inventories.

| Artifact | Included | Missing or separate prerequisite |
| --- | --- | --- |
| Catalog | `data/catalog.json`: sanitized metadata, labels, ordered tile IDs, file hashes and scalar coverage summaries. | Images and historical catalogs. The sanitized bytes differ from the original producer input. |
| Splits | `data/splits/protocol.json` and `titan_contiguous_fold_0.json` through `titan_contiguous_fold_4.json` in that directory. | Historical splits and edited catalogs cannot substitute for the packaged versions. |
| Source products | Product references and notices. | Original mission files, SAR mosaics, map shapefiles and the VIMS/ISS mosaic. Titan and Venus inputs are display mosaics, not raw radar measurements. |
| Numerical arrays | All eighty `results/scaling/*/fitted_arrays.npz` files and associated records, including predictions, probabilities and linear coefficients. | No 128-by-128 image tiles, image-derived feature matrices or complete image-to-result inputs. Fitting arrays are not embeddings or checkpoints. |
| Masks and map raster | Catalog `valid_fraction`, labels, modal fractions and reference loading code. | No pixelwise masks or either full label raster. Scalar fractions cannot recover masks. Zero after normalization can mean fill or valid low DN; map votes were not intersected with SAR validity. |
| Features | Four reference configuration, loader and adapter files under `reference/titansar/`. | No embeddings, classical feature caches or complete generation pipeline. These files are for inspection, not an installable inference package. |
| Weights | Identities in `reference/checkpoint-identities.json` and upstream references in `reference/README.md`. | No pretrained weights or fitted model states. This inventory does not verify current downloadability of exact upstream weight bytes. |
| Training | Saved partial-training predictions and the 80-fit study's outputs. | No complete training/retraining package. The separate CETUS runner supports frozen Titan extraction, probing and kNN. |
| Evaluation | `scripts/replay_metrics.py`, `scripts/prepare_cvpr_tables.py`, predictions and eleven table inputs. | No fresh inference, extraction, refitting, probability/gradient reconstruction or historical Earth/Venus/CNN evaluation. |
| Environment | NumPy 2.4.4 in `requirements.txt`; Python 3.12.3 in the replay instructions. Table checking uses the standard library. | No runtime, wheels, container or complete training environment; no cross-platform equivalence guarantee. |

Preserve the catalog's ordering and matching splits when consuming predictions.
Its exact digest is in the wrapper provenance and internal manifest. Historical
execution remains bound to its original catalog; this release is not a new run
on anonymous catalog bytes.

## Later companions

| Archive | Included and supported | Limits |
| --- | --- | --- |
| Follow-up, 84 entries | `coverage/` supplies the calculation and saved coverage results for 115 vectors. It reads the unchanged scientific ZIP. `label-support/data/vote_inputs.npz` supplies 2,685 sampled map grids and integer weights; `label-support/data/inspection/` contains sixty SAR packets with masks. `label-support/replay_vote_inputs.py` checks the saved integer votes. | No full image collection, complete masks, map raster, embeddings or weights. The 2,643 partial test tiles used for coverage differ from 2,651 partial catalog tiles plus 34 complete inspection tiles used for votes. This is not whole-catalog map reconstruction. |
| Initialization, 19 entries | `data/records.json.gz`, the unchanged anonymous catalog and five splits, and `reconstruct.py`. Sixty native test records, 6,000 validation-history rows from sixty CNN fits and choices reconstruct 90 cells and 180 coverage confusions. | No states, features, full imagery or per-epoch prediction vectors. It cannot repeat training, regenerate probabilities or establish that optimizer updates occurred. Historical helper CLIs are not recipient entrypoints. |
| Reassessment, 54 entries | `data/epoch100.json.gz`, validation-choice records, `cases/` with thirty-seven SAR/map NPZ excerpts, and `check.py`. Requires the unchanged initialization ZIP. | Single-image removal counts use supplied outcomes; omitted per-epoch predictions prevent regeneration of every removal score. No full image collection, model states or whole-catalog map analysis. |

The two inspection counts are not a claim of distinct, non-overlapping samples.
Map grids and integer coverage weights support the documented vote arithmetic;
they do not establish measured SAR/map registration or independent geology.
Original accepted labels remain unchanged. A separate raster reconstruction
differed at seven cells; neither full raster is distributed in these archives.

## Repeating inference or training

Use the [CPU reconstruction commands](README.md#reconstruct-saved-results) for
the supplied records. Image-to-result reproduction additionally requires exact
arrays matching every catalog tile hash, the matching catalog/split version,
complete usable producer source, upstream encoder code and weights, retained
fitted states where applicable, preprocessing and feature definitions,
fold-specific normalization, and the recorded runtime. New runs must identify
their actual source and inputs.

The separate [frozen-runner guide](../../SOFTWARE.md) expects
`tiles/TILE_ID.npy` beside its catalog. Even cached-feature analysis needs
training images to reconstruct normalization. Its public-source replay and
the later matched-model study have separate producers and verification scopes.
The separate [current source package](../current-source-20261109/README.md)
provides execution code and its input contracts. The numerical archives do not
supply complete scientific inputs or private execution supervision. Preserve
the original input and source guards when preparing new runs.

Recorded author-side checks validated anonymous inputs across all five folds
without constructing a model. A separate execution check reproduced validation
and test outputs for three retained models in Region 1: ResNet18 from scratch,
ResNet18 from ImageNet, and pretrained DINOv2 with its saved scaler and linear
classifier. DINOv2 also reproduced training features. Those checks involved no
fitting or new accuracy estimate and do not establish independent-recipient
access to the full images, states, source or environment. Private preservation
does not supply a public download route.

Titan and Venus inputs are display DN; historical Earth inputs are calibrated
linear sigma0. Display scaling cannot recover planetary backscatter calibration.
VIMS fields are ancillary display samples, not classifier inputs or calibrated
composition. DOFA's checkpoint/input convention remains unresolved. Rebuilding
labels also needs the original geometry, rasterization and sampling settings;
saved-result replay does not establish exact raster regeneration or registration.

## Source records and credits

These references identify upstream sources and retain the existing attribution;
they do not provide the processed benchmark collection or certify current access
to all source payloads.

| Source | Record and attribution |
| --- | --- |
| Titan HiSAR | [USGS Cassini SAR HiSAR Global Mosaic 351 m, through T104](https://astrogeology.usgs.gov/search/map/titan_cassini_sar_hisar_global_mosaic_351m). Credit the Cassini SAR Team, Jet Propulsion Laboratory and USGS Astrogeology Science Center. |
| Map labels | [Titan Global Map Shapefiles, version 1](https://data.mendeley.com/datasets/f6jrtyfp66/1), Ashley Schoenfeld, DOI `10.17632/f6jrtyfp66.1`, CC BY 4.0; cite Lopes et al. (2020). Derived grids use rasterization, projection and local sampling; coverage-weighted votes add integer counting. |
| Ancillary VIMS/ISS | [CaltechDATA mosaic](https://data.caltech.edu/records/8q9an-yt176), DOI `10.22002/D1.1173`. Retain the full BSD notice in [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md). |
| Historical Venus | [USGS Magellan FMAP right-look global mosaic 75 m](https://astrogeology.usgs.gov/search/map/venus_magellan_sar_fmap_right_look_global_mosaic_75m). Credit the Magellan mission and retain the source notice. No Venus image collection is included here. |
| Historical Earth | [Copernicus Sentinel-1 product documentation](https://sentiwiki.copernicus.eu/web/s1-products). This does not identify or deliver the exact retained GRD inventory, processed arrays or terrain-correction inputs. Retain modified-Copernicus and WorldDEM notices; exact historical DEM release/tile versions were not recorded. |
| Encoder assets | [Pinned DINOv2 source](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8), [pinned DOFA source](https://github.com/zhu-xlab/DOFA/tree/73ff5c0721da322689ae890bed5b4efd78935f47), and [CROMA](https://github.com/antofuller/CROMA). Recorded weight identities appear in [SOFTWARE.md](../../SOFTWARE.md) and the scientific ZIP's reference record; weights are supplied separately. |

Retain [DATA_LICENSES.md](../../DATA_LICENSES.md),
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md), the archives' own notices
and [NOTICE.md](NOTICE.md). MIT covers author-owned code; it does not replace
source-data licenses, CROMA attribution or asset-specific terms. Permissions
and availability are separate questions.
