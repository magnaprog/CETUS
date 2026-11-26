# CETUS Titan radar images

The collection contains 23,380 processed Cassini HiSAR images. Each NumPy file stores 128 × 128 display digital numbers as float32. The catalog gives each image a location and the most common terrain map class in its footprint. Across the five longitude splits, 23,246 distinct images appear in a test region.

From the extracted directory, verify the collection with Python 3.12:

```sh
python3 -I -B verify_inputs.py
```

On success, the verifier prints `inputs_verified`, the number of distinct test images and the train, validation and test counts for each fold. It checks each file in the archive against its digest, each image against the catalog and each fold against the geographic protocol. Save any output beside the extracted directory so its inventory still matches.

## Use the images

Each catalog record in `data/catalog.json` identifies an array at `data/tiles/TILE_ID.npy`. Join its tile ID to `data/splits/titan_contiguous_fold_F.json`, where F runs from 0 through 4. Use that fold's assignments for training, validation and testing. Buffer exclusions and the Selk holdout have separate roles. The radar images and map labels share a geographic footprint. A classifier score measures agreement with the reference map, which draws partly on the same radar observations.

The benchmark estimates normalization limits from training images. With NumPy default_rng seed 0, it samples up to 200 tiles in catalog-derived order and up to 512 finite, positive pixels per tile. The pooled 1st and 99th percentiles define the range. The loader clips the scaled result to [0,1] using a denominator epsilon of 1e-8 and maps nonfinite inputs to zero. For the matched DINOv2 and ResNet18 comparison, it resizes to 224 × 224 using bilinear interpolation with align_corners=False and antialias=False, repeats the radar channel three times and applies ImageNet normalization. Display values do not establish calibrated backscatter. For the code, model weights and training contract, see the [classifier guide](https://github.com/magnaprog/CETUS/blob/v0.2.0/docs/classifiers.md#supply-the-data-and-weights).

## Sources and terms

Credit the Cassini SAR Team, Jet Propulsion Laboratory and USGS Astrogeology Science Center for the HiSAR mosaic. CETUS projects and resamples the display intensities. [Cassini SAR HiSAR Global Mosaic 351 m](https://astrogeology.usgs.gov/search/map/titan_cassini_sar_hisar_global_mosaic_351m)

Terrain labels derive from Ashley Schoenfeld's Titan Global Map Shapefiles, version 1, and the map of Lopes et al. (2020). CETUS rasterizes, projects and samples the map. Retain the attribution, the license and this description of the modifications with the derived labels. [Map dataset](https://doi.org/10.17632/f6jrtyfp66.1) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Ancillary catalog values sample the VIMS/ISS display mosaic. The classifiers use the radar images. Preserve the BSD 3-Clause copyright notice, conditions and disclaimer in `THIRD_PARTY_NOTICES.md` when redistributing the optical samples. The collection includes `DATA_LICENSES.md`, `NOTICE.md` and the software `LICENSE`. The broader notices also describe Earth and Venus products outside this collection. [VIMS/ISS mosaic](https://doi.org/10.22002/D1.1173)
