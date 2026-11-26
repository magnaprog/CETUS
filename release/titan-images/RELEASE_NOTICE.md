# CETUS v0.2.0 Titan image collection

The collection contains 23,380 Cassini HiSAR radar images, their terrain map labels and five longitude splits. Each NumPy file holds 128 × 128 float32 display values. Across the five splits, 23,246 distinct images occur in a test region. Terrain labels summarize the published map in each footprint. The map draws partly on the same radar observations, so agreement with its labels does not independently establish the underlying geology.

The archive `titan-images.tar.zst` has **1,312,278,780 bytes** and SHA-256 `3ea1c147af5dd6daf663da2d8bc1bdf1c822faf0d6b6764b77ce2a3361392aad`. Download it with `RELEASE_NOTICE.md` and `SHA256SUMS.release` from the [v0.2.0 release](https://github.com/magnaprog/CETUS/releases/tag/v0.2.0). Place all three in the same directory. From that directory, use GNU tar, zstd and Python 3.12:

```sh
set -eu
sha256sum -c SHA256SUMS.release
mkdir titan-images
tar --use-compress-program="zstd -d" --extract --file=titan-images.tar.zst \
  --directory=titan-images --no-same-owner --no-same-permissions
python3 -I -B titan-images/verify_inputs.py > titan-images-verification.json
```

The verifier checks the catalog, split assignments and every array's digest, shape and dtype. A successful run prints `inputs_verified` with the test union and fold counts. Save the output beside the extracted directory, as shown, to keep its inventory intact. To run a model on these files, see the [classifier guide](https://github.com/magnaprog/CETUS/blob/v0.2.0/docs/classifiers.md#supply-the-data-and-weights).

## Sources and terms

**Radar images.** Credit the Cassini SAR Team, Jet Propulsion Laboratory and USGS Astrogeology Science Center for the HiSAR mosaic through flyby T104. The source record requests citation of its authors. CETUS projects and resamples the mosaic. The images retain display digital numbers, which do not provide calibrated radar backscatter. [USGS mosaic record](https://astrogeology.usgs.gov/search/map/titan_cassini_sar_hisar_global_mosaic_351m)

**Terrain labels.** Cite Ashley Schoenfeld's *Titan Global Map Shapefiles*, version 1 (2024), and Lopes et al. (2020). CETUS rasterizes, projects and samples the map to calculate the most common class and its fraction for each tile. The source uses CC BY 4.0. Retain the attribution, license and description of modifications with the derived labels. The license's warranty disclaimer applies. [Map dataset](https://doi.org/10.17632/f6jrtyfp66.1) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

**Ancillary optical samples.** Catalog fields sample *Titan's global map combining VIMS and ISS mosaics*, version 1.1 (2019), by Benoît Seignovert, Stéphane Le Mouélic, Robert H. Brown, Erich Karkoschka, Virginia Pasek, Christophe Sotin and Elizabeth P. Turtle. Credit Le Mouélic et al. (2019) and the Cassini Imaging Team for the ISS source. The classification models use the radar images. The DOI metadata specifies BSD-3-Clause. Preserve the full copyright, conditions and disclaimer in `THIRD_PARTY_NOTICES.md`, including `Caltech-JPL/Univ. of Arizona/LPG-Univ. of Nantes-CNRS | JPL: CL#19-0905`. [Mosaic DOI](https://doi.org/10.22002/D1.1173) · [DOI metadata](https://api.datacite.org/dois/10.22002/D1.1173) · [Source article](https://doi.org/10.1016/j.icarus.2018.09.017) · [ISS source](https://photojournal.jpl.nasa.gov/catalog/PIA22770)

CETUS is authored by Kevin Lee. The MIT license applies to author-owned software. The archive preserves the original `LICENSE`, `DATA_LICENSES.md`, `THIRD_PARTY_NOTICES.md` and `NOTICE.md`. They also document source products outside this Titan collection. Use the DOI metadata above for the VIMS/ISS license.
