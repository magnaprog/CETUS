# Natural Earth coastline

The Earth site locator uses the Natural Earth 1:110m coastline, version 4.1.0, for global orientation. Image centers and window outlines come from the separate [sampling dataset](../dataset-geography.json). Local coverage and terrain labels cannot be inferred from this generalized shoreline.

Credit the source as “Coastline: Natural Earth 1:110m, v4.1.0”. The dataset is in the public domain. [Download page](https://www.naturalearthdata.com/downloads/110m-physical-vectors/110m-coastline/) · [Terms of use](https://www.naturalearthdata.com/about/terms-of-use/)

The [geometry](natural-earth-110m-coastline.json) retains all 134 polyline parts and 5,128 vertices in source order, in WGS84 longitude and latitude. The [provenance record](natural-earth-110m-coastline.provenance.json) identifies the source Shapefile, its archive and member hashes, and the digest of the decoded coordinates. The renderer checks the two JSON files locally.

The extraction used pyshp 3.0.12 and retained the original binary64 values and part boundaries. Record numbers start at one and part numbers at zero. One source endpoint lies at 180.00000044181039 degrees longitude. The locator clips it at 180 degrees because wrapping it to −180 degrees would draw a spurious line across the map.
