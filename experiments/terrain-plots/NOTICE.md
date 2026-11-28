# Software, font and data credits

The author-owned plotting code, tests and display helpers are by Kevin Lee and
use the accompanying MIT [license](LICENSE). NumPy, Matplotlib, Pillow and pytest
are supplied separately under their own terms; their source is not bundled.

The unmodified Nimbus Sans and Nimbus Roman TrueType files are from Artifex
URW Base 35, commit `3c0ba3b5687632dfc66526544a4e811fe0ec0cd9`. Each font
directory retains the upstream `LICENSE`, `COPYING` and `sources.json`:

- [Nimbus Sans license](fonts/nimbus-sans/LICENSE),
  [COPYING](fonts/nimbus-sans/COPYING), and
  [source hashes](fonts/nimbus-sans/sources.json).
- [Nimbus Roman license](fonts/nimbus-roman/LICENSE),
  [COPYING](fonts/nimbus-roman/COPYING), and
  [source hashes](fonts/nimbus-roman/sources.json).

Preserve these separate font terms, including their stated PostScript/PDF
embedding permission. The MIT code license does not replace them. All four
Roman faces are included because the unchanged style configuration registers
each face.

The class and regional displays read the existing accepted ranking and SGD
analysis records. Preserve the ranking companion's
[data and software credits](../class-ranking/NOTICE.md), the repository's
[data licenses](../../licenses/DATA_LICENSES.md) and
[third-party notices](../../licenses/THIRD_PARTY_NOTICES.md), and the notices within the
original numerical archives. Original result files and archived licenses remain
unchanged.

Titan sampling metadata derives from Cassini SAR HiSAR Global Mosaic 351 m
and the current test catalog. Credit the Cassini SAR Team, Jet Propulsion
Laboratory and USGS Astrogeology Science Center. Terrain labels derive from
Lopes et al. (2020) and Ashley Schoenfeld's *Titan Global Map Shapefiles*, version
1, licensed CC BY 4.0. The displayed labels describe modal map classes.

Earth metadata gives the outlines and geometric centers of 398 image windows
extracted from Sentinel-1 radar scenes. Preserve the required notice:
"Contains modified Copernicus Sentinel data 2026."
The underlying processing used calibrated sigma0 and terrain correction; the
repository's notices retain the Copernicus DEM GLO-30/WorldDEM-30 terms and the
recorded limits on exact collection/tile versions. Each outline traces the full
image window, including any pixels without valid radar measurements. Terrain labels describe the intended terrain at each site.

The Earth locator also includes the Natural Earth 1:110m coastline, version
4.1.0, in the public domain. Its [source record](data/coastline/README.md)
retains the upstream attribution and terms. This generalized outline supplies
global context separately from the scientific sampling metadata.

The [Venus catalog](../../data/catalogs/venus_catalog.json) gives the image
centers and valid pixel fractions for tiles from a Magellan FMAP mosaic.
The [data guide](../../data/README.md#catalogs) identifies the right-look
global mosaic as the reference product. The exact mosaic file that supplied
the pixels is unverified. Credit NASA/JPL, the Magellan mission and
Saunders et al. (1992), following the existing source-data notice. The 2,945
samples are unlabeled; their six illustration markers do not assign terrain.

This companion distributes derived ranking arrays and sampling metadata,
including the boundaries of the Earth image windows. It includes no original
image pixels, geological terrain-map geometry or model weights. The MIT software license does
not replace source-data, font or upstream-model terms.
