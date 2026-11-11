# Data and software credits

The author-owned code and analysis are by Kevin Lee and use the accompanying
MIT [license](LICENSE). This companion contains derived ranking results and
reads the unchanged initialization archive; it includes no model weights or
original image pixels.

Terrain labels derive from Lopes et al. (2020) and Ashley Schoenfeld's
*Titan Global Map Shapefiles*, version 1, at
https://data.mendeley.com/datasets/f6jrtyfp66/1. That dataset is licensed under
Creative Commons Attribution 4.0:
https://creativecommons.org/licenses/by/4.0/.
The vector map was rasterized to assign terrain classes to SAR image tiles.

The experiments used the Cassini SAR HiSAR Global Mosaic 351 m. Credit the
Cassini SAR Team, Jet Propulsion Laboratory and USGS Astrogeology Science Center.
The initialization archive retains its data notices and original anonymous MIT
notice. Its public [attribution notice](../titan-followup-20261109/NOTICE.md)
identifies the author without changing the archived bytes.

Preserve the repository's [data licenses](../../DATA_LICENSES.md),
[third-party notices](../../THIRD_PARTY_NOTICES.md), and all corresponding
notices within the input archive. DINOv2, torchvision, NumPy and their separately
supplied source or weights retain their own terms. This package's MIT license
does not replace source-data or upstream-model licenses.
