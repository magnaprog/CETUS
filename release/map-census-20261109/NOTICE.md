# Attribution and data terms

Author-owned code and documentation are by **Kevin Lee**, under the included
[MIT license](LICENSE). The source data retain their own terms.

The map grids derive from Ashley Schoenfeld's *Titan Global Map Shapefiles*,
version 1, and Lopes et al. (2020), *A global geomorphologic map of Saturn's
moon Titan*. Cite both the [dataset](https://data.mendeley.com/datasets/f6jrtyfp66/1)
and the paper when using these labels. The dataset carries
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Processing includes
rasterization to six class codes, projection, nearest-neighbor sampling,
counting under SAR coverage masks, and aggregation. Unknown labels remain
separate.

The validity masks derive from the *Cassini SAR HiSAR Global Mosaic 351 m*.
Credit the **Cassini SAR Team, Jet Propulsion Laboratory, and USGS Astrogeology
Science Center**. The product record requests citation. This companion supplies
validity bits; original SAR image values are distributed separately.

The count functions reuse the separate census checker retained with the
scientific development records. That checker inspected original SAR images;
this companion applies its counting functions to saved grids and masks.

The repository's [data licenses](../../DATA_LICENSES.md) and
[third-party notices](../../THIRD_PARTY_NOTICES.md) retain the full source
credits. The [manuscript](https://arxiv.org/abs/2610.07576) contains the
assistance disclosure. No institutional review or endorsement is implied.
