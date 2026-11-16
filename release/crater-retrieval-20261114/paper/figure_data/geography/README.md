Natural Earth coastline context

`natural-earth-110m-coastline.json` contains all 134 source polyline parts and
5,128 vertices from Natural Earth 1:110m coastline, version 4.1.0. It supplies
only the global Earth locator in `scripts/plot_dataset_geography.py`. Scientific
sample centers, roles, footprints and site tables remain in their separate
hash-bound geography payload.

Source: https://www.naturalearthdata.com/downloads/110m-physical-vectors/110m-coastline/

License: public domain, https://www.naturalearthdata.com/about/terms-of-use/.
Suggested attribution: “Coastline: Natural Earth 1:110m, v4.1.0”.

The adjacent provenance JSON records the upstream archive and member hashes,
version, WGS84 geographic CRS, degree units, decoder identity, transformation,
decoded geometry hash and ordered-parts digest. The original archive is not
included. Rendering reads only these two authenticated JSON files and needs no
Shapefile reader or network access. This bounded context has its own source
contract; it is not a numerical experiment or part of the legacy figure-input
manifest.

Extraction used existing pyshp 3.0.12 and preserved all PolyLine records, part
boundaries and binary64 coordinate values in source order. `source_record` is
one-based; `source_part` is zero-based. JSON uses round-trip float precision,
with sorted keys, compact separators and a trailing newline. No rounding,
simplification, reprojection, joining or longitude wrapping was applied.
Independent parsing of the Shapefile record headers and coordinate bytes
verified every part and vertex.

No adjacent longitude difference within a part exceeds 180 degrees, so no new
seam vertices or splits are needed. A source endpoint at 180.00000044181039
degrees is retained exactly and clipped to the existing −180..180 degree axes.
Wrapping it to −180 would create a false chord. The renderer validates the
complete context and its style separately from the scientific artist digest.

The generalized coastline provides global orientation. It must not establish
local shoreline measurements, image registration, valid SAR coverage, terrain
purity or area statistics. It adds no fills or political boundaries.
