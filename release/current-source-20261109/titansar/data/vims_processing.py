"""VIMS data processing for TitanSAR.

Handles co-registration of an auxiliary Cassini VIMS display composite with
SAR tiles. Samples are display-product metadata only, not calibrated spectra or
footprint-integrated quantitative band ratios.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def vims_coordinate_transform(crs):
    """Validate source coordinates and return its Titan projection, if required.

    Geographic degree rasters retain direct angular sampling. Projected support
    is deliberately limited to the declared Titan_2000 spherical equirectangular
    product: radius 2575000 m, metre axes, and zero projection parameters.
    The source's own geodetic CRS supplies the projection input. No Earth datum
    conversion or automatic radius substitution is permitted.
    """
    from pyproj import CRS, Transformer

    if crs is None:
        raise ValueError("VIMS sampling requires a source geographic or supported Titan projected CRS")
    source = CRS.from_user_input(crs)
    geographic = source.geodetic_crs
    if (source.is_bound or source.is_compound or geographic is None
            or geographic.type_name != "Geographic 2D CRS"
            or len(geographic.axis_info) != 2
            or {axis.direction for axis in geographic.axis_info} != {"north", "east"}
            or any(not np.isclose(axis.unit_conversion_factor, np.pi / 180,
                                 rtol=0, atol=1e-14) for axis in geographic.axis_info)
            or geographic.prime_meridian.longitude != 0):
        raise ValueError("VIMS requires geographic degrees with a zero prime meridian")
    if source.type_name == "Geographic 2D CRS":
        return None

    operation = source.coordinate_operation
    if (source.type_name != "Projected CRS"
            or [axis.direction for axis in source.axis_info] != ["east", "north"]
            or any(axis.unit_conversion_factor != 1 for axis in source.axis_info)
            or geographic.datum.name != "D_Titan_2000"
            or geographic.datum.type_name != "Geodetic Reference Frame"
            or geographic.ellipsoid.semi_major_metre != 2575000
            or geographic.ellipsoid.semi_minor_metre != 2575000
            or operation is None
            or (operation.method_auth_name, operation.method_code) != ("EPSG", "1029")):
        raise ValueError("VIMS supports geographic degrees or the declared Titan_2000 spherical equirectangular CRS only")

    # EPSG:1029 omits latitude of natural origin in the source WKT. Some
    # equivalent encodings include it explicitly; only zero is supported.
    units = {"8823": np.pi / 180, "8802": np.pi / 180, "8806": 1, "8807": 1,
             "8801": np.pi / 180}
    codes = [parameter.code for parameter in operation.params]
    if (len(codes) != len(set(codes))
            or set(codes) not in ({"8823", "8802", "8806", "8807"}, set(units))
            or any(parameter.auth_name != "EPSG" or parameter.value != 0
                   or not np.isclose(parameter.unit_conversion_factor, units[parameter.code],
                                     rtol=0, atol=1e-14)
                   for parameter in operation.params)):
        raise ValueError("VIMS Titan projection requires zero standard parallel, origins and false offsets")
    return Transformer.from_crs(geographic, source, always_xy=True,
                                allow_ballpark=False, force_over=True)


def read_vims_mosaic(vims_path: Path) -> dict:
    """Read a VIMS display composite and retain its explicit file metadata."""
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio required for VIMS mosaic reading")

    with rasterio.open(vims_path) as src:
        # Read all bands
        data = src.read().astype(np.float32)  # (bands, H, W)
        valid_mask = src.read_masks() > 0
        for band, nodata in enumerate(src.nodatavals):
            if nodata is not None:
                valid_mask[band] &= data[band] != nodata
        valid_mask &= np.isfinite(data)
        if data.shape[0] == 1:
            logger.warning(
                "VIMS display mosaic has one band rather than an RGB composite; "
                "recording the available display channel only."
            )
        return {
            "data": data,
            "transform": src.transform,
            "crs": src.crs,
            "width": src.width,
            "height": src.height,
            "bounds": src.bounds,
            "source_path": str(vims_path),
            "dtype": src.dtypes[0],
            "valid_mask": valid_mask,
            "nodata_values": src.nodatavals,
            "mask_flags": [[flag.name for flag in flags] for flags in src.mask_flag_enums],
        }


def coregister_vims_to_tiles(
    tiles: list[dict],
    vims_data: np.ndarray,
    transform,
    bounds,
    *,
    crs=None,
    valid_mask: Optional[np.ndarray] = None,
) -> list[dict]:
    """Sample an auxiliary VIMS display composite at each SAR tile center.

    These values are visualization-product metadata. They must not be interpreted
    as calibrated spectra, footprint means, or quantitative compositional ratios.
    Availability describes numeric sampling under the source mask, without
    establishing valid mission observations or physical coverage.

    Geographic degree rasters use angular coordinates directly. The supported
    Titan_2000 equirectangular raster uses its own geodetic CRS to project each
    longitude candidate into metres before bounds checks and affine inversion.
    No Earth datum transformation is applied. Other projections are rejected.

    Bilinear interpolation uses pixel centers. Samples outside the rectangle
    spanned by those centers are unavailable, including the outer half-pixel
    margins. Exact edge centers are supported. Coordinates within 1e-9 pixel
    of an integer are snapped to it to absorb affine roundoff. Every contributor
    with positive weight must be finite and source-valid in every channel.
    Zero and negative display values are allowed unless the source mask or
    explicit nodata excludes them. Missing support is never renormalized.

    Parameters:
        tiles: list of tile dicts (must have center_lat, center_lon)
        vims_data: (bands, H, W) or (H, W) VIMS display data
        transform: rasterio affine transform for ``vims_data``
        bounds: rasterio bounds for ``vims_data``
        crs: source CRS; missing or unsupported coordinates raise
        valid_mask: optional boolean source mask with the same shape as the data
    """
    projection = vims_coordinate_transform(crs)

    vims_data = np.ma.asarray(vims_data)
    if vims_data.ndim == 2:
        vims_data = vims_data[np.newaxis, :, :]
    if vims_data.ndim != 3 or min(vims_data.shape) == 0:
        raise ValueError("VIMS display data must have shape (bands, height, width) or (height, width)")

    source_valid = ~np.ma.getmaskarray(vims_data)
    vims_data = np.asarray(vims_data.data)
    source_valid &= np.isfinite(vims_data)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask)
        if valid_mask.ndim == 2:
            valid_mask = valid_mask[np.newaxis, :, :]
        if valid_mask.shape != vims_data.shape or valid_mask.dtype != np.bool_:
            raise ValueError("VIMS valid_mask must be boolean and match the display data shape")
        source_valid &= valid_mask

    vims_h, vims_w = vims_data.shape[-2:]
    inverse_transform = ~transform

    for tile in tiles:
        lat = tile["center_lat"]
        lon = tile["center_lon"]
        tile["has_vims"] = False
        tile["vims_display_sample"] = None
        if not np.isfinite([lat, lon]).all() or not -90 <= lat <= 90:
            continue

        # Compare coordinates with bounds only after converting to source units.
        candidates = (lon % 360.0, (lon % 360.0) - 360.0)
        raster_xy = None
        for candidate in candidates:
            x, y = ((candidate, lat) if projection is None
                    else projection.transform(candidate, lat, errcheck=True))
            if bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top:
                raster_xy = x, y
                break
        if raster_xy is None:
            continue
        col, row = inverse_transform * raster_xy
        col, row = col - 0.5, row - 0.5
        if not np.isfinite([row, col]).all():
            continue
        if abs(col - round(col)) <= 1e-9:
            col = float(round(col))
        if abs(row - round(row)) <= 1e-9:
            row = float(round(row))
        if not (0 <= row <= vims_h - 1 and 0 <= col <= vims_w - 1):
            continue

        # Bilinear interpolation
        r0, c0 = int(np.floor(row)), int(np.floor(col))
        r1, c1 = min(r0 + 1, vims_h - 1), min(c0 + 1, vims_w - 1)
        dr, dc = row - r0, col - c0
        weights = np.array([(1 - dr) * (1 - dc), (1 - dr) * dc,
                            dr * (1 - dc), dr * dc])
        rows, cols = np.array([r0, r0, r1, r1]), np.array([c0, c1, c0, c1])
        contributes = weights > 0
        rows, cols = rows[contributes], cols[contributes]
        if not source_valid[:, rows, cols].all():
            continue

        values = vims_data[:, rows, cols] @ weights[contributes]
        if not np.isfinite(values).all():
            continue
        tile["has_vims"] = True
        tile["vims_display_sample"] = [round(float(value), 6) for value in values]

    n_with_vims = sum(1 for t in tiles if t.get("has_vims"))
    logger.info(f"VIMS co-registration: {n_with_vims}/{len(tiles)} tiles have VIMS data")
    return tiles
