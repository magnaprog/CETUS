"""SAR data acquisition and processing pipeline for TitanSAR.

Handles:
- Downloading and reading Cassini BIDR data from PDS
- Reading USGS HiSAR global mosaic
- Tiling SAR mosaics into 128x128 patches
- Quality filtering
- NLDSAR denoised data integration
"""

import hashlib
import json
import logging
import struct
from pathlib import Path
from typing import Optional

import numpy as np

from titansar.configs.defaults import TitanSARConfig
from titansar.data.dataset import assign_split, _parse_label

logger = logging.getLogger(__name__)


def recenter_lon_180_to_360(raster: np.ndarray) -> np.ndarray:
    """Convert a global raster from [-180, 180] to [0, 360] longitude.

    Rasterize label/geometry data in its native [-180, 180] frame so polygons
    that cross longitude 0 stay contiguous; shifting each vertex by +360 instead
    smears those polygons the long way around the globe (review D15). This rolls
    the columns by half-width to re-center into the pipeline's [0, 360] frame.

    Assumes a full-globe raster with longitude on the column axis and no polygon
    crossing the native +/-180 seam (holds for the Lopes et al. map).
    """
    width = raster.shape[1]
    return np.roll(raster, shift=-(width // 2), axis=1)


def geographic_resolution_deg(
    center_lat_deg: float, target_pixel_size_m: float
) -> tuple[float, float]:
    """Per-axis degree resolution giving isotropic ground sampling at a latitude.

    One degree of latitude is ~111320 m everywhere, but one degree of longitude
    is ~111320*cos(lat) m. Using a single degree step for both axes makes the
    latitude ground spacing too coarse by 1/cos(lat) toward the poles (review
    N4). Returns (lon_res_deg, lat_res_deg) so both axes resolve to roughly
    target_pixel_size_m on the ground.
    """
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * float(np.cos(np.radians(center_lat_deg)))
    if m_per_deg_lon < 1000.0:  # avoid blow-up very near the poles
        m_per_deg_lon = m_per_deg_lat
    lon_res_deg = target_pixel_size_m / m_per_deg_lon
    lat_res_deg = target_pixel_size_m / m_per_deg_lat
    return lon_res_deg, lat_res_deg


# Known problematic CORADR volumes (no SAR data or navigation issues)
# Source: NLDSAR documentation (Lucas et al. 2014) and PDS catalog
SKIP_CORADR_VOLUMES = {
    "CORADR_0048",  # No SAR data
    "CORADR_0186",  # No SAR data
    "CORADR_0189",  # No SAR data
    "CORADR_0209",  # No SAR data
    "CORADR_0234",  # No SAR data
}


def read_pds_label(label_path: Path) -> dict:
    """Parse a PDS3 label file to extract key metadata.

    PDS3 labels are KEY=VALUE pairs with some nested structures.
    We extract the fields needed for BIDR processing.
    """
    metadata = {}
    with open(label_path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("/*"):
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"')
                metadata[key] = value
    return metadata


def read_bidr_file(bidr_path: Path, label_path: Optional[Path] = None) -> dict:
    """Read one PDS3 BIDR image in its stored units through GDAL.

    GDAL interprets SAMPLE_TYPE, byte order, and the one-based ^IMAGE pointer
    for attached or detached labels. An explicit label is the authoritative
    dataset; otherwise a local .LBL/.lbl sibling is preferred over the image.
    Attached images and explicit labels also accept GDAL /vsizip/ paths.

    ``data`` preserves stored samples, including negative backscatter, zero,
    and the nodata sentinel. ``valid_mask`` selects finite, unmasked samples.
    ``scale`` and ``offset`` are returned without applying them: the product
    label determines units, and display products must not be treated as linear
    backscatter. ``metadata`` contains GDAL's selected PDS metadata. Transform,
    CRS, and bounds are passed through without reprojection.

    References: https://gdal.org/en/stable/drivers/raster/pds.html and the
    Cassini BIDR SIS, sections 2.5.3 and 5.1.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError("rasterio with the GDAL PDS driver is required to read BIDR files") from exc

    dataset_path = str(label_path) if label_path is not None else str(bidr_path)
    if label_path is None and Path(bidr_path).suffix.lower() != ".lbl":
        for suffix in (".LBL", ".lbl"):
            candidate = Path(bidr_path).with_suffix(suffix)
            if candidate.is_file():
                dataset_path = str(candidate)
                break

    with rasterio.open(dataset_path, driver="PDS") as source:
        if source.count != 1:
            raise ValueError(f"A BIDR file must contain one image band, found {source.count}")
        data = source.read(1)
        valid_mask = (source.read_masks(1) != 0) & np.isfinite(data)
        return {
            "data": data,
            "valid_mask": valid_mask,
            "nodata": source.nodata,
            "lines": source.height,
            "samples": source.width,
            "metadata": source.tags(),
            "scale": source.scales[0],
            "offset": source.offsets[0],
            "transform": source.transform,
            "crs": source.crs,
            "bounds": source.bounds,
        }


def read_geotiff_mosaic(tiff_path: Path) -> dict:
    """Read a GeoTIFF mosaic (e.g., USGS HiSAR or reprojected BIDR).

    Returns dict with 'data', 'transform', 'crs', 'bounds'.
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio is required for reading GeoTIFF files")

    with rasterio.open(tiff_path) as src:
        data = src.read(1).astype(np.float32)
        return {
            "data": data,
            "transform": src.transform,
            "crs": src.crs,
            "bounds": src.bounds,
            "width": src.width,
            "height": src.height,
        }


def tile_geotiff_metric(
    tiff_path: Path,
    planet_radius_m: float,
    footprint_size_m: float = 45_000.0,
    tile_size: int = 128,
    fill_value: float = 0.0,
    stride_m: Optional[float] = None,
    tile_id_prefix: str = "titan",
) -> list[dict]:
    """Sample a square grid in a local azimuthal equidistant projection.

    Each side spans ``footprint_size_m`` metres in the tile-centered projection.
    Pixel spacing is ``footprint_size_m / tile_size``; the default grid spans
    45 km per side. These dimensions describe the projected grid.
    """
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    pixel_size = footprint_size_m / tile_size
    half = footprint_size_m / 2.0
    stride_m = footprint_size_m if stride_m is None else stride_m
    if stride_m < footprint_size_m:
        raise ValueError("stride_m must not be smaller than footprint_size_m")
    angular_step_deg = np.degrees(stride_m / planet_radius_m)
    tiles = []
    tile_idx = 0

    with rasterio.open(tiff_path) as source:
        latitudes = np.arange(-90.0 + angular_step_deg / 2.0, 90.0, angular_step_deg)
        for center_lat in latitudes:
            circumference_scale = max(np.cos(np.radians(center_lat)), 1e-6)
            lon_step = min(360.0, angular_step_deg / circumference_scale)
            longitudes = np.arange(lon_step / 2.0, 360.0, lon_step)
            for center_lon in longitudes:
                local_crs = rasterio.crs.CRS.from_string(
                    f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} "
                    f"+R={planet_radius_m} +units=m +no_defs"
                )
                destination = np.full((tile_size, tile_size), np.nan, dtype=np.float32)
                destination_transform = from_origin(-half, half, pixel_size, pixel_size)
                reproject(
                    source=rasterio.band(source, 1),
                    destination=destination,
                    src_transform=source.transform,
                    src_crs=source.crs,
                    dst_transform=destination_transform,
                    dst_crs=local_crs,
                    src_nodata=source.nodata,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
                valid_mask = (
                    np.isfinite(destination)
                    & (destination != fill_value)
                    & (destination > 0)
                )
                valid_data = destination[valid_mask]
                valid_fraction = float(valid_mask.mean())
                tiles.append({
                    "tile_id": f"{tile_id_prefix}_{tile_idx:06d}",
                    "data": np.nan_to_num(destination, nan=fill_value),
                    "center_lat": round(float(center_lat), 6),
                    "center_lon": round(float(center_lon % 360.0), 6),
                    "valid_fraction": round(valid_fraction, 6),
                    "mean_display_dn": round(float(valid_data.mean()), 4)
                    if valid_data.size else 0.0,
                    "std_display_dn": round(float(valid_data.std()), 4)
                    if valid_data.size else 0.0,
                    "footprint_size_m": footprint_size_m,
                    "pixel_size_m": pixel_size,
                    "local_projection": "azimuthal_equidistant",
                })
                tile_idx += 1

    logger.info("Generated %d local-metric tiles from %s", len(tiles), tiff_path)
    return tiles


def assign_labels_metric(
    tiles: list[dict],
    label_path: Path,
    planet_radius_m: float,
    footprint_size_m: float,
    config: TitanSARConfig,
) -> list[dict]:
    """Assign the modal map class over each local metric footprint.

    The geographic map may use signed or 0 to 360 degree longitudes. Explicit
    wrapping retains samples in that map's coordinate range after projection.
    The winning fraction describes map-class mixture, not annotation reliability.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT

    sample_size = 64
    half = footprint_size_m / 2.0
    resolution = footprint_size_m / sample_size
    with rasterio.open(label_path) as source:
        wrap_center = (source.bounds.left + source.bounds.right) / 2
        label_crs = rasterio.crs.CRS.from_string(
            f"+proj=longlat +R={planet_radius_m} +lon_wrap={wrap_center} +no_defs"
        )
        for tile in tiles:
            local_crs = rasterio.crs.CRS.from_string(
                f"+proj=aeqd +lat_0={tile['center_lat']} +lon_0={tile['center_lon']} "
                f"+R={planet_radius_m} +units=m +no_defs"
            )
            with WarpedVRT(
                source,
                src_crs=label_crs,
                crs=local_crs,
                transform=from_origin(-half, half, resolution, resolution),
                width=sample_size,
                height=sample_size,
                resampling=Resampling.nearest,
                tolerance=1e-12,
                nodata=255,
            ) as warped:
                labels = warped.read(1)
            valid = labels[(labels >= 0) & (labels < config.num_classes)]
            if valid.size == 0:
                tile["label"] = -1
                tile["label_confidence"] = 0.0
            else:
                counts = np.bincount(valid, minlength=config.num_classes)
                tile["label"] = int(np.argmax(counts))
                tile["label_confidence"] = round(
                    float(counts[tile["label"]] / valid.size), 4
                )
    return tiles


def tile_mosaic(
    data: np.ndarray,
    tile_size: int = 128,
    lon_range: tuple = (0.0, 360.0),
    lat_range: tuple = (-90.0, 90.0),
    ppd: int = 128,
    fill_value: float = 0.0,
) -> list[dict]:
    """Tile a global equirectangular mosaic into non-overlapping patches.

    Parameters:
        data: 2D array, equirectangular projection, latitude top-to-bottom
        tile_size: side length of square tiles in pixels
        lon_range, lat_range: geographic extent of the mosaic
        ppd: pixels per degree of the mosaic
        fill_value: value used for no-data pixels (typically 0.0 for BIDR)

    Returns list of dicts with tile data and metadata.
    """
    height, width = data.shape
    tiles = []
    tile_idx = 0

    for row_start in range(0, height - tile_size + 1, tile_size):
        for col_start in range(0, width - tile_size + 1, tile_size):
            tile_data = data[row_start:row_start + tile_size,
                            col_start:col_start + tile_size]

            # Compute geographic center
            center_row = row_start + tile_size / 2
            center_col = col_start + tile_size / 2
            center_lat = lat_range[1] - center_row / ppd
            center_lon = lon_range[0] + center_col / ppd

            # Valid pixel fraction (non-fill, non-zero, non-NaN)
            valid_mask = (tile_data != fill_value) & ~np.isnan(tile_data) & (tile_data > 0)
            valid_fraction = valid_mask.sum() / tile_data.size

            # HiSAR is a logarithmically stretched display product, not calibrated
            # sigma0. Keep its statistics explicitly in display-DN space.
            valid_data = tile_data[valid_mask]
            mean_display_dn = float(valid_data.mean()) if valid_data.size else 0.0
            std_display_dn = float(valid_data.std()) if valid_data.size else 0.0

            tiles.append({
                "tile_id": f"titan_{tile_idx:06d}",
                "data": tile_data,
                "center_lat": round(center_lat, 4),
                "center_lon": round(center_lon, 4),
                "valid_fraction": round(valid_fraction, 4),
                "mean_display_dn": round(mean_display_dn, 4),
                "std_display_dn": round(std_display_dn, 4),
            })
            tile_idx += 1

    logger.info(f"Generated {len(tiles)} raw tiles from mosaic {data.shape}")
    return tiles


def quality_filter(
    tiles: list[dict],
    config: TitanSARConfig,
    incidence_angles: Optional[np.ndarray] = None,
) -> list[dict]:
    """Apply quality filters to tiles.

    Removes tiles based on:
    1. Valid pixel fraction < threshold
    2. Mean incidence angle outside the inclusive bounds, when supplied

    Optional angles must be a finite numeric 1D array in the exact input tile
    order, with one value per tile. Validation precedes any tile mutation.
    Without angles, selection uses only the valid-pixel fraction.
    Returns the filtered list of tiles and logs rejection statistics.
    """
    n_original = len(tiles)
    if incidence_angles is not None:
        if np.ma.isMaskedArray(incidence_angles) and np.ma.getmaskarray(incidence_angles).any():
            raise ValueError("incidence_angles must contain a finite value for every tile")
        incidence_angles = np.asarray(incidence_angles)
        if incidence_angles.ndim != 1 or len(incidence_angles) != n_original:
            raise ValueError("incidence_angles must be 1D with exactly one value per tile")
        if (not np.issubdtype(incidence_angles.dtype, np.number)
                or np.iscomplexobj(incidence_angles)
                or not np.isfinite(incidence_angles).all()):
            raise ValueError("incidence_angles must contain finite real numeric values")
    rejected = {"low_valid_fraction": 0, "bad_incidence_angle": 0}

    filtered = []
    for i, tile in enumerate(tiles):
        # Filter 1: valid pixel fraction
        if tile["valid_fraction"] < config.min_valid_fraction:
            rejected["low_valid_fraction"] += 1
            continue

        # Filter 2: incidence angle (if available)
        if incidence_angles is not None:
            angle = incidence_angles[i]
            if angle < config.incidence_angle_min_deg or angle > config.incidence_angle_max_deg:
                rejected["bad_incidence_angle"] += 1
                continue
            tile["mean_incidence_angle"] = round(float(angle), 2)

        filtered.append(tile)

    logger.info(
        f"Quality filter: {n_original} -> {len(filtered)} tiles. "
        f"Rejected: {rejected}"
    )
    return filtered


def assign_labels(
    tiles: list[dict],
    label_map: np.ndarray,
    label_ppd: int,
    label_lon_range: tuple = (0.0, 360.0),
    label_lat_range: tuple = (-90.0, 90.0),
    config: TitanSARConfig = None,
) -> list[dict]:
    """Assign terrain class labels to tiles from a rasterized label map.

    Assign the most frequent valid class within the sampled label patch.
    The modal class can occupy less than half of the valid samples. The legacy
    ``label_confidence`` field stores its fraction of valid samples, rounded to
    four decimal places. It describes map composition within the patch.

    Parameters:
        label_map: 2D integer array (class indices 0..num_classes-1)
        label_ppd: pixels per degree of the label map
    """
    if config is None:
        config = TitanSARConfig()

    tile_half_deg = (config.tile_size / config.titan_hisar_ppd) / 2  # half tile size in degrees

    for tile in tiles:
        lat = tile["center_lat"]
        lon = tile["center_lon"]

        # Convert geographic coords to label map pixel coords
        label_row_center = int((label_lat_range[1] - lat) * label_ppd)
        label_col_center = int((lon - label_lon_range[0]) * label_ppd)

        # Tile footprint in label map pixels
        half_size_pix = max(1, int(tile_half_deg * label_ppd))
        r0 = max(0, label_row_center - half_size_pix)
        r1 = min(label_map.shape[0], label_row_center + half_size_pix)
        c0 = max(0, label_col_center - half_size_pix)
        c1 = min(label_map.shape[1], label_col_center + half_size_pix)

        label_patch = label_map[r0:r1, c0:c1]

        if label_patch.size == 0:
            tile["label"] = -1
            tile["label_confidence"] = 0.0
            continue

        # Most frequent valid class; ignore background values -1 and 255.
        valid_labels = label_patch[(label_patch >= 0) & (label_patch < config.num_classes)]
        if valid_labels.size == 0:
            tile["label"] = -1
            tile["label_confidence"] = 0.0
        else:
            counts = np.bincount(valid_labels.ravel(), minlength=config.num_classes)
            tile["label"] = int(np.argmax(counts))
            tile["label_confidence"] = round(float(counts[tile["label"]] / valid_labels.size), 4)

    return tiles


def build_catalog(
    tiles: list[dict],
    config: TitanSARConfig,
    output_dir: Path,
    require_labels: bool = True,
    provenance: Optional[dict] = None,
    benchmark_track: str = "TitanSAR-HiSAR-IMG-review-2026-09",
) -> dict:
    """Build a dataset catalog: save tiles as .npy files and metadata as JSON.

    By default this refuses to build a supervised catalog when any tile is
    missing a terrain label, because the previous behaviour silently defaulted
    missing labels to 0 (Plains), producing a plausible-looking but corrupted
    dataset (review N2). Pass ``require_labels=False`` to build an explicitly
    unlabeled catalog (tagged ``labels_available: false``).

    Returns the catalog dict (also saved to disk).
    """
    output_dir = Path(output_dir)
    tiles_dir = output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    labels_available = all("label" in t for t in tiles)
    if require_labels:
        if not labels_available:
            n_missing = sum(1 for t in tiles if "label" not in t)
            raise RuntimeError(
                f"build_catalog: {n_missing}/{len(tiles)} tiles have no 'label'. "
                "Refusing to build a supervised catalog that would silently default "
                "missing labels to 0 (Plains). Assign labels first, or call "
                "build_catalog(..., require_labels=False) for an unlabeled catalog."
            )
        # Supervised catalogs require integer class indices in [0, num_classes).
        bad = []
        for t in tiles:
            try:
                lab = _parse_label(t["label"])
            except (TypeError, ValueError):
                bad.append(t["tile_id"])
                continue
            if not (0 <= lab < config.num_classes):
                bad.append(t["tile_id"])
        if bad:
            raise RuntimeError(
                f"build_catalog: {len(bad)}/{len(tiles)} tiles have labels outside "
                f"[0, {config.num_classes}) (e.g. {bad[:3]}). Supervised catalogs "
                "require valid class indices."
            )
    else:
        # Missing labels become -1. Supplied labels must be -1 or a valid class
        # index in [0, num_classes).
        bad = []
        for t in tiles:
            if "label" not in t:
                continue
            try:
                lab = _parse_label(t["label"])
            except (TypeError, ValueError):
                bad.append(t["tile_id"])
                continue
            if lab != -1 and not (0 <= lab < config.num_classes):
                bad.append(t["tile_id"])
        if bad:
            raise RuntimeError(
                f"build_catalog: {len(bad)} tiles have labels that are neither -1 nor "
                f"a valid class in [0, {config.num_classes}) (e.g. {bad[:3]})."
            )

    catalog_entries = []
    for tile in tiles:
        # Save tile data
        tile_path = tiles_dir / f"{tile['tile_id']}.npy"
        np.save(tile_path, tile["data"])
        tile_sha256 = hashlib.sha256(tile_path.read_bytes()).hexdigest()

        # Assign split
        split = assign_split(tile["center_lat"], tile["center_lon"], config)

        valid_pixels = np.asarray(tile["data"])
        valid_pixels = valid_pixels[np.isfinite(valid_pixels) & (valid_pixels > 0)]
        mean_display_dn = tile.get(
            "mean_display_dn", float(valid_pixels.mean()) if valid_pixels.size else 0.0
        )
        std_display_dn = tile.get(
            "std_display_dn", float(valid_pixels.std()) if valid_pixels.size else 0.0
        )
        entry = {
            "tile_id": tile["tile_id"],
            "tile_sha256": tile_sha256,
            "center_lat": tile["center_lat"],
            "center_lon": tile["center_lon"],
            "valid_fraction": tile["valid_fraction"],
            "mean_display_dn": round(float(mean_display_dn), 4),
            "std_display_dn": round(float(std_display_dn), 4),
            "label": (_parse_label(tile["label"]) if "label" in tile else -1),  # -1 = unlabeled
            "label_confidence": tile.get("label_confidence", 0.0),
            "mean_incidence_angle": tile.get("mean_incidence_angle"),
            "has_vims_display_sample": tile.get("has_vims", False),
            "vims_display_sample": tile.get("vims_display_sample"),
            "has_tb": tile.get("has_tb", False),
            "brightness_temp": tile.get("brightness_temp"),
            "split": split,
        }
        catalog_entries.append(entry)

    # Compute summary statistics
    splits = {}
    for entry in catalog_entries:
        s = entry["split"]
        if s not in splits:
            splits[s] = {"total": 0, "per_class": [0] * config.num_classes}
        splits[s]["total"] += 1
        if 0 <= entry["label"] < config.num_classes:
            splits[s]["per_class"][entry["label"]] += 1

    catalog = {
        "schema_version": "1.0.0",
        "benchmark_track": benchmark_track,
        "domain": "titan",
        "source_product": "USGS Cassini SAR HiSAR Global Mosaic 351m",
        "tile_size": config.tile_size,
        "footprint_size_m": config.footprint_size_m,
        "target_pixel_size_m": config.target_pixel_size_m,
        "num_classes": config.num_classes,
        "class_names": list(config.terrain_classes),
        "labels_available": labels_available,
        "intensity_space": "hisar_log_dn",
        "split_policy": "external_spatial_block_manifest",
        "provenance": provenance or {},
        "splits": splits,
        "tiles": catalog_entries,
    }

    catalog_path = output_dir / "catalog.json"
    with open(catalog_path, "w") as f:
        json.dump(catalog, f, indent=2)

    logger.info(f"Catalog saved to {catalog_path}: {len(catalog_entries)} tiles")
    for s, info in splits.items():
        logger.info(f"  {s}: {info['total']} tiles, per-class: {info['per_class']}")

    return catalog
