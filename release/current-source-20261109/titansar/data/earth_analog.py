"""Earth-analog SAR dataset construction from Sentinel-1.

Collects and processes Sentinel-1 C-band (5.405 GHz) VV-polarization SAR
imagery over terrain analogs for Titan's 6-class terrain ontology.

Processing chain (ESA standard): apply orbit file, thermal noise removal,
radiometric calibration to sigma0, Range-Doppler terrain correction.

The released benchmark tiles were produced by scripts/process_sentinel1_snap.py
running scripts/snap_validation_graph.xml through SNAP gpt (Copernicus 30m
Global DEM, 351.5625 m/pixel). The snappy path in this module is an older
variant (SRTM 1Sec HGT, 175 m/pixel default) and did not produce the
released data.

Analog sites:
  - Dunes: Namib Sand Sea, Grand Erg Oriental, Rub' al Khali
  - Craters: Manicouagan, Mistastin, Haughton (Arctic ice-rich)
  - Lakes: Arctic thermokarst, Salar de Uyuni
  - Labyrinths: Atacama channels, Madagascar tsingy
  - Plains: Arctic tundra, Icelandic sandur, desert pavement

Site labels describe candidate terrain analogs. They do not establish class
membership for every tile or comparable backscatter between Earth and Titan.
Water's microwave permittivity is complex and depends on frequency,
temperature and salinity; a single real constant cannot establish the
radiometric similarity of these sites. See Meissner and Wentz 2004,
doi:10.1109/TGRS.2004.831888.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from titansar.configs.defaults import TitanSARConfig

logger = logging.getLogger(__name__)

# Class index mapping (must match TitanSAR classes)
CLASS_MAP = {
    "plains": 0,
    "dunes": 1,
    "hummocky": 2,
    "labyrinths": 3,
    "lakes": 4,
    "craters": 5,
}


@dataclass
class AnalogSite:
    """Definition of an Earth-analog site for Sentinel-1 collection."""
    name: str
    terrain_class: str
    # Bounding box: (min_lon, min_lat, max_lon, max_lat)
    bbox: tuple[float, float, float, float]
    continent: str  # For cross-geography split


# Analog site definitions with approximate bounding boxes
ANALOG_SITES = [
    # --- Dunes ---
    AnalogSite("namib_sand_sea", "dunes", (14.5, -25.5, 16.0, -24.0), "Africa"),
    AnalogSite("grand_erg_oriental", "dunes", (6.0, 30.0, 8.5, 32.5), "Africa"),
    AnalogSite("rub_al_khali", "dunes", (49.0, 19.0, 52.0, 22.0), "Asia"),
    # --- Craters ---
    AnalogSite("manicouagan", "craters", (-69.0, 51.0, -68.0, 52.0), "N_America"),
    AnalogSite("mistastin", "craters", (-63.5, 55.5, -62.5, 56.5), "N_America"),
    AnalogSite("haughton", "craters", (-90.0, 75.0, -89.0, 76.0), "N_America"),
    # --- Lakes ---
    AnalogSite("arctic_thermokarst", "lakes", (150.0, 68.0, 155.0, 71.0), "Asia"),
    AnalogSite("salar_de_uyuni", "lakes", (-68.0, -21.0, -66.5, -19.5), "S_America"),
    # --- Labyrinths ---
    AnalogSite("atacama_channels", "labyrinths", (-70.5, -24.0, -69.0, -22.5), "S_America"),
    # Wadi Rum is withheld: the former longitude box selected Egypt. A verified
    # Jordan acquisition boundary and footprint labels are still required.
    # See audit/EARTH_SITE_REVIEW_2026-09-29.md; historical catalogs are unchanged.
    AnalogSite("madagascar_tsingy", "labyrinths", (44.5, -19.5, 45.5, -18.5), "Africa"),
    # --- Plains ---
    AnalogSite("arctic_tundra", "plains", (-140.0, 69.0, -135.0, 71.0), "N_America"),
    AnalogSite("icelandic_sandur", "plains", (-18.5, 63.5, -17.0, 64.5), "Europe"),
    AnalogSite("desert_pavement", "plains", (-116.0, 35.0, -115.0, 36.0), "N_America"),
]


def get_sentinel1_search_params(site: AnalogSite) -> dict:
    """Return descriptive Sentinel-1 selection criteria for a site.

    The caller must translate this dictionary to the chosen catalog API.
    Copernicus Data Space OData queries use ``$filter``, ``$orderby`` and
    ``$top``, with ``OData.CSC.Intersects`` for geographic filtering.
    See https://documentation.dataspace.copernicus.eu/APIs/OData.html.
    """
    min_lon, min_lat, max_lon, max_lat = site.bbox
    return {
        "collection": "SENTINEL-1",
        "productType": "GRD",
        "sensorMode": "IW",
        "polarisation": "VV",
        "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "maxRecords": 10,
        "sortBy": "startDate",
        "sortOrder": "descending",
    }


def process_sentinel1_grd(
    input_path: Path,
    output_path: Path,
    target_pixel_size_m: float = 175.0,
) -> Path:
    """Process a Sentinel-1 GRD product to calibrated σ⁰.

    Processing chain (following ESA standard):
    1. Apply orbit file (for precise positioning)
    2. Thermal noise removal
    3. Radiometric calibration (to sigma0)
    4. Terrain correction (using SRTM 1-arcsec DEM)
    5. Resample to target pixel size

    This legacy function requires ESA SNAP/snappy. If snappy is unavailable,
    the rasterio path raises because it cannot calibrate GRD DN to sigma0.
    The released processing route uses scripts/process_sentinel1_snap.py
    with ESA SNAP gpt and scripts/snap_validation_graph.xml.

    Args:
        input_path: Path to Sentinel-1 .zip or .SAFE product
        output_path: Path for output GeoTIFF
        target_pixel_size_m: Target pixel size in meters (legacy default
            175 m; the released tiles use 351.5625 m via the gpt graph)

    Returns:
        Path to processed GeoTIFF
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Try SNAP/snappy processing
        return _process_with_snap(input_path, output_path, target_pixel_size_m)
    except ImportError:
        logger.warning(
            "ESA SNAP/snappy not available. The rasterio path cannot calibrate "
            "GRD to sigma0 and will raise. Use scripts/process_sentinel1_snap.py "
            "with ESA SNAP gpt (--snap_gpt) and "
            "--graph scripts/snap_validation_graph.xml; see scripts/README.md."
        )
        return _process_with_rasterio(input_path, output_path, target_pixel_size_m)


def _process_with_snap(
    input_path: Path,
    output_path: Path,
    target_pixel_size_m: float,
) -> Path:
    """Process Sentinel-1 GRD using ESA SNAP via snappy."""
    import snappy
    from snappy import ProductIO, GPF, HashMap

    product = ProductIO.readProduct(str(input_path))

    # 1. Apply orbit file
    params = HashMap()
    product = GPF.createProduct("Apply-Orbit-File", params, product)

    # 2. Thermal noise removal
    params = HashMap()
    params.put("removeThermalNoise", True)
    product = GPF.createProduct("ThermalNoiseRemoval", params, product)

    # 3. Radiometric calibration to sigma0
    params = HashMap()
    params.put("outputSigmaBand", True)
    params.put("selectedPolarisations", "VV")
    product = GPF.createProduct("Calibration", params, product)

    # 4. Terrain correction
    params = HashMap()
    params.put("demName", "SRTM 1Sec HGT")
    params.put("pixelSpacingInMeter", target_pixel_size_m)
    product = GPF.createProduct("Terrain-Correction", params, product)

    # Write output
    ProductIO.writeProduct(product, str(output_path), "GeoTIFF")
    return output_path


def _process_with_rasterio(
    input_path: Path,
    output_path: Path,
    target_pixel_size_m: float,
) -> Path:
    """Raise an error: rasterio fallback cannot apply radiometric calibration."""
    # Calibration requires the product's annotation lookup tables.
    # The released route applies them through the SNAP gpt graph.
    raise RuntimeError(
        "The rasterio fallback for Sentinel-1 processing does not apply "
        "radiometric calibration. Sentinel-1 GRD DN values are not sigma0. "
        "Use scripts/process_sentinel1_snap.py with ESA SNAP gpt (--snap_gpt) "
        "and --graph scripts/snap_validation_graph.xml; see scripts/README.md."
    )


def tile_earth_analog(
    processed_path: Path,
    site: AnalogSite,
    tile_size: int = 128,
) -> list[dict]:
    """Tile a processed Earth-analog SAR image into 128×128 patches.

    Labels are assigned by site identity (location-based labeling).
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio required for tiling")

    with rasterio.open(processed_path) as src:
        data = src.read(1).astype(np.float32)

    tiles = []
    tile_idx = 0
    label = CLASS_MAP[site.terrain_class]

    for row in range(0, data.shape[0] - tile_size + 1, tile_size):
        for col in range(0, data.shape[1] - tile_size + 1, tile_size):
            tile_data = data[row:row + tile_size, col:col + tile_size]

            # Skip tiles with too many invalid pixels
            valid_mask = (tile_data > 0) & np.isfinite(tile_data)
            valid_fraction = valid_mask.sum() / tile_data.size
            if valid_fraction < 0.80:
                continue

            valid_data = tile_data[valid_mask]
            values_db = 10.0 * np.log10(np.clip(valid_data, 1e-4, None))
            mean_db = float(values_db.mean())
            std_db = float(values_db.std())

            tiles.append({
                "tile_id": f"earth_{site.name}_{tile_idx:05d}",
                "data": tile_data,
                "label": label,
                "site": site.name,
                "terrain_class": site.terrain_class,
                "continent": site.continent,
                "valid_fraction": round(valid_fraction, 4),
                "mean_sigma0_db": round(mean_db, 2),
                "std_sigma0_db": round(std_db, 2),
            })
            tile_idx += 1

    logger.info(f"Site {site.name}: {len(tiles)} tiles (class={site.terrain_class})")
    return tiles


def build_earth_analog_catalog(
    all_tiles: list[dict],
    output_dir: Path,
    test_fraction: float = 0.2,
    cross_geography: bool = False,
    seed: int = 42,
) -> dict:
    """Build an unsplit Earth catalog for external group-aware manifests."""
    if cross_geography or test_fraction != 0.2 or seed != 42:
        raise ValueError(
            "Inline Earth splitting was removed. Build an unsplit catalog, then "
            "use scripts/build_split_manifests.py."
        )
    output_dir = Path(output_dir)

    tiles_dir = output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    catalog_entries = []
    for tile in all_tiles:
        # Save tile
        np.save(tiles_dir / f"{tile['tile_id']}.npy", tile["data"])

        entry = {
            "tile_id": tile["tile_id"],
            "label": tile["label"],
            "site": tile["site"],
            "terrain_class": tile["terrain_class"],
            "continent": tile["continent"],
            "source_product_id": tile.get("source_product_id", tile["site"]),
            "valid_fraction": tile["valid_fraction"],
            "mean_sigma0_db": tile["mean_sigma0_db"],
            "std_sigma0_db": tile["std_sigma0_db"],
        }
        catalog_entries.append(entry)

    catalog = {
        "schema_version": "1.0.0",
        "benchmark_track": "TitanSAR-HiSAR-IMG-v1",
        "domain": "earth",
        "synthetic": False,
        "source_type": "real_sentinel1",
        "intensity_space": "linear_sigma0",
        "labels_available": True,
        "class_names": list(CLASS_MAP.keys()),
        "split_policy": "external_group_manifest",
        "tiles": catalog_entries,
    }

    catalog_path = output_dir / "catalog.json"
    with open(catalog_path, "w") as f:
        json.dump(catalog, f, indent=2)

    logger.info(f"Earth-analog catalog: {len(catalog_entries)} tiles saved to {catalog_path}")
    return catalog
