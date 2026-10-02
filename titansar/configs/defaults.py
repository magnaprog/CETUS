"""Default configuration for TitanSAR experiments."""

from dataclasses import dataclass, field
from typing import Optional


# === Instrument values and reference assumptions ===

# Cassini RADAR: Ku-band
CASSINI_FREQ_GHZ = 13.78  # GHz
CASSINI_WAVELENGTH_CM = 2.176  # cm (c / f = 2.998e10 / 13.78e9 = 2.176 cm)
CASSINI_WAVELENGTH_UM = 21_760  # micrometers (unused: the DOFA path passes GHz)
# CETUS follows the pinned DOFA README's 5.405 Sentinel-1 example and supplies
# 13.78 for Cassini. The paper's version 2 instead describes a Sentinel-1
# identifier of 3.75. Checkpoint-specific pretraining identifiers remain
# unresolved; see DOFAWrapper. These wavelength constants are reference values.

# Sentinel-1: C-band
SENTINEL1_FREQ_GHZ = 5.405  # GHz
SENTINEL1_WAVELENGTH_CM = 5.547  # cm (c / f = 2.998e10 / 5.405e9 = 5.547 cm)
SENTINEL1_WAVELENGTH_UM = 55_470  # micrometers (reference only)

# Venus Magellan: S-band. Note S (2.385) < C (5.405) < Ku (13.78) GHz, so Venus
# is the lowest-frequency of the three SAR domains, not spectrally intermediate.
MAGELLAN_FREQ_GHZ = 2.385  # GHz
MAGELLAN_WAVELENGTH_CM = 12.57  # cm (c / f = 2.998e10 / 2.385e9 = 12.57 cm)
MAGELLAN_PIXEL_SIZE_M = 75.0  # meters per pixel (FMAP mosaic)

# Titan parameters
TITAN_RADIUS_KM = 2574.7  # km; reference sphere used by benchmark geometry
TITAN_SURFACE_TEMP_K = 94  # K (surface temperature)
TITAN_SURFACE_PRESSURE_BAR = 1.47  # bar (about 1.45 atm)

# Rayleigh roughness thresholds at theta=30 deg
# Formula: delta_h < lambda / (8 * cos(theta))
# Ku-band: 2.176 / (8 * cos(30°)) = 2.176 / 6.928 = 0.314 cm = 3.14 mm
RAYLEIGH_THRESHOLD_KU_MM = 3.14
# C-band: 5.547 / (8 * cos(30°)) = 5.547 / 6.928 = 0.801 cm = 8.01 mm
RAYLEIGH_THRESHOLD_C_MM = 8.01

# Unused illustrative relative-permittivity ranges, without universal material bounds.
TITAN_DIELECTRIC_RANGE = (1.5, 3.6)
EARTH_SOIL_DIELECTRIC_RANGE = (3.0, 30.0)
# Unused assumption; its exact source and measurement conditions remain unverified.
WATER_ICE_LOSS_TANGENT_94K = 1e-4


@dataclass
class TitanSARConfig:
    """Main configuration for TitanSAR experiments."""

    # === Data paths ===
    data_root: str = "./data"
    titan_sar_dir: str = "./data/titan_sar"
    titan_vims_dir: str = "./data/titan_vims"
    earth_analog_dir: str = "./data/earth_analog_real"  # real Sentinel-1; synthetic is quarantined elsewhere
    labels_dir: str = "./data/labels"
    output_dir: str = "./outputs"

    # === Tiling ===
    tile_size: int = 128  # pixels per side
    footprint_size_m: float = 45_000.0
    target_pixel_size_m: float = 45_000.0 / 128.0
    titan_radius_m: float = TITAN_RADIUS_KM * 1000.0
    titan_ppd: int = 256  # reference BIDR grid; actual tiling reads product georeferencing
    titan_hisar_ppd: int = 128  # pixels per degree for USGS HiSAR mosaic
    earth_pixel_size_m: float = 45_000.0 / 128.0

    # === Quality filtering ===
    min_valid_fraction: float = 0.80  # minimum fraction of valid (non-fill) pixels
    incidence_angle_min_deg: float = 15.0  # applies only when angle observations are supplied
    incidence_angle_max_deg: float = 35.0  # current HiSAR preparation supplies no angles

    # === Terrain classes ===
    # Lopes et al. (2020) 6-class ontology
    terrain_classes: tuple = (
        "plains",
        "dunes",
        "hummocky",
        "labyrinths",
        "lakes",
        "craters",
    )
    num_classes: int = 6

    # === Split metadata ===
    # Repaired experiments use catalog-bound contiguous-sector manifests with gaps.
    # Historical v1 uses spatial-block manifests without boundary buffers.
    # The center/radius below defines the configured Selk exclusion only.
    selk_center_lat: float = 6.0  # degrees N
    selk_center_lon: float = 161.0  # degrees E (199W = 161E)
    selk_radius_deg: float = 5.0

    # === Model input ===
    model_input_size: int = 224  # resize tiles to this for ViT input
    interpolation: str = "bilinear"

    # === Linear probing ===
    probe_lr: float = 0.01
    probe_epochs: int = 100
    probe_batch_size: int = 256
    probe_seeds: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    use_class_weights: bool = True  # inverse-frequency weighting

    # === k-NN ===
    knn_k: int = 20

    # === Fine-tuning ===
    finetune_lr: float = 1e-4
    finetune_epochs: int = 50
    finetune_unfreeze_blocks: int = 2  # last N transformer blocks

    # === Foundation models ===
    foundation_models: tuple = ("dinov2", "dofa", "croma", "random_init")
    embedding_dim: int = 768  # ViT-B embedding dimension

    # === Auxiliary VIMS display product ===
    # These channels are visualization metadata, not calibrated spectra.
    vims_bands: int = 3  # unused expected display-channel count; reader retains all bands

    # === Earth analog sites ===
    # The selected Earth source set omits Hummocky. Transfer scoring therefore
    # covers five of six classes; this does not establish an absence of analogs.
    earth_analog_sites: dict = field(default_factory=lambda: {
        "dunes": ["namib_sand_sea", "grand_erg_oriental", "rub_al_khali"],
        "craters": ["manicouagan", "mistastin", "haughton"],
        "lakes": ["arctic_thermokarst", "salar_de_uyuni"],
        "labyrinths": ["atacama_channels", "madagascar_tsingy"],
        "plains": ["arctic_tundra", "icelandic_sandur", "desert_pavement"],
    })
