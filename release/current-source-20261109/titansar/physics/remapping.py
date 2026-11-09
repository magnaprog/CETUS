"""Experimental quantile alignment of marginal backscatter distributions.

Surface material, roughness, viewing geometry and processing can affect radar
returns. This transform estimates none of these physical properties and cannot
identify why two image collections differ. Quantile alignment is a statistical
operation; it does not establish physical radiometric comparability.

The Rayleigh criterion gives reference roughness thresholds for a wavelength
and incidence angle. Those calculations do not determine the statistics of the
observed mosaics or validate a dielectric explanation for their differences.

The HiSAR image benchmark does not use this module. Its inputs are display
values, and EarthAnalogDataset rejects a remapper for that track. Any future
application to calibrated backscatter requires a separate product, masking and
acquisition protocol. This module supplies no forward scattering model,
volume-scattering estimate or validated speckle treatment.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from titansar.configs.defaults import TitanSARConfig
from titansar.data.dataset import db_to_sigma0, sigma0_to_db

logger = logging.getLogger(__name__)


def compute_quantile_mapping(
    source_db: np.ndarray,
    target_db: np.ndarray,
    n_quantiles: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute piecewise-linear quantile mapping from source to target distribution.

    For each percentile p ∈ [0, 100], maps source_σ⁰(p) -> target_σ⁰(p).

    Args:
        source_db: 1D array of source domain σ⁰ values (dB)
        target_db: 1D array of target domain σ⁰ values (dB)
        n_quantiles: number of quantile points for the mapping

    Returns:
        source_quantiles: (n_quantiles,) array of source percentile values (dB)
        target_quantiles: (n_quantiles,) array of target percentile values (dB)
    """
    percentiles = np.linspace(0, 100, n_quantiles)
    source_quantiles = np.percentile(source_db, percentiles)
    target_quantiles = np.percentile(target_db, percentiles)
    return source_quantiles, target_quantiles


def apply_quantile_mapping(
    data_db: np.ndarray,
    source_quantiles: np.ndarray,
    target_quantiles: np.ndarray,
) -> np.ndarray:
    """Apply quantile mapping to remap data from source to target distribution.

    Uses piecewise-linear interpolation between quantile points.

    Args:
        data_db: array of σ⁰ values (dB) in the source domain
        source_quantiles: source percentile values from compute_quantile_mapping
        target_quantiles: target percentile values from compute_quantile_mapping

    Returns:
        Values in dB after interpolation through the quantile pairs, with
        values outside the source range clamped to the target endpoints.
    """
    return np.interp(data_db, source_quantiles, target_quantiles)


def collect_pixel_sample_db(
    tiles_linear,
    pixels_per_tile: int = 256,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Pool a pixel-level dB sample from linear-sigma0 tiles for quantile fitting.

    ``remap_tile`` transforms individual pixels, so its cumulative distributions
    should describe pixel values. Tile means omit variation within each tile;
    values outside the fitted quantile range map to its endpoints.

    Args:
        tiles_linear: iterable of 2D arrays of sigma0 in linear scale
        pixels_per_tile: max pixels kept per tile to cap memory (<=0 keeps all)
        rng: NumPy Generator for reproducible sampling (default: fresh default_rng)

    Returns:
        1D array of pooled sigma0 values in dB (empty if no valid pixels)
    """
    rng = np.random.default_rng() if rng is None else rng
    chunks = []
    for tile in tiles_linear:
        flat = np.asarray(tile, dtype=np.float64).ravel()
        flat = flat[np.isfinite(flat) & (flat > 0.0)]
        if flat.size == 0:
            continue
        if 0 < pixels_per_tile < flat.size:
            flat = flat[rng.choice(flat.size, size=pixels_per_tile, replace=False)]
        chunks.append(flat)
    if not chunks:
        return np.zeros(0)
    return sigma0_to_db(np.concatenate(chunks))


def load_tile_pixel_db(
    tiles_dir,
    tile_ids,
    n_tiles: int = 200,
    pixels_per_tile: int = 256,
    seed: int = 0,
) -> np.ndarray:
    """Load a random subset of per-tile .npy arrays and pool a pixel dB sample.

    Sample tile IDs, read their ``{tile_id}.npy`` linear-sigma0 arrays, and
    collect pixel values in dB with ``collect_pixel_sample_db``.

    Args:
        tiles_dir: directory holding ``{tile_id}.npy`` linear-sigma0 tiles
        tile_ids: candidate tile ids (typically one split)
        n_tiles: number of tiles to sample (<=0 or >= available uses all)
        pixels_per_tile: pixels sampled per tile
        seed: RNG seed for both tile and pixel selection

    Returns:
        1D pooled sigma0 dB sample (np.zeros(0) if nothing loadable)
    """
    rng = np.random.default_rng(seed)
    ids = list(tile_ids)
    if 0 < n_tiles < len(ids):
        ids = [ids[i] for i in rng.choice(len(ids), size=n_tiles, replace=False)]
    tiles_dir = Path(tiles_dir)

    def _iter_tiles():
        for tid in ids:
            path = tiles_dir / f"{tid}.npy"
            if path.exists():
                yield np.load(path)

    return collect_pixel_sample_db(
        _iter_tiles(), pixels_per_tile=pixels_per_tile, rng=rng
    )


class BackscatterRemapper:
    """Quantile mapping between two training backscatter distributions.

    Fit one mapping across all terrain classes using samples of calibrated
    Earth and Titan sigma0 in dB. ``remap_tile`` applies it to individual pixels.
    """

    def __init__(self, n_quantiles: int = 1000):
        self.n_quantiles = n_quantiles
        self.source_quantiles = None
        self.target_quantiles = None
        self.is_fitted = False

    def fit(
        self,
        source_db: np.ndarray,
        target_db: np.ndarray,
    ):
        """Fit the quantile mapping between two calibrated sigma0 distributions.

        Use training pixels from both domains, expressed as calibrated sigma0 in
        dB. Display DN cannot supply the required backscatter measurements.

        Args:
            source_db: 1D source-domain training pixel sample in dB, for example
                from ``load_tile_pixel_db``
            target_db: 1D pixel-level target-domain training sigma0 sample (dB)
        """
        self.source_quantiles, self.target_quantiles = compute_quantile_mapping(
            source_db, target_db, self.n_quantiles
        )
        self.is_fitted = True
        logger.info(
            f"Fitted quantile mapping: source [{source_db.min():.1f}, "
            f"{source_db.max():.1f}] dB to target "
            f"[{target_db.min():.1f}, {target_db.max():.1f}] dB"
        )

    def transform(self, sigma0_db: np.ndarray) -> np.ndarray:
        """Apply the fitted quantile mapping to remap σ⁰ values.

        Args:
            sigma0_db: σ⁰ values in dB (any shape)

        Returns:
            Remapped σ⁰ values in dB (same shape)
        """
        if not self.is_fitted:
            raise RuntimeError("Remapper not fitted. Call fit() first.")

        original_shape = sigma0_db.shape
        flat = sigma0_db.ravel()
        remapped = apply_quantile_mapping(flat, self.source_quantiles, self.target_quantiles)
        return remapped.reshape(original_shape)

    def remap_tile(self, tile_linear: np.ndarray) -> np.ndarray:
        """Remap a full 2D tile from linear σ⁰ space.

        Converts to dB, applies mapping, converts back to linear.

        Args:
            tile_linear: (H, W) array of σ⁰ in linear scale

        Returns:
            Remapped tile in linear σ⁰ scale
        """
        tile_db = sigma0_to_db(tile_linear)
        remapped_db = self.transform(tile_db)
        return db_to_sigma0(remapped_db)


class PerClassRemapper:
    """Prototype for per-class quantile remapping using terrain labels.

    Fitting requires labeled training samples from both domains. Transforming
    a value also requires its terrain label. This prototype is not wired into
    a benchmark experiment.
    """

    def __init__(self, num_classes: int = 6, n_quantiles: int = 500):
        self.num_classes = num_classes
        self.n_quantiles = n_quantiles
        self.class_mappers: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.is_fitted = False

    def fit(
        self,
        earth_sigma0_db: np.ndarray,
        earth_labels: np.ndarray,
        titan_sigma0_db: np.ndarray,
        titan_labels: np.ndarray,
    ):
        """Fit per-class quantile mappings.

        Estimate each class's cumulative distribution from sigma0 values and
        their labels. Applying the mapping to pixels requires pixel samples
        with corresponding class labels. The benchmark does not use this class.

        Args:
            earth_sigma0_db: (N,) Earth σ⁰ values (dB), aligned with earth_labels
            earth_labels: (N,) terrain class label per value
            titan_sigma0_db: (N,) Titan σ⁰ values (dB), aligned with titan_labels
            titan_labels: (N,) terrain class label per value
        """
        self.is_fitted = False
        self.class_mappers.clear()
        for c in range(self.num_classes):
            earth_mask = earth_labels == c
            titan_mask = titan_labels == c

            if earth_mask.sum() < 10 or titan_mask.sum() < 10:
                logger.warning(
                    f"Class {c}: insufficient samples (Earth={earth_mask.sum()}, "
                    f"Titan={titan_mask.sum()}). Skipping per-class mapping."
                )
                continue

            src_q, tgt_q = compute_quantile_mapping(
                earth_sigma0_db[earth_mask],
                titan_sigma0_db[titan_mask],
                self.n_quantiles,
            )
            self.class_mappers[c] = (src_q, tgt_q)

        self.is_fitted = True
        logger.info(f"Fitted per-class mappings for {len(self.class_mappers)} classes")

    def transform(self, sigma0_db: float, label: int) -> float:
        """Remap a single σ⁰ value given its terrain class."""
        if not self.is_fitted:
            raise RuntimeError("Per-class remapper not fitted.")
        if label not in self.class_mappers:
            return sigma0_db  # No mapping available; return unchanged
        src_q, tgt_q = self.class_mappers[label]
        return float(np.interp(sigma0_db, src_q, tgt_q))


def rayleigh_roughness_threshold(wavelength_cm: float, incidence_angle_deg: float) -> float:
    """Compute the Rayleigh smooth-surface roughness threshold.

    A surface appears electromagnetically smooth when its rms height
    variation satisfies: Δh < λ / (8 cos θ)

    Args:
        wavelength_cm: radar wavelength in centimeters
        incidence_angle_deg: incidence angle in degrees

    Returns:
        Maximum rms height (in mm) for the surface to appear smooth
    """
    theta_rad = np.deg2rad(incidence_angle_deg)
    threshold_cm = wavelength_cm / (8.0 * np.cos(theta_rad))
    return threshold_cm * 10.0  # convert cm to mm


def compute_roughness_ambiguity_range(
    wavelength_c_cm: float = 5.547,
    wavelength_ku_cm: float = 2.176,
    incidence_angle_deg: float = 30.0,
) -> tuple[float, float]:
    """Compute the roughness range where surfaces appear different at two frequencies.

    Surfaces with roughness in this range appear smooth at the longer wavelength
    (C-band) but rough at the shorter wavelength (Ku-band).

    Returns:
        (lower_mm, upper_mm): the ambiguous roughness range in millimeters
    """
    threshold_ku = rayleigh_roughness_threshold(wavelength_ku_cm, incidence_angle_deg)
    threshold_c = rayleigh_roughness_threshold(wavelength_c_cm, incidence_angle_deg)
    return threshold_ku, threshold_c


def compute_dielectric_contrast_ratio(
    sigma0_per_class: dict[str, float],
) -> float:
    """Compute a descriptive ratio of differences among class mean backscatter.

    The legacy function name does not imply a dielectric inversion.

    Defined as: (σ⁰_max_class - σ⁰_min_class) / σ⁰_mean
    A compressed ratio (small value) means subtle inter-class differences.

    Args:
        sigma0_per_class: dict mapping class name -> mean σ⁰ (linear)
    """
    values = list(sigma0_per_class.values())
    if len(values) < 2:
        return 0.0
    mean_val = np.mean(values)
    if mean_val == 0:
        return 0.0
    return float((max(values) - min(values)) / mean_val)
