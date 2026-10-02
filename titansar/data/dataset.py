"""TitanSAR dataset class and tile management."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from titansar.configs.defaults import TitanSARConfig

logger = logging.getLogger(__name__)


class TitanSARTile:
    """A single SAR tile with metadata."""

    __slots__ = (
        "tile_id", "data", "label", "label_confidence",
        "center_lat", "center_lon", "mean_display_dn", "std_display_dn",
        "mean_incidence_angle", "valid_fraction",
        "vims_display_sample", "has_vims_display_sample", "brightness_temp", "has_tb",
        "split",
    )

    def __init__(self, tile_id: str, data: np.ndarray, label: int, **kwargs):
        self.tile_id = tile_id
        self.data = data  # shape: (H, W), product-native intensity space
        self.label = label
        for k, v in kwargs.items():
            setattr(self, k, v)


def sigma0_to_db(sigma0_linear: np.ndarray) -> np.ndarray:
    """Convert linear sigma0 to dB. Clips at -40 dB floor."""
    sigma0_linear = np.clip(sigma0_linear, 1e-4, None)
    return 10.0 * np.log10(sigma0_linear)


def db_to_sigma0(sigma0_db: np.ndarray) -> np.ndarray:
    """Convert dB sigma0 back to linear scale."""
    return 10.0 ** (sigma0_db / 10.0)


def compute_norm_range(
    mean_db,
    std_db=None,
    n_sigma: float = 3.0,
    default=(-30.0, 0.0),
):
    """Estimate a [lo, hi] dB range for tile normalization from training statistics.

    Center the range on the mean of training-tile mean dB values. Estimate pixel
    variance as mean within-tile variance plus variance of tile means, weighting
    tiles equally:

        sigma_pixel = sqrt( mean(per-tile var) + var(per-tile means) )

    Include the within-tile term when all ``std_sigma0_db`` values are supplied.
    Otherwise, use only the variance of tile means. Set the bounds to the mean
    plus or minus ``n_sigma`` estimated standard deviations.

    Returns ``default`` if no training stats are available.
    """
    if mean_db is None or len(mean_db) == 0:
        return default
    mean_db = np.asarray(mean_db, dtype=float)
    mu = float(mean_db.mean())
    var_means = float(mean_db.var())
    within = 0.0
    if std_db is not None and len(std_db) == len(mean_db):
        within = float(np.mean(np.asarray(std_db, dtype=float) ** 2))
    sigma = float(np.sqrt(max(within + var_means, 0.0)))
    if sigma == 0.0:
        sigma = 1.0  # avoid a zero-width range on degenerate (constant) input
    return mu - n_sigma * sigma, mu + n_sigma * sigma


def _train_db_stats(train_entries):
    """Pull (mean_db list, std_db list-or-None) from training catalog entries."""
    mean_db = [e["mean_sigma0_db"] for e in train_entries]
    std_db = [e.get("std_sigma0_db") for e in train_entries]
    if any(s is None for s in std_db):
        std_db = None
    return mean_db, std_db


def _sample_train_pixels(train_entries, tiles_dir, n_tiles=200, px_per_tile=512, seed=0,
                         return_metadata=False):
    """Pool a sample of valid raw pixel values from train tiles (DN-space range).

    Reads ``{tile_id}.npy`` arrays directly (no log), keeping finite positive
    pixels, capped per tile to bound memory. Used to set the linear normalization
    range for the HiSAR DN domain (see ``compute_linear_norm_range``).
    """
    rng = np.random.default_rng(seed)
    ids = [e["tile_id"] for e in train_entries]
    if 0 < n_tiles < len(ids):
        ids = [ids[i] for i in rng.choice(len(ids), size=n_tiles, replace=False)]
    tiles_dir = Path(tiles_dir)
    chunks = []
    used_ids, missing_ids, empty_ids = [], [], []
    for tid in ids:
        path = tiles_dir / f"{tid}.npy"
        if not path.exists():
            missing_ids.append(tid)
            continue
        a = np.load(path).astype(np.float64).ravel()
        a = a[np.isfinite(a) & (a > 0.0)]
        if a.size == 0:
            empty_ids.append(tid)
            continue
        if 0 < px_per_tile < a.size:
            a = a[rng.choice(a.size, size=px_per_tile, replace=False)]
        chunks.append(a)
        used_ids.append(tid)
    samples = np.concatenate(chunks) if chunks else np.zeros(0)
    if not return_metadata:
        return samples
    return samples, {
        "seed": seed, "tile_limit": n_tiles, "pixels_per_tile_limit": px_per_tile,
        "selection": "without replacement, catalog training order, finite positive pixels",
        "selected_tile_ids": ids, "used_tile_ids": used_ids,
        "missing_tile_ids": missing_ids, "empty_tile_ids": empty_ids,
        "sample_count": int(samples.size),
        "sample_float64_le_sha256": hashlib.sha256(samples.astype('<f8').tobytes()).hexdigest(),
    }


def compute_linear_norm_range(samples, lo_pct=1.0, hi_pct=99.0, default=(0.0, 255.0)):
    """Robust linear [lo, hi] range from a pixel sample (percentile clip).

    The USGS Cassini Titan HiSAR Global Mosaic is an 8-bit logarithmically
    stretched display product. Its authoritative transfer function to calibrated
    backscatter is not available here, so DN is normalized linearly and never
    interpreted as sigma0 or dB. Percentile bounds keep saturated display pixels
    from setting the scale.
    """
    if samples is None or len(samples) == 0:
        return default
    lo = float(np.percentile(samples, lo_pct))
    hi = float(np.percentile(samples, hi_pct))
    if hi - lo < 1e-6:
        return default
    return lo, hi


def _parse_label(value):
    """Coerce a label to a plain int, or raise TypeError for non-integer types.

    Rejects bool (an int subclass), float, str, and None. NumPy integers are
    accepted and normalized. This prevents labels like 1.5, "2", or True from
    passing a loose int() check and then crashing during integer indexing.
    """
    if isinstance(value, bool):
        raise TypeError("bool is not a valid label")
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise TypeError(f"label must be an integer, got {type(value).__name__}: {value!r}")


def _validate_labels(entries, num_classes, allow_unlabeled=False):
    """Raise if any entry has a non-integer or out-of-range label.

    Defense-in-depth at load time: build_catalog validates at creation, but a
    hand-edited or legacy catalog could still carry a corrupt label (e.g. 99, 1.5,
    "2") that would silently mistrain or crash downstream.
    """
    bad = []
    for e in entries:
        try:
            lab = _parse_label(e.get("label"))
        except (TypeError, ValueError):
            bad.append(e.get("tile_id", "?"))
            continue
        if not (0 <= lab < num_classes or (allow_unlabeled and lab == -1)):
            bad.append(e.get("tile_id", "?"))
    if bad:
        allowed = f"an integer in [0, {num_classes})" + (" or -1" if allow_unlabeled else "")
        raise RuntimeError(
            f"{len(bad)} tiles have labels that are not {allowed} (e.g. {bad[:3]})."
        )


def _catalog_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _apply_split_manifest(entries, catalog_path: Path, split_manifest_path: Optional[str]):
    if split_manifest_path is None:
        return entries
    manifest_path = Path(split_manifest_path)
    manifest = json.loads(manifest_path.read_text())
    expected_hash = manifest.get("catalog_sha256")
    actual_hash = _catalog_sha256(catalog_path)
    if expected_hash != actual_hash:
        raise RuntimeError(
            f"Split manifest {manifest_path} is bound to catalog {expected_hash}, "
            f"not {actual_hash}."
        )
    assignments = manifest.get("assignments", {})
    entry_ids = {entry["tile_id"] for entry in entries}
    if set(assignments) != entry_ids:
        raise RuntimeError("Split manifest tile IDs do not exactly match the catalog")
    return [{**entry, "split": assignments[entry["tile_id"]]} for entry in entries]


def assign_split(lat: float, lon: float, config: TitanSARConfig) -> str:
    """Assign a tile to a data split based on its geographic coordinates.

    Split logic (all longitudes in 0-360E convention):
    - Selk holdout: within selk_radius_deg of Selk center
    - Train: Western hemisphere (lon in [180, 360])
    - Val: Eastern equatorial (lon in [0, 180], |lat| <= 30)
    - Test: Eastern polar (lon in [0, 180], |lat| > 30)
    """
    # Normalize longitude to [0, 360) before any checks
    lon = lon % 360

    # Great-circle angular distance. Raw degree-space Euclidean distance inflates
    # east-west distances away from the equator and mishandles longitude geometry.
    lat1 = np.radians(lat)
    lat2 = np.radians(config.selk_center_lat)
    dlat = lat1 - lat2
    dlon = np.radians((lon - config.selk_center_lon + 180.0) % 360.0 - 180.0)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    angular_distance_deg = np.degrees(2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))
    if angular_distance_deg <= config.selk_radius_deg:
        return "selk_holdout"

    if 180 <= lon < 360:
        return "train"
    elif abs(lat) <= 30:
        return "val"
    else:
        return "test"


class TitanSARDataset(Dataset):
    """PyTorch Dataset for TitanSAR tiles.

    Each {tile_id}.npy stores a (128, 128) float32 array in the product's intensity
    space. Catalog metadata declares HiSAR display DN as ``hisar_log_dn`` or
    calibrated linear backscatter as ``linear_sigma0``.
    """

    def __init__(
        self,
        catalog_path: str,
        split: str = "train",
        config: Optional[TitanSARConfig] = None,
        transform=None,
        normalize: bool = True,
        include_vims: bool = False,
        allow_unlabeled: bool = False,
        intensity_space: Optional[str] = None,
        split_manifest_path: Optional[str] = None,
    ):
        self.config = config or TitanSARConfig()
        self.split = split
        self.transform = transform
        self.normalize = normalize
        self.include_vims = include_vims
        # "hisar_log_dn": USGS HiSAR logarithmically stretched display DN, normalized
        # linearly without claiming a documented affine dB transfer function.
        # "linear_sigma0": calibrated linear sigma0, converted to dB before scaling.
        self._intensity_space_arg = intensity_space

        # Load catalog
        catalog_path = Path(catalog_path)
        if not catalog_path.exists():
            raise FileNotFoundError(f"Catalog not found: {catalog_path}")

        with open(catalog_path, "r") as f:
            full_catalog = json.load(f)
        self.benchmark_track = full_catalog.get("benchmark_track")

        if full_catalog.get("split_policy", "").startswith("external_") and split_manifest_path is None:
            raise RuntimeError(
                f"Catalog {catalog_path} requires an immutable split manifest."
            )

        # Refuse explicitly-unlabeled catalogs in this supervised dataset unless the
        # caller opts in (review N2): labels_available=false means the stored tile
        # labels are -1 sentinels, not real terrain classes.
        if full_catalog.get("labels_available") is False and not allow_unlabeled:
            raise RuntimeError(
                f"Catalog {catalog_path} is tagged labels_available=false (unlabeled). "
                "TitanSARDataset is supervised; pass allow_unlabeled=True only for "
                "unsupervised feature extraction or inspection."
            )

        catalog_entries = _apply_split_manifest(
            full_catalog["tiles"], catalog_path, split_manifest_path
        )
        self.entries = [e for e in catalog_entries if e.get("split") == split]
        _validate_labels(self.entries, self.config.num_classes, allow_unlabeled)
        self.tiles_dir = catalog_path.parent / "tiles"

        # Scientific catalogs must state their product semantics explicitly.
        catalog_intensity_space = full_catalog.get("intensity_space")
        if (
            self._intensity_space_arg is not None
            and catalog_intensity_space is not None
            and self._intensity_space_arg != catalog_intensity_space
        ):
            raise RuntimeError(
                "intensity_space override conflicts with catalog product semantics"
            )
        self.intensity_space = self._intensity_space_arg or catalog_intensity_space
        if self.intensity_space is None:
            raise RuntimeError(
                f"Catalog {catalog_path} has no intensity_space. Rebuild or explicitly "
                "identify the product semantics before loading it."
            )
        if self.intensity_space not in {"hisar_log_dn", "linear_sigma0"}:
            raise RuntimeError(
                f"Unsupported Titan intensity_space={self.intensity_space!r}"
            )
        # Compute normalization stats from the training split.
        self._linear_norm = False
        self.db_min, self.db_max = -30.0, 0.0
        self.lin_min, self.lin_max = 0.0, 255.0
        self.normalization_provenance = {"enabled": bool(normalize),
                                         "intensity_space": self.intensity_space}
        if normalize:
            train_entries = [e for e in catalog_entries if e.get("split") == "train"]
            self.normalization_provenance["training_ids_sha256"] = hashlib.sha256(
                json.dumps([e["tile_id"] for e in train_entries], separators=(',', ':')).encode()
            ).hexdigest()
            self.normalization_provenance["training_ids_digest_format"] = "compact JSON array in catalog order, UTF-8"
            if self.intensity_space == "hisar_log_dn":
                # Log-stretched display DN: linear min-max in product space, no log.
                self._linear_norm = True
                samples, sampling = _sample_train_pixels(
                    train_entries, self.tiles_dir, return_metadata=True)
                self.lin_min, self.lin_max = compute_linear_norm_range(samples)
                self.normalization_provenance.update({
                    "method": "linear display DN, clip to unit interval",
                    "percentiles": [1.0, 99.0], "sampling": sampling,
                    "lower": self.lin_min, "upper": self.lin_max,
                    "denominator_epsilon": 1e-8,
                    "fallback_used": bool(samples.size == 0 or
                        np.percentile(samples, 99) - np.percentile(samples, 1) < 1e-6),
                })
            else:
                mean_db, std_db = _train_db_stats(train_entries)
                self.db_min, self.db_max = compute_norm_range(
                    mean_db, std_db, default=(-30.0, 0.0)
                )
                self.normalization_provenance.update({
                    "method": "calibrated sigma0 to dB, training moments, clip to unit interval",
                    "n_sigma": 3.0, "lower": self.db_min, "upper": self.db_max,
                    "denominator_epsilon": 1e-8,
                })

        # Compute class weights (inverse frequency)
        label_counts = np.zeros(self.config.num_classes)
        for e in self.entries:
            if 0 <= e["label"] < self.config.num_classes:  # skip -1 / out-of-range
                label_counts[e["label"]] += 1
        inv_freq = 1.0 / np.maximum(label_counts, 1)
        self.class_weights = torch.tensor(
            inv_freq / inv_freq.sum() * self.config.num_classes,
            dtype=torch.float32,
        )

        logger.info(
            f"TitanSARDataset: split={split}, tiles={len(self.entries)}, "
            f"class_counts={label_counts.astype(int).tolist()}"
        )

        # Warn about classes with too few samples for reliable evaluation
        MIN_RELIABLE_SAMPLES = 30
        if split in ("test", "val"):
            for c_idx in range(self.config.num_classes):
                n = int(label_counts[c_idx])
                if 0 < n < MIN_RELIABLE_SAMPLES:
                    logger.warning(
                        f"Class {self.config.terrain_classes[c_idx]} has only {n} "
                        f"samples in {split} split (< {MIN_RELIABLE_SAMPLES}). "
                        f"Per-class metrics for this class may be unreliable."
                    )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        tile_path = self.tiles_dir / f"{entry['tile_id']}.npy"
        tile_data = np.load(tile_path).astype(np.float32)

        # Replace any NaN/Inf with 0 (fill value) for robustness
        tile_data = np.nan_to_num(tile_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize to [0, 1]. HiSAR is a logarithmically stretched display DN
        # product; only calibrated linear sigma0 is converted via 10*log10.
        if self.normalize:
            if self._linear_norm:
                tile_norm = (tile_data - self.lin_min) / (self.lin_max - self.lin_min + 1e-8)
            else:
                tile_db = sigma0_to_db(tile_data)
                tile_norm = (tile_db - self.db_min) / (self.db_max - self.db_min + 1e-8)
            tile_norm = np.clip(tile_norm, 0.0, 1.0).astype(np.float32)
        else:
            tile_norm = tile_data

        # Convert to tensor: (1, H, W)
        tile_tensor = torch.from_numpy(tile_norm).unsqueeze(0)

        if self.transform is not None:
            tile_tensor = self.transform(tile_tensor)

        result = {
            "image": tile_tensor,
            "label": entry["label"],
            "tile_id": entry["tile_id"],
            "lat": entry["center_lat"],
            "lon": entry["center_lon"],
        }

        if self.include_vims and entry.get("has_vims_display_sample"):
            sample = np.array(entry["vims_display_sample"], dtype=np.float32)
            result["vims_display_sample"] = torch.from_numpy(sample)

        return result

    def get_class_weights(self) -> torch.Tensor:
        """Return inverse-frequency class weights for loss weighting."""
        return self.class_weights


class EarthAnalogDataset(Dataset):
    """PyTorch Dataset for Earth-analog Sentinel-1 tiles.

    Same format as TitanSARDataset but from Sentinel-1 C-band data.
    """

    def __init__(
        self,
        catalog_path: str,
        split: str = "train",
        config: Optional[TitanSARConfig] = None,
        transform=None,
        normalize: bool = True,
        remapper=None,
        titan_catalog_path: str = None,
        allow_synthetic: bool = False,
        allow_unverified: bool = False,
        split_manifest_path: Optional[str] = None,
    ):
        self.config = config or TitanSARConfig()
        self.split = split
        self.transform = transform
        self.normalize = normalize
        if remapper is not None or titan_catalog_path is not None:
            raise RuntimeError(
                "Quantile remapping is invalid for the HiSAR image-domain track "
                "and is not available through EarthAnalogDataset."
            )
        self.remapper = None

        catalog_path = Path(catalog_path)
        with open(catalog_path, "r") as f:
            full_catalog = json.load(f)

        if full_catalog.get("split_policy", "").startswith("external_") and split_manifest_path is None:
            raise RuntimeError(
                f"Catalog {catalog_path} requires an immutable split manifest."
            )
        if full_catalog.get("intensity_space") != "linear_sigma0":
            raise RuntimeError(
                f"Earth catalog {catalog_path} must declare "
                "intensity_space='linear_sigma0'."
            )

        # Provenance gate (review N1 / ERRATA #18): synthetic Earth tiles (Gaussian
        # noise) silently contaminated a past experiment. Refuse synthetic catalogs,
        # and require positive real-data provenance so an untagged catalog (e.g. a
        # legacy synthetic one without the flag) cannot slip through either.
        is_synthetic = bool(full_catalog.get("synthetic")) or \
            full_catalog.get("source_type") == "synthetic"
        if is_synthetic:
            if not allow_synthetic:
                raise RuntimeError(
                    f"Refusing to load synthetic Earth catalog {catalog_path} for a "
                    "real experiment. Synthetic tiles are pipeline-test fixtures only; "
                    "pass allow_synthetic=True only for tests."
                )
        elif not allow_unverified and full_catalog.get("source_type") not in {
            "real_sentinel1", "real_sentinel1_grd"
        }:
            raise RuntimeError(
                f"Earth catalog {catalog_path} lacks recognized real Sentinel-1 "
                "provenance (source_type='real_sentinel1' or "
                "'real_sentinel1_grd'). Pass allow_unverified=True to load an "
                "untagged catalog you have verified, or allow_synthetic=True for "
                "synthetic fixtures."
            )

        catalog_entries = _apply_split_manifest(
            full_catalog["tiles"], catalog_path, split_manifest_path
        )
        self.entries = [e for e in catalog_entries if e.get("split") == split]
        _validate_labels(self.entries, self.config.num_classes)
        self.tiles_dir = catalog_path.parent / "tiles"

        if normalize:
            train_entries = [e for e in catalog_entries if e.get("split") == "train"]
            mean_db, std_db = _train_db_stats(train_entries)
            self.db_min, self.db_max = compute_norm_range(
                mean_db, std_db, default=(-25.0, 5.0)
            )
        else:
            self.db_min = -25.0
            self.db_max = 5.0

        label_counts = np.zeros(self.config.num_classes)
        for e in self.entries:
            if 0 <= e["label"] < self.config.num_classes:  # skip -1 / out-of-range
                label_counts[e["label"]] += 1
        label_counts = np.maximum(label_counts, 1)
        inv_freq = 1.0 / label_counts
        self.class_weights = torch.tensor(
            inv_freq / inv_freq.sum() * self.config.num_classes,
            dtype=torch.float32,
        )

        logger.info(
            f"EarthAnalogDataset: split={split}, tiles={len(self.entries)}"
        )

    def __len__(self) -> int:
        return len(self.entries)

    def get_class_weights(self) -> torch.Tensor:
        """Return inverse-frequency class weights for loss weighting."""
        return self.class_weights

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        tile_path = self.tiles_dir / f"{entry['tile_id']}.npy"
        tile_data = np.load(tile_path).astype(np.float32)

        if self.normalize:
            tile_db = sigma0_to_db(tile_data)
            tile_norm = (tile_db - self.db_min) / (self.db_max - self.db_min + 1e-8)
            tile_norm = np.clip(tile_norm, 0.0, 1.0).astype(np.float32)
        else:
            tile_norm = tile_data

        tile_tensor = torch.from_numpy(tile_norm).unsqueeze(0)

        if self.transform is not None:
            tile_tensor = self.transform(tile_tensor)

        return {
            "image": tile_tensor,
            "label": entry["label"],
            "tile_id": entry["tile_id"],
            "site": entry.get("site", "unknown"),
        }
