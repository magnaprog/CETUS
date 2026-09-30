"""Linear probing and k-NN evaluation for foundation model features.

Implements:
- Linear probing: train linear classifier on frozen features
- k-NN classification: cosine similarity nearest neighbor
- Feature extraction pipeline
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from titansar.configs.defaults import CASSINI_FREQ_GHZ, MAGELLAN_FREQ_GHZ, SENTINEL1_FREQ_GHZ, TitanSARConfig
from titansar.models.foundation_models import FoundationModelWrapper

logger = logging.getLogger(__name__)


def save_feature_metadata(
    feat_dir, array_stem, model_name, weights_source, n_features, feature_dim,
    provenance=None,
):
    """Write provenance for ONE cached feature array (review D8).

    The sidecar is ``{array_stem}.meta.json`` next to ``{array_stem}.npy``, so each
    array carries its own provenance. A model directory can hold arrays produced by
    different runs/weights (e.g. run_probing writes the Titan/Earth/Selk sets; a
    single directory-level record could not prove any individual file. Callers
    bind each array to catalog, split, model revision, checkpoint, and byte hashes;
    the run manifest additionally binds the code commit and all inputs/outputs.
    """
    feat_dir = Path(feat_dir)
    feat_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": "1.0.0",
        "model": model_name,
        "array": array_stem,
        "weights_source": weights_source,
        "n_features": int(n_features),
        "feature_dim": int(feature_dim),
    }
    if provenance:
        meta["provenance"] = dict(provenance)
    with open(feat_dir / f"{array_stem}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_feature_metadata(feat_dir, array_stem):
    """Return the provenance dict for one feature array, or None if absent."""
    path = Path(feat_dir) / f"{array_stem}.meta.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def strict_real_weights_requested():
    """True if TITANSAR_REQUIRE_REAL_WEIGHTS is set (1/true/yes).

    Lets the CPU consumers of cached features (CORAL, label efficiency, physics
    analysis) enforce D8 provenance during a strict rerun without each needing its
    own flag.
    """
    return os.environ.get("TITANSAR_REQUIRE_REAL_WEIGHTS", "").lower() in ("1", "true", "yes")


def assert_cache_real_weights(feat_dir, array_stem, model_name, strict):
    """In strict mode, refuse a cached feature array not proven from real weights.

    Validates the specific array's ``{array_stem}.meta.json``: the sidecar must
    cover this model and array, and ``weights_source`` must be ``pretrained``
    (``random_init`` only for the intentional baseline). Array hashes, shapes, and
    sidecar identity are always checked; ``strict`` controls weight provenance.
    """
    meta = load_feature_metadata(feat_dir, array_stem)
    if meta is None:
        raise RuntimeError(
            f"{feat_dir}/{array_stem}.npy has no provenance "
            f"({array_stem}.meta.json missing): cannot prove it came from real "
            "weights. Re-extract with run_probing or restore matching array metadata."
        )
    if meta.get("model") != model_name or meta.get("array") != array_stem:
        raise RuntimeError(
            f"{feat_dir}/{array_stem}.meta.json describes "
            f"model={meta.get('model')!r} array={meta.get('array')!r}, expected "
            f"{model_name!r}/{array_stem!r}; sidecar mismatched or corrupt. Re-extract."
        )
    array_path = Path(feat_dir) / f"{array_stem}.npy"
    expected_hash = meta.get("provenance", {}).get("feature_sha256")
    if not array_path.exists() or not expected_hash:
        raise RuntimeError(f"Feature cache is missing array/hash binding: {array_path}")
    digest = hashlib.sha256(array_path.read_bytes()).hexdigest()
    if digest != expected_hash:
        raise RuntimeError(f"Feature cache hash mismatch: {array_path}")
    array = np.load(array_path, mmap_mode="r")
    if array.shape != (meta.get("n_features"), meta.get("feature_dim")):
        raise RuntimeError(f"Feature cache shape disagrees with sidecar: {array_path}")
    provenance = meta.get("provenance", {})
    if model_name == "dofa":
        domain = array_stem.split("_", 1)[0]
        expected_identifier = {
            "earth": SENTINEL1_FREQ_GHZ, "venus": MAGELLAN_FREQ_GHZ,
            "titan": CASSINI_FREQ_GHZ, "selk": CASSINI_FREQ_GHZ,
        }.get(domain)
        if expected_identifier is None or provenance.get("band_identifier_ghz") != expected_identifier:
            raise RuntimeError(
                f"DOFA band identifier missing or incorrect for {array_stem}; "
                f"expected {expected_identifier} GHz. Re-extract the domain features."
            )
    for suffix, key in (("_ids", "tile_ids_sha256"), ("_labels", "labels_sha256")):
        companion = Path(feat_dir) / f"{array_stem.replace('_feats', suffix)}.npy"
        expected = provenance.get(key)
        if expected and (
            not companion.exists()
            or hashlib.sha256(companion.read_bytes()).hexdigest() != expected
        ):
            raise RuntimeError(f"Feature companion hash mismatch: {companion}")
    if not strict:
        return
    ws = meta.get("weights_source")
    ok = ws == "pretrained" or (model_name == "random_init" and ws == "random_init")
    if not ok:
        raise RuntimeError(
            f"{feat_dir}/{array_stem}.npy has weights_source={ws!r}; strict mode "
            f"requires real weights for {model_name}. Re-extract."
        )


def load_bound_feature_cache(
    feat_dir, array_stem, model_name, expected_provenance, expected_tile_labels,
):
    """Read a cache only when hashes, tile identities and labels match the task."""
    assert_cache_real_weights(feat_dir, array_stem, model_name, strict=True)
    meta = load_feature_metadata(feat_dir, array_stem)
    provenance = meta["provenance"]
    for key, expected in expected_provenance.items():
        if provenance.get(key) != expected:
            raise RuntimeError(f"Feature cache {array_stem} has a different {key}")
    for key in ("tile_ids_sha256", "labels_sha256"):
        if not provenance.get(key):
            raise RuntimeError(f"Feature cache {array_stem} is missing {key}")
    directory = Path(feat_dir)
    features = np.load(directory / f"{array_stem}.npy")
    labels = np.load(directory / f"{array_stem.replace('_feats', '_labels')}.npy")
    ids = np.load(directory / f"{array_stem.replace('_feats', '_ids')}.npy")
    if ids.ndim != 1 or labels.ndim != 1 or len(ids) != len(features) or len(labels) != len(ids):
        raise RuntimeError(f"Feature cache {array_stem} has misaligned companion arrays")
    if ids.dtype.kind not in {"U", "S"} or len(set(ids.tolist())) != len(ids):
        raise RuntimeError(f"Feature cache {array_stem} has invalid or duplicate tile IDs")
    ids = ids.astype(str).tolist()
    if set(ids) != set(expected_tile_labels):
        raise RuntimeError(f"Feature cache {array_stem} tile IDs differ from the requested split")
    if not np.array_equal(labels, [expected_tile_labels[tid] for tid in ids]):
        raise RuntimeError(f"Feature cache {array_stem} labels differ from the catalog")
    if not np.isfinite(features).all():
        raise RuntimeError(f"Feature cache {array_stem} contains nonfinite features")
    return features, labels, ids, meta


class LinearProbe(nn.Module):
    """Single linear layer for probing frozen features."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def extract_features(
    model: FoundationModelWrapper,
    dataloader: DataLoader,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, list]:
    """Extract features from all tiles in a dataloader.

    Returns:
        features: (N, embedding_dim) array
        labels: (N,) array
        tile_ids: list of tile ID strings
    """
    all_features = []
    all_labels = []
    all_ids = []

    model.model.eval()
    for batch in dataloader:
        images = batch["image"]
        labels = batch["label"]
        ids = batch["tile_id"]

        features = model.extract_features(images)
        all_features.append(features.numpy())
        all_labels.append(np.array(labels))
        all_ids.extend(ids)

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return features, labels, all_ids


def train_linear_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    num_classes: int = 6,
    class_weights: Optional[torch.Tensor] = None,
    config: Optional[TitanSARConfig] = None,
    seed: int = 0,
    device: str = "cuda",
) -> LinearProbe:
    """Train a linear probe on frozen features.

    Uses SGD with cosine annealing and inverse-frequency class weighting.
    """
    if config is None:
        config = TitanSARConfig()

    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = train_features.shape[1]
    probe = LinearProbe(input_dim, num_classes).to(device)

    # Prepare data
    X = torch.tensor(train_features, dtype=torch.float32)
    y = torch.tensor(train_labels, dtype=torch.long)

    # Loss with class weighting
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(probe.parameters(), lr=config.probe_lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.probe_epochs
    )

    # Training loop
    dataset = torch.utils.data.TensorDataset(X, y)
        # Shuffle order comes from the global torch seed set above;
        # adding an explicit generator here would change the shuffle
        # stream and break parity with the canonical published runs.
    loader = DataLoader(dataset, batch_size=config.probe_batch_size, shuffle=True)

    probe.train()
    for epoch in range(config.probe_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = probe(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            logger.debug(f"Probe epoch {epoch+1}/{config.probe_epochs}, loss={epoch_loss:.4f}")

    return probe


def evaluate_linear_probe(
    probe: LinearProbe,
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 6,
    device: str = "cuda",
) -> dict:
    """Evaluate a trained linear probe.

    Returns dict with macro_accuracy, per_class_accuracy, kappa, predictions.
    """
    probe.eval()
    X = torch.tensor(features, dtype=torch.float32).to(device)
    y = labels

    with torch.no_grad():
        logits = probe(X)
        preds = logits.argmax(dim=1).cpu().numpy()
        probs = F.softmax(logits, dim=1).cpu().numpy()

    # Per-class accuracy
    per_class_acc = []
    for c in range(num_classes):
        mask = y == c
        if mask.sum() > 0:
            per_class_acc.append(float((preds[mask] == c).mean()))
        else:
            per_class_acc.append(float("nan"))

    # Macro accuracy (mean of per-class accuracies, excluding NaN)
    valid_accs = [a for a in per_class_acc if not np.isnan(a)]
    macro_acc = float(np.mean(valid_accs)) if valid_accs else 0.0

    # Cohen's kappa
    kappa = _cohens_kappa(y, preds, num_classes)

    # Confusion matrix (normalized by true class)
    confusion = np.zeros((num_classes, num_classes))
    for true, pred in zip(y, preds):
        confusion[true, pred] += 1
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1)
    confusion_norm = confusion / row_sums

    return {
        "macro_accuracy": macro_acc,
        "per_class_accuracy": per_class_acc,
        "kappa": kappa,
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_normalized": confusion_norm.tolist(),
        "predictions": preds,
        "probabilities": probs,
    }


def knn_classify(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    k: int = 20,
    num_classes: int = 6,
) -> dict:
    """k-NN classification using cosine similarity.

    No training required - uses all training features as the reference set.
    """
    # Clamp k to training set size
    k = min(k, len(train_labels))

    # Normalize features for cosine similarity
    train_norm = train_features / (np.linalg.norm(train_features, axis=1, keepdims=True) + 1e-8)
    test_norm = test_features / (np.linalg.norm(test_features, axis=1, keepdims=True) + 1e-8)

    # Compute similarities in batches to manage memory
    batch_size = 512
    all_preds = []

    for start in range(0, len(test_norm), batch_size):
        end = min(start + batch_size, len(test_norm))
        batch = test_norm[start:end]

        # Cosine similarity: (batch, train)
        sims = batch @ train_norm.T
        # Top-k indices
        topk_idx = np.argsort(-sims, axis=1)[:, :k]

        # Majority vote
        for i in range(len(batch)):
            neighbor_labels = train_labels[topk_idx[i]]
            counts = np.bincount(neighbor_labels, minlength=num_classes)
            all_preds.append(int(np.argmax(counts)))

    preds = np.array(all_preds)

    # Compute metrics
    per_class_acc = []
    for c in range(num_classes):
        mask = test_labels == c
        if mask.sum() > 0:
            per_class_acc.append(float((preds[mask] == c).mean()))
        else:
            per_class_acc.append(float("nan"))

    valid_accs = [a for a in per_class_acc if not np.isnan(a)]
    macro_acc = float(np.mean(valid_accs)) if valid_accs else 0.0
    kappa = _cohens_kappa(test_labels, preds, num_classes)

    return {
        "macro_accuracy": macro_acc,
        "per_class_accuracy": per_class_acc,
        "kappa": kappa,
        "k": k,
        "predictions": preds,
    }


def _cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Compute Cohen's kappa statistic."""
    confusion = np.zeros((num_classes, num_classes))
    for true, pred in zip(y_true, y_pred):
        confusion[true, pred] += 1

    n = confusion.sum()
    if n == 0:
        return 0.0

    p_o = np.diag(confusion).sum() / n  # observed agreement
    p_e = sum(
        confusion[c, :].sum() * confusion[:, c].sum() for c in range(num_classes)
    ) / (n * n)  # expected agreement

    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0

    return float((p_o - p_e) / (1.0 - p_e))


def train_finetuned_model(
    model: FoundationModelWrapper,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int = 6,
    unfreeze_blocks: int = 2,
    lr: float = 1e-4,
    epochs: int = 50,
    class_weights: Optional[torch.Tensor] = None,
    device: str = "cuda",
    use_amp: bool = True,
    grad_accum_steps: int = 4,
    init_probe: Optional[LinearProbe] = None,
    return_history: bool = False,
) -> tuple:
    """Fine-tune the last N transformer blocks + linear head.

    Freezes all backbone parameters, then selectively unfreezes the last
    `unfreeze_blocks` transformer blocks. Trains with AdamW + cosine
    annealing and optional mixed precision (AMP).

    Args:
        model: Foundation model wrapper (will be modified in-place).
        train_loader: Training DataLoader yielding dicts with "image" and "label".
        val_loader: Validation DataLoader for early stopping / best checkpoint.
        num_classes: Number of terrain classes.
        unfreeze_blocks: Number of trailing transformer blocks to unfreeze.
        lr: Learning rate for AdamW.
        epochs: Maximum training epochs.
        class_weights: Optional per-class loss weights.
        device: CUDA device string.
        use_amp: Whether to use automatic mixed precision (GPU only).
        grad_accum_steps: Gradient accumulation steps (effective batch =
            batch_size * grad_accum_steps).
        init_probe: Optional LinearProbe supplied by the caller. It may be
            freshly initialized or previously trained.
        return_history: Also return per-epoch loss and validation accuracy.

    Returns:
        The fine-tuned model wrapper, trained probe head, and best validation
        accuracy. If ``return_history`` is true, a fourth item contains the
        per-epoch training history. After return, all backbone parameters are
        re-frozen.
    """
    if unfreeze_blocks < 0 or grad_accum_steps < 1 or epochs < 1:
        raise ValueError("unfreeze_blocks must be nonnegative; accumulation and epochs must be positive")
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("Fine-tuning requires non-empty training and validation loaders")
    # --- Step 1: Freeze ALL backbone parameters ---
    for p in model.model.parameters():
        p.requires_grad = False

    # --- Step 2: Unfreeze last N transformer blocks ---
    blocks = model.get_transformer_blocks()
    num_to_unfreeze = min(unfreeze_blocks, len(blocks))
    unfrozen_blocks = blocks[-num_to_unfreeze:] if num_to_unfreeze else []
    for block in unfrozen_blocks:
        for p in block.parameters():
            p.requires_grad = True

    trainable_backbone = sum(
        p.numel() for p in model.model.parameters() if p.requires_grad
    )
    total_backbone = sum(p.numel() for p in model.model.parameters())
    logger.info(
        f"Fine-tuning {num_to_unfreeze}/{len(blocks)} blocks: "
        f"{trainable_backbone:,}/{total_backbone:,} backbone params trainable"
    )

    # --- Step 3: Create or reuse probe head ---
    if init_probe is not None:
        probe = init_probe
        logger.info("Using the linear probe head supplied by the caller")
    else:
        probe = LinearProbe(model.embedding_dim, num_classes)
    probe = probe.to(device)

    # --- Step 4: Set up optimizer, scheduler, loss ---
    # Collect trainable parameters: unfrozen backbone blocks + probe head
    param_groups = [
        {"params": [p for b in unfrozen_blocks for p in b.parameters() if p.requires_grad],
         "lr": lr},
        {"params": probe.parameters(), "lr": lr * 10},  # Higher LR for head
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), reduction="sum")
    else:
        criterion = nn.CrossEntropyLoss(reduction="sum")

    # AMP setup (only for CUDA)
    use_amp = use_amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- Step 5: Training loop ---
    best_val_acc = -np.inf
    best_model_state = None
    best_probe_state = None
    history = {
        "train_loss": [], "val_balanced_accuracy": [],
        "optimizer_steps": [], "amp_skipped_steps": [], "loss_scale": [],
    }

    model.model.to(device)
    model.model.train()

    for epoch in range(epochs):
        # -- Training --
        model.model.train()
        probe.train()
        epoch_loss = 0.0
        epoch_weight = 0.0
        accumulated_weight = 0.0
        optimizer_steps = 0
        skipped_steps = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            images = batch["image"]
            labels = batch["label"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                x = model.prepare_input(images).to(device)
                features = model._forward(x)
                logits = probe(features)
                loss = criterion(logits, labels)

            # Match the weighted mean loss of the whole effective batch,
            # including a final partial batch or accumulation window.
            batch_weight = (
                float(criterion.weight[labels].sum())
                if criterion.weight is not None else len(labels)
            )
            accumulated_weight += batch_weight

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                for group in optimizer.param_groups:
                    for parameter in group["params"]:
                        if parameter.grad is not None:
                            parameter.grad.div_(accumulated_weight)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if not np.isfinite(scaler.get_scale()) or scaler.get_scale() <= 0:
                    raise RuntimeError("AMP loss scale must remain finite and positive")
                if scaler.get_scale() < previous_scale:
                    skipped_steps += 1
                else:
                    optimizer_steps += 1
                optimizer.zero_grad()
                accumulated_weight = 0.0

            epoch_loss += loss.item()
            epoch_weight += batch_weight

        if optimizer_steps:
            scheduler.step()
        avg_loss = epoch_loss / epoch_weight

        # -- Validation (balanced accuracy to match test-time evaluation) --
        model.model.eval()
        probe.eval()
        val_per_class_correct = np.zeros(num_classes)
        val_per_class_total = np.zeros(num_classes)
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"]
                labels = batch["label"].to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    x = model.prepare_input(images).to(device)
                    features = model._forward(x)
                    logits = probe(features)
                preds = logits.argmax(dim=1)
                for c in range(num_classes):
                    mask = labels == c
                    val_per_class_total[c] += mask.sum().item()
                    val_per_class_correct[c] += (preds[mask] == c).sum().item()

        # Balanced accuracy: mean of per-class recall (same as test metric)
        per_class_acc = []
        for c in range(num_classes):
            if val_per_class_total[c] > 0:
                per_class_acc.append(val_per_class_correct[c] / val_per_class_total[c])
        val_acc = float(np.mean(per_class_acc)) if per_class_acc else 0.0
        history["train_loss"].append(avg_loss)
        history["val_balanced_accuracy"].append(val_acc)
        history["optimizer_steps"].append(optimizer_steps)
        history["amp_skipped_steps"].append(skipped_steps)
        history["loss_scale"].append(scaler.get_scale())

        if optimizer_steps and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                k: v.clone() for k, v in model.model.state_dict().items()
            }
            best_probe_state = {
                k: v.clone() for k, v in probe.state_dict().items()
            }

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, "
                f"val_acc={val_acc:.4f} (best={best_val_acc:.4f})"
            )

    if best_model_state is None:
        raise RuntimeError("Fine-tuning completed without any optimizer updates")

    # --- Step 6: Restore best checkpoint ---
    if best_model_state is not None:
        model.model.load_state_dict(best_model_state)
    if best_probe_state is not None:
        probe.load_state_dict(best_probe_state)

    # --- Step 7: Re-freeze everything ---
    for p in model.model.parameters():
        p.requires_grad = False
    model.model.eval()

    logger.info(f"Fine-tuning complete. Best val accuracy: {best_val_acc:.4f}")

    if return_history:
        return model, probe, best_val_acc, history
    return model, probe, best_val_acc


def compute_silhouette_score(
    features: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 5000,
    seed: int = 0,
) -> float:
    """Compute silhouette score on feature embeddings.

    Measures how well terrain classes cluster in feature space without
    any training, providing insight into intrinsic structure captured
    by the representations.
    """
    from sklearn.metrics import silhouette_score

    # Subsample for efficiency if needed
    if len(features) > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(features), sample_size, replace=False)
        features = features[idx]
        labels = labels[idx]

    # Need at least 2 classes with samples
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    return float(silhouette_score(features, labels, metric="cosine"))
