"""Foundation model wrappers for TitanSAR probing experiments.

Wraps DINOv2, DOFA, CROMA, and random-init ViT-B as frozen feature extractors.
Each wrapper handles the model-specific input channel requirements.
"""

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from titansar.configs.defaults import (
    CASSINI_FREQ_GHZ,
    TitanSARConfig,
)

logger = logging.getLogger(__name__)

DINOV2_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DOFA_REVISION = "73ff5c0721da322689ae890bed5b4efd78935f47"
DINOV2_SHA256 = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
DOFA_SHA256 = "4720985e42b918ac0307009eb06121a3435d9bbce6fd95446f84824a538165b1"
CROMA_SHA256 = "0238d814b53108f3574bf1ea240e38a0a6edd46173816d9a6962070561893b63"


def _verify_checkpoint(path: Path, expected_sha256: str):
    if not path.exists():
        raise RuntimeError(f"Expected checkpoint is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"Checkpoint hash mismatch for {path}: {actual} != {expected_sha256}"
        )


class FoundationModelWrapper(ABC):
    """Base class for foundation model feature extractors."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.embedding_dim = 768  # ViT-B default
        # Set by load(): "pretrained" (real checkpoint), "placeholder" (random
        # timm fallback), or "random_init" (intentional baseline). Recorded so a
        # run can prove which models actually used pre-trained weights (D8).
        self.weights_source = None
        self.model_revision = None
        self.weights_sha256 = None

    @abstractmethod
    def load(self, allow_placeholder: bool = True):
        """Load model weights.

        allow_placeholder=False makes wrappers that depend on external real
        weights (DOFA, CROMA) raise rather than silently fall back to a random
        timm placeholder (review D8). Implementations set self.weights_source.
        """
        pass

    @abstractmethod
    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare single-channel SAR input for the specific model.

        Args:
            x: (B, 1, H, W) normalized SAR tile tensor

        Returns:
            Model-ready input tensor
        """
        pass

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract one feature vector per input tile.

        DINOv2 and Random Init use CLS tokens. DOFA uses normalized mean patch
        features. CROMA applies a learned feed-forward module after mean pooling.

        Args:
            x: (B, 1, H, W) normalized SAR tile tensor

        Returns:
            (B, embedding_dim) feature vectors
        """
        self.model.eval()
        x = self.prepare_input(x).to(self.device)
        features = self._forward(x)
        return features.cpu()

    @abstractmethod
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Model-specific forward pass returning one feature vector per tile."""
        pass

    def get_transformer_blocks(self) -> list[nn.Module]:
        """Return ordered list of transformer blocks for selective unfreezing.

        Used by fine-tuning to unfreeze the last N blocks while keeping
        earlier layers frozen. Subclasses must implement this based on
        their model architecture.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_transformer_blocks()"
        )


class DINOv2Wrapper(FoundationModelWrapper):
    """DINOv2 ViT-B/14 (Oquab et al. 2024).

    Pretrained on LVD-142M images. The wrapper repeats the single SAR channel
    in all three input channels expected by the released model.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.name = "dinov2"
        self.patch_size = 14

    def load(self, allow_placeholder: bool = True):
        # DINOv2 has no placeholder path: it loads real weights or raises, so
        # allow_placeholder is accepted for a uniform factory API but unused.
        logger.info("Loading DINOv2 ViT-B/14 from torch.hub...")
        self.model = torch.hub.load(
            f"facebookresearch/dinov2:{DINOV2_REVISION}",
            "dinov2_vitb14",
            pretrained=True,
            trust_repo=True,
        )
        _verify_checkpoint(
            Path(torch.hub.get_dir()) / "checkpoints" / "dinov2_vitb14_pretrain.pth",
            DINOV2_SHA256,
        )
        self.model.to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.embed_dim
        self.weights_source = "pretrained"
        self.model_revision = DINOV2_REVISION
        self.weights_sha256 = DINOV2_SHA256
        logger.info(f"DINOv2 loaded. Embedding dim: {self.embedding_dim}")

    # ImageNet normalization constants (from DINOv2 pre-training)
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Replicate 1-channel -> 3-channel and apply ImageNet normalization for DINOv2."""
        # x: (B, 1, H, W) in [0, 1] -> (B, 3, H, W)
        x = x.expand(-1, 3, -1, -1)
        # Resize to 224x224 (DINOv2 default, patch_size=14 -> 16x16 patches)
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        # Apply each channel's ImageNet mean and standard deviation. The repeated
        # input channels therefore differ after normalization.
        mean = self.IMAGENET_MEAN.to(x.device).view(1, 3, 1, 1)
        std = self.IMAGENET_STD.to(x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # Returns [CLS] token by default

    def get_transformer_blocks(self) -> list[nn.Module]:
        """DINOv2 ViT uses self.model.blocks (nn.Sequential of Block modules)."""
        return list(self.model.blocks)


class DOFAWrapper(FoundationModelWrapper):
    """DOFA ViT-B/16 (Xiong et al. 2024).

    Pretraining includes Sentinel-1, Sentinel-2, NAIP, Gaofen and EnMAP data.
    A dynamic network generates patch embedding weights from numeric band
    identifiers. The wrapper supplies one SAR channel.

    The authors' README at DOFA_REVISION uses 5.405 for Sentinel-1. CETUS
    extends that example's frequency convention to 13.78 for Cassini. The
    paper's version 2, Appendix D, instead describes the Sentinel-1 identifier
    as 3.75: https://arxiv.org/html/2403.15356v2#A4. The identifier history of
    the downloaded checkpoint remains unresolved. These values record our
    input convention; they do not validate conditioning on an unseen sensor.

    The accepted Titan evaluation supplies image values in [0, 1] after fitting
    the range on training pixels. The pinned README's Earth example also applies channel
    standardization. Its example constants establish neither an appropriate
    Titan normalization nor the downloaded checkpoint's pretraining recipe.

    Weights available at: earthflow/DOFA on HuggingFace
    Code at: github.com/zhu-xlab/DOFA
    """

    def __init__(self, device: str = "cuda", wavelength: float = CASSINI_FREQ_GHZ):
        super().__init__(device)
        self.name = "dofa"
        # Preserve the recorded numeric convention for this evaluation.
        self.wavelength = wavelength
        self.patch_size = 16

    def load(self, allow_placeholder: bool = True):
        """Load DOFA model.

        DOFA code is at github.com/zhu-xlab/DOFA, weights on HuggingFace.
        Loaded via torch.hub (hubconf.py) which downloads the checkpoint
        from huggingface.co/earthflow/DOFA/DOFA_ViT_base_e100.pth.
        """
        logger.info("Loading DOFA ViT-B/16...")
        try:
            # Load via torch.hub (downloads model code + weights automatically)
            self.model = torch.hub.load(
                f"zhu-xlab/DOFA:{DOFA_REVISION}",
                "vit_base_dofa",
                pretrained=True,
                trust_repo=True,
            )
            _verify_checkpoint(
                Path(torch.hub.get_dir()) / "checkpoints" / "DOFA_ViT_base_e100.pth",
                DOFA_SHA256,
            )
            self.weights_source = "pretrained"
            self.model_revision = DOFA_REVISION
            self.weights_sha256 = DOFA_SHA256
        except Exception as e:
            if not allow_placeholder:
                raise RuntimeError(
                    f"DOFA real weights failed to load ({e}) and "
                    "allow_placeholder=False; refusing to extract features from a "
                    "random placeholder. Install the DOFA weights or drop the model."
                ) from e
            logger.warning(
                f"DOFA torch.hub load failed: {e}\n"
                "Falling back to placeholder ViT-B/16."
            )
            self.model = self._build_placeholder()
            self.weights_source = "placeholder"

        self.model.to(self.device)
        self.model.eval()
        logger.info(
            f"DOFA loaded ({self.weights_source}). "
            f"Band identifier: {self.wavelength} (GHz for SAR)"
        )

    def _build_placeholder(self) -> nn.Module:
        """Build a placeholder ViT-B/16 with single-channel input for testing."""
        try:
            import timm
            model = timm.create_model("vit_base_patch16_224", pretrained=False, in_chans=1)
            return model
        except ImportError:
            raise ImportError("Either DOFA or timm is required")

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Resize the dataset's single-channel tensor without further normalization."""
        # x: (B, 1, H, W) - keep as-is
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return x

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with wavelength conditioning.

        DOFA's forward signature: forward_features(x, wave_list)
        where wave_list is a list of band identifiers (frequency in GHz for SAR).
        """
        try:
            # Real DOFA API: forward_features(x, wave_list)
            features = self.model.forward_features(
                x, wave_list=[self.wavelength]
            )
            # The pinned DOFA factory defaults to mean patch pooling followed by
            # fc_norm, returning a (batch, 768) tensor.
            if features.ndim == 3:
                return features[:, 0]
            return features
        except TypeError:
            # Fallback for placeholder model (standard timm API)
            features = self.model.forward_features(x)
            if features.ndim == 3:
                return features[:, 0]
            return features

    def get_transformer_blocks(self) -> list[nn.Module]:
        """DOFA ViT uses self.model.blocks (nn.Sequential of Block modules).

        Falls back to timm blocks for placeholder model.
        """
        return list(self.model.blocks)


class CROMAWrapper(FoundationModelWrapper):
    """CROMA SAR encoder (Fuller et al. 2023, NeurIPS 36).

    Pre-trained on co-located Sentinel-1/2 via contrastive + reconstructive
    objectives. SAR encoder expects 2-channel (VV+VH) input.

    Limitation: We replicate Titan's single channel to both VV and VH inputs,
    producing identical channels. This prevents the model from leveraging any
    cross-polarization features learned during pre-training, since cross-pol
    information requires genuinely different polarization measurements.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.name = "croma"
        self.patch_size = 8  # CROMA SAR encoder uses patch size 8 at 120px input

    def load(self, allow_placeholder: bool = True):
        logger.info("Loading CROMA SAR encoder...")
        weights_path = Path.home() / ".cache" / "croma" / "CROMA_base.pt"
        try:
            from titansar.models.croma_model import PretrainedCROMA

            if not weights_path.exists():
                logger.info("Downloading CROMA weights from HuggingFace...")
                weights_path.parent.mkdir(parents=True, exist_ok=True)
                import urllib.request
                urllib.request.urlretrieve(
                    "https://huggingface.co/antofuller/CROMA/resolve/main/CROMA_base.pt",
                    str(weights_path),
                )

            _verify_checkpoint(weights_path, CROMA_SHA256)
            self.model = PretrainedCROMA(
                pretrained_path=str(weights_path),
                size="base", modality="SAR", image_resolution=120,
            )
            self._use_real_croma = True
            self.weights_source = "pretrained"
            self.model_revision = "local:titansar.models.croma_model"
            self.weights_sha256 = CROMA_SHA256
        except Exception as e:
            if not allow_placeholder:
                raise RuntimeError(
                    f"CROMA real weights failed to load ({e}) and "
                    "allow_placeholder=False; refusing to extract features from a "
                    "random placeholder. Install the CROMA weights or drop the model."
                ) from e
            logger.warning(
                f"CROMA load failed: {e}\n"
                "Falling back to placeholder ViT-B/16."
            )
            self.model = self._build_placeholder()
            self._use_real_croma = False
            self.weights_source = "placeholder"

        self.model.to(self.device)
        self.model.eval()
        logger.info(f"CROMA SAR encoder loaded ({self.weights_source}).")

    def _build_placeholder(self) -> nn.Module:
        try:
            import timm
            model = timm.create_model("vit_base_patch16_224", pretrained=False, in_chans=2)
            return model
        except ImportError:
            raise ImportError("Either CROMA or timm is required")

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Replicate 1-channel -> 2-channel (VV+VH) for CROMA."""
        # x: (B, 1, H, W) -> (B, 2, H, W)
        x = x.expand(-1, 2, -1, -1)
        # CROMA uses 120x120 input (patch_size=8 -> 15×15=225 patches)
        target_size = 120 if getattr(self, '_use_real_croma', False) else 224
        if x.shape[-1] != target_size or x.shape[-2] != target_size:
            x = F.interpolate(x, size=(target_size, target_size), mode="bilinear", align_corners=False)
        return x

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, '_use_real_croma', False):
            # SAR_GAP applies a learned feed-forward module after mean patch pooling.
            result = self.model(SAR_images=x)
            return result['SAR_GAP']
        else:
            # Placeholder timm ViT
            features = self.model.forward_features(x)
            if features.ndim == 3:
                return features[:, 0]
            return features

    def get_transformer_blocks(self) -> list[nn.Module]:
        """CROMA SAR encoder transformer blocks.

        Real CROMA: blocks are in self.model.s1_encoder.transformer.layers
        (nn.ModuleList of [Attention, FFN] pairs).
        Placeholder (timm ViT): uses self.model.blocks.
        """
        if getattr(self, '_use_real_croma', False):
            return list(self.model.s1_encoder.transformer.layers)
        else:
            return list(self.model.blocks)


class RandomInitWrapper(FoundationModelWrapper):
    """Random initialization ViT-B/14 baseline.

    Configured with single-channel input embedding (1->patch_dim).
    Measures what a linear probe can learn from an untrained encoder. Its score
    is a control, not a mathematical lower bound on another model's score.
    """

    def __init__(self, device: str = "cuda", seed: int = 42):
        super().__init__(device)
        self.name = "random_init"
        self.seed = seed

    def load(self, allow_placeholder: bool = True):
        # Random-init is an intentional baseline (no pretrained weights), so
        # allow_placeholder does not apply; weights_source records the intent.
        logger.info("Loading randomly initialized ViT-B/14...")
        import timm

        architecture = "vit_base_patch14_dinov2.lvd142m"
        # Keep a fixed architecture and preserve the caller's random state.
        # A fallback with register tokens changes the experiment.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            self.model = timm.create_model(
                architecture,
                pretrained=False,
                in_chans=1,
                img_size=224,
            )

        digest = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}\n".encode())
            digest.update(value.numpy().tobytes())
        self.model_revision = f"timm:{timm.__version__}:{architecture}:seed={self.seed}"
        self.weights_sha256 = digest.hexdigest()

        self.model.to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.embed_dim
        self.weights_source = "random_init"
        logger.info(f"Random init ViT-B/14. Embedding dim: {self.embedding_dim}")

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Single-channel input, no replication needed."""
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return x

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(x)
        if features.ndim == 3:
            return features[:, 0]
        return features

    def get_transformer_blocks(self) -> list[nn.Module]:
        """Random init ViT (timm) uses self.model.blocks."""
        return list(self.model.blocks)


def get_model(
    model_name: str,
    device: str = "cuda",
    allow_placeholder: bool = True,
    **kwargs,
) -> FoundationModelWrapper:
    """Factory function to get a foundation model wrapper by name.

    Set allow_placeholder=False for a real evaluation run: DOFA/CROMA then raise
    instead of silently substituting a random timm placeholder when their real
    weights are unavailable (review D8). Inspect wrapper.weights_source after
    load() to record provenance ("pretrained" / "placeholder" / "random_init").

    The environment variable TITANSAR_REQUIRE_REAL_WEIGHTS=1 forces
    allow_placeholder=False for every caller, so a whole rerun can forbid
    placeholders without threading a flag through each script.
    """
    # Global strict override (covers scripts that do not expose an explicit flag).
    if os.environ.get("TITANSAR_REQUIRE_REAL_WEIGHTS", "").lower() in ("1", "true", "yes"):
        allow_placeholder = False

    models = {
        "dinov2": DINOv2Wrapper,
        "dofa": DOFAWrapper,
        "croma": CROMAWrapper,
        "random_init": RandomInitWrapper,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")

    wrapper = models[model_name](device=device, **kwargs)
    wrapper.load(allow_placeholder=allow_placeholder)
    return wrapper
