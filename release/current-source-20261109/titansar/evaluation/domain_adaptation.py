"""Domain adaptation methods for TitanSAR.

Implements:
- CORAL (CORrelation ALignment): unsupervised covariance alignment
  Reference: Sun, Feng and Saenko, "Return of Frustratingly Easy Domain
  Adaptation", AAAI 2016.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class CORALAlignment:
    """CORAL: align source feature covariance to target feature covariance.

    Centers source features, whitens and recolors them using regularized source
    and target covariances, then adds the target mean. Fitting uses no labels.

    The transformation is:
        x_aligned = (x - mean_source) @ C_source^{-1/2} @ C_target^{1/2} + mean_target

    Each C includes diagonal regularization. Matrix square roots are computed
    via eigendecomposition of these symmetric positive-definite matrices.
    Regularization means the transformed empirical source covariance need not
    equal the empirical target covariance.

    Reference: Sun, Feng and Saenko, "Return of Frustratingly Easy Domain
    Adaptation", AAAI 2016.
    """

    def __init__(self, regularization: float = 1e-5):
        """Initialize CORAL alignment.

        Args:
            regularization: Small value added to covariance diagonal for
                numerical stability. Must be positive.
        """
        if regularization <= 0:
            raise ValueError(f"regularization must be positive, got {regularization}")
        self.reg = regularization
        self.transform_matrix = None  # (D, D) transformation
        self.source_mean = None  # (D,) source centroid
        self.target_mean = None  # (D,) target centroid

    def _matrix_sqrt_and_inv_sqrt(
        self, cov: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute matrix square root and inverse square root via eigendecomposition.

        For a symmetric PD matrix C with eigendecomposition C = V diag(λ) V^T:
            C^{1/2}   = V diag(sqrt(λ)) V^T
            C^{-1/2}  = V diag(1/sqrt(λ)) V^T

        Args:
            cov: (D, D) symmetric positive-definite covariance matrix.

        Returns:
            Tuple of (C^{1/2}, C^{-1/2}), each (D, D).
        """
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Clip eigenvalues for numerical stability (should already be positive
        # due to regularization, but guard against floating-point drift)
        eigvals = np.maximum(eigvals, self.reg)

        sqrt_eigvals = np.sqrt(eigvals)
        inv_sqrt_eigvals = 1.0 / sqrt_eigvals

        C_sqrt = (eigvecs * sqrt_eigvals[np.newaxis, :]) @ eigvecs.T
        C_inv_sqrt = (eigvecs * inv_sqrt_eigvals[np.newaxis, :]) @ eigvecs.T

        return C_sqrt, C_inv_sqrt

    def fit(self, source_features: np.ndarray, target_features: np.ndarray) -> "CORALAlignment":
        """Learn alignment from source to target covariance.

        Computes the transformation matrix A = C_source^{-1/2} @ C_target^{1/2}
        where each C is a regularized empirical covariance matrix
        (D x D, D=768 for ViT-B).

        Args:
            source_features: (N_s, D) source domain feature matrix.
            target_features: (N_t, D) target domain feature matrix.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If inputs are empty or have mismatched feature dimensions.
        """
        # Input validation
        if source_features.ndim != 2 or target_features.ndim != 2:
            raise ValueError(
                f"Expected 2D arrays, got source shape {source_features.shape} "
                f"and target shape {target_features.shape}"
            )
        if source_features.shape[0] == 0 or target_features.shape[0] == 0:
            raise ValueError(
                f"Empty arrays: source has {source_features.shape[0]} samples, "
                f"target has {target_features.shape[0]} samples"
            )
        D = source_features.shape[1]
        if target_features.shape[1] != D:
            raise ValueError(
                f"Feature dimension mismatch: source has {D}, "
                f"target has {target_features.shape[1]}"
            )

        n_s = source_features.shape[0]
        n_t = target_features.shape[0]
        logger.info(
            f"Fitting CORAL: source ({n_s} x {D}), target ({n_t} x {D}), "
            f"reg={self.reg}"
        )

        # Store means for centering during transform
        self.source_mean = source_features.mean(axis=0)
        self.target_mean = target_features.mean(axis=0)

        # Center features
        source_centered = source_features - self.source_mean
        target_centered = target_features - self.target_mean

        # Covariance matrices with regularization
        # Use max(n-1, 1) to handle single-sample edge case
        C_s = (source_centered.T @ source_centered) / max(n_s - 1, 1) + self.reg * np.eye(D)
        C_t = (target_centered.T @ target_centered) / max(n_t - 1, 1) + self.reg * np.eye(D)

        # Matrix square roots via eigendecomposition
        _, C_s_inv_sqrt = self._matrix_sqrt_and_inv_sqrt(C_s)
        C_t_sqrt, _ = self._matrix_sqrt_and_inv_sqrt(C_t)

        # Transform: A = C_s^{-1/2} @ C_t^{1/2}
        self.transform_matrix = C_s_inv_sqrt @ C_t_sqrt

        logger.info("CORAL fit complete")
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply CORAL alignment to source-domain features.

        Centers with source mean, applies the learned transformation, then
        re-centers with target mean.

        Args:
            features: (N, D) feature matrix from the source domain.

        Returns:
            (N, D) aligned feature matrix.

        Raises:
            RuntimeError: If fit() has not been called.
            ValueError: If feature dimension doesn't match fitted dimension.
        """
        if self.transform_matrix is None:
            raise RuntimeError("Must call fit() before transform()")

        if features.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {features.shape}")
        if features.shape[1] != self.transform_matrix.shape[0]:
            raise ValueError(
                f"Feature dimension {features.shape[1]} doesn't match "
                f"fitted dimension {self.transform_matrix.shape[0]}"
            )

        centered = features - self.source_mean
        aligned = centered @ self.transform_matrix
        return aligned + self.target_mean

    def fit_transform(self, source_features: np.ndarray, target_features: np.ndarray) -> np.ndarray:
        """Fit CORAL and transform source features in one step.

        Args:
            source_features: (N_s, D) source domain features (used for fitting AND transformed).
            target_features: (N_t, D) target domain features (used for fitting only).

        Returns:
            (N_s, D) aligned source features.
        """
        self.fit(source_features, target_features)
        return self.transform(source_features)

    def coral_distance(self, source_features: np.ndarray, target_features: np.ndarray) -> float:
        """Compute the scaled squared Frobenius norm of the covariance difference.

        The covariance discrepancy is measured as:
            d_CORAL = (1 / 4D^2) * ||C_s - C_t||_F^2

        Useful for measuring how much covariance shift remains after alignment.

        Args:
            source_features: (N_s, D) feature matrix.
            target_features: (N_t, D) feature matrix.

        Returns:
            Scalar CORAL distance (non-negative).
        """
        if source_features.shape[1] != target_features.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: {source_features.shape[1]} vs "
                f"{target_features.shape[1]}"
            )

        D = source_features.shape[1]
        n_s = source_features.shape[0]
        n_t = target_features.shape[0]

        source_centered = source_features - source_features.mean(axis=0)
        target_centered = target_features - target_features.mean(axis=0)

        C_s = (source_centered.T @ source_centered) / max(n_s - 1, 1)
        C_t = (target_centered.T @ target_centered) / max(n_t - 1, 1)

        diff = C_s - C_t
        frob_sq = np.sum(diff ** 2)
        return float(frob_sq / (4.0 * D * D))
