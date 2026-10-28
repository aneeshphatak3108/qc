"""
Leakage-free preprocessing: raw CICFlowMeter features → qubit rotation angles
============================================================================

QuantumPreprocessor (new, default)
  fit on the TRAINING split only:
    1. drop zero-variance columns            (train statistics)
    2. signed log transform  x → sign(x)·log(1+|x|)
         CICFlowMeter features are extremely heavy-tailed (Flow Bytes/s spans
         0 … 2.9e9). Under plain min–max scaling almost every flow is squeezed
         into a tiny corner of [0, π], so most qubits receive near-identical
         angles. The log compresses the tail before scaling.
    3. StandardScaler                        (train statistics)
    4. PCA → n_qubits components             (train statistics)
    5. per-component min–max to [0, π]       (train statistics)
       test values outside the train range are clipped to [0, π]

LegacyPreprocessor (reproduces the original pipeline for comparison)
    global MinMax(0, π) over the *whole* sampled dataset (before the split —
    this is the leak), then PCA fitted on train, then min–max to [0, π]
    with train statistics and no clipping.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def signed_log1p(X: np.ndarray) -> np.ndarray:
    return np.sign(X) * np.log1p(np.abs(X))


class QuantumPreprocessor:
    """Fit on train, transform train/test into angles in [0, π]."""

    def __init__(self, n_components: int = 16, log_transform: bool = True,
                 clip: bool = True, seed: int = 42):
        self.n_components = n_components
        self.log_transform = log_transform
        self.clip = clip
        self.seed = seed

    def _pre(self, X: np.ndarray) -> np.ndarray:
        X = X[:, self.keep_]
        if self.log_transform:
            X = signed_log1p(X)
        return self.scaler_.transform(X)

    def fit(self, X: np.ndarray) -> "QuantumPreprocessor":
        self.keep_ = X.std(axis=0) > 0
        Xk = X[:, self.keep_]
        if self.log_transform:
            Xk = signed_log1p(Xk)
        self.scaler_ = StandardScaler().fit(Xk)
        n_comp = min(self.n_components, int(self.keep_.sum()))
        self.pca_ = PCA(n_components=n_comp, random_state=self.seed)
        Z = self.pca_.fit_transform(self.scaler_.transform(Xk))
        self.min_, self.max_ = Z.min(axis=0), Z.max(axis=0)
        return self

    def transform_pca(self, X: np.ndarray) -> np.ndarray:
        """PCA scores (used by the classical baselines)."""
        return self.pca_.transform(self._pre(X))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Rotation angles in [0, π], shape (n, n_components)."""
        Z = self.transform_pca(X)
        A = (Z - self.min_) / (self.max_ - self.min_ + 1e-12) * np.pi
        return np.clip(A, 0.0, np.pi) if self.clip else A

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def explained_variance(self) -> float:
        return float(self.pca_.explained_variance_ratio_.sum())

    @property
    def n_input_features(self) -> int:
        return int(self.keep_.sum())


def legacy_global_scale(X: np.ndarray) -> np.ndarray:
    """
    The original loader's encoding, applied to the full dataset before the
    split: drop zero-variance columns, pad/trim to 58, MinMax to [0, π].
    """
    X = X[:, X.std(axis=0) > 0]
    if X.shape[1] < 58:
        X = np.hstack([X, np.zeros((len(X), 58 - X.shape[1]))])
    else:
        X = X[:, :58]
    return MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X)


class LegacyPreprocessor:
    """Original QSVM-internal preprocessing: PCA on train + min–max, no clip."""

    def __init__(self, n_components: int = 16, seed: int = 42):
        self.n_components = n_components
        self.seed = seed

    def fit(self, X: np.ndarray) -> "LegacyPreprocessor":
        self.pca_ = PCA(n_components=self.n_components, random_state=self.seed)
        Z = self.pca_.fit_transform(X)
        self.min_, self.max_ = Z.min(axis=0), Z.max(axis=0)
        return self

    def transform_pca(self, X: np.ndarray) -> np.ndarray:
        return self.pca_.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self.pca_.transform(X)
        return (Z - self.min_) / (self.max_ - self.min_ + 1e-8) * np.pi

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def explained_variance(self) -> float:
        return float(self.pca_.explained_variance_ratio_.sum())

    @property
    def n_input_features(self) -> int:
        return int(self.pca_.n_features_in_)
