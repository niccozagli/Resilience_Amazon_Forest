from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


def _validate_kernel_inputs(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same feature dimension")
    return X, Y


def _pairwise_sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X_norm = np.sum(X * X, axis=1)[:, None]
    Y_norm = np.sum(Y * Y, axis=1)[None, :]
    dists = X_norm + Y_norm - 2.0 * (X @ Y.T)
    return np.maximum(dists, 0.0)


class Kernel(ABC):
    """Abstract kernel interface for kernelized DMD."""

    def fit(self, data: np.ndarray) -> "Kernel":
        _ = data
        return self

    @abstractmethod
    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        """Evaluate kernel matrix K(X, Y)."""

    def grad_x(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        """
        Gradient of the kernel with respect to X.

        Returns an array of shape (n_samples_X, n_samples_Y, n_features).
        """
        raise NotImplementedError("grad_x is not implemented for this kernel")


class GaussianKernel(Kernel):
    """
    Gaussian (RBF) kernel:

        k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
    """

    def __init__(self, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.sigma = float(sigma)

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        if Y is None:
            Y = X
        X, Y = _validate_kernel_inputs(X, Y)
        dists = _pairwise_sq_dists(X, Y)
        return np.exp(-dists / (2.0 * self.sigma**2))

    def grad_x(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        if Y is None:
            Y = X
        X, Y = _validate_kernel_inputs(X, Y)
        K = self.__call__(X, Y)
        diff = X[:, None, :] - Y[None, :, :]
        return -(K[:, :, None] * diff) / (self.sigma**2)


class PolynomialKernel(Kernel):
    """
    Polynomial kernel:

        k(x, y) = (gamma * x^T y + coef0) ** degree
    """

    def __init__(
        self,
        degree: int = 2,
        gamma: float | None = None,
        coef0: float = 1.0,
    ):
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.degree = int(degree)
        self.gamma = gamma
        self.coef0 = float(coef0)
        self._gamma: Optional[float] = None

    def fit(self, data: np.ndarray) -> "PolynomialKernel":
        if self.gamma is None:
            if data.ndim != 2 or data.shape[1] < 1:
                raise ValueError("data must be 2D with a valid feature dimension")
            self._gamma = 1.0 / data.shape[1]
        else:
            self._gamma = float(self.gamma)
        return self

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        if Y is None:
            Y = X
        X, Y = _validate_kernel_inputs(X, Y)
        if self._gamma is None and self.gamma is None:
            raise ValueError("PolynomialKernel must be fit before calling or provide gamma")
        gamma = self._gamma if self.gamma is None else float(self.gamma)
        return (gamma * (X @ Y.T) + self.coef0) ** self.degree

    def grad_x(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError("PolynomialKernel grad_x is not yet implemented")
