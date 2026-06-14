from __future__ import annotations

from typing import Tuple

import numpy as np


def minmax_scale(
    data: np.ndarray,
    feature_range: Tuple[float, float] = (-1.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Min-max scale each column of data to a target range.

    Parameters:
        data: (n_samples, d) input array.
        feature_range: target range (min, max), default (-1, 1).

    Returns:
        scaled_data: normalized data in feature_range
        data_min: minimum values per column (shape: d,)
        data_max: maximum values per column (shape: d,)
    """
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    a, b = feature_range
    scaled = (b - a) * (data - data_min) / (data_max - data_min) + a
    return scaled, data_min, data_max


def standardize(
    data: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize each column of data to zero mean and unit variance.

    Parameters:
        data: (n_samples, d) input array.
        eps: small value to avoid division by zero.

    Returns:
        scaled_data: standardized data
        mean: mean values per column (shape: d,)
        std: standard deviation per column (shape: d,), with zeros replaced by 1.0
    """
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std_safe = np.where(std < eps, 1.0, std)
    scaled = (data - mean) / std_safe
    return scaled, mean, std_safe


def standardize_global(
    data: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float, float]:
    """
    Standardize data using a single global mean and std over all entries.

    Parameters:
        data: (n_samples, d) input array.
        eps: small value to avoid division by zero.

    Returns:
        scaled_data: standardized data
        mean: global mean (scalar)
        std: global standard deviation (scalar), with zeros replaced by 1.0
    """
    mean = float(np.mean(data))
    std = float(np.std(data))
    std_safe = 1.0 if std < eps else std
    scaled = (data - mean) / std_safe
    return scaled, mean, std_safe


def make_snapshots(
    data: np.ndarray,
    lag: int = 1,
    stride: int = 1,
    dt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Create snapshot pairs (X, Y) from a single trajectory.

    Parameters:
        data: array of shape (n_samples, dim)
        lag: positive integer time lag
        stride: positive integer subsampling stride (use every stride-th sample)
        dt: base time step between consecutive samples in data

    Returns:
        X: data[:-lag]
        Y: data[lag:]
        dt_eff: effective time step (dt * lag * stride)
    """
    if lag < 1:
        raise ValueError("lag must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if stride > 1:
        data = data[::stride]
    n_samples = data.shape[0]
    if lag >= n_samples:
        raise ValueError(f"lag={lag} is too large for data length {n_samples}")
    dt_eff = float(dt) * int(lag) * int(stride)
    return data[: -lag], data[lag:], dt_eff
