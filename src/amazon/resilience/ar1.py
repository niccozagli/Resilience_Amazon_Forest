"""Sliding-window lag-1 autocorrelation estimators."""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau
import xarray as xr


def compute_sliding_ar1(
    residuals: xr.DataArray,
    mask: xr.DataArray | None = None,
    *,
    window: int = 60,
    step: int = 1,
    method: str = "pearson",
) -> xr.DataArray:
    """Compute lag-1 AR(1) coefficients in sliding windows.

    Parameters
    ----------
    residuals
        Residual field with dimensions ``time``, ``lat`` and ``lon``.
    mask
        Optional 2D boolean mask over ``lat`` and ``lon``. Cells outside the
        mask are returned as ``NaN``.
    window
        Sliding-window length in time steps.
    step
        Number of time steps between consecutive windows.
    method
        Estimator to use. ``"pearson"`` computes the Pearson lag-1
        autocorrelation; ``"ols"`` computes the slope from ``x[t+1] ~ x[t]``
        with an intercept.

    Returns
    -------
    xarray.DataArray
        AR(1) field with the same dimensions and coordinates as ``residuals``.
        Each window's estimate is stored at the final time step of the window.
    """
    _validate_inputs(residuals, window, step, method)

    if mask is not None:
        residuals = residuals.where(mask)

    stacked = residuals.stack(cell=("lat", "lon")).transpose("time", "cell")
    values = stacked.to_numpy()
    ar1_values = np.full(values.shape, np.nan, dtype=float)

    for end_index in range(window - 1, values.shape[0], step):
        window_values = values[end_index - window + 1 : end_index + 1]
        ar1_values[end_index] = _estimate_window_ar1(window_values, method)

    return (
        xr.DataArray(
            ar1_values,
            coords=stacked.coords,
            dims=stacked.dims,
            name=f"AR(1) {method}",
        )
        .unstack("cell")
        .transpose("time", "lat", "lon")
    )


def compute_ar1_tendency(
    ar1: xr.DataArray,
    mask: xr.DataArray | None = None,
    *,
    min_valid: int = 2,
) -> xr.DataArray:
    """Compute Kendall tau tendency for each grid-cell AR(1) time series.

    Parameters
    ----------
    ar1
        Sliding-window AR(1) field with dimensions ``time``, ``lat`` and
        ``lon``.
    mask
        Optional 2D boolean mask over ``lat`` and ``lon``. Cells outside the
        mask are returned as ``NaN``.
    min_valid
        Minimum number of finite AR(1) values required to estimate Kendall tau.

    Returns
    -------
    xarray.DataArray
        Kendall rank-correlation coefficient between time and AR(1), with
        dimensions ``lat`` and ``lon``.
    """
    _validate_tendency_inputs(ar1, min_valid)

    if mask is not None:
        ar1 = ar1.where(mask)

    stacked = ar1.stack(cell=("lat", "lon")).transpose("time", "cell")
    values = stacked.to_numpy()
    time_values = np.arange(values.shape[0], dtype=float)
    tendency_values = np.full(values.shape[1], np.nan, dtype=float)

    for cell_index in range(values.shape[1]):
        cell_values = values[:, cell_index]
        finite = np.isfinite(cell_values)
        if finite.sum() >= min_valid:
            tendency_values[cell_index] = kendalltau(
                time_values[finite],
                cell_values[finite],
            ).statistic

    return (
        xr.DataArray(
            tendency_values,
            coords={"cell": stacked["cell"]},
            dims=("cell",),
            name="AR(1) Kendall tau",
        )
        .unstack("cell")
        .transpose("lat", "lon")
    )


def _estimate_window_ar1(window_values: np.ndarray, method: str) -> np.ndarray:
    x = window_values[:-1]
    y = window_values[1:]
    valid = np.isfinite(x).all(axis=0) & np.isfinite(y).all(axis=0)

    estimates = np.full(window_values.shape[1], np.nan, dtype=float)
    if not valid.any():
        return estimates

    x_valid = x[:, valid]
    y_valid = y[:, valid]
    x_anom = x_valid - x_valid.mean(axis=0)
    y_anom = y_valid - y_valid.mean(axis=0)
    covariance = np.sum(x_anom * y_anom, axis=0)
    x_sum_squares = np.sum(x_anom**2, axis=0)

    if method == "ols":
        estimable = x_sum_squares > 0
        valid_estimates = np.full(x_valid.shape[1], np.nan, dtype=float)
        valid_estimates[estimable] = covariance[estimable] / x_sum_squares[estimable]
    else:
        y_sum_squares = np.sum(y_anom**2, axis=0)
        denominator = np.sqrt(x_sum_squares * y_sum_squares)
        estimable = denominator > 0
        valid_estimates = np.full(x_valid.shape[1], np.nan, dtype=float)
        valid_estimates[estimable] = covariance[estimable] / denominator[estimable]

    estimates[valid] = valid_estimates
    return estimates


def _validate_inputs(
    residuals: xr.DataArray,
    window: int,
    step: int,
    method: str,
) -> None:
    required_dims = {"time", "lat", "lon"}
    missing_dims = required_dims.difference(residuals.dims)
    if missing_dims:
        missing = ", ".join(sorted(missing_dims))
        raise ValueError(f"residuals must include dimensions: {missing}")
    if window < 2:
        raise ValueError("window must be at least 2")
    if window > residuals.sizes["time"]:
        raise ValueError("window cannot be longer than the time dimension")
    if step < 1:
        raise ValueError("step must be at least 1")
    if method not in {"ols", "pearson"}:
        raise ValueError("method must be 'ols' or 'pearson'")


def _validate_tendency_inputs(ar1: xr.DataArray, min_valid: int) -> None:
    required_dims = {"time", "lat", "lon"}
    missing_dims = required_dims.difference(ar1.dims)
    if missing_dims:
        missing = ", ".join(sorted(missing_dims))
        raise ValueError(f"ar1 must include dimensions: {missing}")
    if min_valid < 2:
        raise ValueError("min_valid must be at least 2")
