"""STL decomposition utilities for gridded monthly vegetation data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from statsmodels.tsa.seasonal import STL


@dataclass(frozen=True)
class STLComponents:
    """Full-field STL decomposition output."""

    trend: xr.DataArray
    seasonal: xr.DataArray
    residual: xr.DataArray
    complete_cell_mask: xr.DataArray


def compute_stl_components(
    data: xr.DataArray,
    mask: xr.DataArray | None = None,
    *,
    period: int = 12,
    robust: bool = True,
) -> STLComponents:
    """Compute STL components independently for each complete grid cell.

    Parameters
    ----------
    data
        Monthly gridded data with dimensions ``time``, ``lat`` and ``lon``.
    mask
        Optional 2D boolean mask over ``lat`` and ``lon``. Cells outside the
        mask are returned as ``NaN`` in all components.
    period
        Seasonal period passed to :class:`statsmodels.tsa.seasonal.STL`.
    robust
        Whether to use robust STL fitting.

    Returns
    -------
    STLComponents
        Trend, seasonal, residual and complete-cell mask. Cells with missing
        values at any time step are skipped and returned as ``NaN``.
    """
    _validate_gridded_time_series(data)

    if mask is not None:
        data = data.where(mask)

    stacked = data.stack(cell=("lat", "lon")).transpose("time", "cell")
    complete_cell_mask_stacked = stacked.notnull().all("time")
    complete = stacked.where(complete_cell_mask_stacked, drop=True)

    trend_values = np.full(complete.shape, np.nan, dtype=float)
    seasonal_values = np.full(complete.shape, np.nan, dtype=float)
    residual_values = np.full(complete.shape, np.nan, dtype=float)

    for cell_index in range(complete.sizes["cell"]):
        cell_values = complete.isel(cell=cell_index).to_numpy()
        fit = STL(cell_values, period=period, robust=robust).fit()
        trend_values[:, cell_index] = fit.trend
        seasonal_values[:, cell_index] = fit.seasonal
        residual_values[:, cell_index] = fit.resid

    complete_cell_indices = np.flatnonzero(complete_cell_mask_stacked.to_numpy())

    return STLComponents(
        trend=_to_spatial_field(
            trend_values,
            stacked=stacked,
            complete_cell_indices=complete_cell_indices,
            name="computed STL trend",
        ),
        seasonal=_to_spatial_field(
            seasonal_values,
            stacked=stacked,
            complete_cell_indices=complete_cell_indices,
            name="computed STL seasonal",
        ),
        residual=_to_spatial_field(
            residual_values,
            stacked=stacked,
            complete_cell_indices=complete_cell_indices,
            name="computed STL residual",
        ),
        complete_cell_mask=complete_cell_mask_stacked.unstack("cell").transpose(
            "lat",
            "lon",
        ),
    )


def _validate_gridded_time_series(data: xr.DataArray) -> None:
    required_dims = {"time", "lat", "lon"}
    missing_dims = required_dims.difference(data.dims)
    if missing_dims:
        missing = ", ".join(sorted(missing_dims))
        raise ValueError(f"data must include dimensions: {missing}")


def _to_spatial_field(
    component_values: np.ndarray,
    *,
    stacked: xr.DataArray,
    complete_cell_indices: np.ndarray,
    name: str,
) -> xr.DataArray:
    component_values_stacked = np.full(stacked.shape, np.nan, dtype=float)
    component_values_stacked[:, complete_cell_indices] = component_values

    return (
        xr.DataArray(
            component_values_stacked,
            coords=stacked.coords,
            dims=stacked.dims,
            name=name,
        )
        .unstack("cell")
        .transpose("time", "lat", "lon")
    )
