"""Windowed kernel Gram decompositions for Amazon residual fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import xarray as xr

from koopman_response.algorithms.kernels import GaussianKernel


@dataclass(frozen=True)
class WindowedGramTSVD:
    """Full windowed Gram eigendecompositions for later TSVD truncation."""

    gram_eigenvalues: xr.DataArray
    gram_eigenvectors: xr.DataArray
    sigma: xr.DataArray
    valid_cell_mask: xr.DataArray
    gram_matrices: xr.DataArray | None = None
    cross_gram_matrices: xr.DataArray | None = None


def compute_subdominant_eigenvalue_trajectory(
    windowed_gram_tsvd: WindowedGramTSVD,
    *,
    truncations: Sequence[int | Literal["all"]] = (20, 40, 60, "all"),
    min_relative_eigenvalue: float = 1e-12,
    continuous_real_tolerance: float = 1e-10,
) -> xr.Dataset:
    """Compute subdominant KDMD eigenvalues for several TSVD truncations.

    Parameters
    ----------
    windowed_gram_tsvd
        Full windowed Gram eigendecompositions and stored cross-Gram matrices
        from :func:`compute_windowed_gram_tsvd`.
    truncations
        TSVD ranks to evaluate. ``"all"`` keeps every Gram mode above
        ``min_relative_eigenvalue`` times the leading Gram eigenvalue.
    min_relative_eigenvalue
        Numerical floor used to avoid dividing by zero or near-zero Gram
        eigenvalues.
    continuous_real_tolerance
        Tolerance used to collapse eigenvalues with the same continuous-time
        real part. This prevents complex-conjugate pairs from being counted as
        two different decay-rate branches.

    Returns
    -------
    xarray.Dataset
        Contains complex ``subdominant_eigenvalue`` and ``next_eigenvalue``
        together with integer ``effective_rank`` with dimensions ``time`` and
        ``truncation``.
    """
    if windowed_gram_tsvd.cross_gram_matrices is None:
        raise ValueError("cross_gram_matrices must be stored to solve KDMD operators")
    if min_relative_eigenvalue < 0:
        raise ValueError("min_relative_eigenvalue must be non-negative")
    if continuous_real_tolerance < 0:
        raise ValueError("continuous_real_tolerance must be non-negative")

    gram_eigenvalues = windowed_gram_tsvd.gram_eigenvalues.to_numpy()
    gram_eigenvectors = windowed_gram_tsvd.gram_eigenvectors.to_numpy()
    cross_gram_matrices = windowed_gram_tsvd.cross_gram_matrices.to_numpy()

    truncation_labels = [str(truncation) for truncation in truncations]
    subdominant = np.full(
        (gram_eigenvalues.shape[0], len(truncations)),
        np.nan + 0j,
        dtype=complex,
    )
    next_eigenvalue = np.full(
        (gram_eigenvalues.shape[0], len(truncations)),
        np.nan + 0j,
        dtype=complex,
    )
    effective_rank = np.zeros(
        (gram_eigenvalues.shape[0], len(truncations)),
        dtype=int,
    )

    for window_index in range(gram_eigenvalues.shape[0]):
        eigenvalues = gram_eigenvalues[window_index]
        eigenvectors = gram_eigenvectors[window_index]
        cross_gram = cross_gram_matrices[window_index]
        numerical_floor = min_relative_eigenvalue * eigenvalues[0]
        max_usable_rank = int(np.sum(eigenvalues > numerical_floor))

        for truncation_index, truncation in enumerate(truncations):
            rank = (
                max_usable_rank
                if truncation == "all"
                else min(int(truncation), max_usable_rank)
            )
            effective_rank[window_index, truncation_index] = rank
            if rank < 2:
                continue

            reduced_koopman = _reduced_koopman_matrix(
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                cross_gram=cross_gram,
                rank=rank,
            )
            koopman_eigenvalues = np.linalg.eigvals(reduced_koopman)
            ordered_eigenvalues = _distinct_continuous_real_eigenvalues(
                koopman_eigenvalues,
                tolerance=continuous_real_tolerance,
            )
            if ordered_eigenvalues.size > 1:
                subdominant[window_index, truncation_index] = ordered_eigenvalues[1]
            if ordered_eigenvalues.size > 2:
                next_eigenvalue[window_index, truncation_index] = ordered_eigenvalues[2]

    coords = {
        "time": windowed_gram_tsvd.gram_eigenvalues["time"],
        "truncation": truncation_labels,
    }
    return xr.Dataset(
        data_vars={
            "subdominant_eigenvalue": (
                ("time", "truncation"),
                subdominant,
            ),
            "next_eigenvalue": (
                ("time", "truncation"),
                next_eigenvalue,
            ),
            "effective_rank": (
                ("time", "truncation"),
                effective_rank,
            ),
        },
        coords=coords,
    )


def compute_eigenvalue_spectra(
    windowed_gram_tsvd: WindowedGramTSVD,
    *,
    truncations: Sequence[int | Literal["all"]] = (10, 20, 30),
    min_relative_eigenvalue: float = 1e-12,
) -> xr.Dataset:
    """Compute full KDMD eigenvalue spectra for several TSVD truncations."""
    if windowed_gram_tsvd.cross_gram_matrices is None:
        raise ValueError("cross_gram_matrices must be stored to solve KDMD operators")
    if min_relative_eigenvalue < 0:
        raise ValueError("min_relative_eigenvalue must be non-negative")

    gram_eigenvalues = windowed_gram_tsvd.gram_eigenvalues.to_numpy()
    gram_eigenvectors = windowed_gram_tsvd.gram_eigenvectors.to_numpy()
    cross_gram_matrices = windowed_gram_tsvd.cross_gram_matrices.to_numpy()

    truncation_labels = [str(truncation) for truncation in truncations]
    numeric_truncations = [
        int(truncation)
        for truncation in truncations
        if truncation != "all"
    ]
    max_rank = max(numeric_truncations) if numeric_truncations else 0
    if "all" in truncations:
        max_rank = gram_eigenvalues.shape[1]

    spectra = np.full(
        (gram_eigenvalues.shape[0], len(truncations), max_rank),
        np.nan + 0j,
        dtype=complex,
    )
    effective_rank = np.zeros(
        (gram_eigenvalues.shape[0], len(truncations)),
        dtype=int,
    )

    for window_index in range(gram_eigenvalues.shape[0]):
        eigenvalues = gram_eigenvalues[window_index]
        eigenvectors = gram_eigenvectors[window_index]
        cross_gram = cross_gram_matrices[window_index]
        numerical_floor = min_relative_eigenvalue * eigenvalues[0]
        max_usable_rank = int(np.sum(eigenvalues > numerical_floor))

        for truncation_index, truncation in enumerate(truncations):
            rank = (
                max_usable_rank
                if truncation == "all"
                else min(int(truncation), max_usable_rank)
            )
            effective_rank[window_index, truncation_index] = rank
            if rank < 1:
                continue

            reduced_koopman = _reduced_koopman_matrix(
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                cross_gram=cross_gram,
                rank=rank,
            )
            koopman_eigenvalues = np.linalg.eigvals(reduced_koopman)
            order = np.argsort(np.abs(koopman_eigenvalues))[::-1]
            spectra[window_index, truncation_index, :rank] = koopman_eigenvalues[order]

    coords = {
        "time": windowed_gram_tsvd.gram_eigenvalues["time"],
        "truncation": truncation_labels,
        "eigenvalue_index": np.arange(max_rank),
    }
    return xr.Dataset(
        data_vars={
            "eigenvalue": (
                ("time", "truncation", "eigenvalue_index"),
                spectra,
            ),
            "effective_rank": (
                ("time", "truncation"),
                effective_rank,
            ),
        },
        coords=coords,
    )


def compute_spatial_modes(
    residuals: xr.DataArray,
    mask: xr.DataArray | None,
    windowed_gram_tsvd: WindowedGramTSVD,
    *,
    window: int,
    lag: int = 1,
    rank: int = 10,
    eigenvalue_index: int = 0,
    window_indices: Sequence[int] = (0, -1),
) -> xr.DataArray:
    """Compute spatial Koopman mode maps for selected windows.

    The mode is computed for the vector-valued observable equal to the
    flattened residual field. It is the spatial pattern associated with the
    requested reduced KDMD eigenvalue.
    """
    _validate_inputs(residuals, window, step=1, lag=lag, sigma=1.0)
    if rank < 1:
        raise ValueError("rank must be at least 1")
    if eigenvalue_index < 0:
        raise ValueError("eigenvalue_index must be non-negative")
    if windowed_gram_tsvd.cross_gram_matrices is None:
        raise ValueError("cross_gram_matrices must be stored to compute modes")

    if mask is not None:
        residuals = residuals.where(mask)

    stacked = residuals.stack(cell=("lat", "lon")).transpose("time", "cell")
    valid_cell_mask_stacked = stacked.notnull().all("time")
    data = stacked.where(valid_cell_mask_stacked, drop=True).to_numpy()
    valid_cell_indices = np.flatnonzero(valid_cell_mask_stacked.to_numpy())

    resolved_window_indices = [
        index if index >= 0 else windowed_gram_tsvd.gram_eigenvalues.sizes["time"] + index
        for index in window_indices
    ]
    mode_values = np.full(
        (len(resolved_window_indices), stacked.sizes["cell"]),
        np.nan + 0j,
        dtype=complex,
    )
    window_labels = []

    for output_index, window_index in enumerate(resolved_window_indices):
        window_time = windowed_gram_tsvd.gram_eigenvalues["time"].isel(time=window_index)
        end_index = _time_index_for_value(residuals["time"], window_time)
        start_index = end_index - window + 1
        x_snapshots = data[start_index : end_index + 1 - lag]

        gram_eigenvalues = windowed_gram_tsvd.gram_eigenvalues.isel(
            time=window_index
        ).to_numpy()
        gram_eigenvectors = windowed_gram_tsvd.gram_eigenvectors.isel(
            time=window_index
        ).to_numpy()
        cross_gram = windowed_gram_tsvd.cross_gram_matrices.isel(
            time=window_index
        ).to_numpy()

        rank_value = min(rank, int(np.sum(gram_eigenvalues > 0)))
        if eigenvalue_index >= rank_value:
            raise ValueError("eigenvalue_index must be smaller than the effective rank")

        reduced_koopman = _reduced_koopman_matrix(
            eigenvalues=gram_eigenvalues,
            eigenvectors=gram_eigenvectors,
            cross_gram=cross_gram,
            rank=rank_value,
        )
        koopman_eigenvalues, right_eigvecs = np.linalg.eig(reduced_koopman)
        left_eigvecs = np.linalg.eig(reduced_koopman.T.conj())[1]
        order = np.argsort(np.abs(koopman_eigenvalues))[::-1]
        left_eigvecs = _match_left_eigenvectors(
            reduced_koopman,
            koopman_eigenvalues[order],
            left_eigvecs,
        )

        selected = order[eigenvalue_index]
        basis = gram_eigenvectors[:, :rank_value]
        singular_values = gram_eigenvalues[:rank_value]
        projected_observable = (basis.T.conj() @ x_snapshots) / np.sqrt(
            singular_values
        )[:, None]
        mode = left_eigvecs[:, eigenvalue_index].conj().T @ projected_observable
        mode_values[output_index, valid_cell_indices] = mode
        window_labels.append(f"{float(window_time):.3f}")

    return (
        xr.DataArray(
            mode_values,
            coords={
                "window": window_labels,
                "cell": stacked["cell"],
            },
            dims=("window", "cell"),
            name="Koopman spatial mode",
        )
        .unstack("cell")
        .transpose("window", "lat", "lon")
    )


def compute_windowed_gram_tsvd(
    residuals: xr.DataArray,
    mask: xr.DataArray | None = None,
    *,
    window: int = 96,
    step: int = 1,
    lag: int = 1,
    sigma: float | Literal["median"] = "median",
    store_matrices: bool = True,
) -> WindowedGramTSVD:
    """Compute Gaussian-kernel Gram matrices and full eigenspectra by window.

    Parameters
    ----------
    residuals
        Residual field with dimensions ``time``, ``lat`` and ``lon``.
    mask
        Optional 2D boolean mask over ``lat`` and ``lon``. Cells outside the
        mask are excluded from the flattened state vector.
    window
        Window length in months.
    step
        Number of months between window end points.
    lag
        Snapshot lag in months. With ``lag=1``, each window produces
        ``window - 1`` snapshot pairs.
    sigma
        Gaussian-kernel bandwidth. ``"median"`` uses the median pairwise
        snapshot distance inside each window.
    store_matrices
        Whether to store full ``G = k(X, X)`` and ``A = k(Y, X)`` matrices for
        later truncation/Koopman solves.

    Returns
    -------
    WindowedGramTSVD
        Full descending eigenvalue/eigenvector decompositions of each window's
        Gram matrix, with window coordinates stored at the end month.
    """
    _validate_inputs(residuals, window, step, lag, sigma)

    if mask is not None:
        residuals = residuals.where(mask)

    stacked = residuals.stack(cell=("lat", "lon")).transpose("time", "cell")
    valid_cell_mask_stacked = stacked.notnull().all("time")
    data = stacked.where(valid_cell_mask_stacked, drop=True).to_numpy()

    n_pairs = window - lag
    window_end_indices = np.arange(window - 1, data.shape[0], step)
    window_times = residuals["time"].isel(time=window_end_indices)
    snapshot_index = np.arange(n_pairs)
    mode_index = np.arange(n_pairs)

    gram_eigenvalues = np.full((window_end_indices.size, n_pairs), np.nan, dtype=float)
    gram_eigenvectors = np.full(
        (window_end_indices.size, n_pairs, n_pairs),
        np.nan,
        dtype=float,
    )
    sigma_values = np.full(window_end_indices.size, np.nan, dtype=float)
    gram_matrices = (
        np.full((window_end_indices.size, n_pairs, n_pairs), np.nan, dtype=float)
        if store_matrices
        else None
    )
    cross_gram_matrices = (
        np.full((window_end_indices.size, n_pairs, n_pairs), np.nan, dtype=float)
        if store_matrices
        else None
    )

    for window_number, end_index in enumerate(window_end_indices):
        start_index = end_index - window + 1
        window_values = data[start_index : end_index + 1]
        x_snapshots = window_values[:-lag]
        y_snapshots = window_values[lag:]

        sigma_value = (
            _median_pairwise_distance(x_snapshots)
            if sigma == "median"
            else float(sigma)
        )
        sigma_values[window_number] = sigma_value

        kernel = GaussianKernel(sigma=sigma_value)
        gram = kernel(x_snapshots, x_snapshots)
        cross_gram = kernel(y_snapshots, x_snapshots)

        eigvals, eigvecs = _full_tsvd(gram)
        gram_eigenvalues[window_number] = eigvals
        gram_eigenvectors[window_number] = eigvecs

        if store_matrices:
            gram_matrices[window_number] = gram
            cross_gram_matrices[window_number] = cross_gram

    coords = {
        "time": window_times,
        "snapshot": snapshot_index,
        "mode": mode_index,
    }
    return WindowedGramTSVD(
        gram_eigenvalues=xr.DataArray(
            gram_eigenvalues,
            coords={"time": window_times, "mode": mode_index},
            dims=("time", "mode"),
            name="kernel Gram eigenvalues",
        ),
        gram_eigenvectors=xr.DataArray(
            gram_eigenvectors,
            coords=coords,
            dims=("time", "snapshot", "mode"),
            name="kernel Gram eigenvectors",
        ),
        sigma=xr.DataArray(
            sigma_values,
            coords={"time": window_times},
            dims=("time",),
            name="Gaussian kernel sigma",
        ),
        valid_cell_mask=valid_cell_mask_stacked.unstack("cell").transpose(
            "lat",
            "lon",
        ),
        gram_matrices=(
            xr.DataArray(
                gram_matrices,
                coords={"time": window_times, "snapshot_i": snapshot_index, "snapshot_j": snapshot_index},
                dims=("time", "snapshot_i", "snapshot_j"),
                name="kernel Gram matrix",
            )
            if store_matrices
            else None
        ),
        cross_gram_matrices=(
            xr.DataArray(
                cross_gram_matrices,
                coords={"time": window_times, "snapshot_i": snapshot_index, "snapshot_j": snapshot_index},
                dims=("time", "snapshot_i", "snapshot_j"),
                name="kernel cross-Gram matrix",
            )
            if store_matrices
            else None
        ),
    )


def _full_tsvd(gram: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sym_gram = 0.5 * (gram + gram.T)
    eigvals, eigvecs = np.linalg.eigh(sym_gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    return eigvals, eigvecs


def _distinct_continuous_real_eigenvalues(
    eigenvalues: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    finite_eigenvalues = eigenvalues[np.isfinite(eigenvalues) & (np.abs(eigenvalues) > 0)]
    if finite_eigenvalues.size == 0:
        return finite_eigenvalues

    continuous_real = np.log(finite_eigenvalues).real
    order = np.argsort(continuous_real)[::-1]
    sorted_eigenvalues = finite_eigenvalues[order]
    sorted_real = continuous_real[order]

    distinct = []
    start = 0
    while start < sorted_eigenvalues.size:
        group_real = sorted_real[start]
        end = start + 1
        while (
            end < sorted_eigenvalues.size
            and abs(sorted_real[end] - group_real) <= tolerance
        ):
            end += 1

        group = sorted_eigenvalues[start:end]
        positive_imaginary = group[group.imag >= 0]
        candidates = positive_imaginary if positive_imaginary.size else group
        representative = candidates[np.argmin(np.abs(candidates.imag))]
        distinct.append(representative)
        start = end

    return np.asarray(distinct, dtype=complex)


def _reduced_koopman_matrix(
    *,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    cross_gram: np.ndarray,
    rank: int,
) -> np.ndarray:
    basis = eigenvectors[:, :rank]
    singular_values = eigenvalues[:rank]
    inv_sqrt = 1.0 / np.sqrt(singular_values)
    return (inv_sqrt[:, None] * (basis.T @ cross_gram @ basis)) * inv_sqrt[None, :]


def _match_left_eigenvectors(
    matrix: np.ndarray,
    sorted_eigenvalues: np.ndarray,
    left_eigvecs: np.ndarray,
) -> np.ndarray:
    matched = np.empty_like(left_eigvecs)
    for index, eigenvalue in enumerate(sorted_eigenvalues):
        residuals = np.linalg.norm(matrix.T.conj() @ left_eigvecs - eigenvalue.conjugate() * left_eigvecs, axis=0)
        best = int(np.argmin(residuals))
        matched[:, index] = left_eigvecs[:, best]
    return matched


def _time_index_for_value(time: xr.DataArray, value: xr.DataArray) -> int:
    matches = np.flatnonzero(time.to_numpy() == value.to_numpy())
    if matches.size != 1:
        raise ValueError("window time could not be matched to residual time coordinate")
    return int(matches[0])


def _median_pairwise_distance(data: np.ndarray) -> float:
    squared_distances = _pairwise_squared_distances(data)
    distances = np.sqrt(squared_distances)
    upper_triangle = distances[np.triu_indices_from(distances, k=1)]
    positive_distances = upper_triangle[upper_triangle > 0]
    if positive_distances.size == 0:
        return 1.0
    return float(np.median(positive_distances))


def _pairwise_squared_distances(data: np.ndarray) -> np.ndarray:
    row_norms = np.sum(data * data, axis=1)
    squared_distances = row_norms[:, None] + row_norms[None, :] - 2.0 * (data @ data.T)
    return np.maximum(squared_distances, 0.0)


def _validate_inputs(
    residuals: xr.DataArray,
    window: int,
    step: int,
    lag: int,
    sigma: float | Literal["median"],
) -> None:
    required_dims = {"time", "lat", "lon"}
    missing_dims = required_dims.difference(residuals.dims)
    if missing_dims:
        missing = ", ".join(sorted(missing_dims))
        raise ValueError(f"residuals must include dimensions: {missing}")
    if window < 3:
        raise ValueError("window must be at least 3")
    if lag < 1:
        raise ValueError("lag must be at least 1")
    if lag >= window:
        raise ValueError("lag must be smaller than window")
    if window > residuals.sizes["time"]:
        raise ValueError("window cannot exceed the time dimension")
    if step < 1:
        raise ValueError("step must be at least 1")
    if sigma != "median" and float(sigma) <= 0:
        raise ValueError("sigma must be positive or 'median'")
