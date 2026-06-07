import numpy as np
import xarray as xr

from amazon.preprocessing.stl import compute_stl_components


def test_compute_stl_components_preserves_grid_and_masks_incomplete_cells():
    time = np.arange(24)
    lat = np.array([-1.0, 1.0])
    lon = np.array([10.0, 20.0])

    seasonal = np.sin(2 * np.pi * time / 12)
    data = np.empty((time.size, lat.size, lon.size), dtype=float)
    for lat_index in range(lat.size):
        for lon_index in range(lon.size):
            data[:, lat_index, lon_index] = (
                5.0
                + 0.1 * time
                + seasonal
                + lat_index
                + lon_index
            )
    data[3, 1, 1] = np.nan

    values = xr.DataArray(
        data,
        coords={"time": time, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
    )
    mask = xr.DataArray(
        [[True, False], [True, True]],
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
    )

    components = compute_stl_components(values, mask, period=12, robust=True)

    assert components.residual.dims == ("time", "lat", "lon")
    assert components.residual.shape == values.shape
    assert components.trend.shape == values.shape
    assert components.seasonal.shape == values.shape
    assert bool(components.complete_cell_mask.sel(lat=-1.0, lon=10.0))
    assert bool(components.complete_cell_mask.sel(lat=1.0, lon=10.0))
    assert not bool(components.complete_cell_mask.sel(lat=-1.0, lon=20.0))
    assert not bool(components.complete_cell_mask.sel(lat=1.0, lon=20.0))
    assert components.residual.sel(lat=-1.0, lon=20.0).isnull().all()
    assert components.residual.sel(lat=1.0, lon=20.0).isnull().all()
    assert components.residual.sel(lat=-1.0, lon=10.0).notnull().all()
