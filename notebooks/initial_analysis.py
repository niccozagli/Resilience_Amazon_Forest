import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import matplotlib.pyplot as plt
    from amazon.preprocessing.stl import compute_stl_components
    import xarray as xr

    return (
        BoundaryNorm,
        ListedColormap,
        Path,
        ccrs,
        cfeature,
        compute_stl_components,
        plt,
        xr,
    )


@app.cell
def _():
    #### Hyper parameters
    stl_period = 12
    stl_robust = True
    threshold_BL = 80
    threshold_HLU = 0
    return stl_period, stl_robust, threshold_BL, threshold_HLU


@app.cell
def _(Path, xr):
    #### Loading dataset
    project_root = Path(__file__).resolve().parents[1]
    old_data_path = project_root / "data" / "processed" / "old_dataset" 
    monthly_data_path = old_data_path / "VOD_Month_full.nc"
    amazon_mask_path = old_data_path / "Amazon_Basin_Mask.nc"
    broadleaf_fraction_path = old_data_path / "BL_2001_fraction.nc"
    human_land_use_path = old_data_path / "HLU_fraction.nc"


    monthly_ds = xr.open_dataset(monthly_data_path, engine="h5netcdf")
    amazon_mask_ds = xr.open_dataset(amazon_mask_path, engine="h5netcdf")
    broadleaf_fraction_ds = xr.open_dataset(broadleaf_fraction_path, engine="h5netcdf")
    human_land_use_ds = xr.open_dataset(human_land_use_path, engine="h5netcdf")
    return amazon_mask_ds, broadleaf_fraction_ds, human_land_use_ds, monthly_ds


@app.cell
def _(
    amazon_mask_ds,
    broadleaf_fraction_ds,
    human_land_use_ds,
    threshold_BL,
    threshold_HLU,
):
    #### Masking 
    amazon_mask = amazon_mask_ds["Amazon Basin Mask"].notnull() # mask for the amazon basin
    forest_mask = broadleaf_fraction_ds["BL Frac"] >= threshold_BL # mask for Broafleef forest in 2001
    no_human_land_use_mask = human_land_use_ds["HLU Frac"].max("time") <= threshold_HLU  # no human land use in the whole period
    return amazon_mask, forest_mask, no_human_land_use_mask


@app.cell
def _(amazon_mask, forest_mask, no_human_land_use_mask):
    analysis_mask = amazon_mask & forest_mask & no_human_land_use_mask
    return (analysis_mask,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plotting
    """)
    return


@app.cell
def _(amazon_mask, analysis_mask, ccrs, cfeature, monthly_ds, plt):
    _variable = "VOD"
    _time_index = 10
    _selected = monthly_ds[_variable].isel(time=_time_index)
    _selected_amazon = _selected.where(amazon_mask)
    _selected_analysis_domain = _selected.where(analysis_mask)

    _fig, _axes = plt.subplots(
        ncols=2,
        figsize=(18, 8),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    _vmin = float(_selected_amazon.min())
    _vmax = float(_selected_amazon.max())

    _image = _selected_amazon.plot(
        ax=_axes[0],
        x="lon",
        y="lat",
        transform=ccrs.PlateCarree(),
        vmin=_vmin,
        vmax=_vmax,
        add_colorbar=False,
    ) #type:ignore
    _selected_analysis_domain.plot(
        ax=_axes[1],
        x="lon",
        y="lat",
        transform=ccrs.PlateCarree(),
        vmin=_vmin,
        vmax=_vmax,
        add_colorbar=False,
    ) #type:ignore

    for _ax in _axes:
        _ax.coastlines(resolution="50m", linewidth=0.8)
        _ax.add_feature(cfeature.BORDERS, linewidth=0.6)
        _gridlines = _ax.gridlines(
            draw_labels=True,
            linewidth=0.3,
            color="0.5",
            alpha=0.5,
            linestyle="--",
        )
        _gridlines.top_labels = False
        _gridlines.right_labels = False
        _ax.set_xlabel("longitude")
        _ax.set_ylabel("latitude")

    _axes[0].set_title(f"{_variable}: Amazon basin")
    _axes[1].set_title(f"{_variable}: Amazon, BL >= 80%, HLU = 0")
    _fig.colorbar(
        _image,
        ax=_axes,
        orientation="horizontal",
        label=_variable,
        shrink=0.85,
        pad=0.05,
    )
    _fig
    return


@app.cell
def _(
    BoundaryNorm,
    ListedColormap,
    analysis_mask,
    ccrs,
    cfeature,
    monthly_ds,
    plt,
):
    _early_mean = (
        monthly_ds["VOD"]
        .where(analysis_mask)
        .sel(time=(monthly_ds["time"] >= 1991) & (monthly_ds["time"] < 1996))
        .mean("time")
    )
    _late_mean = (
        monthly_ds["VOD"]
        .where(analysis_mask)
        .sel(time=(monthly_ds["time"] >= 2012) & (monthly_ds["time"] < 2017))
        .mean("time")
    )
    _vod_change = _late_mean - _early_mean
    _boundaries = [
        -0.13,
        -0.11,
        -0.09,
        -0.07,
        -0.05,
        -0.03,
        0.03,
        0.05,
        0.07,
        0.09,
        0.11,
        0.13,
    ]
    _change_cmap = ListedColormap(
        [
            "#ff0000",
            "#ff4d4d",
            "#ff8080",
            "#ffb3b3",
            "#ffe0e0",
            "#bdbdbd",
            "#e0f0ff",
            "#b3d9ff",
            "#80bfff",
            "#4da6ff",
            "#008cff",
        ]
    )
    _change_norm = BoundaryNorm(_boundaries, _change_cmap.N, clip=True)

    _fig, _ax = plt.subplots(
        figsize=(10, 7),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    _image = _vod_change.plot(
        ax=_ax,
        x="lon",
        y="lat",
        transform=ccrs.PlateCarree(),
        cmap=_change_cmap,
        norm=_change_norm,
        add_colorbar=False,
    ) #type:ignore
    _ax.coastlines(resolution="50m", linewidth=0.8)
    _ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    _gridlines = _ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        color="0.5",
        alpha=0.5,
        linestyle="--",
    )
    _gridlines.top_labels = False
    _gridlines.right_labels = False
    _ax.set_title("Mean VOD change: 2012-2016 minus 1991-1995")
    _ax.set_xlabel("longitude")
    _ax.set_ylabel("latitude")
    _fig.colorbar(
        _image,
        ax=_ax,
        orientation="horizontal",
        label="VOD change",
        boundaries=_boundaries,
        ticks=_boundaries[1:-1],
        shrink=0.85,
        pad=0.08,
    )
    _fig
    return


@app.cell
def _(monthly_ds):
    monthly_ds["VOD"]
    return


@app.cell
def _(analysis_mask, monthly_ds, plt):
    _fig , _ax = plt.subplots()

    _mean_spatial_vod = (monthly_ds["VOD"]
                        .where(cond=analysis_mask)
                        .mean(dim=("lat","lon"))
    )
    _mean_spatial_vod.plot(x="time",ax=_ax)
    _ax.set_ylabel("Mean VOD")
    _ax.grid(linestyle='--',alpha=0.4)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Detrending analysis
    """)
    return


@app.cell
def _(
    analysis_mask,
    compute_stl_components,
    monthly_ds,
    stl_period,
    stl_robust,
):
    stl_components = compute_stl_components(
        monthly_ds["VOD"],
        analysis_mask,
        period=stl_period,
        robust=stl_robust,
    )
    computed_stl_trend = stl_components.trend
    computed_stl_seasonal = stl_components.seasonal
    computed_stl_residuals = stl_components.residual
    stl_complete_cell_mask = stl_components.complete_cell_mask
    return computed_stl_residuals, computed_stl_seasonal, computed_stl_trend


@app.cell
def _(
    computed_stl_residuals,
    computed_stl_seasonal,
    computed_stl_trend,
    monthly_ds,
):
    _valid_cells = computed_stl_residuals.notnull().all("time").stack(
        cell=("lat", "lon")
    )
    _selected_cell = _valid_cells.where(_valid_cells, drop=True).cell.isel(cell=50)
    selected_lat = float(_selected_cell["lat"])
    selected_lon = float(_selected_cell["lon"])

    cell_timeseries = monthly_ds["VOD"].sel(lat=selected_lat, lon=selected_lon)
    cell_stl_trend = computed_stl_trend.sel(lat=selected_lat, lon=selected_lon)
    cell_stl_seasonal = computed_stl_seasonal.sel(lat=selected_lat, lon=selected_lon)
    cell_stl_residual = computed_stl_residuals.sel(lat=selected_lat, lon=selected_lon)
    stored_residual = monthly_ds["VOD Resid"].sel(lat=selected_lat, lon=selected_lon)
    return (
        cell_stl_residual,
        cell_stl_seasonal,
        cell_stl_trend,
        cell_timeseries,
        selected_lat,
        selected_lon,
        stored_residual,
    )


@app.cell
def _(cell_timeseries, plt, selected_lat, selected_lon):
    _fig, _ax = plt.subplots(figsize=(11, 4))
    cell_timeseries.plot(ax=_ax)
    _ax.set_title(f"VOD time series at lat={selected_lat:.3f}, lon={selected_lon:.3f}")
    _ax.set_xlabel("decimal year")
    _ax.set_ylabel("VOD")
    _fig
    return


@app.cell
def _(
    cell_stl_residual,
    cell_stl_seasonal,
    cell_stl_trend,
    cell_timeseries,
    plt,
    selected_lat,
    selected_lon,
):
    _fig, _axes = plt.subplots(nrows=4, figsize=(12, 9), sharex=True)

    cell_timeseries.plot(ax=_axes[0])
    _axes[0].set_title(
        f"Python STL decomposition at lat={selected_lat:.3f}, lon={selected_lon:.3f}"
    )
    _axes[0].set_ylabel("VOD")

    cell_stl_trend.plot(ax=_axes[1], color="tab:orange")
    _axes[1].set_ylabel("trend")

    cell_stl_seasonal.plot(ax=_axes[2], color="tab:green")
    _axes[2].set_ylabel("seasonal")

    cell_stl_residual.plot(ax=_axes[3], color="tab:red")
    _axes[3].axhline(0, color="black", linewidth=0.8)
    _axes[3].set_xlabel("decimal year")
    _axes[3].set_ylabel("residual")

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(cell_stl_residual, plt, selected_lat, selected_lon, stored_residual):
    _fig, _ax = plt.subplots(figsize=(11, 4))
    cell_stl_residual.plot(ax=_ax, label="Python STL residual")
    stored_residual.plot(ax=_ax, label="Stored VOD Resid")
    _ax.set_title(
        f"STL residual comparison at lat={selected_lat:.3f}, lon={selected_lon:.3f}"
    )
    _ax.set_xlabel("decimal year")
    _ax.set_ylabel("VOD residual")
    _ax.legend()
    _fig
    return


if __name__ == "__main__":
    app.run()
