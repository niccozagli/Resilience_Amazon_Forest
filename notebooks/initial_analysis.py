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
    from amazon.resilience.ar1 import compute_ar1_tendency, compute_sliding_ar1
    import xarray as xr

    return (
        BoundaryNorm,
        ListedColormap,
        Path,
        ccrs,
        cfeature,
        compute_ar1_tendency,
        compute_sliding_ar1,
        compute_stl_components,
        plt,
        xr,
    )


@app.cell
def _():
    #### Hyper parameters
    ar1_step = 1
    ar1_window = 60
    stl_period = 12
    stl_seasonal_window = "periodic"
    stl_robust = True
    threshold_BL = 80
    threshold_HLU = 0
    return (
        ar1_step,
        ar1_window,
        stl_period,
        stl_robust,
        stl_seasonal_window,
        threshold_BL,
        threshold_HLU,
    )


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
            "#4d0000",
            "#7f0000",
            "#b2182b",
            "#d6604d",
            "#f4a582",
            "#bdbdbd",
            "#92c5de",
            "#4393c3",
            "#2166ac",
            "#053061",
            "#021a33",
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
    stl_seasonal_window,
):
    stl_components = compute_stl_components(
        monthly_ds["VOD"],
        analysis_mask,
        period=stl_period,
        seasonal_window=stl_seasonal_window,
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
    SELECTED_CELL = 100

    _valid_cells = computed_stl_residuals.notnull().all("time").stack(
        cell=("lat", "lon")
    )
    _selected_cell = _valid_cells.where(_valid_cells, drop=True).cell.isel(cell=SELECTED_CELL)
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
        f"STL decomposition at lat={selected_lat:.3f}, lon={selected_lon:.3f}"
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Resilience AR(1) analysis
    """)
    return


@app.cell
def _(
    analysis_mask,
    ar1_step,
    ar1_window,
    compute_sliding_ar1,
    computed_stl_residuals,
):
    computed_ar1_ols = compute_sliding_ar1(
        computed_stl_residuals,
        analysis_mask,
        window=ar1_window,
        step=ar1_step,
        method="ols",
    )
    computed_ar1_pearson = compute_sliding_ar1(
        computed_stl_residuals,
        analysis_mask,
        window=ar1_window,
        step=ar1_step,
        method="pearson",
    )
    return computed_ar1_ols, computed_ar1_pearson


@app.cell
def _(
    computed_ar1_ols,
    computed_ar1_pearson,
    monthly_ds,
    plt,
    selected_lat,
    selected_lon,
):
    _cell_ar1_ols = computed_ar1_ols.sel(lat=selected_lat, lon=selected_lon)
    _cell_ar1_pearson = computed_ar1_pearson.sel(lat=selected_lat, lon=selected_lon)
    _stored_ar1 = monthly_ds["VOD AR(1)"].sel(lat=selected_lat, lon=selected_lon)

    _fig, _ax = plt.subplots(figsize=(11, 4))
    _cell_ar1_ols.plot(ax=_ax, label="Computed AR(1), OLS")
    _cell_ar1_pearson.plot(ax=_ax, label="Computed AR(1), Pearson")
    _stored_ar1.plot(ax=_ax, label="Stored VOD AR(1)", linestyle="--", alpha=0.8)
    _ax.axhline(0, color="black", linewidth=0.8)
    _ax.set_title(f"Sliding-window AR(1) at lat={selected_lat:.3f}, lon={selected_lon:.3f}")
    _ax.set_xlabel("decimal year")
    _ax.set_ylabel("AR(1)")
    _ax.legend()
    _fig
    return


@app.cell
def _(analysis_mask, computed_ar1_ols, plt):
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _masked_ar1 = computed_ar1_ols.where(cond=analysis_mask)
    _mean_ar1 = _masked_ar1.mean(dim=("lat", "lon"))
    _spatial_std_ar1 = _masked_ar1.std(dim=("lat", "lon"))

    _mean_ar1.plot(x="time", ax=_ax, label="Mean AR(1), OLS") #type:ignore
    (_mean_ar1 + _spatial_std_ar1).plot(
        x="time",
        ax=_ax,
        color="grey",
        linestyle="--",
    ) #type:ignore
    (_mean_ar1 - _spatial_std_ar1).plot(
        x="time",
        ax=_ax,
        color="grey",
        linestyle="--",
    )#type:ignore
    _ax.set_title("Spatial mean AR(1) ")
    _ax.set_xlabel("decimal year")
    _ax.set_ylabel("AR(1)")
    _ax.grid(linestyle="--", alpha=0.4)
    _ax.legend()
    _ax.set_ylim()
    _fig

    return


@app.cell
def _(
    analysis_mask,
    ccrs,
    cfeature,
    compute_ar1_tendency,
    computed_ar1_ols,
    plt,
):
    computed_ar1_ols_tendency = compute_ar1_tendency(
        computed_ar1_ols,
        analysis_mask,
    )
    _masked_ar1_ols_tendency = computed_ar1_ols_tendency.where(analysis_mask)

    _fig, _ax = plt.subplots(
        figsize=(10, 7),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    _image = _masked_ar1_ols_tendency.plot(
        ax=_ax,
        x="lon",
        y="lat",
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
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
    _ax.set_title("Kendall tau tendency of AR(1)")
    _ax.set_xlabel("longitude")
    _ax.set_ylabel("latitude")
    _fig.colorbar(
        _image,
        ax=_ax,
        orientation="horizontal",
        label="Kendall tau",
        shrink=0.85,
        pad=0.08,
    )
    _fig
    return (computed_ar1_ols_tendency,)


@app.cell
def _(analysis_mask, computed_ar1_ols_tendency, plt):
    _fig, _ax = plt.subplots(figsize=(7, 4))

    _ar1_tendency_values = (
        computed_ar1_ols_tendency
        .where(analysis_mask)
        .stack(cell=("lat", "lon"))
        .dropna("cell")
        .to_numpy()
    )

    _counts, _bins, _patches = _ax.hist(
        _ar1_tendency_values,
        bins=15,
        edgecolor="white",
    )

    _norm = plt.Normalize(vmin=-1, vmax=1)
    _cmap = plt.get_cmap("RdBu_r")

    for _bin_left, _bin_right, _patch in zip(_bins[:-1], _bins[1:], _patches):
        _bin_center = 0.5 * (_bin_left + _bin_right)
        _patch.set_facecolor(_cmap(_norm(_bin_center)))

    _ax.axvline(0, color="black", linewidth=1)
    _ax.set_xlabel("Kendall tau tendency")
    _ax.set_ylabel("Number of grid cells")
    _ax.set_title("Distribution of AR(1) tendency")
    _ax.grid(linestyle="--", alpha=0.3)

    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
