import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import matplotlib.pyplot as plt
    import numpy as np
    from amazon.koopman.windowed import (
        compute_eigenvalue_spectra,
        compute_spatial_modes,
        compute_subdominant_eigenvalue_trajectory,
        compute_windowed_gram_tsvd,
    )
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
        compute_spatial_modes,
        compute_stl_components,
        compute_subdominant_eigenvalue_trajectory,
        compute_windowed_gram_tsvd,
        np,
        plt,
        xr,
    )


@app.cell
def _():
    #### Hyper parameters
    ar1_step = 1
    ar1_window = 60
    koopman_lag = 1
    koopman_sigma = "median"
    koopman_step = 1
    koopman_window = 120
    koopman_store_matrices = True
    koopman_truncations = (10,20)
    koopman_robustness_truncations = (5, 10, 15, 20, 25, 30)
    stl_period = 12
    stl_seasonal_window = "periodic"
    stl_robust = True
    threshold_BL = 80
    threshold_HLU = 0
    return (
        ar1_step,
        ar1_window,
        koopman_lag,
        koopman_robustness_truncations,
        koopman_sigma,
        koopman_step,
        koopman_store_matrices,
        koopman_truncations,
        koopman_window,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Koopman moving-window Gram TSVD
    """)
    return


@app.cell
def _(
    analysis_mask,
    compute_windowed_gram_tsvd,
    computed_stl_residuals,
    koopman_lag,
    koopman_sigma,
    koopman_step,
    koopman_store_matrices,
    koopman_window,
):
    windowed_gram_tsvd = compute_windowed_gram_tsvd(
        computed_stl_residuals,
        analysis_mask,
        window=koopman_window,
        step=koopman_step,
        lag=koopman_lag,
        sigma=koopman_sigma,
        store_matrices=koopman_store_matrices,
    )
    return (windowed_gram_tsvd,)


@app.cell
def _(windowed_gram_tsvd):
    windowed_gram_tsvd.gram_eigenvalues
    return


@app.cell
def _(plt, windowed_gram_tsvd):
    _eigenvalues = windowed_gram_tsvd.gram_eigenvalues
    _window_indices = [0, _eigenvalues.sizes["time"] // 2, _eigenvalues.sizes["time"] - 1]

    _fig, _ax = plt.subplots(figsize=(7, 4))
    for _window_index in _window_indices:
        _spectrum = _eigenvalues.isel(time=_window_index)
        _label_time = float(_spectrum["time"])
        _ax.semilogy(
            _spectrum["mode"].to_numpy(),
            _spectrum.to_numpy(),
            marker=".",
            label=f"{_label_time:.1f}",
        )

    _ax.set_title("Full kernel Gram spectra by moving window")
    _ax.set_xlabel("TSVD mode")
    _ax.set_ylabel("Gram eigenvalue")
    _ax.grid(linestyle="--", alpha=0.3)
    _ax.legend(title="window end")
    _fig
    return


@app.cell
def _(
    analysis_mask,
    compute_spatial_modes,
    computed_stl_residuals,
    koopman_lag,
    koopman_window,
    windowed_gram_tsvd,
):
    koopman_spatial_modes_rank10 = compute_spatial_modes(
        computed_stl_residuals,
        analysis_mask,
        windowed_gram_tsvd,
        window=koopman_window,
        lag=koopman_lag,
        rank=10,
        eigenvalue_index=0,
        window_indices=(0, -1),
    )
    koopman_spatial_mode_initial = koopman_spatial_modes_rank10.isel(window=0)
    koopman_spatial_mode_final = koopman_spatial_modes_rank10.isel(window=-1)
    return koopman_spatial_mode_final, koopman_spatial_mode_initial


@app.cell
def _(
    BoundaryNorm,
    ListedColormap,
    ccrs,
    cfeature,
    koopman_spatial_mode_initial,
    np,
    plt,
):
    _mode_real = -koopman_spatial_mode_initial.real
    _abs_limit = float(np.nanmax(np.abs(_mode_real.to_numpy())))
    _boundaries = np.array(
        [
            -_abs_limit,
            -0.65 * _abs_limit,
            -0.45 * _abs_limit,
            -0.25 * _abs_limit,
            -0.08 * _abs_limit,
            0.08 * _abs_limit,
            0.25 * _abs_limit,
            0.45 * _abs_limit,
            0.65 * _abs_limit,
            _abs_limit,
        ]
    )
    _cmap = ListedColormap(
        [
            "#67000d",
            "#a50f15",
            "#cb181d",
            "#ef3b2c",
            "#bdbdbd",
            "#9ecae1",
            "#6baed6",
            "#3182bd",
            "#08519c",
        ]
    )
    _norm = BoundaryNorm(_boundaries, _cmap.N, clip=True)

    _fig, _ax = plt.subplots(
        figsize=(7, 5),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    _image = _mode_real.plot(
        ax=_ax,
        x="lon",
        y="lat",
        transform=ccrs.PlateCarree(),
        cmap=_cmap,
        norm=_norm,
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
    _ax.set_title("Leading Koopman mode, initial time")
    _fig.colorbar(
        _image,
        ax=_ax,
        orientation="horizontal",
        label="",
        boundaries=_boundaries,
        ticks=_boundaries,
        shrink=0.85,
        pad=0.08,
        format="%.3f",
    )
    _fig
    return


@app.cell
def _(
    BoundaryNorm,
    ListedColormap,
    ccrs,
    cfeature,
    koopman_spatial_mode_final,
    np,
    plt,
):
    _mode_real = -koopman_spatial_mode_final.real
    _abs_limit = float(np.nanmax(np.abs(_mode_real.to_numpy())))
    _boundaries = np.array(
        [
            -_abs_limit,
            -0.65 * _abs_limit,
            -0.45 * _abs_limit,
            -0.25 * _abs_limit,
            -0.08 * _abs_limit,
            0.08 * _abs_limit,
            0.25 * _abs_limit,
            0.45 * _abs_limit,
            0.65 * _abs_limit,
            _abs_limit,
        ]
    )
    _cmap = ListedColormap(
        [
            "#67000d",
            "#a50f15",
            "#cb181d",
            "#ef3b2c",
            "#bdbdbd",
            "#9ecae1",
            "#6baed6",
            "#3182bd",
            "#08519c",
        ]
    )
    _norm = BoundaryNorm(_boundaries, _cmap.N, clip=True)

    _fig, _ax = plt.subplots(
        figsize=(7, 5),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    _image = _mode_real.plot(
        ax=_ax,
        x="lon",
        y="lat",
        transform=ccrs.PlateCarree(),
        cmap=_cmap,
        norm=_norm,
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
    _ax.set_title("Leading Koopman mode, final window")
    _fig.colorbar(
        _image,
        ax=_ax,
        orientation="horizontal",
        boundaries=_boundaries,
        ticks=_boundaries,
        shrink=0.85,
        pad=0.08,
        format="%.3f",
    )
    _fig
    return


@app.cell
def _(
    compute_subdominant_eigenvalue_trajectory,
    koopman_truncations,
    windowed_gram_tsvd,
):
    koopman_subdominant_eigenvalues = compute_subdominant_eigenvalue_trajectory(
        windowed_gram_tsvd,
        truncations=koopman_truncations,
    )
    return (koopman_subdominant_eigenvalues,)


@app.cell
def _(koopman_subdominant_eigenvalues, np, plt, xr):
    _continuous_time_subdominant = xr.apply_ufunc(
        np.log,
        koopman_subdominant_eigenvalues["subdominant_eigenvalue"],
    )
    _continuous_time_next = xr.apply_ufunc(
        np.log,
        koopman_subdominant_eigenvalues["next_eigenvalue"],
    )
    _subdominant_decay = _continuous_time_subdominant.real
    _next_decay = _continuous_time_next.real
    _subdominant_timescale = (-1 / _subdominant_decay).where(_subdominant_decay < 0)
    _next_timescale = (-1 / _next_decay).where(_next_decay < 0)
    _rank = "10"

    _fig, _ax = plt.subplots(figsize=(9, 8),nrows=2,sharex=True)

    _subdominant_decay.sel(truncation=_rank).plot(
        ax=_ax[0],
        x="time",
        color="tab:blue",
        label="leading",
    ) #type:ignore
    _next_decay.sel(truncation=_rank).plot(
        ax=_ax[0],
        x="time",
        color="tab:green",
        label="second",
    ) #type:ignore

    _subdominant_timescale.sel(truncation=_rank).plot(
        ax=_ax[1],
        x="time",
        color="tab:blue",
        label="leading",
    ) #type:ignore
    _next_timescale.sel(truncation=_rank).plot(
        ax=_ax[1],
        x="time",
        color="tab:green",
        label="second",
    ) #type:ignore
    _ax[0].set_title("")
    _ax[1].set_title("")
    _ax[0].set_ylabel(r"$\lambda$",size=12)
    _ax[0].grid(linestyle="--", alpha=0.3)
    _ax[0].legend(
    )
    _ax[1].set_xlabel("decimal year")
    _ax[1].set_ylabel("Koopman timescale [months]",size=12)
    _ax[1].grid(linestyle="--", alpha=0.3)
    _ax[1].legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Robustness Section
    """)
    return


@app.cell
def _(
    compute_subdominant_eigenvalue_trajectory,
    koopman_robustness_truncations,
    windowed_gram_tsvd,
):
    koopman_truncation_robustness = compute_subdominant_eigenvalue_trajectory(
        windowed_gram_tsvd,
        truncations=koopman_robustness_truncations,
    )
    return (koopman_truncation_robustness,)


@app.cell
def _(koopman_truncation_robustness, np, plt, xr):
    _continuous_time_leading = xr.apply_ufunc(
        np.log,
        koopman_truncation_robustness["subdominant_eigenvalue"],
    )
    _leading_timescale = (-1 / _continuous_time_leading.real).where(
        _continuous_time_leading.real < 0
    )
    _colors = plt.get_cmap("viridis")(
        np.linspace(0.12, 0.88, _leading_timescale.sizes["truncation"])
    )

    _fig, _ax = plt.subplots(nrows=2, figsize=(9, 7), sharex=True)
    for _color, _rank in zip(_colors, _leading_timescale["truncation"].to_numpy()):
        _axis = _ax[0] if int(_rank) <= 15 else _ax[1]
        _leading_timescale.sel(truncation=_rank).plot(
            ax=_axis,
            x="time",
            color=_color,
            label=f"rank {_rank}",
        ) #type:ignore

    _ax[0].set_title("Truncation robustness")
    _ax[1].set_title("")
    _ax[0].set_ylabel("Koopman timescale [months]")
    _ax[1].set_xlabel(" time [year]")
    _ax[1].set_ylabel("Koopman timescale [months]")
    for _axis in _ax:
        _axis.grid(linestyle="--", alpha=0.5)
        _axis.legend()

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Window-Length Robustness
    """)
    return


@app.cell
def _(
    analysis_mask,
    compute_subdominant_eigenvalue_trajectory,
    compute_windowed_gram_tsvd,
    computed_stl_residuals,
    koopman_lag,
    koopman_sigma,
    koopman_step,
    xr,
):
    _window_length_years = (5, 6, 7, 8, 9, 10)
    _rank = 10
    _trajectories = []

    for _years in _window_length_years:
        _windowed_gram_tsvd = compute_windowed_gram_tsvd(
            computed_stl_residuals,
            analysis_mask,
            window=_years * 12,
            step=koopman_step,
            lag=koopman_lag,
            sigma=koopman_sigma,
            store_matrices=True,
        )
        _trajectory = compute_subdominant_eigenvalue_trajectory(
            _windowed_gram_tsvd,
            truncations=(_rank,),
        )
        _leading_eigenvalue = (
            _trajectory["subdominant_eigenvalue"]
            .sel(truncation=str(_rank), drop=True)
            .assign_coords(window_years=_years)
            .expand_dims("window_years")
        )
        _trajectories.append(_leading_eigenvalue)

    koopman_window_length_robustness = xr.concat(
        _trajectories,
        dim="window_years",
        join="outer",
    ).to_dataset(name="leading_eigenvalue")
    return (koopman_window_length_robustness,)


@app.cell
def _(koopman_window_length_robustness, np, plt, xr):
    _continuous_time_leading = xr.apply_ufunc(
        np.log,
        koopman_window_length_robustness["leading_eigenvalue"],
    )
    _leading_timescale = (-1 / _continuous_time_leading.real).where(
        _continuous_time_leading.real < 0
    )
    _colors = plt.get_cmap("viridis")(
        np.linspace(0.12, 0.88, _leading_timescale.sizes["window_years"])
    )[::-1]

    _fig, _ax = plt.subplots(figsize=(9, 4))
    for _color, _years in zip(_colors, _leading_timescale["window_years"].to_numpy()):
        _leading_timescale.sel(window_years=_years).plot(
            ax=_ax,
            x="time",
            color=_color,
            label=f"{int(_years)} years",
        ) #type:ignore

    _ax.set_title("Window-length robustness")
    _ax.set_xlabel("time [year]")
    _ax.set_ylabel("Koopman timescale [months]")
    _ax.grid(linestyle="--", alpha=0.5)
    _ax.legend(title="Window")

    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
