# Amazon Rainforest Resilience — Koopman Spectral Analysis

Koopman operator analysis (Kernel DMD) of Amazon rainforest vegetation dynamics,
extending Boulton et al. (2022) *Nature Climate Change*.

## What this project does

Boulton et al. showed that >76% of the Amazon has been losing dynamical
resilience since the early 2000s, using pointwise lag-1 autocorrelation on
satellite vegetation data. This project replaces the scalar, grid-cell-level
indicator with a spatially coherent Koopman spectral decomposition of the full
vegetation field, extracting global dynamical modes, their timescales, and their
spatial structure.

## Data

- **Primary:** VODCA v2 CXKu (Zotta et al., 2024) — vegetation optical depth,
  0.25°, daily, 1987–2021. [Access](https://doi.org/10.48436/t74ty-tcx62)
- **Ancillary:** MODIS land cover (MCD12C1), CHIRPS precipitation, Amazon
  basin shapefile.

## Method

1. Preprocess VOD to monthly residuals (STL decomposition) on Amazon forest cells.
2. Full-series Kernel DMD to extract baseline Koopman modes and timescales.
3. Sliding-window Kernel DMD to track eigenvalue migration toward the unit circle
   (critical slowing down).

## References

- Boulton, C.A., Lenton, T.M. & Boers, N. (2022). Pronounced loss of Amazon
  rainforest resilience since the early 2000s. *Nat. Clim. Change* 12, 271–278.
- Zotta, R.-M. et al. (2024). VODCA v2. *Earth Syst. Sci. Data* 16, 4573–4617.

## For AI coding agents

See `AGENTS.md` for full project context, dataset specifications, analysis
plan, and implementation constraints.
