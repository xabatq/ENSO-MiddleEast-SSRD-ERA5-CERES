# Code and derived data: ENSO imprint on surface solar radiation over the Middle East (ERA5, with a CERES EBAF check)

Authors: Khabat Ghamari, Asrin Seyedzahedi. Manuscript submitted to Atmospheric Research.

## Contents
- `stages_combine.py` - ERA5 analysis: regional means, lagged correlation, phase-randomised surrogates, block bootstrap, seasonal analysis, cloud/clear-sky decomposition, robustness, maps.
- `ceres_enso.py` - ERA5 vs CERES EBAF validation and the same ENSO analysis for 2001-2025, including gridded regression maps.
- `ceres.py` - initial ERA5-CERES comparison (monthly metrics, 2001-2015).
- `regions.json` (if used) - region boxes R1-R10.
- `data/` - derived monthly series: `regional_monthly_ssrd_enso.csv` (ONI and regional SSRD), `regional_monthly_ssrdc.csv`, `regional_monthly_cloud_radiative_effect.csv`, `regional_monthly_tisr.csv`, `climate_indices.csv` (DMI, NAO), `regional_monthly_tcc.csv`, `regional_monthly_tcwv.csv`.
- `results/` - result tables used in the paper (`summary_v3.csv`, `seasonal_ONI_NDJ.csv`, `seasonal_leave_one_year_out.csv`, `robustness.csv`, `optional_checks.csv`, `enso_summary_ERA5_vs_CERES.csv`, `validation_metrics_ERA5_vs_CERES.csv`, `seasonal_ONI_NDJ_ERA5_CERES.csv`, `key_numbers*.txt`).

## Input data (not redistributed; public)
- ERA5 monthly means (ssrd, ssrdc, tisr, tcc, tcwv), Copernicus Climate Data Store, 1996-2025.
- CERES EBAF Edition 4.2.1 (DOI 10.5067/TERRA-AQUA-NOAA20/CERES/EBAF_L3B004.2.1), NASA LaRC ASDC.
- ONI and NAO: NOAA Climate Prediction Center. DMI (HadISST1.1): NOAA PSL.

## Usage
Set `BASE` to the data folder (default `C:/Users/Think/Desktop/new-data`) with `ssrd-ssrdc.nc`, `CERES_200003-202512.nc`, `stage2_outputs/` and `extra/`.

    python stages_combine.py        # maps + ERA5 analysis -> BASE/combined_outputs
    python ceres_enso.py            # CERES check        -> BASE/ceres_outputs

Environment variables: `DETREND` (1/0), `N_BOOT`, `N_SURR` (default 1000), `SYNTH=1` (synthetic test data). Random seeds are fixed per variable and region, so results are reproducible.

## Requirements
Python 3.10+, numpy, pandas, scipy, matplotlib, xarray, netCDF4; optional basemap or cartopy for coastlines.

## License
Code: MIT. Derived data: CC BY 4.0.
