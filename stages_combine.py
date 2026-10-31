#!/usr/bin/env python3
# =====================================================================
# stages_combine.py - maps (stage 1) + ENSO/SSRD analysis (stage 3) in one run
# ---------------------------------------------------------------------
# Inputs
#   BASE/ssrd-ssrdc.nc                       ERA5 monthly SSRD (variable 'ssrd')
#   BASE/stage2_outputs/regional_monthly_{ssrd,ssrdc,cloud_radiative_effect,tisr}.csv
#   BASE/stage2_outputs/regional_monthly_ssrd_enso.csv   (column ONI)
#   optional  BASE/regions.json                  {"R1": ["name", latS, latN, lonW, lonE], ...}
#   optional  BASE/extra/tcc-tcwv.nc (ERA5 monthly means of tcc and tcwv in one file; regional means are computed here;
#             alternatively era5_tcc.nc and era5_tcwv.nc)
#             or BASE/extra/regional_monthly_{tcc,tcwv,aod_cams}.csv, climate_indices.csv (DMI, NAO),
#             era5_lsm.nc (variable lsm)
# Outputs  BASE/combined_outputs/   (BASE/combined_outputs_nodetrend when DETREND=0)
#   Fig1_domain_trend   (a) mean SSRD + regions   (b) trend per decade (no regions)
#   Fig2_overview       SSRD anomalies/ONI, lag correlation, seasonal correlation
#   Fig3_attribution    cloud vs clear-sky
#   Fig4_robustness     jackknife, peak lag CI, sub-periods, effect size
#   FigS1_CRE_SSRDc_lag, FigS2_trend_seasonal (4 maps, one shared colour bar),
#   FigS3_trend_relative; table_regions.tex, table_summary.tex; CSV tables
# Environment switches: RUN_MAPS=0/1, RUN_ANALYSIS=0/1, DETREND=0/1, N_BOOT, N_SURR
#
# Significance of the regional SSRD-ONI link (used for the tables and for the grey bars in
# Fig. 3d / open symbols in Fig. 4a): phase-randomised surrogates, peak searched in the a-priori
# ENSO-leading window 0..+6 months, Benjamini-Hochberg FDR over the ten regions (q = 0.05).
# The full-range (-18..+18) peak lag is reported separately (Fig. 2b, summary CSV).
# key_numbers.txt collects the numbers needed to fill the \tbd{} items of the manuscript.
#
# REGION IDS: stage-2 columns are matched to the map boxes by comparing monthly means
# (match_regions). If the IDs in stage 2 are a permutation of the IDs on the map, the
# stage-2 series are renamed to the map IDs automatically and region_id_map.csv records
# the mapping. If no box reproduces a stage-2 series the analysis stops (set FORCE=1 to
# override) instead of silently mislabelling regions.
# =====================================================================
import os
import json
import zlib
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = os.environ.get("BASE", r"C:/Users/Think/Desktop/new-data")
FILE = os.path.join(BASE, "ssrd-ssrdc.nc")
STAGE2 = os.path.join(BASE, "stage2_outputs")
EXTRA = os.path.join(BASE, "extra")
RUN_MAPS = os.environ.get("RUN_MAPS", "1") == "1"
RUN_ANALYSIS = os.environ.get("RUN_ANALYSIS", "1") == "1"
DETREND = os.environ.get("DETREND", "1") == "1"
OUT = os.path.join(BASE, "combined_outputs" + ("" if DETREND else "_nodetrend"))

Y0, Y1 = 1996, 2025
J2KWH = 1.0 / 3.6e6
FDR_Q = 0.10                                  # grid-point trend FDR (Wilks 2016)
QUARTERS = {"JFM": (1, 2, 3), "AMJ": (4, 5, 6), "JAS": (7, 8, 9), "OND": (10, 11, 12)}
MAX_LAG = 18
LAGS = np.arange(-MAX_LAG, MAX_LAG + 1)
LEAD = (0, 6)
LEAD_IDX = np.where((LAGS >= LEAD[0]) & (LAGS <= LEAD[1]))[0]
ALPHA = 0.05
N_BOOT = int(os.environ.get("N_BOOT", 1000))
N_SURR = int(os.environ.get("N_SURR", 1000))
BLOCK = 12
rng = np.random.default_rng(42)


def reseed(tag):
    """deterministic random stream per (analysis, variable, region): bootstrap / surrogate results
    do not depend on the order in which regions are processed or on how regions are numbered"""
    global rng
    rng = np.random.default_rng([42, zlib.crc32(tag.encode())])

# name, latS, latN, lonW, lonE  (map layout of stage1.py)
REGIONS = {
    "R1": ("Western Yemen", 12, 17, 40, 45), "R2": ("Oman", 16, 24, 53, 58),
    "R3": ("Egypt", 22, 30, 25, 35), "R4": ("Southeastern Iran", 25, 29, 56, 64),
    "R5": ("Northern Saudi Arabia", 26, 32, 40, 48), "R6": ("Southern Syria", 32, 38, 36, 43),
    "R7": ("Zagros Mountains", 34, 37, 45, 49), "R8": ("Northern Iran", 36, 39, 49, 56),
    "R9": ("Northern Turkey", 38, 43, 27, 41), "R10": ("Caucasus", 39, 44, 46, 49)}
SHORT = {"R1": "W. Yemen", "R2": "Oman", "R3": "Egypt", "R4": "SE Iran", "R5": "N. Saudi",
         "R6": "S. Syria", "R7": "Zagros", "R8": "N. Iran", "R9": "N. Turkey", "R10": "Caucasus"}
_rj = os.path.join(BASE, "regions.json")
if os.path.exists(_rj):
    REGIONS = {k: tuple(v) for k, v in json.load(open(_rj)).items()}
    SHORT = {k: v[0] for k, v in REGIONS.items()}
RK = list(REGIONS)
LAB = {k: f"{k} {SHORT[k]}" for k in RK}
KEY = [k for k in ("R1", "R6", "R7", "R10") if k in RK]
VARS = ["SSRD", "SSRDc", "CRE"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5,
                     "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.5, "figure.dpi": 150, "axes.linewidth": 0.6})


def save(fig, name):
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{e}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def month_start(df):
    """put every monthly index on the first day of the month so that files from different
    stages (month-start / month-end / mid-month stamps) align exactly"""
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index).to_period("M").to_timestamp()
    return df


# =====================================================================
# PART A - data, trends, maps
# =====================================================================
def load_era5():
    """-> times, lat (ascending), lon, data[t, lat, lon] in kWh m-2 d-1"""
    if os.environ.get("SYNTH"):                      # synthetic data to test the code
        r = np.random.default_rng(0)
        t = pd.date_range(f"{Y0}-01-01", f"{Y1}-12-01", freq="MS")
        lat, lon = np.arange(10, 45.01, 0.5), np.arange(25, 70.01, 0.5)
        seas = np.sin(2 * np.pi * (t.month.values - 3) / 12)[:, None, None]
        base = 5.0 + 2.0 * np.cos(np.deg2rad(lat))[None, :, None]
        tr = (np.arange(len(t)) / len(t))[:, None, None] * (0.15 * np.sin(np.deg2rad(lon * 4))[None, None, :])
        return t, lat, lon, base + 2 * seas + tr + r.normal(0, 0.15, (len(t), len(lat), len(lon)))
    import xarray as xr
    ds = xr.open_dataset(FILE)
    tdim = "valid_time" if "valid_time" in ds.coords else "time"
    ydim = "latitude" if "latitude" in ds.coords else "lat"
    xdim = "longitude" if "longitude" in ds.coords else "lon"
    name = next((n for n in ("ssrd", "SSRD", "surface_solar_radiation_downwards") if n in ds.data_vars), None)
    if name is None:
        raise SystemExit("SSRD not found. Available: " + str(list(ds.data_vars)))
    if float(ds[xdim].max()) > 180:
        ds = ds.assign_coords({xdim: (((ds[xdim] + 180) % 360) - 180)}).sortby(xdim)
    ds = ds.sortby(ydim)
    da = ds[name] * J2KWH
    yr = da[tdim].dt.year
    da = da.isel({tdim: ((yr >= Y0) & (yr <= Y1)).values})
    for extra in [d for d in da.dims if d not in (tdim, ydim, xdim)]:      # e.g. 'expver'
        da = da.mean(extra)
    da = da.transpose(tdim, ydim, xdim)
    return (pd.DatetimeIndex(da[tdim].values), da[ydim].values, da[xdim].values,
            da.values.astype(np.float64))


def load_nc_regional(path, names, times):
    """regional monthly series (map IDs, cos-lat weighted boxes) from an ERA5 monthly NetCDF (e.g. tcc, tcwv)"""
    import xarray as xr
    ds = xr.open_dataset(path)
    tdim = "valid_time" if "valid_time" in ds.coords else "time"
    ydim = "latitude" if "latitude" in ds.coords else "lat"
    xdim = "longitude" if "longitude" in ds.coords else "lon"
    name = next((n for n in names if n in ds.data_vars), None)
    if name is None:
        raise SystemExit(f"{path}: none of {names} found. Available: {list(ds.data_vars)}")
    if float(ds[xdim].max()) > 180:
        ds = ds.assign_coords({xdim: (((ds[xdim] + 180) % 360) - 180)}).sortby(xdim)
    ds = ds.sortby(ydim)
    da = ds[name]
    yr = da[tdim].dt.year
    da = da.isel({tdim: ((yr >= Y0) & (yr <= Y1)).values})
    for extra in [d for d in da.dims if d not in (tdim, ydim, xdim)]:
        da = da.mean(extra)
    da = da.transpose(tdim, ydim, xdim)
    t = pd.DatetimeIndex(da[tdim].values).to_period("M").to_timestamp()
    v, la, lo = da.values.astype(np.float64), da[ydim].values, da[xdim].values
    out = pd.DataFrame({k: box_series(v, la, lo, b) for k, b in REGIONS.items()}, index=t)
    return out.reindex(times.to_period("M").to_timestamp())


def load_land(lat, lon):
    p = os.path.join(EXTRA, "era5_lsm.nc")
    if os.environ.get("SYNTH") or not os.path.exists(p):
        return None
    import xarray as xr
    l = xr.open_dataset(p)["lsm"]
    l = l.isel({d: 0 for d in l.dims if d not in ("latitude", "longitude")})
    if float(l.longitude.max()) > 180:
        l = l.assign_coords(longitude=(((l.longitude + 180) % 360) - 180)).sortby("longitude")
    return l.sortby("latitude").reindex(latitude=lat, longitude=lon, method="nearest").values > 0.5


def trend_fields(A, yrs):
    """Theil-Sen slope per year and AR(1)-adjusted Mann-Kendall p for A[year, lat, lon]."""
    n, ny, nx = A.shape
    iu = np.triu_indices(n, 1)
    dt = (yrs[iu[1]] - yrs[iu[0]]).astype(float)
    slope, S, rho = np.empty((ny, nx)), np.empty((ny, nx)), np.empty((ny, nx))
    t = yrs - yrs.mean()
    for j in range(ny):
        row = A[:, j, :]
        d = row[iu[1], :] - row[iu[0], :]
        slope[j] = np.median(d / dt[:, None], axis=0)
        S[j] = np.sign(d).sum(axis=0)
        b = (t[:, None] * (row - row.mean(0))).sum(0) / (t ** 2).sum()
        res = row - row.mean(0) - b * t[:, None]
        a, c = res[:-1], res[1:]
        den = np.sqrt((a ** 2).sum(0) * (c ** 2).sum(0))
        rho[j] = np.where(den > 0, (a * c).sum(0) / den, 0.0)
    infl = np.where(rho > 0, (1 + rho) / (1 - np.minimum(rho, 0.9)), 1.0)
    z = (S - np.sign(S)) / np.sqrt(n * (n - 1) * (2 * n + 5) / 18.0 * infl)
    return slope, 2 * stats.norm.sf(np.abs(z))


def bh(p, q):
    flat = np.asarray(p, float).ravel()
    ok = np.isfinite(flat)
    out = np.zeros(flat.shape, bool)
    pv = flat[ok]
    if pv.size == 0:
        return out.reshape(np.shape(p))
    o = np.argsort(pv)
    passed = pv[o] <= q * np.arange(1, pv.size + 1) / pv.size
    if passed.any():
        out[ok] = pv <= pv[o][np.max(np.where(passed))]
    return out.reshape(np.shape(p))


def box_mask(lat, lon, b):
    _, s, n, w, e = b
    return (lat[:, None] >= s) & (lat[:, None] <= n) & (lon[None, :] >= w) & (lon[None, :] <= e)


def box_series(data, lat, lon, b, weighted=True, land=None):
    m = box_mask(lat, lon, b)
    if land is not None:
        m = m & land
    w = (np.cos(np.deg2rad(lat))[:, None] * np.ones(len(lon))[None, :]) if weighted else np.ones(m.shape)
    w = w * m
    return (data * w[None]).sum((1, 2)) / w.sum()


def annual_from(D, times, months=None):
    yr, mo = times.year.values, times.month.values
    ys, out = [], []
    for y in range(Y0, Y1 + 1):
        sel = (yr == y) & (np.isin(mo, months) if months else True)
        if sel.sum() == (len(months) if months else 12):
            ys.append(y)
            out.append(D[sel].mean(0))
    return np.array(ys, float), np.stack(out)


def run_maps(times, lat, lon, D, land):
    DOM = (lon.min(), lon.max(), lat.min(), lat.max())
    yrs, A = annual_from(D, times)
    mean_map = A.mean(0)
    slope, p = trend_fields(A, yrs)
    sig = bh(p, FDR_Q)
    dec = slope * 10
    rel = 100 * dec / mean_map
    print(f"[maps] years {int(yrs[0])}-{int(yrs[-1])}; mean {mean_map.min():.2f}-{mean_map.max():.2f}; "
          f"trend/decade {dec.min():+.3f}..{dec.max():+.3f}; unadjusted p<0.05: {100 * (p < 0.05).mean():.0f}% "
          f"of cells, FDR(q={FDR_Q}): {100 * sig.mean():.0f}%")
    np.save(os.path.join(OUT, "mean_ssrd.npy"), mean_map)
    np.save(os.path.join(OUT, "trend_per_decade.npy"), dec)
    np.save(os.path.join(OUT, "trend_fdr_sig.npy"), sig)

    # region statistics: area-weighted box means of annual SSRD; Mann-Kendall p; FDR across regions
    rows = []
    for k, b in REGIONS.items():
        ser = np.array([box_series(A[i:i + 1], lat, lon, b)[0] for i in range(len(yrs))])
        sl, pp = trend_fields(ser[:, None, None], yrs)
        m = box_mask(lat, lon, b)
        row = dict(region=k, name=b[0], latS=b[1], latN=b[2], lonW=b[3], lonE=b[4], mean_ssrd=ser.mean(),
                   trend_per_decade=sl[0, 0] * 10, trend_pct_per_decade=100 * sl[0, 0] * 10 / ser.mean(),
                   p_MK=pp[0, 0], n_cells=int(m.sum()))
        if land is not None:
            row["land_fraction"] = float((land & m).sum() / m.sum())
        rows.append(row)
    fdr_reg = bh(np.array([r["p_MK"] for r in rows]), ALPHA)
    tex = []
    for row, f_ in zip(rows, fdr_reg):
        row["p_MK_fdr_sig"] = bool(f_)
        pf = "$<$0.001" if row["p_MK"] < 0.001 else f"{row['p_MK']:.3f}"
        tex.append(f"{row['region']} & {row['name']} & {row['latS']}--{row['latN']}\\textdegree N & "
                   f"{row['lonW']}--{row['lonE']}\\textdegree E & {row['mean_ssrd']:.2f} & "
                   f"${row['trend_pct_per_decade']:+.1f}$ ({pf}){'$^{*}$' if f_ else ''} \\\\")
    pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "region_trends.csv"), index=False)
    with open(os.path.join(OUT, "table_regions.tex"), "w") as f:
        f.write("\\begin{tabular}{llccrc}\n\\topline\nID & Region & Latitude & Longitude & Mean SSRD & "
                "Trend (\\% decade$^{-1}$) ($p$)\\\\\n\\midline\n" + "\n".join(tex) + "\n\\botline\n\\end{tabular}\n")
    print(f"[maps] regional trends: p<0.05 in {sum(r['p_MK'] < 0.05 for r in rows)} of {len(rows)} regions; "
          f"{int(fdr_reg.sum())} after FDR across regions (q={ALPHA})")

    def add_geo(ax):
        try:
            from mpl_toolkits.basemap import Basemap
            m = Basemap(projection="cyl", llcrnrlon=DOM[0], llcrnrlat=DOM[2], urcrnrlon=DOM[1],
                        urcrnrlat=DOM[3], resolution="i", ax=ax)
            m.drawcoastlines(linewidth=0.4)
            m.drawcountries(linewidth=0.4)
        except Exception:
            try:
                import cartopy.feature as cf
                ax.add_feature(cf.COASTLINE, lw=0.4)
                ax.add_feature(cf.BORDERS, lw=0.4)
            except Exception:
                print("   (Basemap/cartopy unavailable: coastlines omitted)")

    def boxes(ax):
        for k, (nm, s, n, w, e) in REGIONS.items():
            ax.add_patch(Rectangle((w, s), e - w, n - s, fill=False, ec="k", lw=0.8, zorder=8))
            cx, cy = (w + e) / 2, (s + n) / 2
            ax.text(cx, cy + 0.45, k, ha="center", va="center", fontsize=6.5, fontweight="bold",
                    zorder=9, bbox=dict(fc="w", ec="none", alpha=0.6, pad=0.4))
            ax.text(cx, cy - 0.85, SHORT[k], ha="center", va="center", fontsize=4.8, style="italic",
                    zorder=9, bbox=dict(fc="w", ec="none", alpha=0.55, pad=0.25))

    def finish(ax, labels=True):
        ax.set_xlim(DOM[0], DOM[1])
        ax.set_ylim(DOM[2], DOM[3])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="gray", lw=0.3, ls=":")
        if labels:
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")

    def draw_mean(ax):
        lv = np.arange(np.floor(mean_map.min() * 4) / 4, mean_map.max() + 0.26, 0.25)
        cf = ax.contourf(lon, lat, mean_map, levels=lv, cmap="YlOrRd")
        add_geo(ax)
        boxes(ax)                                   # regions only on the mean map
        finish(ax)
        return cf

    def draw_trend(ax, field, sigmask, vm, step=3, labels=True):
        cf = ax.contourf(lon, lat, field, levels=np.linspace(-vm, vm, 21), cmap="RdBu_r", extend="both")
        yy, xx = np.where(sigmask[::step, ::step])
        ax.scatter(lon[::step][xx], lat[::step][yy], s=1.2, c="k", lw=0, alpha=0.7, zorder=5)
        add_geo(ax)
        finish(ax, labels)
        return cf

    VM = max(0.02, np.ceil(np.nanpercentile(np.abs(dec), 99) * 50) / 50)
    TL = "SSRD trend (kWh m$^{-2}$ d$^{-1}$ decade$^{-1}$)"
    ML = "Mean SSRD (kWh m$^{-2}$ d$^{-1}$)"
    yl = f"{int(yrs[0])}–{int(yrs[-1])}"

    fig, axs = plt.subplots(2, 1, figsize=(6.5, 8.6))
    cf = draw_mean(axs[0])
    fig.colorbar(cf, ax=axs[0], shrink=0.85, pad=0.02).set_label(ML)
    axs[0].set_title(f"(a) Mean SSRD {yl} and study regions", loc="left", fontweight="bold")
    cf = draw_trend(axs[1], dec, sig, VM)
    fig.colorbar(cf, ax=axs[1], shrink=0.85, pad=0.02, extend="both").set_label(TL)
    axs[1].set_title("(b) SSRD trend (Theil–Sen); dots: Mann–Kendall, FDR $q=0.10$", loc="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "Fig1_domain_trend")

    for nm, fn in (("Fig1a_mean_map", lambda f, a: (draw_mean(a), ML)),
                   ("Fig1b_trend_map", lambda f, a: (draw_trend(a, dec, sig, VM), TL))):
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        cf, lab = fn(fig, ax)
        fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02).set_label(lab)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"{nm}.pdf"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # quarterly trends: four large panels, ONE shared horizontal colour bar, no region boxes
    QN = {"JFM": "Jan–Mar, winter", "AMJ": "Apr–Jun, spring", "JAS": "Jul–Sep, summer", "OND": "Oct–Dec, autumn"}
    qdat = {q: annual_from(D, times, mo) for q, mo in QUARTERS.items()}
    qtr = {q: trend_fields(a, y) for q, (y, a) in qdat.items()}
    vq = max(0.02, np.ceil(max(np.nanpercentile(np.abs(s * 10), 99) for s, _ in qtr.values()) * 50) / 50)
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.6), sharex=True, sharey=True, layout="constrained")
    for i, (ax, q) in enumerate(zip(axs.ravel(), QUARTERS)):
        sq, pq = qtr[q]
        cf = draw_trend(ax, sq * 10, bh(pq, FDR_Q), vq, step=4, labels=False)
        ax.set_title(f"({'abcd'[i]}) {q} ({QN[q]})", loc="left", fontweight="bold")
        ax.set_xticks([30, 40, 50, 60])
        ax.set_yticks([10, 20, 30, 40])
        if i % 2 == 0:
            ax.set_ylabel("Latitude (°N)")
        if i >= 2:
            ax.set_xlabel("Longitude (°E)")
    cb = fig.colorbar(cf, ax=axs, location="bottom", shrink=0.6, aspect=40, pad=0.02, extend="both")
    cb.set_label("SSRD trend (kWh m$^{-2}$ d$^{-1}$ decade$^{-1}$); dots: Mann–Kendall, FDR $q=0.10$")
    save(fig, "FigS2_trend_seasonal")

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    vr = max(0.5, np.ceil(np.nanpercentile(np.abs(rel), 99) * 2) / 2)
    cf = draw_trend(ax, rel, sig, vr)
    fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02, extend="both").set_label("SSRD trend (% decade$^{-1}$)")
    ax.set_title("Annual SSRD trend relative to the mean", loc="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "FigS3_trend_relative")
    return pd.DataFrame(rows)


# =====================================================================
# PART B - match stage-2 region IDs to the map boxes
# =====================================================================
def match_regions(s2, mine_w, mine_u):
    """difference matrix (rows: map boxes, cols: stage-2 columns), best box per column, ok flag"""
    cols = [c for c in s2.columns if c in mine_w.columns]
    M = pd.DataFrame({c: {k: min(np.nanmean(np.abs(s2[c] - mine_w[k])), np.nanmean(np.abs(s2[c] - mine_u[k])))
                          for k in mine_w.columns} for c in cols})
    best = M.idxmin()
    ok = bool((M.min() < 0.05).all() and best.nunique() == len(best))
    return M, best.to_dict(), ok


# =====================================================================
# PART C - ENSO / SSRD analysis (former stage3_v3_analysis.py)
# =====================================================================
def run_analysis(ssrd, ssrdc, cre, tisr, oni_raw, extras, pr_regions):
    maybe_detrend = None

    def detrend(s):
        v = s.values.astype(float)
        x = np.arange(len(v), dtype=float)
        m = np.isfinite(v)
        sl, ic, *_ = stats.linregress(x[m], v[m])
        return pd.Series(v - (sl * x + ic), index=s.index)

    maybe_detrend = detrend if DETREND else (lambda s: s)
    deseason = lambda s: s - s.groupby(s.index.month).transform("mean")
    clim = lambda s: s.groupby(s.index.month).transform("mean")
    anom = lambda s: maybe_detrend(deseason(s.astype(float)))
    idx = ssrd.index
    oni_a = anom(oni_raw)
    A = {v: {k: anom(df[k]) for k in RK} for v, df in (("SSRD", ssrd), ("SSRDc", ssrdc), ("CRE", cre))}
    ac1 = lambda a: np.corrcoef(a[:-1], a[1:])[0, 1]

    def corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        d = np.sqrt((a @ a) * (b @ b))
        return (a @ b) / d if d > 0 else np.nan

    def rvec(x, y):
        n = len(x)
        out = np.full(len(LAGS), np.nan)
        for i, k in enumerate(LAGS):
            a, b = (x[:n - k], y[k:]) if k >= 0 else (x[-k:], y[:n + k])
            if len(a) > 24:
                out[i] = corr(a, b)
        return out

    def lagged(x, y):
        rs, ps = np.full(len(LAGS), np.nan), np.full(len(LAGS), np.nan)
        for i, k in enumerate(LAGS):
            p = pd.concat([x.shift(k), y], axis=1).dropna()
            if len(p) < 24:
                continue
            a, b = p.iloc[:, 0].values, p.iloc[:, 1].values
            r = np.corrcoef(a, b)[0, 1]
            if not np.isfinite(r):
                continue
            den = 1 + ac1(a) * ac1(b)
            neff = len(p) * (1 - ac1(a) * ac1(b)) / den if den > 0 else len(p)
            neff = float(np.clip(neff, 5, len(p)))
            t = r * np.sqrt((neff - 2) / (1 - min(abs(r), 0.999999) ** 2))
            rs[i], ps[i] = r, 2 * stats.t.sf(abs(t), neff - 2)
        return rs, ps

    def block_idx(n):
        nb = int(np.ceil(n / BLOCK))
        st = rng.integers(0, n - BLOCK + 1, nb)
        return (st[:, None] + np.arange(BLOCK)).ravel()[:n]

    def phase_surrogates(x, n):
        x = x - x.mean()
        N = len(x)
        F = np.fft.rfft(x)
        out = np.empty((n, N))
        for i in range(n):
            ph = rng.uniform(0, 2 * np.pi, len(F))
            ph[0] = 0
            if N % 2 == 0:
                ph[-1] = 0
            out[i] = np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=N)
        return out

    def peak_tools(x, y, tag=""):
        reseed("peak|" + tag)
        d = pd.concat([x, y], axis=1).dropna()
        xv, yv = d.iloc[:, 0].values, d.iloc[:, 1].values
        r = rvec(xv, yv)
        kf = int(np.nanargmax(np.abs(r)))
        kl = int(LEAD_IDX[np.nanargmax(np.abs(r[LEAD_IDX]))])
        of, ol = abs(r[kf]), abs(r[kl])
        cf = cl = 0
        for s in phase_surrogates(xv, N_SURR):
            rs = np.abs(rvec(s, yv))
            cf += np.nanmax(rs) >= of
            cl += np.nanmax(rs[LEAD_IDX]) >= ol
        sf, sl = np.sign(r[kf]), np.sign(r[kl])
        bf, bl = [], []
        n = len(xv)
        for _ in range(N_BOOT):
            ii = block_idx(n)
            rb = rvec(xv[ii], yv[ii])
            bf.append(LAGS[np.nanargmax(rb * sf)])
            bl.append(LAGS[LEAD_IDX][np.nanargmax((rb * sl)[LEAD_IDX])])
        return dict(lag_full=int(LAGS[kf]), r_full=r[kf], p_surr_full=(1 + cf) / (1 + N_SURR),
                    ci_full=np.percentile(bf, [2.5, 97.5]), lag_lead=int(LAGS[kl]), r_lead=r[kl],
                    p_surr_lead=(1 + cl) / (1 + N_SURR), ci_lead=np.percentile(bl, [2.5, 97.5]))

    def r_at_lag(x, y, k, mask=None):
        d = pd.concat([x.shift(k), y], axis=1)
        if mask is not None:
            d = d[~mask]
        d = d.dropna()
        return np.corrcoef(d.iloc[:, 0], d.iloc[:, 1])[0, 1] if len(d) > 24 else np.nan

    # ---- 1 lag correlations
    print("[analysis 1] lagged correlations ...")
    RES = {}
    for v in VARS:
        for r in RK:
            rs, ps = lagged(oni_a, A[v][r])
            RES[(v, r)] = dict(rs=rs, ps=ps, sig=bh(ps, ALPHA))
        pd.DataFrame({r: RES[(v, r)]["rs"] for r in RK}, index=LAGS).T.to_csv(
            os.path.join(OUT, f"lag_matrix_r_{v}.csv"))
    # ---- 2 peaks / surrogates / bootstrap
    print("[analysis 2] peaks, surrogate tests, bootstrap ...")
    PK = {(v, r): peak_tools(oni_a, A[v][r], f"{v}|{r}") for v in ("SSRD", "CRE", "SSRDc") for r in RK}
    # regional significance: surrogate p of the 0..+6 month peak, BH-FDR over the ten regions
    SIG = dict(zip(RK, bh(np.array([PK[("SSRD", r)]["p_surr_lead"] for r in RK]), ALPHA)))
    print("   significant regions (lead window, FDR):", [r for r in RK if SIG[r]])
    # ---- 3 seasonal
    print("[analysis 3] seasonal ...")
    SEAS_LAB = ["O", "N", "D", "J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    SEAS_T = [(-1, m) for m in (10, 11, 12)] + [(0, m) for m in range(1, 13)]
    years = list(range(idx.year.min() + 1, idx.year.max() + 1))
    OD = {(d.year, d.month): v for d, v in oni_raw.items()}
    ndj = {y: np.nanmean([OD.get((y - 1, 11), np.nan), OD.get((y - 1, 12), np.nan), OD.get((y, 1), np.nan)])
           for y in years}

    def seasonal(Y):
        Yd = {(d.year, d.month): v for d, v in Y.items()}
        R, P, S = np.full(15, np.nan), np.full(15, np.nan), np.full(15, np.nan)
        for j, (off, m) in enumerate(SEAS_T):
            pr = [(ndj[y], Yd.get((y + off, m), np.nan)) for y in years]
            pr = [(a, b) for a, b in pr if np.isfinite(a) and np.isfinite(b)]
            if len(pr) > 12:
                lr = stats.linregress([a for a, _ in pr], [b for _, b in pr])
                R[j], P[j], S[j] = lr.rvalue, lr.pvalue, lr.slope
        return R, P, S

    SEAS = {}
    for v in VARS:
        for r in RK:
            SEAS[(v, r)] = list(seasonal(A[v][r]))
        sg = bh(np.array([SEAS[(v, r)][1] for r in RK]), ALPHA)
        for i, r in enumerate(RK):
            SEAS[(v, r)].append(sg[i])
    pd.DataFrame([dict(var=v, region=r, year_offset=SEAS_T[j][0], month=SEAS_T[j][1], r=R[j], p=P[j],
                       slope=S[j], fdr_sig=bool(sg[j]))
                  for (v, r), (R, P, S, sg) in SEAS.items() for j in range(15)]).to_csv(
        os.path.join(OUT, "seasonal_ONI_NDJ.csv"), index=False)
    # leave-one-year-out (and DMI control) for the FDR-significant seasonal cells (SSRD).
    # DMI = Indian Ocean Dipole index averaged over Sep-Nov of the preceding year; the autumn cells
    # are the ones in which an ENSO and an IOD signal cannot be separated without this control.
    CI0 = extras.get("indices")
    dmi_son = {}
    if CI0 is not None and "DMI" in CI0:
        Dd = {(d.year, d.month): v for d, v in CI0["DMI"].items()}
        dmi_son = {y: np.nanmean([Dd.get((y - 1, m_), np.nan) for m_ in (9, 10, 11)]) for y in years}
    loo = []
    for r in RK:
        R, P, S, sg = SEAS[("SSRD", r)]
        Yd = {(d.year, d.month): v for d, v in A["SSRD"][r].items()}
        for j in np.where(sg)[0]:
            off, m = SEAS_T[j]
            pr = [(y, ndj[y], Yd.get((y + off, m), np.nan)) for y in years]
            pr = [t for t in pr if np.isfinite(t[1]) and np.isfinite(t[2])]
            rr = [np.corrcoef([a for yy, a, _ in pr if yy != y], [b for yy, _, b in pr if yy != y])[0, 1]
                  for y, _, _ in pr]
            row = dict(region=r, year_offset=off, month=m, r=R[j], loo_min=min(rr), loo_max=max(rr),
                       most_influential_year=pr[int(np.argmax(np.abs(np.array(rr) - R[j])))][0])
            if dmi_son:
                tri = [(a, b, dmi_son[yy]) for yy, a, b in pr if np.isfinite(dmi_son.get(yy, np.nan))]
                if len(tri) > 12:
                    xo, yo, zo = (np.array(c) for c in zip(*tri))
                    Zm = np.column_stack([np.ones(len(zo)), zo])
                    resid = lambda v, Zm=Zm: v - Zm @ np.linalg.lstsq(Zm, v, rcond=None)[0]
                    row["r_ONI_vs_DMI_SON"] = np.corrcoef(xo, zo)[0, 1]
                    row["r_partial_given_DMI_SON"] = corr(resid(xo), resid(yo))
            loo.append(row)
    pd.DataFrame(loo).to_csv(os.path.join(OUT, "seasonal_leave_one_year_out.csv"), index=False)
    # ---- 4 robustness
    print("[analysis 4] robustness ...")
    ey = pd.Series(idx.year + (idx.month >= 7), index=idx)
    first = pd.Series(idx < pd.Timestamp("2011-01-01"), index=idx)
    CI = extras.get("indices")
    ROB = {}
    for r in RK:
        L, y = PK[("SSRD", r)]["lag_lead"], A["SSRD"][r]
        r_all = r_at_lag(oni_a, y, L)
        jk = {int(e): r_at_lag(oni_a, y, L, mask=(ey == e)) for e in ey.unique()}
        jk = {e: v for e, v in jk.items() if np.isfinite(v)}
        row = dict(lag=L, r_all=r_all, jk_min=min(jk.values()), jk_max=max(jk.values()),
                   jk_most_influential_year=max(jk, key=lambda e: abs(jk[e] - r_all)),
                   r_first_half=r_at_lag(oni_a, y, L, mask=~first),
                   r_second_half=r_at_lag(oni_a, y, L, mask=first))
        if CI is not None:
            cols = [c for c in ("DMI", "NAO") if c in CI]
            ctrl = pd.concat([CI[c].reindex(idx).shift(L) for c in cols], axis=1)
            d = pd.concat([oni_a.shift(L).rename("x"), y.rename("y"), ctrl], axis=1).dropna()
            Z = np.column_stack([np.ones(len(d))] + [d[c].values for c in ctrl.columns])
            res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
            row["r_partial_DMI_NAO"] = corr(res(d["x"].values), res(d["y"].values))
        ROB[r] = row
    pd.DataFrame(ROB).T.to_csv(os.path.join(OUT, "robustness.csv"))
    # ---- 5 decomposition
    print("[analysis 5] decomposition ...")
    oni6 = oni_a.rolling(7).mean()
    slope = lambda x, y: np.cov(x, y, bias=True)[0, 1] / np.var(x)

    def decompose(r, x, tag=""):
        reseed(f"dec|{tag}|{r}")
        clear = maybe_detrend((1 - clim(cre[r])) * deseason(ssrdc[r]))
        cloud = maybe_detrend(-clim(ssrdc[r]) * deseason(cre[r]))
        d = pd.concat([x.rename("x"), A["SSRD"][r].rename("t"), clear.rename("a"), cloud.rename("c")],
                      axis=1).dropna()
        X, T, Ca, Cc = (d[c].values for c in "xtac")
        st, sa, sc = slope(X, T), slope(X, Ca), slope(X, Cc)
        bs = []
        for _ in range(N_BOOT):
            ii = block_idx(len(X))
            t_ = slope(X[ii], T[ii])
            bs.append(slope(X[ii], Cc[ii]) / t_ if abs(t_) > 1e-9 else np.nan)
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        return dict(slope_SSRD=st, slope_clear=sa, slope_cloud=sc, residual=st - sa - sc, cloud_share=sc / st,
                    cloud_share_lo=lo, cloud_share_hi=hi, clear_share=sa / st,
                    pct_per_unit=100 * st / ssrd[r].mean())

    DEC = {"lag": {r: decompose(r, oni_a.shift(PK[("SSRD", r)]["lag_lead"]), "lag") for r in RK},
           "oni6": {r: decompose(r, oni6, "oni6") for r in RK}}
    for k in DEC:
        pd.DataFrame(DEC[k]).T.round(4).to_csv(os.path.join(OUT, f"decomposition_{k}.csv"))
    # ---- 6 optional checks
    chk = []
    for r in RK:
        L = PK[("SSRD", r)]["lag_lead"]
        row = {"region": r, "lag": L, "tisr_anom_std_pct_of_mean": 100 * anom(tisr[r]).std() / tisr[r].mean()}
        for nm, key in (("TCC", "tcc"), ("TCWV", "tcwv"), ("AOD", "aod")):
            df = extras.get(key)
            if df is not None and r in df:
                a = anom(df[r].reindex(idx))
                row[f"r_{nm}_ONI_at_lag"] = r_at_lag(oni_a, a, L)
                row[f"r_{nm}_vs_{'CRE' if nm == 'TCC' else 'SSRDc'}"] = a.corr(A["CRE" if nm == "TCC" else "SSRDc"][r])
        chk.append(row)
    pd.DataFrame(chk).round(3).to_csv(os.path.join(OUT, "optional_checks.csv"), index=False)
    # ---- 7 tables
    fp = lambda p: "$<$0.001" if p <= 0.001 else f"{p:.3f}"
    fci = lambda c: f"[{c[0]:.0f},{c[1]:.0f}]"
    S_rows, T_rows = [], []
    for r in RK:
        a, c, s, d = PK[("SSRD", r)], PK[("CRE", r)], PK[("SSRDc", r)], DEC["oni6"][r]
        sg = bool(SIG[r])
        S_rows.append(dict(region=r, label=SHORT[r], SSRD_sig_fdr_lead=sg,
                           SSRD_lag_full=a["lag_full"], SSRD_r_full=a["r_full"],
                           SSRD_lag_full_CI=fci(a["ci_full"]), SSRD_p_surr_full=a["p_surr_full"],
                           SSRD_lag_lead=a["lag_lead"], SSRD_r_lead=a["r_lead"], SSRD_lag_lead_CI=fci(a["ci_lead"]),
                           SSRD_p_surr_lead=a["p_surr_lead"], CRE_r_lead=c["r_lead"], CRE_lag_lead=c["lag_lead"],
                           SSRDc_r_lead=s["r_lead"], SSRDc_lag_lead=s["lag_lead"],
                           **{f"dec_{k}": v for k, v in d.items()}))
        share = (f"{100 * d['cloud_share']:.0f} [{100 * d['cloud_share_lo']:.0f},{100 * d['cloud_share_hi']:.0f}]"
                 if sg else "n.s.")
        T_rows.append(f"{LAB[r]} & {a['lag_lead']:+d} {fci(a['ci_lead'])} & ${a['r_lead']:+.2f}$ & "
                      f"{fp(a['p_surr_lead'])}{'$^{*}$' if sg else ''} & ${c['r_lead']:+.2f}$ & "
                      f"${s['r_lead']:+.2f}$ & {share} & ${d['pct_per_unit']:+.1f}$ \\\\")
    pd.DataFrame(S_rows).to_csv(os.path.join(OUT, "summary_v3.csv"), index=False)
    with open(os.path.join(OUT, "table_summary.tex"), "w") as f:
        f.write("\\begin{tabular}{lccccccc}\n\\topline\nRegion & Peak lag [95\\% CI] & $r_{\\mathrm{SSRD}}$ & "
                "$p_{\\mathrm{surr}}$ & $r_{\\mathrm{CRE}}$ & $r_{\\mathrm{SSRDc}}$ & "
                "Cloud share (\\%) & SSRD (\\% per ONI)\\\\\n\\midline\n" + "\n".join(T_rows)
                + "\n\\botline\n\\end{tabular}\n")
    print(pd.DataFrame(S_rows)[["region", "label", "SSRD_sig_fdr_lead", "SSRD_lag_lead", "SSRD_lag_lead_CI",
                                "SSRD_r_lead", "SSRD_p_surr_lead", "SSRD_lag_full", "SSRD_lag_full_CI",
                                "dec_cloud_share", "dec_pct_per_unit"]].round(3).to_string(index=False))

    # numbers for the manuscript
    sigr = [r for r in RK if SIG[r]]
    nsr = [r for r in RK if not SIG[r]]
    rng_ = lambda v, f="{:.2f}": f.format(min(v)) + " to " + f.format(max(v))
    kn = [f"significant regions (lead window 0..+6, surrogate p, BH-FDR 0.05, {len(sigr)} of {len(RK)}): "
          + ", ".join(LAB[r] for r in sigr),
          "not significant: " + (", ".join(LAB[r] for r in nsr) or "none")]
    if sigr:
        kn += [f"peak r (0..+6) among significant: {rng_([PK[('SSRD', r)]['r_lead'] for r in sigr])}",
               f"SSRD change per +1 ONI, % of mean: {rng_([DEC['oni6'][r]['pct_per_unit'] for r in sigr], '{:.1f}')}",
               f"SSRD change per +1 ONI, kWh m-2 d-1: {rng_([DEC['oni6'][r]['slope_SSRD'] for r in sigr], '{:.3f}')}",
               f"cloud share %: {rng_([100 * DEC['oni6'][r]['cloud_share'] for r in sigr], '{:.0f}')}; "
               f"lowest in {LAB[min(sigr, key=lambda r: DEC['oni6'][r]['cloud_share'])]}",
               f"clear-sky share %: {rng_([100 * DEC['oni6'][r]['clear_share'] for r in sigr], '{:.0f}')}"]
    kn += [f"max |second-order residual| / |SSRD response| (significant regions): "
           f"{100 * max(abs(DEC['oni6'][r]['residual'] / DEC['oni6'][r]['slope_SSRD']) for r in (sigr or RK)):.2f} %",
           f"std of TISR anomaly, % of mean: {rng_([x['tisr_anom_std_pct_of_mean'] for x in chk], '{:.2f}')}",
           "", "full-range peak lag [95% CI] (all regions):"]
    kn += [f"  {LAB[r]}: {PK[('SSRD', r)]['lag_full']:+d} {fci(PK[('SSRD', r)]['ci_full'])}, "
           f"r={PK[('SSRD', r)]['r_full']:+.2f}, p_surr={PK[('SSRD', r)]['p_surr_full']:.3f}" for r in RK]
    nseas = sum(int(np.nansum(SEAS[('SSRD', r)][3])) for r in RK)
    kn += ["", f"FDR-significant seasonal cells (SSRD): {nseas}"]
    kn += [f"  {LAB[r]}: " + ", ".join(f"{SEAS_LAB[j]}({SEAS_T[j][0]:+d}) r={SEAS[('SSRD', r)][0][j]:+.2f}"
                                         for j in np.where(SEAS[('SSRD', r)][3])[0])
           for r in RK if SEAS[('SSRD', r)][3].any()]
    open(os.path.join(OUT, "key_numbers.txt"), "w", encoding="utf-8").write("\n".join(kn) + "\n")
    print("\n".join(kn))

    # ---- 8 figures
    print("[analysis 8] figures ...")

    def asym(p):
        return [[max(p["lag_full"] - p["ci_full"][0], 0)], [max(p["ci_full"][1] - p["lag_full"], 0)]]

    def heat(ax, M, xs, sigmask=None, vmax=0.45):
        im = ax.pcolormesh(xs, np.arange(len(RK)), M, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           shading="nearest", rasterized=True)
        if sigmask is not None:
            yy, xx = np.where(sigmask)
            ax.scatter(np.asarray(xs)[xx], yy, s=2.2, c="k", marker=".", zorder=5, lw=0)
        ax.set_ylim(len(RK) - 0.5, -0.5)
        ax.set_yticks(range(len(RK)))
        ax.set_yticklabels([LAB[r] for r in RK])
        return im

    cbar = lambda fig, im, cax: (fig.colorbar(im, cax=cax).set_label("Pearson $r$"), cax.tick_params(length=2))
    cols = dict(zip(KEY, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]))

    fig = plt.figure(figsize=(6.5, 8.0))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1, 0.025], hspace=0.42, wspace=0.04, height_ratios=[0.9, 1.15, 1.15])
    ax = fig.add_subplot(gs[0, 0])
    for r in KEY:
        ax.plot(idx, A["SSRD"][r].rolling(3, center=True).mean(), color=cols[r], lw=0.7, label=LAB[r])
    ax.set_ylim(-0.7, 0.7)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_ylabel("SSRD anomaly (kWh m$^{-2}$ d$^{-1}$)")
    ax.legend(ncol=4, frameon=False, loc="upper left")
    a2 = ax.twinx()
    a2.fill_between(idx, 0, oni_raw, where=oni_raw >= 0.5, color="#C44E52", alpha=0.25, lw=0)
    a2.fill_between(idx, 0, oni_raw, where=oni_raw <= -0.5, color="#4C72B0", alpha=0.25, lw=0)
    a2.set_ylim(-3.5, 3.5)
    a2.set_ylabel("ONI ($^\\circ$C)")
    ax.set_title("(a) Regional SSRD anomalies (3-month mean) and ONI", loc="left", fontweight="bold")
    ax = fig.add_subplot(gs[1, 0])
    im = heat(ax, np.array([RES[("SSRD", r)]["rs"] for r in RK]), LAGS, np.array([RES[("SSRD", r)]["sig"] for r in RK]))
    for i, r in enumerate(RK):
        p = PK[("SSRD", r)]
        ax.errorbar(p["lag_full"], i, xerr=asym(p), fmt="D", ms=2.8, mfc="w", mec="k", color="k", lw=0.7,
                    capsize=1.5, zorder=6)
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.axvspan(LEAD[0] - 0.5, LEAD[1] + 0.5, color="gray", alpha=0.10, zorder=0)
    ax.set_xlabel("Lag (months); positive = ONI leads")
    ax.set_title("(b) Lagged correlation SSRD–ONI (dots: FDR-significant; diamonds: peak lag, 95% bootstrap CI)",
                 loc="left", fontweight="bold", fontsize=6.8)
    cbar(fig, im, fig.add_subplot(gs[1, 1]))
    ax = fig.add_subplot(gs[2, 0])
    im = heat(ax, np.array([SEAS[("SSRD", r)][0] for r in RK]), np.arange(15),
              np.array([SEAS[("SSRD", r)][3] for r in RK]))
    ax.set_xticks(np.arange(15))
    ax.set_xticklabels(SEAS_LAB)
    ax.axvline(2.5, color="k", lw=0.8)
    ax.set_xlabel("Target month of SSRD   (left of line: year before ENSO peak; right: year of peak)")
    ax.set_title("(c) ONI (Nov–Jan mean) vs SSRD in individual months (dots: FDR-significant)", loc="left",
                 fontweight="bold", fontsize=6.8)
    cbar(fig, im, fig.add_subplot(gs[2, 1]))
    save(fig, "Fig2_overview")

    star = [r for r in ("R7", "R1") if r in RK] or RK[:2]
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 5.6))
    for ax, r, lab in [(axs[0, 0], star[0], "(a)"), (axs[0, 1], star[-1], "(b)")]:
        for v, c, ls in [("SSRD", "#1f77b4", "-"), ("SSRDc", "#2ca02c", "-"), ("CRE", "#ff7f0e", "--")]:
            ax.plot(LAGS, RES[(v, r)]["rs"], color=c, ls=ls, lw=1.3,
                    label={"SSRD": "SSRD", "SSRDc": "SSRDc (clear-sky)", "CRE": "CRE$_{rel}$ (cloud)"}[v])
        ax.axhline(0, color="k", lw=0.4)
        ax.axvline(0, color="gray", lw=0.4, ls=":")
        ax.axvspan(LEAD[0] - 0.5, LEAD[1] + 0.5, color="gray", alpha=0.10, zorder=0)
        ax.set_xlabel("Lag (months); positive = ONI leads")
        ax.set_ylabel("Pearson $r$")
        ax.set_title(f"{lab} {LAB[r]}", loc="left", fontweight="bold")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
    xp = np.arange(len(RK))
    sigreg = np.array([SIG[r] for r in RK])
    D6 = DEC["oni6"]
    ax = axs[1, 0]
    cd, cl_, tt = (np.array([D6[r][k] for r in RK]) for k in ("slope_cloud", "slope_clear", "slope_SSRD"))
    ax.bar(xp, cd, 0.6, color="#ff7f0e", label="cloud term")
    ax.bar(xp, cl_, 0.6, bottom=cd, color="#8c564b", label="clear-sky term")
    ax.plot(xp, tt, "k_", ms=7, mew=1.2, label="total SSRD")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xticks(xp)
    ax.set_xticklabels(RK)
    ax.set_ylabel("SSRD change per +1 ONI\n(kWh m$^{-2}$ d$^{-1}$)")
    lo_ = min(np.min(cd), np.min(cd + cl_), np.min(tt), 0) * 1.12      # stacked bars: both ends count
    hi_ = max(np.max(cd), np.max(cd + cl_), np.max(tt), 0)
    ax.set_ylim(lo_, max(abs(lo_) * 0.2, hi_ * 1.2 + abs(lo_) * 0.12))
    ax.set_title("(c) Linearised decomposition", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper left", columnspacing=0.8, handlelength=1.2)
    ax = axs[1, 1]
    sh = np.array([D6[r]["cloud_share"] for r in RK])
    lo = np.array([D6[r]["cloud_share_lo"] for r in RK])
    hi = np.array([D6[r]["cloud_share_hi"] for r in RK])
    ax.bar(xp, np.where(sigreg, sh, np.nan), 0.6, color="#ff7f0e",
           yerr=[np.where(sigreg, np.clip(sh - lo, 0, None), 0), np.where(sigreg, np.clip(hi - sh, 0, None), 0)],
           ecolor="k", capsize=1.5, error_kw={"lw": 0.6})
    for i_ in np.where(~sigreg)[0]:        # cloud share of an unresolved response is undefined
        ax.text(i_, 0.04, "n.s.", ha="center", va="bottom", fontsize=6, color="0.35", rotation=90)
    ax.axhline(0.5, color="k", lw=0.6, ls="--")
    ax.set_xticks(xp)
    ax.set_xticklabels(RK)
    ax.set_ylabel("Cloud share of SSRD response")
    ax.set_ylim(0, 1.6)
    ax.set_title("(d) Cloud share, 95% CI (n.s.: SSRD–ONI link not significant)", loc="left", fontweight="bold", fontsize=6.5)
    fig.tight_layout()
    save(fig, "Fig3_attribution")

    fig, axs = plt.subplots(2, 2, figsize=(6.5, 5.6))
    ax = axs[0, 0]
    for i, r in enumerate(RK):
        p, rb = PK[("SSRD", r)], ROB[r]
        ax.plot([i, i], [rb["jk_min"], rb["jk_max"]], color="gray", lw=2, alpha=0.6)
        ax.plot(i, p["r_lead"], "o", ms=4, mfc="k" if SIG[RK[i]] else "w", mec="k")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xticks(range(len(RK)))
    ax.set_xticklabels(RK)
    ax.set_ylabel("$r$ at lead-window peak lag")
    ax.set_title("(a) Peak $r$, 0–6 mo\n(filled: FDR-significant; bar: leave-one-year-out)", loc="left",
                 fontweight="bold", fontsize=6.8)
    ax = axs[0, 1]
    for i, r in enumerate(RK):
        p = PK[("SSRD", r)]
        ax.errorbar(i, p["lag_full"], yerr=asym(p), fmt="o", color="k", ms=3.5, lw=0.7, capsize=1.5,
                    mfc="k" if SIG[RK[i]] else "w")
    ax.axhspan(LEAD[0], LEAD[1], color="gray", alpha=0.12)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xticks(range(len(RK)))
    ax.set_xticklabels(RK)
    ax.set_ylabel("Peak lag (months)")
    ax.set_title("(b) Peak lag, full range, 95% bootstrap CI\n(open: SSRD–ONI link not significant)", loc="left",
                 fontweight="bold", fontsize=6.8)
    ax = axs[1, 0]
    w = 0.38
    ax.bar(xp - w / 2, [ROB[r]["r_first_half"] for r in RK], w, color="#9ecae1", label="1996–2010")
    ax.bar(xp + w / 2, [ROB[r]["r_second_half"] for r in RK], w, color="#3182bd", label="2011–2025")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xticks(xp)
    ax.set_xticklabels(RK)
    ax.set_ylabel("$r$ at lead-window peak lag")
    vals_ = np.array([ROB[r][k] for r in RK for k in ("r_first_half", "r_second_half")])
    ax.set_ylim(min(vals_.min(), 0) * 1.35, max(vals_.max(), 0) * 1.15 + 0.01)
    ax.legend(frameon=False, ncol=2, loc="lower left")
    ax.set_title("(c) Sub-period stability", loc="left", fontweight="bold")
    ax = axs[1, 1]
    ax.bar(xp, [D6[r]["pct_per_unit"] for r in RK], 0.6, color="#4C72B0")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xticks(xp)
    ax.set_xticklabels(RK)
    ax.set_ylabel("SSRD change (% of mean per +1 ONI)")
    ax.set_title("(d) Effect size", loc="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "Fig4_robustness")

    fig = plt.figure(figsize=(6.5, 5.6))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 0.025], hspace=0.4, wspace=0.04)
    for row, (v, lab) in enumerate([("CRE", "(a) Relative cloud attenuation CRE$_{rel}$"), ("SSRDc", "(b) Clear-sky SSRDc")]):
        ax = fig.add_subplot(gs[row, 0])
        im = heat(ax, np.array([RES[(v, r)]["rs"] for r in RK]), LAGS, np.array([RES[(v, r)]["sig"] for r in RK]))
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("Lag (months); positive = ONI leads")
        ax.set_title(lab + " vs ONI", loc="left", fontweight="bold")
        cbar(fig, im, fig.add_subplot(gs[row, 1]))
    save(fig, "FigS1_CRE_SSRDc_lag")


# =====================================================================
def main():
    os.makedirs(OUT, exist_ok=True)
    times, lat, lon, D = load_era5()
    land = load_land(lat, lon)
    if RUN_MAPS:
        run_maps(times, lat, lon, D, land)
    if not RUN_ANALYSIS:
        return
    rd = lambda f: month_start(pd.read_csv(os.path.join(STAGE2, f), index_col=0, parse_dates=True))
    ssrd, ssrdc = rd("regional_monthly_ssrd.csv"), rd("regional_monthly_ssrdc.csv")
    cre, tisr = rd("regional_monthly_cloud_radiative_effect.csv"), rd("regional_monthly_tisr.csv")
    enso = rd("regional_monthly_ssrd_enso.csv")
    if np.nanmax(cre.values) > 1.5:
        raise SystemExit("CRE looks like percent, not fraction: divide by 100 first.")
    # ---- are stage-2 IDs the same as the map IDs?
    tm = times.to_period("M").to_timestamp()
    mine_w = pd.DataFrame({k: box_series(D, lat, lon, b) for k, b in REGIONS.items()}, index=tm)
    mine_u = pd.DataFrame({k: box_series(D, lat, lon, b, weighted=False) for k, b in REGIONS.items()}, index=tm)
    M, best, ok = match_regions(ssrd, mine_w, mine_u)
    M.round(4).to_csv(os.path.join(OUT, "region_label_check.csv"))
    pd.Series(best, name="map_id").rename_axis("stage2_column").to_csv(os.path.join(OUT, "region_id_map.csv"))
    print("[match] stage-2 column -> map box:", best, "| ok" if ok else "| NOT OK")
    if not ok:
        print(M.round(3).to_string())
        if os.environ.get("FORCE") != "1":
            raise SystemExit("No unique box reproduces the stage-2 series (diff >= 0.05 kWh m-2 d-1). Put the "
                             "stage-2 boxes into regions.json (or set FORCE=1 to continue with stage-2 IDs).")
        best = {c: c for c in ssrd.columns}
    elif any(c != k for c, k in best.items()):
        print("   stage-2 IDs differ from map IDs: series are renamed to the map IDs.")
    ren = lambda df: df.rename(columns=best)[RK]
    ssrd, ssrdc, cre, tisr = ren(ssrd), ren(ssrdc), ren(cre), ren(tisr)
    idx = ssrd.index
    extras = {}
    for key, f in (("tcc", "regional_monthly_tcc.csv"), ("tcwv", "regional_monthly_tcwv.csv"),
                   ("aod", "regional_monthly_aod_cams.csv")):
        p = os.path.join(EXTRA, f)
        # raw ERA5 NetCDF: one combined file (tcc-tcwv.nc) or one file per variable (era5_<key>.nc)
        pn = next((q for q in (os.path.join(EXTRA, "tcc-tcwv.nc"), os.path.join(EXTRA, f"era5_{key}.nc"))
                   if os.path.exists(q) and key in ("tcc", "tcwv")), None)
        if os.path.exists(p):
            extras[key] = month_start(pd.read_csv(p, index_col=0, parse_dates=True)).rename(columns=best)
        elif pn:
            extras[key] = month_start(load_nc_regional(pn, (key, key.upper()), times))
            print(f"[extra] {key}: regional means computed from {pn}")
    p = os.path.join(EXTRA, "climate_indices.csv")
    if os.path.exists(p):
        extras["indices"] = month_start(pd.read_csv(p, index_col=0, parse_dates=True))
    run_analysis(ssrd, ssrdc, cre, tisr, enso.reindex(idx)["ONI"].astype(float), extras, REGIONS)
    print("Done ->", OUT)


if __name__ == "__main__":
    main()
