#!/usr/bin/env python3
# =====================================================================
# STAGE 1 (new map) - mean SSRD, SSRD trend, region statistics
# ---------------------------------------------------------------------
# Replaces stage1.py. Input: BASE/ssrd-ssrdc.nc (variable 'ssrd').
# Outputs (BASE/stage1_outputs_newmap/):
#   Fig1_domain_trend.pdf/png     (a) mean SSRD + regions, (b) trend per decade + regions
#   Fig1a_mean_map.png, Fig1b_trend_map.png   the two panels separately
#   FigS2_trend_seasonal.pdf/png  trend per decade for JFM, AMJ, JAS, OND
#   FigS3_trend_relative.pdf/png  annual trend in % per decade
#   table_regions.tex, region_trends.csv      box mean SSRD and trend (auto-generated)
#   region_label_check.csv        compares boxes below with stage2 regional_monthly_ssrd.csv
#
# What was wrong in stage1.py and is changed here
#  1. Significance: unadjusted p<0.05 at thousands of correlated grid points, OLS t-test
#     ignoring autocorrelation. Now: Theil-Sen slope + Mann-Kendall (AR(1)-adjusted) and
#     Benjamini-Hochberg FDR across grid points (q=0.10, Wilks 2016).
#  2. Stars were drawn wherever ANY of the 9 neighbouring cells was significant
#     (np.any over a 3x3 window), overstating the significant area. Now each marker is a
#     truly significant grid point (thinned for legibility, never extended).
#  3. 'N-year change' = slope x 30 overstates the change between first and last year (29
#     intervals) and is not a standard unit. Now: change per decade.
#  4. Colour levels from min/max (3.424, 0.1690 ...) -> fixed round levels; sequential
#     colormap for the mean, symmetric diverging for trends.
#  5. Region boxes: one dictionary, checked against the stage-2 monthly series.
#  6. Extra: relative trend (% per decade) and quarterly trends.
#
# !! REGION LAYOUT: the boxes below are the "new layout" of stage1.py. The stage-3 results
#    (summary_v3.csv) use different names for R3-R6, R8, R9. Run this script, open
#    region_label_check.csv, and if the diagonal is not ~0 put the boxes actually used in
#    stage 2 into BASE/regions.json  {"R1": ["name", latS, latN, lonW, lonE], ...}
# =====================================================================
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE = os.environ.get("BASE", r"C:/Users/Think/Desktop/new-data")
FILE = os.path.join(BASE, "ssrd-ssrdc.nc")
STAGE2 = os.path.join(BASE, "stage2_outputs")
LSM_FILE = os.path.join(BASE, "extra", "era5_lsm.nc")
OUT = os.path.join(BASE, "stage1_outputs_newmap")
os.makedirs(OUT, exist_ok=True)

Y0, Y1 = 1996, 2025
J2KWH = 1.0 / 3.6e6
FDR_Q = 0.10
QUARTERS = {"JFM": (1, 2, 3), "AMJ": (4, 5, 6), "JAS": (7, 8, 9), "OND": (10, 11, 12)}

# name, latS, latN, lonW, lonE  -- "new layout" copied from stage1.py
REGIONS = {
    "R1": ("Western Yemen", 12, 17, 40, 45), "R2": ("Oman", 16, 24, 53, 58),
    "R3": ("Egypt", 22, 30, 25, 35), "R4": ("Southeastern Iran", 25, 29, 56, 64),
    "R5": ("Northern Saudi Arabia", 26, 32, 40, 48), "R6": ("Southern Syria", 32, 38, 36, 43),
    "R7": ("Zagros Mountains", 34, 37, 45, 49), "R8": ("Northern Iran", 36, 39, 49, 56),
    "R9": ("Northern Turkey", 38, 43, 27, 41), "R10": ("Caucasus", 39, 44, 46, 49)}
rj = os.path.join(BASE, "regions.json")
if os.path.exists(rj):
    REGIONS = {k: tuple(v) for k, v in json.load(open(rj)).items()}
    print("Regions read from regions.json")


# ------------------------------------------------------------------ data
def load_era5():
    """-> times, lat (ascending), lon, data[t, lat, lon] in kWh m-2 d-1"""
    if os.environ.get("SYNTH"):                      # synthetic data, for testing the code
        rng = np.random.default_rng(0)
        t = pd.date_range(f"{Y0}-01-01", f"{Y1}-12-01", freq="MS")
        lat, lon = np.arange(10, 45.01, 0.5), np.arange(25, 70.01, 0.5)
        seas = np.sin(2 * np.pi * (t.month.values - 3) / 12)[:, None, None]
        base = 5.0 + 2.0 * np.cos(np.deg2rad(lat))[None, :, None]
        tr = (np.arange(len(t)) / len(t))[:, None, None] * \
            (0.15 * np.sin(np.deg2rad(lon * 4))[None, None, :])
        return t, lat, lon, base + 2 * seas + tr + rng.normal(0, 0.15, (len(t), len(lat), len(lon)))
    import xarray as xr
    ds = xr.open_dataset(FILE)
    tdim = "valid_time" if "valid_time" in ds.coords else "time"
    ydim = "latitude" if "latitude" in ds.coords else "lat"
    xdim = "longitude" if "longitude" in ds.coords else "lon"
    name = next((n for n in ("ssrd", "SSRD", "surface_solar_radiation_downwards")
                 if n in ds.data_vars), None)
    if name is None:
        raise SystemExit("SSRD not found. Available: " + str(list(ds.data_vars)))
    if float(ds[xdim].max()) > 180:
        ds = ds.assign_coords({xdim: (((ds[xdim] + 180) % 360) - 180)}).sortby(xdim)
    ds = ds.sortby(ydim)
    da = ds[name] * J2KWH
    yr = da[tdim].dt.year
    da = da.isel({tdim: ((yr >= Y0) & (yr <= Y1)).values})
    for extra in [d for d in da.dims if d not in (tdim, ydim, xdim)]:   # e.g. 'expver'
        da = da.mean(extra)
    da = da.transpose(tdim, ydim, xdim)
    return (pd.DatetimeIndex(da[tdim].values), da[ydim].values, da[xdim].values,
            da.values.astype(np.float64))


def load_land(lat, lon):
    if os.environ.get("SYNTH") or not os.path.exists(LSM_FILE):
        return None
    import xarray as xr
    l = xr.open_dataset(LSM_FILE)["lsm"]
    l = l.isel({d: 0 for d in l.dims if d not in ("latitude", "longitude")})
    if float(l.longitude.max()) > 180:
        l = l.assign_coords(longitude=(((l.longitude + 180) % 360) - 180)).sortby("longitude")
    return l.sortby("latitude").reindex(latitude=lat, longitude=lon, method="nearest").values > 0.5


# ------------------------------------------------------------------ trends
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


def ols_slope(A, yrs):
    t = yrs - yrs.mean()
    return (t[:, None, None] * (A - A.mean(0))).sum(0) / (t ** 2).sum()


def bh(p, q):
    flat = np.asarray(p, float).ravel()
    ok = np.isfinite(flat)
    out = np.zeros(flat.shape, bool)
    pv = flat[ok]
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


# -------------------------------------------------------------------- main
times, lat, lon, D = load_era5()
land = load_land(lat, lon)
DOM = (lon.min(), lon.max(), lat.min(), lat.max())
yrs, A = annual_from(D, times)
if len(yrs) < Y1 - Y0 + 1:
    print("WARNING: incomplete years dropped; years used:", yrs.astype(int))
mean_map = A.mean(0)
slope, p = trend_fields(A, yrs)
sig = bh(p, FDR_Q)
dec = slope * 10
rel = 100 * dec / mean_map
ols = ols_slope(A, yrs) * 10
print(f"Years used: {int(yrs[0])}-{int(yrs[-1])} (N={len(yrs)})")
print(f"Mean SSRD {mean_map.min():.2f}-{mean_map.max():.2f}; trend/decade "
      f"{dec.min():+.3f}..{dec.max():+.3f} (mean {dec.mean():+.3f})")
print(f"Unadjusted p<0.05: {100 * (p < 0.05).mean():.0f}% of cells; FDR(q={FDR_Q}): "
      f"{100 * sig.mean():.0f}%; corr(Theil-Sen, OLS) = {np.corrcoef(dec.ravel(), ols.ravel())[0, 1]:.3f}")
for nm, a in (("brightening", (dec > 0) & sig), ("dimming", (dec < 0) & sig)):
    print(f"   significant {nm}: {100 * a.mean():.0f}% of cells")
np.save(os.path.join(OUT, "mean_ssrd.npy"), mean_map)
np.save(os.path.join(OUT, "trend_per_decade.npy"), dec)
np.save(os.path.join(OUT, "trend_fdr_sig.npy"), sig)

# ---- region statistics
rows, tex = [], []
for k, b in REGIONS.items():
    ser = np.array([box_series(A[i:i + 1], lat, lon, b)[0] for i in range(len(yrs))])
    sl, pp = trend_fields(ser[:, None, None], yrs)
    m = box_mask(lat, lon, b)
    row = dict(region=k, name=b[0], latS=b[1], latN=b[2], lonW=b[3], lonE=b[4],
               mean_ssrd=ser.mean(), trend_per_decade=sl[0, 0] * 10,
               trend_pct_per_decade=100 * sl[0, 0] * 10 / ser.mean(), p_MK=pp[0, 0], n_cells=int(m.sum()))
    if land is not None:
        row["land_fraction"] = float((land & m).sum() / m.sum())
        ls = np.array([box_series(A[i:i + 1], lat, lon, b, land=land)[0] for i in range(len(yrs))])
        s2, p2 = trend_fields(ls[:, None, None], yrs)
        row.update(mean_ssrd_land=ls.mean(), trend_pct_per_decade_land=100 * s2[0, 0] * 10 / ls.mean(),
                   p_MK_land=p2[0, 0])
    rows.append(row)
    pf = "$<$0.001" if pp[0, 0] < 0.001 else f"{pp[0, 0]:.3f}"
    tex.append(f"{k} & {b[0]} & {b[1]}--{b[2]}\\textdegree N & {b[3]}--{b[4]}\\textdegree E & "
               f"{ser.mean():.2f} & ${row['trend_pct_per_decade']:+.1f}$ ({pf}) \\\\")
pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "region_trends.csv"), index=False)
with open(os.path.join(OUT, "table_regions.tex"), "w") as f:
    f.write("\\begin{tabular}{llccrc}\n\\topline\nID & Region & Latitude & Longitude & "
            "Mean SSRD & Trend (\\% decade$^{-1}$) ($p$)\\\\\n\\midline\n"
            + "\n".join(tex) + "\n\\botline\n\\end{tabular}\n")
print(pd.DataFrame(rows)[["region", "name", "mean_ssrd", "trend_per_decade",
                          "trend_pct_per_decade", "p_MK"]].round(3).to_string(index=False))

# ---- do the boxes match the stage-2 series?
p2f = os.path.join(STAGE2, "regional_monthly_ssrd.csv")
if os.path.exists(p2f) or os.environ.get("SYNTH"):
    mine = pd.DataFrame({k: box_series(D, lat, lon, b) for k, b in REGIONS.items()}, index=times)
    mine_u = pd.DataFrame({k: box_series(D, lat, lon, b, weighted=False) for k, b in REGIONS.items()}, index=times)
    s2 = mine.copy() if os.environ.get("SYNTH") else pd.read_csv(p2f, index_col=0, parse_dates=True).reindex(times)
    M = pd.DataFrame({c: {k: min(np.nanmean(np.abs(s2[c] - mine[k])), np.nanmean(np.abs(s2[c] - mine_u[k])))
                          for k in REGIONS} for c in s2.columns if c in REGIONS})
    M.index.name = "box_here \\ stage2_column"
    M.round(4).to_csv(os.path.join(OUT, "region_label_check.csv"))
    print("\nMean |difference| (kWh m-2 d-1): rows = boxes in this script, columns = stage-2 series")
    print(M.round(3).to_string())
    bad = [c for c in M.columns if M[c].idxmin() != c or M[c].min() > 0.02]
    for c in bad:
        print(f"  !! stage-2 column {c} best matches box {M[c].idxmin()} (diff {M[c].min():.3f})")
    if bad:
        print("  -> the boxes (or their IDs) differ from stage 2. Put the stage-2 boxes into "
              "regions.json and rerun.")
    else:
        print("  OK: boxes reproduce the stage-2 series.")


# ---- figures
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


def boxes(ax, fs=6.5, names=False):
    for k, (nm, s, n, w, e) in REGIONS.items():
        ax.add_patch(Rectangle((w, s), e - w, n - s, fill=False, ec="k", lw=0.8, zorder=8))
        ax.text((w + e) / 2, (s + n) / 2, k, ha="center", va="center", fontsize=fs, fontweight="bold",
                zorder=9, bbox=dict(fc="w", ec="none", alpha=0.6, pad=0.5))


def finish(ax, labels=True):
    ax.set_xlim(DOM[0], DOM[1])
    ax.set_ylim(DOM[2], DOM[3])
    ax.set_aspect("equal")
    ax.grid(color="gray", lw=0.3, ls=":")
    if labels:
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")


def mean_panel(fig, ax, title="(a) Mean SSRD"):
    lv = np.arange(np.floor(mean_map.min() * 4) / 4, mean_map.max() + 0.26, 0.25)
    cf = ax.contourf(lon, lat, mean_map, levels=lv, cmap="YlOrRd", extend="neither")
    add_geo(ax)
    boxes(ax)
    cb = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Mean SSRD (kWh m$^{-2}$ d$^{-1}$)")
    ax.set_title(f"{title} {int(yrs[0])}–{int(yrs[-1])} and study regions", loc="left", fontweight="bold")
    finish(ax)


def trend_panel(fig, ax, field, sigmask, title, vm, label, step=3, box_fs=6.5, nlev=21):
    cf = ax.contourf(lon, lat, field, levels=np.linspace(-vm, vm, nlev), cmap="RdBu_r", extend="both")
    yy, xx = np.where(sigmask[::step, ::step])
    ax.scatter(lon[::step][xx], lat[::step][yy], s=1.2, c="k", lw=0, alpha=0.7, zorder=5)
    add_geo(ax)
    boxes(ax, fs=box_fs)
    cb = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(label)
    ax.set_title(title, loc="left", fontweight="bold")
    finish(ax)


plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6})
VM = max(0.02, np.ceil(np.nanpercentile(np.abs(dec), 99) * 50) / 50)
TL = "SSRD trend (kWh m$^{-2}$ d$^{-1}$ decade$^{-1}$)"
TT = "(b) SSRD trend (Theil–Sen); dots: Mann–Kendall, FDR $q=0.10$"


def save(fig, name):
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{e}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


fig, axs = plt.subplots(2, 1, figsize=(6.5, 8.6))
mean_panel(fig, axs[0])
trend_panel(fig, axs[1], dec, sig, TT, VM, TL)
fig.tight_layout()
save(fig, "Fig1_domain_trend")

fig, ax = plt.subplots(figsize=(6.5, 4.6))
mean_panel(fig, ax, "Mean SSRD")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "Fig1a_mean_map.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
fig, ax = plt.subplots(figsize=(6.5, 4.6))
trend_panel(fig, ax, dec, sig, "SSRD trend (Theil–Sen); dots: FDR $q=0.10$", VM, TL)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "Fig1b_trend_map.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# supplement: quarterly trends (same scale) and relative trend
fig, axs = plt.subplots(2, 2, figsize=(6.5, 5.6))
for ax, (q, mo) in zip(axs.ravel(), QUARTERS.items()):
    yq, Aq = annual_from(D, times, mo)
    sq, pq = trend_fields(Aq, yq)
    trend_panel(fig, ax, sq * 10, bh(pq, FDR_Q), f"{q}", max(VM, 0.02), "kWh m$^{-2}$ d$^{-1}$ decade$^{-1}$",
                step=4, box_fs=5)
fig.tight_layout()
save(fig, "FigS2_trend_seasonal")

fig, ax = plt.subplots(figsize=(6.5, 4.6))
vr = max(0.5, np.ceil(np.nanpercentile(np.abs(rel), 99) * 2) / 2)
trend_panel(fig, ax, rel, sig, "Annual SSRD trend relative to the mean", vr, "% decade$^{-1}$")
fig.tight_layout()
save(fig, "FigS3_trend_relative")
print("Done ->", OUT)
