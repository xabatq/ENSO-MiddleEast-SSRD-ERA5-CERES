#!/usr/bin/env python3
# =====================================================================
# ceres_enso.py - independent check of the ERA5 ENSO--SSRD results with CERES EBAF
# ---------------------------------------------------------------------
# Runs next to stages_combine.py (it imports its region boxes, FDR routine and region-mean routine).
#
# Inputs
#   BASE/ssrd-ssrdc.nc                     ERA5 monthly ssrd, ssrdc (same file as stages_combine.py)
#   BASE/CERES_200003-202512.nc            CERES EBAF surface fluxes (env CERES_PATH to change)
#                                          variables: sfc_sw_down_all_mon, sfc_sw_down_clr_t_mon (W m-2)
#   BASE/stage2_outputs/regional_monthly_ssrd_enso.csv   (column ONI)
#   optional BASE/extra/climate_indices.csv (DMI, NAO) - used for the partial correlation
# Outputs  BASE/ceres_outputs/  (BASE/ceres_outputs_nodetrend when DETREND=0)
#   validation_metrics_ERA5_vs_CERES.csv, table_validation.tex
#   enso_summary_ERA5_vs_CERES.csv, table_ceres_enso.tex, seasonal_ONI_NDJ_ERA5_CERES.csv
#   key_numbers_ceres.txt
#   Fig5_validation, Fig6_ceres_enso_regions, Fig7_ceres_enso_maps, Fig8_ceres_enso_maps_season,
#   Fig9_ceres_seasonal
# Environment: BASE, CERES_PATH, Y0 (2001), Y1 (2025), DETREND, N_BOOT, N_SURR, SYNTH (test data)
# What it answers (reviewer questions): (1) is the ERA5 interannual variability realistic?  (2) does an
# independent satellite product show the same ENSO signal in sign, size, season, cloud/clear-sky split?
# (3) do gridded regression maps show the same spatial pattern?
# =====================================================================
import os
import sys
import glob
import zlib
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stages_combine as sc                      # noqa: E402  (region boxes, bh, box_series, month_start)

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = sc.BASE
STAGE2, EXTRA = sc.STAGE2, sc.EXTRA
CERES_PATH = os.environ.get("CERES_PATH", os.path.join(BASE, "CERES_200003-202512.nc"))
Y0 = int(os.environ.get("Y0", 2001))
Y1 = int(os.environ.get("Y1", 2025))
DETREND = sc.DETREND
N_BOOT, N_SURR = sc.N_BOOT, sc.N_SURR
OUT = os.path.join(BASE, "ceres_outputs" + ("" if DETREND else "_nodetrend"))
SYNTH = bool(os.environ.get("SYNTH"))
REGIONS, SHORT, RK = sc.REGIONS, sc.SHORT, sc.RK
LAGS, LEAD_IDX, BLOCK, ALPHA = sc.LAGS, sc.LEAD_IDX, sc.BLOCK, sc.ALPHA
FDR_GRID = 0.10
DOM_LAT, DOM_LON = (10.0, 45.0), (25.0, 70.0)
J2KWH, W2KWH = 1.0 / 3.6e6, 24.0 / 1000.0
QUARTERS = {"JFM": (1, 2, 3), "AMJ": (4, 5, 6), "JAS": (7, 8, 9), "OND": (10, 11, 12)}
bh = sc.bh
rng = np.random.default_rng(42)


def reseed(tag):
    global rng
    rng = np.random.default_rng([42, zlib.crc32(tag.encode())])


def save(fig, name):
    import io
    import time
    for e in ("pdf", "png"):
        buf = io.BytesIO()
        fig.savefig(buf, format=e, dpi=300, bbox_inches="tight")
        path = os.path.join(OUT, f"{name}.{e}")
        for attempt in range(3):
            try:
                with open(path, "wb") as fh:
                    fh.write(buf.getvalue())
                break
            except OSError as err:
                time.sleep(1.0)
                if attempt == 2:
                    alt = os.path.join(OUT, f"{name}_{time.strftime('%H%M%S')}.{e}")
                    with open(alt, "wb") as fh:
                        fh.write(buf.getvalue())
                    print(f"[warn] could not write {path} ({err}); saved as {alt}")
    plt.close(fig)


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "figure.dpi": 150,
                     "axes.linewidth": 0.6})
C_ERA, C_CER = "#1f77b4", "#d62728"


# =====================================================================
# 1. data
# =====================================================================
def _clean_lon(ds, xdim):
    if float(ds[xdim].max()) > 180:
        ds = ds.assign_coords({xdim: (((ds[xdim] + 180) % 360) - 180)}).sortby(xdim)
    return ds


def load_data():
    """-> times (month start), (lat_e, lon_e, era_all, era_clr), (lat_c, lon_c, cer_all, cer_clr); kWh m-2 d-1"""
    if SYNTH:
        r = np.random.default_rng(1)
        t = pd.date_range(f"{Y0}-01-01", f"{Y1}-12-01", freq="MS")
        lat_e, lon_e = np.arange(DOM_LAT[0], DOM_LAT[1] + 0.01, 0.5), np.arange(DOM_LON[0], DOM_LON[1] + 0.01, 0.5)
        lat_c, lon_c = np.arange(10.5, 45, 1.0), np.arange(25.5, 70, 1.0)
        oni = pd.read_csv(os.path.join(STAGE2, "regional_monthly_ssrd_enso.csv"), index_col=0, parse_dates=True)
        oni = sc.month_start(oni)["ONI"].reindex(t).ffill().bfill().rolling(7, min_periods=1).mean().values
        seas = np.sin(2 * np.pi * (t.month.values - 3) / 12)[:, None, None]

        def make(lat, lon, noise, bias):
            base = 5.0 + 1.5 * np.cos(np.deg2rad(lat))[None, :, None] + 0 * lon[None, None, :]
            amp = (0.04 + 0.03 * (lat[None, :, None] - 10) / 35) * 0
            sig = -0.06 * oni[:, None, None] * (1 + 0 * lat[None, :, None])
            allsky = base + 2 * seas + sig + r.normal(0, noise, (len(t), len(lat), len(lon))) + bias
            clr = base + 2.4 * seas + 0.35 + 0.4 * sig + r.normal(0, noise / 3, allsky.shape)
            return allsky, np.maximum(clr, allsky + 0.2)
        ea, ec = make(lat_e, lon_e, 0.12, 0.1)
        ca, cc = make(lat_c, lon_c, 0.18, 0.0)
        return t, (lat_e, lon_e, ea, ec), (lat_c, lon_c, ca, cc)
    import xarray as xr
    ds = xr.open_dataset(sc.FILE)
    tdim = "valid_time" if "valid_time" in ds.coords else "time"
    ydim = "latitude" if "latitude" in ds.coords else "lat"
    xdim = "longitude" if "longitude" in ds.coords else "lon"
    ds = _clean_lon(ds, xdim).sortby(ydim)
    yr = ds[tdim].dt.year
    ds = ds.isel({tdim: ((yr >= Y0) & (yr <= Y1)).values})

    def era(v):
        da = ds[v]
        for extra in [d for d in da.dims if d not in (tdim, ydim, xdim)]:
            da = da.mean(extra)
        return da.transpose(tdim, ydim, xdim).values.astype(np.float64) * J2KWH
    ea, ec = era("ssrd"), era("ssrdc")
    te = pd.DatetimeIndex(ds[tdim].values).to_period("M").to_timestamp()
    lat_e, lon_e = ds[ydim].values, ds[xdim].values
    path = CERES_PATH
    if not os.path.exists(path):
        cand = sorted(glob.glob(os.path.join(BASE, "CERES*.nc")))
        if not cand:
            raise SystemExit(f"CERES file not found: {CERES_PATH}")
        path = cand[0]
    cd = xr.open_dataset(path)
    cx = "lon" if "lon" in cd.coords else "longitude"
    cy = "lat" if "lat" in cd.coords else "latitude"
    ct = "time" if "time" in cd.coords else "valid_time"
    cd = _clean_lon(cd, cx).sortby(cy)
    cd = cd.sel({cy: slice(DOM_LAT[0] - 1, DOM_LAT[1] + 1), cx: slice(DOM_LON[0] - 1, DOM_LON[1] + 1)})
    cyr = cd[ct].dt.year
    cd = cd.isel({ct: ((cyr >= Y0) & (cyr <= Y1)).values})
    va = next((n for n in ("sfc_sw_down_all_mon",) if n in cd.data_vars), None)
    vc = next((n for n in ("sfc_sw_down_clr_t_mon", "sfc_sw_down_clr_c_mon") if n in cd.data_vars), None)
    if va is None or vc is None:
        raise SystemExit("CERES variables not found. Available: " + str(list(cd.data_vars)))
    print(f"[data] CERES variables: all-sky {va}, clear-sky {vc}")
    ca = cd[va].transpose(ct, cy, cx).values.astype(np.float64) * W2KWH
    cc = cd[vc].transpose(ct, cy, cx).values.astype(np.float64) * W2KWH
    tc = pd.DatetimeIndex(cd[ct].values).to_period("M").to_timestamp()
    common = te.intersection(tc)
    ie, ic = te.get_indexer(common), tc.get_indexer(common)
    return common, (lat_e, lon_e, ea[ie], ec[ie]), (cd[cy].values, cd[cx].values, ca[ic], cc[ic])


def regional(times, lat, lon, D):
    return pd.DataFrame({k: sc.box_series(D, lat, lon, b[:5] if len(b) == 5 else b) for k, b in REGIONS.items()},
                        index=times)


def regrid_mean(D, lat_s, lon_s, lat_t, lon_t):
    """mean of the source cells that fall into each target cell (target = 1-degree CERES grid)"""
    def wmat(src, tgt):
        d = np.median(np.diff(tgt))
        idx = np.floor((src - (tgt[0] - d / 2)) / d).astype(int)
        W = np.zeros((len(tgt), len(src)))
        ok = (idx >= 0) & (idx < len(tgt))
        W[idx[ok], np.where(ok)[0]] = 1.0
        s = W.sum(1, keepdims=True)
        return np.divide(W, s, out=np.zeros_like(W), where=s > 0)
    Wy, Wx = wmat(lat_s, lat_t), wmat(lon_s, lon_t)
    return np.einsum("ij,tjk,lk->til", Wy, D, Wx)


# =====================================================================
# 2. validation (ERA5 vs CERES, region means)
# =====================================================================
def neff_ar1(x):
    x = np.asarray(x, float) - np.mean(x)
    if x.std() == 0 or len(x) < 3:
        return len(x)
    a = np.corrcoef(x[:-1], x[1:])[0, 1]
    return max(5.0, len(x) * (1 - a) / (1 + a))


def deseason(s):
    return s - s.groupby(s.index.month).transform("mean")


def detrend(s):
    v = s.values.astype(float)
    x = np.arange(len(v), dtype=float)
    m = np.isfinite(v)
    sl, ic, *_ = stats.linregress(x[m], v[m])
    return pd.Series(v - (sl * x + ic), index=s.index)


def anom(s):
    a = deseason(s.astype(float))
    return detrend(a) if DETREND else a


def metrics(e, c):
    d = pd.concat([e.rename("era"), c.rename("cer")], axis=1).dropna()
    diff = (d.era - d.cer).values
    bias, rmse, mae = diff.mean(), np.sqrt((diff ** 2).mean()), np.abs(diff).mean()
    ne = neff_ar1(diff)
    p_bias = 2 * stats.t.sf(abs(bias / (np.std(diff, ddof=1) / np.sqrt(ne))), ne - 1)
    ea, ca = deseason(d.era), deseason(d.cer)
    dan = ea - ca
    r_an = ea.corr(ca)
    nea = neff_ar1(ea.values)
    p_an = 2 * stats.t.sf(abs(r_an) * np.sqrt(max(nea - 2, 1) / max(1 - r_an ** 2, 1e-6)), max(nea - 2, 1))
    return dict(n=len(d), mean_era=d.era.mean(), mean_cer=d.cer.mean(), bias=bias, bias_pct=100 * bias / d.cer.mean(),
                p_bias=p_bias, rmse=rmse, mae=mae, nrmse_pct=100 * rmse / d.cer.mean(), r_raw=d.era.corr(d.cer),
                rmse_anom=np.sqrt((dan ** 2).mean()), r_anom=r_an, p_anom=p_an,
                std_era_anom=ea.std(), std_cer_anom=ca.std(), std_ratio=ea.std() / ca.std())


def run_validation(era_all, era_clr, cer_all, cer_clr):
    era_cre, cer_cre = 1 - era_all / era_clr, 1 - cer_all / cer_clr
    rows = []
    for var, e, c in (("all-sky", era_all, cer_all), ("clear-sky", era_clr, cer_clr), ("CRE_rel", era_cre, cer_cre)):
        for k in RK:
            rows.append(dict(variable=var, region=k, name=SHORT[k], **metrics(e[k], c[k])))
    df = pd.DataFrame(rows)
    df.round(4).to_csv(os.path.join(OUT, "validation_metrics_ERA5_vs_CERES.csv"), index=False)
    # compact LaTeX table: all-sky and clear-sky
    tex = []
    for k in RK:
        a = df[(df.variable == "all-sky") & (df.region == k)].iloc[0]
        c = df[(df.variable == "clear-sky") & (df.region == k)].iloc[0]
        tex.append(f"{k} {SHORT[k]} & ${a.bias:+.2f}$ ({a.bias_pct:+.1f}) & {a.r_anom:.2f} & {a.std_ratio:.2f} & "
                   f"${c.bias:+.2f}$ ({c.bias_pct:+.1f}) & {c.r_anom:.2f} & {c.std_ratio:.2f} \\\\")
    with open(os.path.join(OUT, "table_validation.tex"), "w") as f:
        f.write("\\begin{tabular}{lcccccc}\n\\topline\n & \\multicolumn{3}{c}{All-sky SSRD} & "
                "\\multicolumn{3}{c}{Clear-sky SSRDc}\\\\\nRegion & Bias (\\%) & $r_{anom}$ & $\\sigma_E/\\sigma_C$ & "
                "Bias (\\%) & $r_{anom}$ & $\\sigma_E/\\sigma_C$\\\\\n\\midline\n" + "\n".join(tex) +
                "\n\\botline\n\\end{tabular}\n")
    return df


def fig_validation(df):
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4))
    x = np.arange(len(RK))
    w = 0.38
    for j, (var, col, lab) in enumerate((("all-sky", "#1f77b4", "all-sky"), ("clear-sky", "#d62728", "clear-sky"))):
        s = df[df.variable == var].set_index("region").loc[RK]
        for ax, c, t in ((axs[0, 0], "bias_pct", "(a) Bias ERA5 − CERES (% of CERES mean)"),
                         (axs[0, 1], "r_anom", "(b) Correlation of monthly anomalies"),
                         (axs[1, 0], "std_ratio", "(c) σ(ERA5 anomaly) / σ(CERES anomaly)"),
                         (axs[1, 1], "rmse_anom", "(d) RMSE of anomalies (kWh m$^{-2}$ d$^{-1}$)")):
            ax.bar(x - w / 2 + j * w, s[c].values, w, color=col, label=lab)
            ax.set_title(t, loc="left", fontweight="bold")
    axs[1, 0].axhline(1, color="k", lw=0.6, ls="--")
    axs[0, 0].axhline(0, color="k", lw=0.5)
    for ax in axs.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(RK)
        ax.grid(axis="y", alpha=0.3)
    axs[0, 0].legend(frameon=False)
    fig.tight_layout()
    save(fig, "Fig5_validation")


# =====================================================================
# 3. region-level ENSO analysis (same method as stages_combine.py)
# =====================================================================
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


def peak_tools(x, y, tag):
    reseed("ceres|peak|" + tag)
    d = pd.concat([x, y], axis=1).dropna()
    xv, yv = d.iloc[:, 0].values, d.iloc[:, 1].values
    r = rvec(xv, yv)
    kl = int(LEAD_IDX[np.nanargmax(np.abs(r[LEAD_IDX]))])
    ol = abs(r[kl])
    cl = sum(np.nanmax(np.abs(rvec(s, yv))[LEAD_IDX]) >= ol for s in phase_surrogates(xv, N_SURR))
    sl = np.sign(r[kl])
    bl = []
    for _ in range(N_BOOT):
        ii = block_idx(len(xv))
        rb = rvec(xv[ii], yv[ii])
        bl.append(LAGS[LEAD_IDX][np.nanargmax((rb * sl)[LEAD_IDX])])
    return dict(r_curve=r, lag_lead=int(LAGS[kl]), r_lead=r[kl], p_surr_lead=(1 + cl) / (1 + N_SURR),
                ci_lead=np.percentile(bl, [2.5, 97.5]))


def r_at_lag(x, y, k):
    d = pd.concat([x.shift(k), y], axis=1).dropna()
    return np.corrcoef(d.iloc[:, 0], d.iloc[:, 1])[0, 1] if len(d) > 24 else np.nan


def analyse(label, ssrd, ssrdc, oni_raw, extras):
    """lag correlations, lead-window peak (surrogates, FDR over regions), ONI7 regression and decomposition,
    seasonal NDJ correlations; one dataset"""
    cre = 1 - ssrd / ssrdc
    clim = lambda s: s.groupby(s.index.month).transform("mean")
    maybe = detrend if DETREND else (lambda s: s)
    oni_a = anom(oni_raw)
    oni7 = oni_a.rolling(7).mean()
    A = {"SSRD": {k: anom(ssrd[k]) for k in RK}, "SSRDc": {k: anom(ssrdc[k]) for k in RK},
         "CRE": {k: anom(cre[k]) for k in RK}}
    print(f"[{label}] lagged correlations, surrogates, bootstrap ...")
    PK = {(v, k): peak_tools(oni_a, A[v][k], f"{label}|{v}|{k}") for v in A for k in RK}
    SIG = dict(zip(RK, bh(np.array([PK[("SSRD", k)]["p_surr_lead"] for k in RK]), ALPHA)))
    slope = lambda x, y: np.cov(x, y, bias=True)[0, 1] / np.var(x)
    DEC = {}
    for k in RK:
        reseed(f"ceres|dec|{label}|{k}")
        clear = maybe((1 - clim(cre[k])) * deseason(ssrdc[k]))
        cloud = maybe(-clim(ssrdc[k]) * deseason(cre[k]))
        d = pd.concat([oni7.rename("x"), A["SSRD"][k].rename("t"), clear.rename("a"), cloud.rename("c")],
                      axis=1).dropna()
        X, T, Ca, Cc = (d[c].values for c in "xtac")
        st, sa, sc_ = slope(X, T), slope(X, Ca), slope(X, Cc)
        bs = []
        for _ in range(N_BOOT):
            ii = block_idx(len(X))
            t_ = slope(X[ii], T[ii])
            bs.append(slope(X[ii], Cc[ii]) / t_ if abs(t_) > 1e-9 else np.nan)
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        DEC[k] = dict(slope_SSRD=st, slope_clear=sa, slope_cloud=sc_, cloud_share=sc_ / st, cloud_share_lo=lo,
                      cloud_share_hi=hi, clear_share=sa / st, pct_per_unit=100 * st / ssrd[k].mean())
    # seasonal: NDJ ONI vs monthly SSRD anomaly, October(y-1) .. December(y)
    idx = ssrd.index
    years = list(range(idx.year.min() + 1, idx.year.max() + 1))
    OD = {(d.year, d.month): v for d, v in oni_raw.items()}
    ndj = {y: np.nanmean([OD.get((y - 1, 11), np.nan), OD.get((y - 1, 12), np.nan), OD.get((y, 1), np.nan)])
           for y in years}
    T15 = [(-1, m) for m in (10, 11, 12)] + [(0, m) for m in range(1, 13)]
    SEAS = {}
    for v in ("SSRD", "SSRDc"):
        for k in RK:
            Yd = {(d.year, d.month): z for d, z in A[v][k].items()}
            R, P = np.full(15, np.nan), np.full(15, np.nan)
            for j, (off, m) in enumerate(T15):
                pr = [(ndj[y], Yd.get((y + off, m), np.nan)) for y in years]
                pr = [(a, b) for a, b in pr if np.isfinite(a) and np.isfinite(b)]
                if len(pr) > 12:
                    lr = stats.linregress([a for a, _ in pr], [b for _, b in pr])
                    R[j], P[j] = lr.rvalue, lr.pvalue
            SEAS[(v, k)] = [R, P]
        sg = bh(np.array([SEAS[(v, k)][1] for k in RK]), ALPHA)
        for i, k in enumerate(RK):
            SEAS[(v, k)].append(sg[i])
    # DMI / NAO partial correlation at the lead-window lag
    part = {}
    CI = extras.get("indices")
    if CI is not None:
        cols = [c for c in ("DMI", "NAO") if c in CI]
        for k in RK:
            L = PK[("SSRD", k)]["lag_lead"]
            ctrl = pd.concat([CI[c].reindex(idx).shift(L) for c in cols], axis=1)
            d = pd.concat([oni_a.shift(L).rename("x"), A["SSRD"][k].rename("y"), ctrl], axis=1).dropna()
            Z = np.column_stack([np.ones(len(d))] + [d[c].values for c in ctrl.columns])
            res = lambda v, Z=Z: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
            part[k] = corr(res(d["x"].values), res(d["y"].values))
    return dict(label=label, A=A, oni_a=oni_a, PK=PK, SIG=SIG, DEC=DEC, SEAS=SEAS, T15=T15, part=part,
                mean=ssrd.mean())


def summary_table(RE, RC):
    rows = []
    for k in RK:
        row = dict(region=k, name=SHORT[k])
        for tag, R in (("ERA5", RE), ("CERES", RC)):
            p, d = R["PK"][("SSRD", k)], R["DEC"][k]
            row.update({f"{tag}_lag": p["lag_lead"], f"{tag}_lag_CI": f"[{p['ci_lead'][0]:.0f},{p['ci_lead'][1]:.0f}]",
                        f"{tag}_r": p["r_lead"], f"{tag}_p_surr": p["p_surr_lead"], f"{tag}_sig_fdr": bool(R["SIG"][k]),
                        f"{tag}_pct_per_ONI": d["pct_per_unit"], f"{tag}_slope": d["slope_SSRD"],
                        f"{tag}_cloud_share": d["cloud_share"], f"{tag}_cloud_lo": d["cloud_share_lo"],
                        f"{tag}_cloud_hi": d["cloud_share_hi"], f"{tag}_clear_share": d["clear_share"],
                        f"{tag}_r_CRE": p and R["PK"][("CRE", k)]["r_lead"],
                        f"{tag}_r_SSRDc": R["PK"][("SSRDc", k)]["r_lead"]})
            if R["part"]:
                row[f"{tag}_r_partial_DMI_NAO"] = R["part"][k]
        # cross-check at the other dataset's lag
        row["CERES_r_at_ERA5_lag"] = r_at_lag(RC["oni_a"], RC["A"]["SSRD"][k], RE["PK"][("SSRD", k)]["lag_lead"])
        row["ERA5_r_at_CERES_lag"] = r_at_lag(RE["oni_a"], RE["A"]["SSRD"][k], RC["PK"][("SSRD", k)]["lag_lead"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.round(4).to_csv(os.path.join(OUT, "enso_summary_ERA5_vs_CERES.csv"), index=False)
    tex = []
    for _, r in df.iterrows():
        def cs(tag):
            if r[f"{tag}_sig_fdr"]:
                return f"{100 * r[f'{tag}_cloud_share']:.0f} [{100 * r[f'{tag}_cloud_lo']:.0f},{100 * r[f'{tag}_cloud_hi']:.0f}]"
            return "n.s."
        st = lambda tag: "$^{*}$" if r[f"{tag}_sig_fdr"] else ""
        tex.append(f"{r.region} {r['name']} & ${r.ERA5_r:+.2f}${st('ERA5')} & ${r.CERES_r:+.2f}${st('CERES')} & "
                   f"${r.ERA5_pct_per_ONI:+.1f}$ & ${r.CERES_pct_per_ONI:+.1f}$ & {cs('ERA5')} & {cs('CERES')} \\\\")
    with open(os.path.join(OUT, "table_ceres_enso.tex"), "w") as f:
        f.write("\\begin{tabular}{lcccccc}\n\\topline\n & \\multicolumn{2}{c}{Peak $r$ (0--6\\,mo)} & "
                "\\multicolumn{2}{c}{SSRD (\\% per ONI)} & \\multicolumn{2}{c}{Cloud share (\\%)}\\\\\n"
                "Region & ERA5 & CERES & ERA5 & CERES & ERA5 & CERES\\\\\n\\midline\n" + "\n".join(tex) +
                "\n\\botline\n\\end{tabular}\n")
    rows = []
    for tag, R in (("ERA5", RE), ("CERES", RC)):
        for v in ("SSRD", "SSRDc"):
            for k in RK:
                Rr, P, sg = R["SEAS"][(v, k)]
                for j, (off, m) in enumerate(R["T15"]):
                    rows.append(dict(dataset=tag, var=v, region=k, year_offset=off, month=m, r=Rr[j], p=P[j],
                                     fdr_sig=bool(sg[j])))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "seasonal_ONI_NDJ_ERA5_CERES.csv"), index=False)
    return df


def fig_regions(RE, RC, df):
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.0))
    x = np.arange(len(RK))
    w = 0.38
    ax = axs[0, 0]
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    lim = 0.05 + max(df.ERA5_r.abs().max(), df.CERES_r.abs().max())
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.6)
    for _, r in df.iterrows():
        both = r.ERA5_sig_fdr and r.CERES_sig_fdr
        one = r.ERA5_sig_fdr or r.CERES_sig_fdr
        ax.scatter(r.ERA5_r, r.CERES_r, s=22, facecolor=("k" if both else "w"), edgecolor=("k" if one else "0.6"),
                   zorder=4)
        ax.annotate(r.region, (r.ERA5_r, r.CERES_r), xytext=(2, 2), textcoords="offset points", fontsize=5.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Peak correlation with ONI, ERA5")
    ax.set_ylabel("Peak correlation with ONI, CERES")
    ax.set_title("(a) ENSO correlation (filled: significant in both)", loc="left", fontweight="bold")
    ax = axs[0, 1]
    ax.bar(x - w / 2, df.ERA5_pct_per_ONI, w, color=C_ERA, label="ERA5")
    ax.bar(x + w / 2, df.CERES_pct_per_ONI, w, color=C_CER, label="CERES")
    ax.set_xticks(x)
    ax.set_xticklabels(RK)
    ax.set_ylabel("SSRD change per unit ONI (% of mean)")
    ax.set_title("(b) Effect size", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax = axs[1, 0]
    for j, (tag, col) in enumerate((("ERA5", C_ERA), ("CERES", C_CER))):
        for i, r in df.iterrows():
            if r[f"{tag}_sig_fdr"]:
                cl, ce = r[f"{tag}_cloud_share"], r[f"{tag}_clear_share"]
                ax.bar(i - w / 2 + j * w, cl, w, color=col, alpha=0.95)
                ax.bar(i - w / 2 + j * w, ce, w, bottom=cl, color=col, alpha=0.35)
            else:
                ax.text(i - w / 2 + j * w, 0.02, "n.s.", rotation=90, ha="center", va="bottom", fontsize=6.5, color=col, alpha=0.8)
    ax.axhline(1, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(RK)
    ax.set_ylabel("Share of the ONI-regressed SSRD change")
    ax.set_title("(c) Cloud (dark) and clear-sky (light) share", loc="left", fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax = axs[1, 1]
    for kk, ls in (("R7", "-"), ("R1", "--")):
        if kk in RK:
            ax.plot(LAGS, RE["PK"][("SSRD", kk)]["r_curve"], color=C_ERA, ls=ls, label=f"ERA5 {kk}")
            ax.plot(LAGS, RC["PK"][("SSRD", kk)]["r_curve"], color=C_CER, ls=ls, label=f"CERES {kk}")
    ax.axvspan(0, 6, color="0.9", zorder=0)
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4, ls=":")
    ax.set_xlabel("Lag (months); positive = ONI leads")
    ax.set_ylabel("Pearson r")
    ax.legend(frameon=False, ncol=2)
    ax.set_title("(d) Lag correlation, SSRD vs ONI", loc="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "Fig6_ceres_enso_regions")


def fig_seasonal(RE, RC):
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.2), sharey=True)
    lab = ["O", "N", "D", "J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    M = {tag: np.array([R["SEAS"][("SSRD", k)][0] for k in RK]) for tag, R in (("ERA5", RE), ("CERES", RC))}
    S = {tag: np.array([R["SEAS"][("SSRD", k)][2] for k in RK]) for tag, R in (("ERA5", RE), ("CERES", RC))}
    for ax, tag in zip(axs[:2], ("ERA5", "CERES")):
        im = ax.imshow(M[tag], vmin=-0.9, vmax=0.9, cmap="RdBu_r", aspect="auto")
        yy, xx = np.where(S[tag])
        ax.scatter(xx, yy, s=4, c="k")
        ax.set_title(f"{tag}: r(ONI NDJ, SSRD)", loc="left", fontweight="bold")
    ax = axs[2]
    im2 = ax.imshow(M["CERES"] - M["ERA5"], vmin=-0.6, vmax=0.6, cmap="PuOr", aspect="auto")
    ax.set_title("CERES − ERA5", loc="left", fontweight="bold")
    for ax in axs:
        ax.set_xticks(range(15))
        ax.set_xticklabels(lab)
        ax.axvline(2.5, color="k", lw=0.6)
        ax.set_xlabel("Target month (left of line: year before ENSO year)")
    axs[0].set_yticks(range(len(RK)))
    axs[0].set_yticklabels([f"{k} {SHORT[k]}" for k in RK])
    fig.colorbar(im, ax=axs[:2], shrink=0.8, pad=0.01).set_label("Pearson r (dots: FDR-significant)")
    fig.colorbar(im2, ax=axs[2], shrink=0.8, pad=0.02).set_label("Δ r")
    save(fig, "Fig9_ceres_seasonal")


# =====================================================================
# 4. gridded ENSO regression maps (1-degree CERES grid)
# =====================================================================
def anom_grid(D, times):
    a = D.copy()
    for m in range(1, 13):
        s = times.month.values == m
        a[s] -= a[s].mean(0)
    if DETREND:
        t = np.arange(len(a), dtype=float)
        t -= t.mean()
        a -= t[:, None, None] * ((t[:, None, None] * a).sum(0) / (t ** 2).sum())[None]
        a -= a.mean(0)[None]
    return a


def grid_regress(Y, x):
    ok = np.isfinite(x)
    Y, x = Y[ok], x[ok]
    n = len(x)
    xm = x - x.mean()
    Ym = Y - Y.mean(0)
    sxx = (xm ** 2).sum()
    sxy = (xm[:, None, None] * Ym).sum(0)
    slope = sxy / sxx
    r = sxy / np.sqrt(sxx * (Ym ** 2).sum(0))
    rx = np.corrcoef(xm[:-1], xm[1:])[0, 1]
    ry = (Ym[:-1] * Ym[1:]).sum(0) / np.sqrt((Ym[:-1] ** 2).sum(0) * (Ym[1:] ** 2).sum(0))
    den = 1 + rx * ry
    neff = np.clip(n * (1 - rx * ry) / np.where(den > 0, den, 1), 5, n)
    t = r * np.sqrt((neff - 2) / np.maximum(1 - r ** 2, 1e-9))
    return slope, r, 2 * stats.t.sf(np.abs(t), neff - 2)


def add_geo(ax, dom):
    try:
        from mpl_toolkits.basemap import Basemap
        m = Basemap(projection="cyl", llcrnrlon=dom[0], llcrnrlat=dom[2], urcrnrlon=dom[1], urcrnrlat=dom[3],
                    resolution="i", ax=ax)
        m.drawcoastlines(linewidth=0.4)
        m.drawcountries(linewidth=0.4)
    except Exception:
        try:
            import cartopy.feature as cf
            ax.add_feature(cf.COASTLINE, lw=0.4)
            ax.add_feature(cf.BORDERS, lw=0.4)
        except Exception:
            pass


def boxes(ax):
    for k, b in REGIONS.items():
        _, s, n, w, e = b[:5]
        ax.add_patch(Rectangle((w, s), e - w, n - s, fill=False, ec="0.25", lw=0.5, zorder=8))
        ax.text((w + e) / 2, (s + n) / 2, k, ha="center", va="center", fontsize=5, zorder=9,
                bbox=dict(fc="w", ec="none", alpha=0.55, pad=0.2))


def draw_map(ax, lat, lon, field, sig, vm, title):
    dom = (lon.min(), lon.max(), lat.min(), lat.max())
    cf = ax.contourf(lon, lat, field, levels=np.linspace(-vm, vm, 21), cmap="RdBu", extend="both")
    yy, xx = np.where(sig)
    ax.scatter(lon[xx], lat[yy], s=1.5, c="k", lw=0, alpha=0.7, zorder=5)
    add_geo(ax, dom)
    boxes(ax)
    ax.set_xlim(dom[0], dom[1])
    ax.set_ylim(dom[2], dom[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="gray", lw=0.3, ls=":")
    ax.set_title(title, loc="left", fontweight="bold")
    return cf


def maps(times, oni_raw, era_g, cer_g, lat, lon):
    """ERA5 (averaged to 1 degree) and CERES: SSRD regression on ONI7, all months and by season"""
    oni_a = anom(oni_raw.reindex(times))
    oni7 = oni_a.rolling(7).mean().values
    out = {}
    mean_c = cer_g.mean(0)
    for tag, G in (("ERA5", era_g), ("CERES", cer_g)):
        Y = anom_grid(G, times)
        sl, r, p = grid_regress(Y, oni7)
        sg = bh(p, FDR_GRID)
        out[tag] = dict(pct=100 * sl / G.mean(0), r=r, sig=sg, kwh=sl)
        # seasons: seasonal mean of anomalies vs seasonal mean ONI7, n = number of years
        for q, mo in QUARTERS.items():
            ys = sorted(set(times.year))
            Ys, xs = [], []
            for y in ys:
                s = (times.year.values == y) & np.isin(times.month.values, mo)
                if s.sum() == 3 and np.all(np.isfinite(oni7[s])):
                    Ys.append(Y[s].mean(0))
                    xs.append(oni7[s].mean())
            if len(xs) > 12:
                sl_q, r_q, p_q = grid_regress(np.stack(Ys), np.array(xs))
                out[tag][q] = dict(pct=100 * sl_q / G[np.isin(times.month.values, mo)].mean(0), sig=bh(p_q, FDR_GRID))
        print(f"[maps] {tag}: {100 * sg.mean():.0f}% of cells FDR-significant (q={FDR_GRID}); "
              f"slope {out[tag]['pct'].min():+.2f}..{out[tag]['pct'].max():+.2f} % per ONI")
    # figure 7: ERA5, CERES, difference
    vm = max(0.5, np.ceil(np.nanpercentile(np.abs(np.concatenate([out[t]["pct"].ravel() for t in out])), 99) * 4) / 4)
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 3.3), sharey=True)
    cf = draw_map(axs[0], lat, lon, out["ERA5"]["pct"], out["ERA5"]["sig"], vm, "(a) ERA5")
    draw_map(axs[1], lat, lon, out["CERES"]["pct"], out["CERES"]["sig"], vm, "(b) CERES")
    diff = out["CERES"]["pct"] - out["ERA5"]["pct"]
    cf2 = draw_map(axs[2], lat, lon, diff, np.zeros_like(diff, bool), vm, "(c) CERES − ERA5")
    for ax in axs:
        ax.set_xlabel("Longitude (°E)")
    axs[0].set_ylabel("Latitude (°N)")
    fig.colorbar(cf, ax=axs, location="bottom", shrink=0.6, aspect=40, pad=0.1, extend="both").set_label(
        "SSRD change per unit ONI (% of local mean); dots: FDR $q=0.10$")
    save(fig, "Fig7_ceres_enso_maps")
    pc = np.corrcoef(out["ERA5"]["pct"].ravel(), out["CERES"]["pct"].ravel())[0, 1]
    # seasons
    qs = [q for q in QUARTERS if q in out["ERA5"] and q in out["CERES"]]
    fig, axs = plt.subplots(2, len(qs), figsize=(1.9 * len(qs) + 0.8, 4.4), sharex=True, sharey=True,
                            layout="constrained", squeeze=False)
    vq = max(1.0, np.ceil(np.nanpercentile(np.abs(np.concatenate(
        [out[t][q]["pct"].ravel() for t in out for q in qs])), 99) * 2) / 2)
    for i, tag in enumerate(("ERA5", "CERES")):
        for j, q in enumerate(qs):
            cfq = draw_map(axs[i, j], lat, lon, out[tag][q]["pct"], out[tag][q]["sig"], vq,
                           f"({'ab'[i]}{j + 1}) {tag} {q}")
    cb = fig.colorbar(cfq, ax=axs, location="bottom", shrink=0.6, aspect=40, pad=0.02, extend="both")
    cb.set_label("SSRD change per unit ONI (% of seasonal mean); dots: FDR $q=0.10$")
    save(fig, "Fig8_ceres_enso_maps_season")
    return out, pc


# =====================================================================
def main():
    os.makedirs(OUT, exist_ok=True)
    times, (lat_e, lon_e, ea, ec), (lat_c, lon_c, ca, cc) = load_data()
    print(f"[data] {len(times)} common months {times[0]:%Y-%m}..{times[-1]:%Y-%m}; "
          f"ERA5 grid {len(lat_e)}x{len(lon_e)}, CERES grid {len(lat_c)}x{len(lon_c)}; DETREND={DETREND}")
    era_all, era_clr = regional(times, lat_e, lon_e, ea), regional(times, lat_e, lon_e, ec)
    cer_all, cer_clr = regional(times, lat_c, lon_c, ca), regional(times, lat_c, lon_c, cc)
    rd = lambda f: sc.month_start(pd.read_csv(os.path.join(STAGE2, f), index_col=0, parse_dates=True))
    oni_raw = rd("regional_monthly_ssrd_enso.csv")["ONI"].astype(float)
    # ONI is needed for the full period of the data, not only for the common months
    oni_raw = oni_raw.reindex(times)
    extras = {}
    p = os.path.join(EXTRA, "climate_indices.csv")
    if os.path.exists(p):
        extras["indices"] = sc.month_start(pd.read_csv(p, index_col=0, parse_dates=True)).reindex(times)

    dv = run_validation(era_all, era_clr, cer_all, cer_clr)
    fig_validation(dv)
    RE = analyse("ERA5", era_all, era_clr, oni_raw, extras)
    RC = analyse("CERES", cer_all, cer_clr, oni_raw, extras)
    df = summary_table(RE, RC)
    fig_regions(RE, RC, df)
    fig_seasonal(RE, RC)

    # gridded maps on the CERES grid (ERA5 averaged into 1-degree cells)
    sel_lat = (lat_c >= DOM_LAT[0]) & (lat_c <= DOM_LAT[1])
    sel_lon = (lon_c >= DOM_LON[0]) & (lon_c <= DOM_LON[1])
    lat_g, lon_g = lat_c[sel_lat], lon_c[sel_lon]
    cer_g = ca[:, sel_lat][:, :, sel_lon]
    era_g = regrid_mean(ea, lat_e, lon_e, lat_g, lon_g)
    mp, pcorr = maps(times, oni_raw, era_g, cer_g, lat_g, lon_g)

    # ---- key numbers
    a = dv[dv.variable == "all-sky"]
    c = dv[dv.variable == "clear-sky"]
    rng_ = lambda v, f="{:.2f}": f"{f.format(np.min(v))} to {f.format(np.max(v))}"
    sig_e = [k for k in RK if RE["SIG"][k]]
    sig_c = [k for k in RK if RC["SIG"][k]]
    both = [k for k in RK if RE["SIG"][k] and RC["SIG"][k]]
    same_sign = int(((df.ERA5_r < 0) == (df.CERES_r < 0)).sum())
    cn = [f"period {times[0]:%Y-%m}..{times[-1]:%Y-%m}, DETREND={int(DETREND)}",
          "VALIDATION (monthly region means, ERA5 - CERES)",
          f"  all-sky bias % of CERES mean: {rng_(a.bias_pct, '{:+.1f}')}; anomaly r: {rng_(a.r_anom)}; "
          f"sigma ratio ERA5/CERES: {rng_(a.std_ratio)}",
          f"  clear-sky bias %: {rng_(c.bias_pct, '{:+.1f}')}; anomaly r: {rng_(c.r_anom)}; "
          f"sigma ratio ERA5/CERES: {rng_(c.std_ratio)}",
          "ENSO (lead window 0..+6 months, surrogate p, BH-FDR across regions)",
          f"  ERA5 significant ({len(sig_e)}): {sig_e}",
          f"  CERES significant ({len(sig_c)}): {sig_c}",
          f"  significant in both: {both}",
          f"  sign of peak r agrees in {same_sign} of {len(RK)} regions; "
          f"r(ERA5 peak r, CERES peak r) across regions = {np.corrcoef(df.ERA5_r, df.CERES_r)[0, 1]:.2f}",
          f"  peak r ERA5 {rng_(df.ERA5_r)}; CERES {rng_(df.CERES_r)}",
          f"  % per ONI: ERA5 {rng_(df.ERA5_pct_per_ONI, '{:+.1f}')}; CERES {rng_(df.CERES_pct_per_ONI, '{:+.1f}')}",
          f"  ratio of effect size CERES/ERA5 across regions: median "
          f"{np.median(df.CERES_pct_per_ONI / df.ERA5_pct_per_ONI):.2f}",
          f"  cloud share in significant regions: ERA5 "
          f"{rng_([100 * df.loc[df.region == k, 'ERA5_cloud_share'].iloc[0] for k in sig_e], '{:.0f}') if sig_e else '-'}"
          f" %; CERES "
          f"{rng_([100 * df.loc[df.region == k, 'CERES_cloud_share'].iloc[0] for k in sig_c], '{:.0f}') if sig_c else '-'} %",
          f"  gridded: spatial correlation of the ERA5 and CERES regression maps = {pcorr:.2f}; "
          f"FDR-significant cells ERA5 {100 * mp['ERA5']['sig'].mean():.0f}%, CERES {100 * mp['CERES']['sig'].mean():.0f}%"]
    for tag, R in (("ERA5", RE), ("CERES", RC)):
        cnt = {k: int(R["SEAS"][("SSRD", k)][2].sum()) for k in RK}
        cn.append(f"  seasonal FDR-significant SSRD cells (of 15 per region) {tag}: total {sum(cnt.values())}; " +
                  ", ".join(f"{k}:{v}" for k, v in cnt.items()))
    open(os.path.join(OUT, "key_numbers_ceres.txt"), "w").write("\n".join(cn) + "\n")
    print("\n".join(cn))
    print("Done ->", OUT)


if __name__ == "__main__":
    main()
