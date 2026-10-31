# ============================================================
# Monthly ERA5 vs CERES comparison (2001-2015), per region
# Metrics: bias, RMSE, MAE, r  (raw + anomaly), with AR(1) tests
# ============================================================
import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ERA5_PATH  = r"C:\Users\Think\Desktop\new-data\ssrd-ssrdc.nc"
CERES_PATH = r"C:\Users\Think\Desktop\new-data\CERES_200003-202512.nc"
OUT        = r"C:\Users\Think\Desktop\new-data"
Y0, Y1     = 2001, 2015

J2KWH = 1.0 / 3.6e6
W2KWH = 24.0 / 1000.0

REGIONS = {
    "R1":  ("Western Yemen",        12, 17, 40, 45),
    "R2":  ("Oman",                 16, 24, 53, 58),
    "R3":  ("Egypt",                22, 30, 25, 35),
    "R4":  ("Southeastern Iran",    25, 29, 56, 64),
    "R5":  ("Northern Saudi Arabia",26, 32, 40, 48),
    "R6":  ("Southern Syria",       32, 38, 36, 43),
    "R7":  ("Zagros Mountains",     34, 37, 45, 49),
    "R8":  ("Northern Iran",        36, 39, 49, 56),
    "R9":  ("Northern Turkey",      38, 43, 27, 41),
    "R10": ("Caucasus",             39, 44, 46, 49),
}

# ---------- helpers ----------
def box_ts(da, lat_name, lon_name, box):
    """cos-lat weighted monthly time series inside a box; index on month-start."""
    _, s, n, w, e = box
    x = da.sel({lat_name: slice(s, n), lon_name: slice(w, e)})
    if x.sizes[lat_name] == 0 or x.sizes[lon_name] == 0:
        return None
    wgt = xr.DataArray(np.cos(np.deg2rad(x[lat_name].values)),
                       dims=lat_name, coords={lat_name: x[lat_name]})
    ts = x.weighted(wgt).mean((lat_name, lon_name)).to_series()
    ts.index = pd.DatetimeIndex(ts.index).to_period("M").to_timestamp()
    return ts

def neff_ar1(x):
    """effective sample size for AR(1) series (Bretherton et al. 1999)."""
    x = np.asarray(x, float) - np.mean(x)
    if x.std() == 0 or len(x) < 3:
        return len(x)
    a = np.corrcoef(x[:-1], x[1:])[0, 1]
    n = len(x)
    return max(5.0, n * (1 - a) / (1 + a))

def compute_metrics(era_s, cer_s):
    """all 4 metrics + both anomaly/raw versions, bias = mean(ERA5 - CERES)."""
    d = pd.concat([era_s.rename("era"), cer_s.rename("cer")], axis=1).dropna()
    diff = (d.era - d.cer).values                       # + means ERA5 overestimates
    bias, rmse, mae = float(np.mean(diff)), float(np.sqrt(np.mean(diff**2))), float(np.mean(np.abs(diff)))
    n_eff = neff_ar1(diff)
    t = bias / (np.std(diff, ddof=1) / np.sqrt(n_eff))
    p_bias = float(2 * stats.t.sf(abs(t), n_eff - 1))

    era_an = d.era - d.era.groupby(d.index.month).transform("mean")
    cer_an = d.cer - d.cer.groupby(d.index.month).transform("mean")
    dan = era_an - cer_an
    rmse_an, mae_an = float(np.sqrt((dan**2).mean())), float(dan.abs().mean())
    r_an = float(era_an.corr(cer_an))
    n_eff_a = neff_ar1(era_an.values)
    t_a = r_an * np.sqrt(max(n_eff_a - 2, 1) / max(1 - r_an**2, 1e-6))
    p_an = float(2 * stats.t.sf(abs(t_a), max(n_eff_a - 2, 1)))

    clim_e = d.era.groupby(d.index.month).mean()
    clim_c = d.cer.groupby(d.index.month).mean()
    rms_clim = float(np.sqrt(((clim_e - clim_c)**2).mean()))

    return dict(
        n=len(d),
        mean_era=float(d.era.mean()), mean_cer=float(d.cer.mean()),
        bias=bias, p_bias=p_bias,
        rmse=rmse, mae=mae, nrmse=100 * rmse / d.cer.mean(),
        r_raw=float(d.era.corr(d.cer)),
        rmse_clim=rms_clim,
        rmse_anom=rmse_an, mae_anom=mae_an, nrmse_anom=100 * rmse_an / d.cer.mean(),
        r_anom=r_an, p_anom=p_an,
        std_era_anom=float(era_an.std()), std_cer_anom=float(cer_an.std()),
    )

# ---------- 1) load both datasets ----------
era = xr.open_dataset(ERA5_PATH)
tdim = "valid_time" if "valid_time" in era.coords else "time"
ydim = "latitude"   if "latitude"   in era.coords else "lat"
xdim = "longitude"  if "longitude"  in era.coords else "lon"
if float(era[xdim].max()) > 180:
    era = era.assign_coords({xdim: (((era[xdim] + 180) % 360) - 180)}).sortby(xdim)
era = era.sortby(ydim)
yr = era[tdim].dt.year
era_sub = era.isel({tdim: ((yr >= Y0) & (yr <= Y1)).values})

def era_ts(v):
    da = era_sub[v]
    for extra in [d for d in da.dims if d not in (tdim, ydim, xdim)]:
        da = da.mean(extra)
    return {k: box_ts(da, ydim, xdim, b) * J2KWH for k, b in REGIONS.items()}

era_all_ts = era_ts("ssrd")
era_clr_ts = era_ts("ssrdc")

cer = xr.open_dataset(CERES_PATH)
mask = (cer.time.dt.year >= Y0) & (cer.time.dt.year <= Y1)
cer_sub = cer.isel(time=mask)

def cer_ts(v):
    da = cer_sub[v]
    return {k: box_ts(da, "lat", "lon", b) * W2KWH for k, b in REGIONS.items()}

cer_all_ts = cer_ts("sfc_sw_down_all_mon")
cer_clr_ts = cer_ts("sfc_sw_down_clr_t_mon")

# ---------- 2) metrics per region and variable ----------
rows = []
for var, e, c in [("all-sky", era_all_ts, cer_all_ts),
                  ("clear-sky", era_clr_ts, cer_clr_ts)]:
    for k in REGIONS:
        m = compute_metrics(e[k], c[k])
        rows.append(dict(variable=var, region=k, name=REGIONS[k][0], **m))

df = pd.DataFrame(rows)

# print clean table
pd.set_option("display.width", 200)
print("\n=== ERA5 vs CERES, monthly, 2001-2015 (bias = mean[ERA5 - CERES]; + means ERA5 overestimates) ===\n")
for var in ("all-sky", "clear-sky"):
    sub = df[df.variable == var].copy()
    sub["sig"] = np.where(sub.p_bias < 0.05, "*", "")
    print(f"--- {var} ---")
    print(sub[["region", "name", "mean_era", "mean_cer", "bias", "sig",
               "rmse", "mae", "nrmse", "r_raw", "r_anom",
               "rmse_anom", "nrmse_anom", "p_anom"]]
          .round(3).to_string(index=False))
    print()

df.round(4).to_csv(os.path.join(OUT, f"metrics_ERA5_vs_CERES_{Y0}_{Y1}.csv"), index=False)

# compact manuscript-ready text table
with open(os.path.join(OUT, f"table_metrics_ERA5_vs_CERES_{Y0}_{Y1}.txt"), "w") as f:
    f.write(f"ERA5 vs CERES, monthly 2001-2015, bias = mean(ERA5 - CERES)\n")
    f.write(f"{'Var':10s} {'Reg':4s} {'bias':>8s} {'p':>7s} {'RMSE':>7s} "
            f"{'MAE':>7s} {'nRMSE%':>7s} {'r_raw':>7s} {'r_anom':>7s} {'RMSE_an':>8s}\n")
    for _, r in df.iterrows():
        sig = "*" if r.p_bias < 0.05 else " "
        f.write(f"{r.variable:10s} {r.region:4s} {r.bias:+8.3f} {r.p_bias:7.3f} "
                f"{r.rmse:7.3f} {r.mae:7.3f} {r.nrmse:7.2f} {r.r_raw:7.3f} "
                f"{r.r_anom:7.3f} {r.rmse_anom:8.3f}{sig}\n")

# ---------- 3) short per-region summary ----------
lines = []
for k in REGIONS:
    ra = df[(df.variable=="all-sky")  & (df.region==k)].iloc[0]
    rc = df[(df.variable=="clear-sky")& (df.region==k)].iloc[0]
    direction = "ERA5 HIGHER" if ra.bias > 0 else "ERA5 LOWER"
    lines.append(
        f"{k} {REGIONS[k][0]:22s}  "
        f"all: bias={ra.bias:+.2f} ({direction}), RMSE={ra.rmse:.2f}, "
        f"MAE={ra.mae:.2f}, r_anom={ra.r_anom:+.2f} | "
        f"clr: bias={rc.bias:+.2f}, RMSE={rc.rmse:.2f}, r_anom={rc.r_anom:+.2f}")
open(os.path.join(OUT, f"summary_metrics_{Y0}_{Y1}.txt"), "w").write("\n".join(lines))
print("\n".join(lines))

# ---------- 4) Fig 1 & 2 : monthly time series + 12-month climatology ----------
def panel_series(era_d, cer_d, title, fname):
    fig, axs = plt.subplots(5, 2, figsize=(10, 11), sharex=True)
    for i, k in enumerate(REGIONS):
        ax = axs.flat[i]
        d = pd.concat([era_d[k].rename("era"), cer_d[k].rename("cer")], axis=1).dropna()
        ce = d.era.groupby(d.index.month).mean()
        cc = d.cer.groupby(d.index.month).mean()
        m = d.index.month
        ax.plot(d.index, d.era, color="#1f77b4", lw=0.5, alpha=0.55)
        ax.plot(d.index, d.cer, color="#d62728", lw=0.5, alpha=0.55)
        ax.plot(d.index, ce.loc[m].values, color="#1f77b4", lw=1.6, label="ERA5 clim.")
        ax.plot(d.index, cc.loc[m].values, color="#d62728", lw=1.6, label="CERES clim.")
        ra = df[(df.variable == ("all-sky" if "all-sky" in title else "clear-sky")) &
                (df.region == k)].iloc[0]
        ax.set_title(f"{k} {REGIONS[k][0]}   bias={ra.bias:+.2f}  "
                     f"RMSE={ra.rmse:.2f}  r_anom={ra.r_anom:+.2f}",
                     loc="left", fontsize=7)
        if i == 0:
            ax.legend(frameon=False, loc="upper right", fontsize=6)
        ax.grid(alpha=0.25)
    for ax in axs[-1]:
        ax.set_xlabel("Year")
    for i in range(5):
        axs[i, 0].set_ylabel("kWh m$^{-2}$ d$^{-1}$")
    fig.suptitle(title, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)

panel_series(era_all_ts, cer_all_ts,
             f"All-sky SSRD, monthly 2001\u20132015 (thin = monthly, thick = 12-month climatology)",
             f"Fig_monthly_allsky_{Y0}_{Y1}.png")
panel_series(era_clr_ts, cer_clr_ts,
             f"Clear-sky SSRDc, monthly 2001\u20132015 (thin = monthly, thick = 12-month climatology)",
             f"Fig_monthly_clearsky_{Y0}_{Y1}.png")

# ---------- 5) Fig 3 : summary bars ----------
fig, axs = plt.subplots(2, 2, figsize=(11, 7))
rk = list(REGIONS)
x = np.arange(len(rk))
w = 0.38
for j, var in enumerate(("all-sky", "clear-sky")):
    sub = df[df.variable == var].set_index("region").loc[rk]
    for ax, col, lab in [
            (axs[0, 0], "bias",       "Bias = mean(ERA5 - CERES)\n(kWh m$^{-2}$ d$^{-1}$)"),
            (axs[0, 1], "rmse",       "RMSE (kWh m$^{-2}$ d$^{-1}$)"),
            (axs[1, 0], "rmse_anom",  "RMSE of anomaly (kWh m$^{-2}$ d$^{-1}$)"),
            (axs[1, 1], "r_anom",     "Anomaly correlation $r$")]:
        vals = sub[col].values
        offset = -w/2 + j*w
        bars = ax.bar(x + offset, vals, w,
                      color="#1f77b4" if j == 0 else "#d62728",
                      label="all-sky" if j == 0 else "clear-sky")
        if col == "bias":
            for xi, v in zip(x + offset, vals):
                ax.text(xi, v, "*" if abs(v) > 0 else "", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
for ax, lab in zip(axs.ravel(), ["Bias = mean(ERA5 - CERES)",
                                  "RMSE", "RMSE of anomaly", "Anomaly correlation"]):
    ax.set_xticks(x); ax.set_xticklabels(rk)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_title(lab, loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
axs[0, 0].set_ylabel("kWh m$^{-2}$ d$^{-1}$")
axs[0, 1].set_ylabel("kWh m$^{-2}$ d$^{-1}$")
axs[1, 0].set_ylabel("kWh m$^{-2}$ d$^{-1}$")
axs[1, 1].set_ylabel("Pearson $r$")
axs[0, 0].legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, f"Fig_metrics_summary_{Y0}_{Y1}.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- 6) Fig 4 : scatter of monthly anomalies (all-sky) ----------
fig, axs = plt.subplots(5, 2, figsize=(9, 11))
for i, k in enumerate(REGIONS):
    ax = axs.flat[i]
    d = pd.concat([era_all_ts[k].rename("era"), cer_all_ts[k].rename("cer")], axis=1).dropna()
    ea = d.era - d.era.groupby(d.index.month).transform("mean")
    ca = d.cer - d.cer.groupby(d.index.month).transform("mean")
    m = df[(df.variable == "all-sky") & (df.region == k)].iloc[0]
    ax.scatter(ea, ca, s=8, c="tab:purple", alpha=0.6, edgecolors="none")
    lim = max(abs(ea).max(), abs(ca).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.6)
    ax.axhline(0, color="k", lw=0.3); ax.axvline(0, color="k", lw=0.3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(f"{k} {REGIONS[k][0]}   $r_{{anom}}$={m.r_anom:+.2f}",
                 loc="left", fontsize=7)
    ax.grid(alpha=0.25)
for ax in axs[-1]:
    ax.set_xlabel("ERA5 SSRD anomaly")
for i in range(5):
    axs[i, 0].set_ylabel("CERES SSRD anomaly")
fig.suptitle(f"Monthly SSRD anomalies, 2001\u20132015 (all-sky)", fontweight="bold", y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, f"Fig_anomaly_scatter_allsky_{Y0}_{Y1}.png"),
            dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved: metrics CSV, text table, per-region summary, 4 figures -> {OUT}")