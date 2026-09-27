"""Task 3: equilibration stationarity and residual stress.

Protocol (from 001_Anneal.in, 00k_Ext_0k.in; all NPT, aniso barostat, P=0, Tdamp 0.2, Pdamp 2, dt 0.002):
  anneal: 0.1M warm-up 0.01->1.7 (P=1) | 1M soak 1.7 | 1M cool 1.7->1.35 | 1M hold 1.35 | 1M cool 1.35->1.0
  production: five blocks of 2e6 steps at T=1 (4,000 tau each; thermo every 5000 steps = 10 tau).
Thermo columns: step temp pe ke etotal press pxx pyy pzz lx ly lz density (no off-diagonal stress logged;
the snapshot shear stress of equilibrated_network.data is recomputed from positions+velocities and
validated against the logged diagonal components, see extract_per_network.py).
Inputs: cache/per_network_raw.csv and cache/thermo.npz (extract_per_network.py, parsed from slurm-*.out and
log.lammps, which are not deposited) and data/derived/mechanics/mechanics_12pull.csv.
Outputs go to data/, tables/, figures/ and cache/ in NPJ_ANALYSIS_OUT/cg_structure/.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import OUT, MECH12

CACHE = os.path.join(OUT, "cache")
raw = pd.read_csv(os.path.join(CACHE, "per_network_raw.csv"))
mech = pd.read_csv(MECH12)[["folder_name", "uts_mean", "toughness_mean", "Quadrant"]]
th = np.load(os.path.join(CACHE, "thermo.npz"))
nets = list(th["nets"]); assert nets == list(raw.folder_name)
assert (th["prod_len"] == 401).all()
COLS = ["step", "temp", "pe", "ke", "etotal", "press", "pxx", "pyy", "pzz", "lx", "ly", "lz", "density"]
ci = {c: i for i, c in enumerate(COLS)}
prod = th["prod"].astype(np.float64)
ann = th["anneal"].astype(np.float64)
NN = len(nets)
# blocks without the duplicated boundary row: (327, 5, 400, 13)
blocks = np.stack([prod[:, 401 * k + 1: 401 * (k + 1), :] for k in range(5)], 1)
assert np.allclose(blocks[:, 1:, 0, 0] - blocks[:, :-1, -1, 0], 5000)


def derived(a):
    """a[..., 13] -> dict of named series."""
    q = {c: a[..., ci[c]] for c in ["temp", "pe", "etotal", "press", "pxx", "pyy", "pzz", "lx", "ly", "lz", "density"]}
    q["vol"] = q["lx"] * q["ly"] * q["lz"]
    q["dev_xy"] = q["pxx"] - q["pyy"]; q["dev_yz"] = q["pyy"] - q["pzz"]; q["dev_zx"] = q["pzz"] - q["pxx"]
    q["box_aniso"] = np.maximum(np.maximum(q["lx"], q["ly"]), q["lz"]) / np.minimum(np.minimum(q["lx"], q["ly"]), q["lz"]) - 1
    return q


Q = derived(blocks)
QUANT = ["pe", "temp", "etotal", "press", "density", "vol", "pxx", "pyy", "pzz", "dev_xy", "dev_yz", "dev_zx", "lx", "lz", "box_aniso"]

# ---- block and sub-block means ----------------------------------------------------
bm = {k: Q[k].mean(2) for k in QUANT}                                          # (327, 5)
sb = {k: Q[k].reshape(NN, 5, 4, 100).mean(3).reshape(NN, 20) for k in QUANT}  # 20 sub-blocks of 500k steps (1000 tau)


def ols_t(y, x=None):
    """Row-wise OLS slope and t for y (n_net, m)."""
    m = y.shape[1]
    x = np.arange(m, dtype=float) if x is None else x
    xc = x - x.mean()
    b = (y - y.mean(1, keepdims=True)) @ xc / (xc @ xc)
    res = y - y.mean(1, keepdims=True) - b[:, None] * xc[None, :]
    s2 = (res ** 2).sum(1) / (m - 2)
    se = np.sqrt(s2 / (xc @ xc))
    return b, se, b / se, res


per = pd.DataFrame({"folder_name": nets})
summ = []
tc3, tc18, tc10 = stats.t.ppf(0.975, 3), stats.t.ppf(0.975, 18), stats.t.ppf(0.975, 10)
for k in QUANT:
    b5, se5, t5, _ = ols_t(bm[k])                   # 5 block means, df=3
    b20, se20, t20, res20 = ols_t(sb[k])            # 20 sub-blocks, df=18 (slope per sub-block)
    b12, se12, t12, _ = ols_t(sb[k][:, 8:])         # last 3 blocks (12 sub-blocks), df=10
    lag1 = np.mean([np.corrcoef(r[:-1], r[1:])[0, 1] for r in res20])
    bsd = bm[k].std(1, ddof=1)                      # block-to-block SD
    per[f"{k}_block5_mean"] = bm[k][:, 4]
    per[f"{k}_blocks_sd"] = bsd
    per[f"{k}_slope_per_block"] = b5; per[f"{k}_t5"] = t5
    per[f"{k}_t20"] = t20; per[f"{k}_t_last3"] = t12
    per[f"{k}_change_b1_to_b5"] = bm[k][:, 4] - bm[k][:, 0]
    tt = stats.ttest_1samp(b20 * 4, 0.0)           # ensemble test of the mean slope (per block)
    summ.append(dict(quantity=k, ens_mean_block5=bm[k][:, 4].mean(), ens_sd_block5=bm[k][:, 4].std(ddof=1),
                     mean_within_block_sd=Q[k][:, 4].std(1).mean(),
                     mean_block_to_block_sd=bsd.mean(),
                     ens_mean_slope_per_block=(b20 * 4).mean(), ens_se_slope_per_block=(b20 * 4).std(ddof=1) / np.sqrt(NN),
                     mean_abs_change_b1_b5=np.abs(bm[k][:, 4] - bm[k][:, 0]).mean(),
                     frac_t5_sig=np.mean(np.abs(t5) > tc3), frac_t20_sig=np.mean(np.abs(t20) > tc18),
                     frac_tlast3_sig=np.mean(np.abs(t12) > tc10),
                     mean_t20=t20.mean(), sd_t20=t20.std(ddof=1),
                     ens_p_slope_zero=tt.pvalue, lag1_resid_acf_sub=lag1))
summ = pd.DataFrame(summ)

# ---- anneal vs production ---------------------------------------------------------
A = derived(ann)
for k in ["pe", "density", "press"]:
    per[f"{k}_anneal_last100"] = A[k][:, -100:].mean(1)   # last 1000 tau of the final cooling ramp (T 1.07 -> 1.0)

# ---- final state and residual stress ------------------------------------------------
fin = raw.set_index("folder_name").loc[nets]
per["final_press_snapshot"] = fin["snap_press"].values
for c in ["pxx", "pyy", "pzz", "pxy", "pxz", "pyz"]:
    per[f"final_{c}_snapshot"] = fin[f"snap_{c}"].values
# block-5 means (time averages) and standard errors. Stress samples 10 tau apart are uncorrelated
# (lag-1 ACF ~0, see acf below), so 20 sub-blocks of 200 tau give a df=19 SE.
for k in ["press", "dev_xy", "dev_yz", "dev_zx", "pxx", "pyy", "pzz"]:
    s20 = Q[k][:, 4].reshape(NN, 20, 20).mean(2)
    per[f"{k}_block5_se"] = s20.std(1, ddof=1) / np.sqrt(20)
T19 = stats.t.ppf(0.975, 19)
# thermal SD of instantaneous shear-like stress: SD over time of (pxx-pyy)/2 within block 5 (10-tau samples)
therm_sd_shear = (Q["dev_xy"][:, 4] / 2).std(1)
therm_sd_press = Q["press"][:, 4].std(1)
per["thermal_sd_halfdev"] = therm_sd_shear
per["thermal_sd_press"] = therm_sd_press
shear = fin[["snap_pxy", "snap_pxz", "snap_pyz"]].values
halfdev_snap = np.c_[(fin.snap_pxx - fin.snap_pyy) / 2, (fin.snap_pyy - fin.snap_pzz) / 2, (fin.snap_pzz - fin.snap_pxx) / 2]
tsd = therm_sd_shear.mean()
var_ratio_shear = (shear ** 2).mean() / tsd ** 2                   # mean-zero test: E[x^2]/thermal var
n_sh = shear.size
# chi-square CI on the ratio (treating the 981 values as independent)
ci_lo = n_sh * var_ratio_shear / stats.chi2.ppf(0.975, n_sh); ci_hi = n_sh * var_ratio_shear / stats.chi2.ppf(0.025, n_sh)
resid_rms_upper = tsd * np.sqrt(max(0.0, ci_hi - 1))
shear_stats = dict(mean_pxy=shear[:, 0].mean(), mean_pxz=shear[:, 1].mean(), mean_pyz=shear[:, 2].mean(),
                   rms_shear_snapshot=np.sqrt((shear ** 2).mean()), thermal_sd_halfdev_mean=tsd,
                   var_ratio=var_ratio_shear, var_ratio_ci=(ci_lo, ci_hi), resid_rms_upper95=resid_rms_upper,
                   rms_halfdev_snapshot=np.sqrt((halfdev_snap ** 2).mean()),
                   p_mean_shear_zero=stats.ttest_1samp(shear.ravel(), 0).pvalue)


# ---- autocorrelation (production, 10-tau spacing) -----------------------------------
def acf_mean(y, maxlag):
    y = y - y.mean(1, keepdims=True)
    v = (y ** 2).mean(1)
    return np.array([np.mean((y[:, :y.shape[1] - L] * y[:, L:]).mean(1) / v) for L in range(maxlag)])


def tau_int(acf, dt=10.0):
    z = np.argmax(acf <= 0) if np.any(acf <= 0) else len(acf)   # integrate to first zero crossing
    return dt * (0.5 + acf[1:z].sum())


acf_box = acf_mean(np.log(Q["lz"].reshape(NN, -1) / Q["lx"].reshape(NN, -1)), 400)
acf_dev = acf_mean(Q["dev_xy"].reshape(NN, -1), 50)
acf_pe = acf_mean(Q["pe"].reshape(NN, -1), 50)
fin_aniso = np.asarray(np.maximum.reduce([fin.eq_Lx, fin.eq_Ly, fin.eq_Lz]) / np.minimum.reduce([fin.eq_Lx, fin.eq_Ly, fin.eq_Lz]) - 1)
La = np.stack([A["lx"], A["ly"], A["lz"]], -1)
ann_aniso = La.max(-1) / La.min(-1) - 1
box_stats = dict(tau_int_ln_lz_over_lx_lowerbound=tau_int(acf_box), acf_box_at_1000tau=acf_box[100], acf_box_at_2000tau=acf_box[200],
                 acf_box_at_3990tau=acf_box[399], acf_devxy_lag10tau=acf_dev[1], acf_pe_lag10tau=acf_pe[1],
                 final_box_aniso_median=float(np.median(fin_aniso)), final_box_aniso_p90=float(np.quantile(fin_aniso, 0.9)),
                 final_box_aniso_max=float(fin_aniso.max()), anneal_end_aniso_mean=float(np.nanmean(ann_aniso[:, -100:])))
per["final_box_aniso"] = fin_aniso
Lb = np.stack([Q["lx"], Q["ly"], Q["lz"]], -1)                      # (N,5,400,3)
lnrel = np.log(Lb / np.exp(np.log(Lb).mean(-1, keepdims=True)))
box_blocks = pd.DataFrame(dict(block=np.arange(1, 6),
                               mean_timeavg_aniso=bm["box_aniso"].mean(0),
                               sd_between_nets_timeavg_lnLz=lnrel[..., 2].mean(2).std(0, ddof=1),
                               mean_within_block_sd_lnLz=lnrel[..., 2].std(2).mean(0)))
for k in range(5):
    per[f"box_aniso_block{k+1}"] = bm["box_aniso"][:, k]
per = per.merge(mech, on="folder_name")
per.to_csv(os.path.join(OUT, "data", "task3_stationarity_per_network.csv"), index=False)
summ.to_csv(os.path.join(OUT, "tables", "task3_drift_summary.csv"), index=False)
box_blocks.to_csv(os.path.join(OUT, "tables", "task3_box_shape_by_block.csv"), index=False)

# ---- prints ---------------------------------------------------------------------
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(summ.to_string(float_format=lambda v: f"{v:.4g}"))
print("\nshear:", {k: (np.round(v, 5) if np.ndim(v) == 0 else np.round(v, 4)) for k, v in shear_stats.items()})
print("box:", {k: round(float(v), 4) for k, v in box_stats.items()})
print(box_blocks.to_string(float_format=lambda v: f"{v:.4f}"))
uts = mech.uts_mean.mean()
fb = per
lines = []
lines.append(f"final-block mean pressure: mean {fb.press_block5_mean.mean():+.5f} sd {fb.press_block5_mean.std():.5f} "
             f"max|.| {fb.press_block5_mean.abs().max():.5f}; mean SE {fb.press_block5_se.mean():.5f}; "
             f"frac |P/SE|>t19 {(np.abs(fb.press_block5_mean / fb.press_block5_se) > T19).mean():.3f}; "
             f"SD across nets / mean SE {fb.press_block5_mean.std() / fb.press_block5_se.mean():.3f}")
for k in ["dev_xy", "dev_yz", "dev_zx"]:
    lines.append(f"final-block mean {k}: mean {fb[k + '_block5_mean'].mean():+.5f} sd {fb[k + '_block5_mean'].std():.5f} "
                 f"max|.| {fb[k + '_block5_mean'].abs().max():.5f}; mean SE {fb[k + '_block5_se'].mean():.5f}; "
                 f"frac |x/SE|>t19 {(np.abs(fb[k + '_block5_mean'] / fb[k + '_block5_se']) > T19).mean():.3f}; "
                 f"SD/SE {fb[k + '_block5_mean'].std() / fb[k + '_block5_se'].mean():.3f}")
lines.append(f"snapshot pressure: mean {fb.final_press_snapshot.mean():+.4f} sd {fb.final_press_snapshot.std():.4f}; "
             f"thermal SD of P {therm_sd_press.mean():.4f}")
lines.append(f"UTS mean {uts:.3f}: max |block-5 normal-stress difference| / UTS = "
             f"{max(fb[k + '_block5_mean'].abs().max() for k in ['dev_xy', 'dev_yz', 'dev_zx']) / uts:.4f}")
lines.append(f"anneal end (last 1000 tau, T->1) vs block 5: PE diff mean {(fb.pe_anneal_last100 - fb.pe_block5_mean).mean():+.5f}, "
             f"density diff {(fb.density_anneal_last100 - fb.density_block5_mean).mean():+.5f}")
for k in ["pe", "density", "press"]:
    r = stats.spearmanr(fb[f"{k}_t20"], fb.uts_mean)
    lines.append(f"rho(drift t of {k}, UTS) = {r.statistic:+.3f} (p={r.pvalue:.2g})")
r = stats.spearmanr(fb.final_box_aniso, fb.uts_mean); lines.append(f"rho(final box anisotropy, UTS) = {r.statistic:+.3f} (p={r.pvalue:.2g})")
r = stats.spearmanr(fb.final_box_aniso, fb.toughness_mean); lines.append(f"rho(final box anisotropy, toughness) = {r.statistic:+.3f} (p={r.pvalue:.2g})")
print("\n".join(lines))
with open(os.path.join(OUT, "tables", "task3_extra.txt"), "w") as fh:
    fh.write(repr(shear_stats) + "\n" + repr(box_stats) + "\n" + "\n".join(lines) + "\n")
np.savez(os.path.join(CACHE, "task3_acf.npz"), acf_box=acf_box, acf_dev=acf_dev, acf_pe=acf_pe)

# =============================== figures =========================================
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 300})
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK2 = "#52514e"
DT = 0.002
# representative network: closest to the ensemble median of block-5 PE and density (rank distance)
rk = (per.pe_block5_mean.rank() - NN / 2).abs() + (per.density_block5_mean.rank() - NN / 2).abs()
rep = int(np.argmin(rk.values)); rep_net = nets[rep]
t_ann = ann[rep, :, 0] * DT / 1000; t_pr = prod[rep, :, 0] * DT / 1000   # in 10^3 tau
fig, axs = plt.subplots(5, 1, figsize=(7.0, 8.2), sharex=True)
panels = [("temp", "T"), ("pe", "PE per bead"), ("density", r"Density ($\sigma^{-3}$)"), ("press", "Pressure"), ("box", r"Box length ($\sigma$)")]
for axx, (k, lab) in zip(axs, panels):
    if k == "press":
        for c, col, nm in (("pxx", C1, r"$P_{xx}$"), ("pyy", C2, r"$P_{yy}$"), ("pzz", C3, r"$P_{zz}$")):
            axx.plot(t_pr, prod[rep, :, ci[c]], lw=0.35, color=col, alpha=0.5, label=nm)
        axx.plot(t_ann, ann[rep, :, ci["press"]], lw=0.5, color="0.6")
        axx.plot(t_pr, prod[rep, :, ci["press"]], lw=0.4, color="0.25", label="P")
        rm = np.convolve(prod[rep, :, ci["press"]], np.ones(40) / 40, mode="same")
        axx.plot(t_pr[20:-20], rm[20:-20], lw=1.2, color="k", label=r"P, 400 $\tau$ mean")
        axx.set_ylim(-1.0, 1.0)
        axx.legend(ncol=5, frameon=False, loc="upper right", fontsize=6.5)
    elif k == "box":
        for c, col, nm in (("lx", C1, r"$L_x$"), ("ly", C2, r"$L_y$"), ("lz", C3, r"$L_z$")):
            axx.plot(t_ann, ann[rep, :, ci[c]], lw=0.5, color=col, alpha=0.45)
            axx.plot(t_pr, prod[rep, :, ci[c]], lw=0.5, color=col, label=nm)
        axx.set_ylim(18, 30)
        axx.legend(ncol=3, frameon=False, loc="upper right", fontsize=7)
    else:
        axx.plot(t_ann, ann[rep, :, ci[k]], lw=0.5, color="0.6")
        axx.plot(t_pr, prod[rep, :, ci[k]], lw=0.5, color=C1)
        for b in range(5):
            tb = blocks[rep, b, :, 0] * DT / 1000
            axx.hlines(bm[k][rep, b], tb[0], tb[-1], color="k", lw=1.2)
    if k == "pe":
        axx.set_ylim(np.nanmin(prod[rep, :, ci[k]]) - 0.3, np.nanmax(prod[rep, :, ci[k]]) + 0.3)
    if k == "density":
        axx.set_ylim(0.84, 0.94)
    if k == "temp":
        axx.set_ylim(0.9, 1.1)
    axx.set_ylabel(lab)
    for b in range(6):
        axx.axvline((4.1e6 + 2e6 * b) * DT / 1000, color="0.85", lw=0.5, zorder=0)
axs[-1].set_xlabel(r"Time ($10^3\,\tau$); grey = anneal (T 1.7 to 1.35 to 1.0), colour = five 2e6-step production blocks at T = 1")
axs[0].set_title(f"Representative network {rep_net}: equilibration trace (black bars = block means)", fontsize=8, loc="left")
axs[1].text(0.01, 0.1, "anneal starts from the dilute build (density 0.11); early values off-scale", transform=axs[1].transAxes,
            fontsize=6.5, color=INK2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figures", "task3_trace_representative.png"))
plt.close(fig)

# ensemble panels
fig, axs = plt.subplots(2, 3, figsize=(7.4, 5.2))
xb = np.arange(1, 6)


def band(axx, y, col, lab=True):
    for i in range(NN):
        axx.plot(xb, y[i], color="0.8", lw=0.3, alpha=0.5, zorder=1)
    lo, hi = np.quantile(y, [0.05, 0.95], axis=0)
    axx.fill_between(xb, lo, hi, color=col, alpha=0.18, lw=0, zorder=2, label="5-95%" if lab else None)
    axx.plot(xb, np.median(y, 0), color=col, lw=1.6, marker="o", ms=3.5, zorder=3, label="median" if lab else None)
    axx.set_xticks(xb); axx.set_xlabel(r"Production block (4000 $\tau$ each)")


for axx, k, lab in ((axs[0, 0], "pe", "a  PE per bead - network mean"),
                    (axs[0, 1], "density", r"b  density - network mean ($\sigma^{-3}$)")):
    y = bm[k] - bm[k].mean(1, keepdims=True)
    band(axx, y, C1, lab=(k == "pe"))
    axx.axhline(0, color="k", lw=0.5)
    axx.set_title(lab, fontsize=8, loc="left")
axs[0, 0].legend(frameon=False, fontsize=6.5, loc="lower left", ncol=2)
axx = axs[0, 2]
band(axx, bm["box_aniso"], C2, lab=False)
axx.axhline(box_stats["anneal_end_aniso_mean"], color="k", ls=":", lw=0.8)
axx.text(5, 0.42, "dotted: anneal end (mean)", fontsize=6, color=INK2, ha="right")
axx.set_title("c  box shape: max(L)/min(L) - 1", fontsize=8, loc="left")
axx = axs[1, 0]
tt = np.linspace(-6, 6, 300)
for k, col, nm in (("pe", C1, "PE"), ("density", C2, "density"), ("press", C3, "P")):
    axx.hist(per[f"{k}_t20"], bins=np.linspace(-6, 6, 41), histtype="step", density=True, color=col, lw=1.1,
             label=f"{nm} ({np.mean(np.abs(per[f'{k}_t20']) > tc18):.0%})")
axx.plot(tt, stats.t.pdf(tt, 18), color="k", lw=0.8, ls="--", label="t(18) null (5%)")
axx.set_xlabel("Drift t (OLS on 20 sub-block means)"); axx.set_ylabel("Density")
axx.set_ylim(0, 0.62)
axx.legend(frameon=False, fontsize=6, loc="upper left", ncol=2, title=r"share $|t| > t_{0.975}$", title_fontsize=6)
axx.set_title("d  drift tests, 327 networks", fontsize=8, loc="left")
axx = axs[1, 1]
bins = np.linspace(-0.04, 0.04, 33)
axx.hist(per.press_block5_mean, bins=bins, histtype="stepfilled", color=C1, alpha=0.35, density=True, label="P")
dev_all = np.concatenate([per[f"{k}_block5_mean"] for k in ["dev_xy", "dev_yz", "dev_zx"]])
axx.hist(dev_all, bins=bins, histtype="step", color=C2, lw=1.1, density=True, label=r"$P_{xx}-P_{yy}$ etc.")
xx = np.linspace(-0.04, 0.04, 200)
axx.plot(xx, stats.norm.pdf(xx, 0, per.press_block5_se.mean()), color=C1, ls="--", lw=0.8, label="sampling SE, P")
axx.plot(xx, stats.norm.pdf(xx, 0, per.dev_xy_block5_se.mean()), color=C2, ls="--", lw=0.8, label="sampling SE, diff.")
axx.set_ylim(0, 1.45 * axx.get_ylim()[1])
axx.set_xlabel(r"Block-5 time average ($\epsilon\sigma^{-3}$)"); axx.set_ylabel("Density")
axx.legend(frameon=False, fontsize=6, loc="upper left", ncol=2)
axx.set_title(f"e  residual stress (mean UTS = {uts:.2f})", fontsize=8, loc="left")
axx = axs[1, 2]
bins = np.linspace(-0.4, 0.4, 41)
axx.hist(shear.ravel(), bins=bins, density=True, histtype="stepfilled", color=C3, alpha=0.4,
         label=r"$P_{xy},P_{xz},P_{yz}$ (3x327)")
xx = np.linspace(-0.4, 0.4, 300)
axx.plot(xx, stats.norm.pdf(xx, 0, tsd), color="k", lw=0.8, ls="--", label=f"thermal N(0, {tsd:.3f})")
axx.set_ylim(0, 1.35 * axx.get_ylim()[1])
axx.set_xlabel("Shear stress, pull start state"); axx.set_ylabel("Density")
axx.legend(frameon=False, fontsize=6, loc="upper left")
axx.set_title(f"f  shear <x²>/thermal {var_ratio_shear:.2f} [{ci_lo:.2f}, {ci_hi:.2f}]", fontsize=7.2, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figures", "task3_ensemble_summary.png"))
plt.close(fig)
print("representative:", rep_net)
