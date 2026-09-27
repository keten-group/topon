#!/usr/bin/env python3
"""
Reproduce manuscript Figure 5 ("Dynamic glass transition analysis",
file Figure_4_Tg) with the three independent cooling histories averaged.

Identical to generate_figures.ipynb code cell 4 in style, colours, markers,
vertical offsets and fitting routine (scipy curve_fit of a continuous
piecewise-linear function from starting guesses of 150 K / 200 K). Only the
data change:
  * points      = MSD at the chosen lag, averaged over the three histories
  * error bars  = standard deviation across the three histories
  * fit line    = bilinear fit to the averaged curve
  * Tg label    = Tg of the averaged curve +/- SD of the three per-history Tg

Usage:  python make_figure5_replicates.py          (4 ps, the manuscript lag)
        TG_LAG=3 python make_figure5_replicates.py (any other lag, 1-10 ps)

Inputs: msd.dat of the three cooling histories, deposited in data/raw/tg_cooling.tar.xz and read as
RAW_ROOT/<PDMS|PMTFPS>/DP<n>/<original|replicate2|replicate3>/msd.dat (see ../paths.py), and the
plot style data/npj_style_v1.py of this companion. The figures go to NPJ_ANALYSIS_OUT/tg/.
"""
import os
import sys
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))    # raw_data_analyses/, for paths.py
from paths import DATA, RAW_ROOT, TG_HISTORY, TG_LABEL, out_dir  # noqa: E402
OUT = str(out_dir("tg"))
STYLE = str(DATA)
sys.path.append(STYLE)
from npj_style_v1 import set_npj_style  # noqa: E402

LAG = int(os.environ.get("TG_LAG", 4))
CHEMS = {"PDMS": ("PDMS", 150), "PMTFPS": ("FPDMS", 200)}   # label -> (folder, guess)
DPS = ["DP_10", "DP_30", "DP_100"]
HISTS = ("orig", "rep2", "rep3")


# --- the manuscript's own fitting routine, verbatim --------------------------
def piecewise_linear(T, Tg, m1, m2, c1):
    condlist = [T < Tg, T >= Tg]
    funclist = [lambda x: m1 * x + c1,
                lambda x: m2 * (x - Tg) + (m1 * Tg + c1)]
    return np.piecewise(T, condlist, funclist)


def get_optimized_bilinear_tg(temp, msd, guess_tg=150):
    try:
        p0 = [guess_tg, 0.005, 0.02, np.min(msd)]
        bounds = ([np.min(temp) + 5, -np.inf, -np.inf, -np.inf],
                  [np.max(temp) - 5, np.inf, np.inf, np.inf])
        popt, pcov = curve_fit(piecewise_linear, temp, msd, p0=p0,
                               bounds=bounds, maxfev=10000)
        return popt[0], (popt, pcov)
    except Exception:
        return np.nan, None


# --- data --------------------------------------------------------------------
def msd_path(folder, dp, hist):
    # deposited layout. The original folders were <folder>/<dp>/06_TgSimulation/msd.dat
    # and replicate<k>_data/<folder>_<DPn>_msd.dat (FPDMS = PMTFPS).
    return os.path.join(RAW_ROOT, TG_LABEL[folder], dp.replace('_', ''), TG_HISTORY[hist], "msd.dat")


def load_lag(path, lag):
    """MSD at `lag` ps per temperature: mean over the 10 ps windows."""
    blocks = collections.OrderedDict()
    with open(path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            q = line.split()
            if len(q) < 8:
                continue
            blocks.setdefault(int(float(q[0])), []).append([float(x) for x in q])
    out = {}
    for t, rows in blocks.items():
        nw = len(rows) // 11
        wins = [rows[i * 11:(i + 1) * 11] for i in range(nw)]
        if len(wins) >= 10:          # a temperature run twice keeps all windows
            out[t] = np.mean([w[lag][2] for w in wins])
    return out


res = {}
for label, (folder, guess) in CHEMS.items():
    for dp in DPS:
        per = [load_lag(msd_path(folder, dp, h), LAG) for h in HISTS]
        Ts = np.array(sorted(set.intersection(*[set(p) for p in per])), float)
        M = np.array([[p[t] for t in Ts] for p in per])             # (3, nT)
        tg_i = [get_optimized_bilinear_tg(Ts, m, guess)[0] for m in M]
        mean, sd = M.mean(0), M.std(0, ddof=1)
        tg_avg, params = get_optimized_bilinear_tg(Ts, mean, guess)
        fit_err = float(np.sqrt(np.diag(params[1]))[0]) if params else np.nan
        res[(label, dp)] = dict(T=Ts, mean=mean, sd=sd, params=params,
                                tg_avg=tg_avg, tg_i=tg_i,
                                tg_sd=float(np.std(tg_i, ddof=1)), fit_err=fit_err)


# --- figure: same geometry and styling as the manuscript ---------------------
width, _ = set_npj_style(column_type='double')
height = width * 0.45
plt.rcParams['figure.constrained_layout.use'] = True
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height), constrained_layout=True)
colors = {'DP_10': '#E69F00', 'DP_30': '#CC79A7', 'DP_100': '#0072B2'}
markers = {'DP_10': 'o', 'DP_30': 's', 'DP_100': '^'}


def plot_tg_panel(ax, chem):
    ax.text(0.05, 0.9, chem, transform=ax.transAxes, fontsize=8)
    for i, dp in enumerate(DPS):
        r = res[(chem, dp)]
        off = i * 1.0
        dpv = dp.split('_')[1]
        lab = fr'$DP={dpv}$' + (f" (+{off} $\\mathrm{{\\AA}}^2$)" if i > 0 else "")
        ax.errorbar(r['T'], r['mean'] + off, yerr=r['sd'], marker=markers[dp],
                    linestyle='None', color=colors[dp], alpha=0.6, markersize=3.5,
                    elinewidth=0.6, capsize=0, label=lab)
        if r['params'] is None:
            continue
        popt = r['params'][0]
        tf = np.linspace(r['T'].min(), r['T'].max(), 200)
        ax.plot(tf, piecewise_linear(tf, *popt) + off, '-', color=colors[dp],
                linewidth=1.2, alpha=0.8)
        tg = r['tg_avg']
        yt = piecewise_linear(tg, *popt) + off
        ax.vlines(tg, yt - 0.5, yt, color=colors[dp], linestyle=':',
                  linewidth=0.8, alpha=0.7)
        ax.text(tg + 5, yt - 0.35, fr"$T_g={tg:.1f}\pm{r['tg_sd']:.1f}$ K",
                color=colors[dp], fontsize=7,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    ax.set_xlabel('Temperature (K)')
    ax.set_xlim(95, 310)
    ax.yaxis.set_tick_params(left=True, labelleft=False)
    ax.set_ylim(bottom=-0.5, top=4.5)


plot_tg_panel(ax1, 'PDMS')
ax1.set_ylabel(fr'MSD @ {LAG}ps ($\mathrm{{\AA}}^2$)')
ax1.text(0.02, 0.02, 'Data vertically shifted for clarity', transform=ax1.transAxes,
         fontsize=7, fontstyle='italic', color='grey')
ax1.legend(frameon=False, loc='lower right', fontsize=7)
plot_tg_panel(ax2, 'PMTFPS')
ax2.text(0.02, 0.02, r'Mean $\pm$ SD of 3 independent cooling histories',
         transform=ax2.transAxes, fontsize=7, fontstyle='italic', color='grey')


def align_label_to_ylabel(ax, text):
    fig.canvas.draw()
    bb = ax.yaxis.label.get_window_extent(renderer=fig.canvas.get_renderer())
    ab = ax.get_window_extent()
    ax.text(((bb.x0 + bb.x1) / 2 - ab.x0) / ab.width, 1.0, text,
            transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='bottom', ha='center')


align_label_to_ylabel(ax1, 'a')
ax2.text(-0.05, 1.0, 'b', transform=ax2.transAxes, fontsize=9,
         fontweight='bold', va='bottom', ha='right')

# every lag is written to the tg output folder
outdirs = [OUT]
stem = "Figure_4_Tg_replicates" + ("" if LAG == 4 else f"_{LAG}ps")
for od in outdirs:
    os.makedirs(od, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(od, f"{stem}.{ext}"), dpi=600)

print(f"lag {LAG} ps: Tg of averaged curve | per-history Tg | SD across histories | curve_fit error")
for chem in CHEMS:
    for dp in DPS:
        r = res[(chem, dp)]
        hist = " ".join(f"{v:6.1f}" for v in r['tg_i'])
        print(f"{chem:7s}{dp:7s} avg {r['tg_avg']:6.1f} | {hist} | "
              f"SD {r['tg_sd']:5.1f} | fit err {r['fit_err']:4.1f} | "
              f"max point SD {r['sd'].max():.3f}")
for od in outdirs:
    print("written:", os.path.join(od, stem + ".pdf"))
