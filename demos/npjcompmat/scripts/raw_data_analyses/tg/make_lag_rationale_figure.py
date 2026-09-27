#!/usr/bin/env python3
"""
Why 4 ps? Three panels built from the DP100 data (three histories averaged):
  a  MSD versus time at several temperatures: the cage is reached within ~1 ps,
     after which the atom only rattles; the MSD at 4 ps measures that rattling.
  b  local exponent dlnMSD/dlnt: its minimum marks the caging plateau, and
     4 ps sits inside it for both chemistries.
  c  Tg versus probe window: PMTFPS barely moves, PDMS slides down, exactly the
     rate dependence every experimental Tg method shows.

Inputs: the DP 100 msd.dat files of the three cooling histories, deposited in data/raw/tg_cooling.tar.xz and
read as RAW_ROOT/<PDMS|PMTFPS>/DP100/<original|replicate2|replicate3>/msd.dat (see ../paths.py), and the plot
style data/npj_style_v1.py of this companion. The figure goes to NPJ_ANALYSIS_OUT/tg/.
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
sys.path.append(str(DATA))
from npj_style_v1 import set_npj_style  # noqa: E402

HISTS = ("orig", "rep2", "rep3")
DP = "DP_100"
CHEM = {"PDMS": ("PDMS", 150, "#0173B2"), "PMTFPS": ("FPDMS", 200, "#DE8F05")}


def piecewise_linear(T, Tg, m1, m2, c1):
    return np.piecewise(T, [T < Tg, T >= Tg],
                        [lambda x: m1 * x + c1,
                         lambda x: m2 * (x - Tg) + (m1 * Tg + c1)])


def fit_tg(temp, msd, guess):
    try:
        p0 = [guess, 0.005, 0.02, np.min(msd)]
        bounds = ([np.min(temp) + 5, -np.inf, -np.inf, -np.inf],
                  [np.max(temp) - 5, np.inf, np.inf, np.inf])
        popt, _ = curve_fit(piecewise_linear, temp, msd, p0=p0, bounds=bounds, maxfev=10000)
        return popt[0]
    except Exception:
        return np.nan


def msd_path(folder, hist):
    # deposited layout. The original folders were <folder>/DP_100/06_TgSimulation/msd.dat
    # and replicate<k>_data/<folder>_DP100_msd.dat (FPDMS = PMTFPS).
    return os.path.join(RAW_ROOT, TG_LABEL[folder], DP.replace("_", ""), TG_HISTORY[hist], "msd.dat")


def load_curves(path):
    """{T: MSD at lags 0..10 ps}, averaged over the 10 ps windows."""
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
        if len(wins) >= 10:
            out[t] = np.array([np.mean([w[l][2] for w in wins]) for l in range(11)])
    return out


data = {}
for label, (folder, guess, col) in CHEM.items():
    per = [load_curves(msd_path(folder, h)) for h in HISTS]
    Ts = sorted(set.intersection(*[set(p) for p in per]))
    data[label] = dict(T=np.array(Ts, float),
                       per=[np.array([p[t] for t in Ts]) for p in per])   # 3 x (nT, 11)
    data[label]["avg"] = np.mean(data[label]["per"], axis=0)

t = np.arange(11, dtype=float)                     # lags 0..10 ps
tm = np.sqrt(t[1:-1] * t[2:])                      # geometric mid-lags 1.4..9.5
LAGS = list(range(1, 10))

width, _ = set_npj_style(column_type='double')
fig, (a, b, c) = plt.subplots(1, 3, figsize=(width, width * 0.36), constrained_layout=True)

# ---- a: the cage, seen directly ---------------------------------------------
d = data["PDMS"]
show = [(110, "#08519C", "110 K"), (150, "#8C8C8C", "150 K ≈ Tg"),
        (220, "#E6550D", "220 K"), (300, "#A63603", "300 K")]
for T, col, lab in show:
    i = list(d["T"]).index(T)
    a.plot(t, d["avg"][i], color=col, lw=1.3, marker="o", ms=2.6)
    a.annotate(lab, (10, d["avg"][i][-1]), textcoords="offset points",
               xytext=(4, 0), color=col, fontsize=6.5, va="center")
a.axvspan(2.5, 4.5, color="#bdbdbd", alpha=0.25, lw=0)
a.axvline(4, color="#555555", lw=0.7, ls=(0, (3, 2)))
a.text(0.25, 2.2, "fast: the atom hits its\ncage within ~1 ps", fontsize=6.5,
       color="#333333", ha="left", va="top")
a.text(5.0, 1.03, "then only slow creep\ninside the cage", fontsize=6.5,
       color="#333333", ha="left", va="center")
a.set_xlim(0, 13.2)
a.set_ylim(0, 2.25)
a.set_xlabel("time (ps)")
a.set_ylabel(r"MSD ($\mathrm{\AA}^2$)")
a.set_title("PDMS, DP 100", fontsize=7.5, loc="left")

# ---- b: where the rattling regime is -----------------------------------------
for label, (folder, guess, col) in CHEM.items():
    d = data[label]
    for T, ls in ((110, (0, (3, 2))), (150 if label == "PDMS" else 190, "-")):
        i = list(d["T"]).index(T)
        m = d["avg"][i]
        alpha = np.diff(np.log(m[1:])) / np.diff(np.log(t[1:]))
        b.plot(tm, alpha, color=col, lw=1.2, ls=ls, marker="o", ms=2.4)
b.axvspan(2.5, 4.5, color="#bdbdbd", alpha=0.25, lw=0)
b.axvline(4, color="#555555", lw=0.7, ls=(0, (3, 2)))
b.text(3.5, 0.012, "plateau: slowest growth", ha="center", va="bottom",
       fontsize=6.5, color="#333333")
b.plot([], [], color="#0173B2", lw=1.2, label="PDMS")
b.plot([], [], color="#DE8F05", lw=1.2, label="PMTFPS")
b.plot([], [], color="#666666", lw=1.2, label="at its Tg")
b.plot([], [], color="#666666", lw=1.2, ls=(0, (3, 2)), label="deep glass, 110 K")
b.legend(frameon=False, fontsize=6, loc="upper right", ncol=1)
b.set_xlabel("time (ps)")
b.set_ylabel(r"growth rate  d ln MSD / d ln $t$")
b.set_xlim(1, 10)
b.set_ylim(0.0, 0.47)

# ---- c: Tg depends on the probe window, like every experimental Tg -----------
for label, (folder, guess, col) in CHEM.items():
    d = data[label]
    mean, sd = [], []
    for L in LAGS:
        tg_h = [fit_tg(d["T"], p[:, L], guess) for p in d["per"]]
        mean.append(fit_tg(d["T"], d["avg"][:, L], guess))
        sd.append(np.nanstd(tg_h, ddof=1))
    mean, sd = np.array(mean), np.array(sd)
    c.fill_between(LAGS, mean - sd, mean + sd, color=col, alpha=0.18, lw=0)
    c.plot(LAGS, mean, color=col, lw=1.4, marker="o", ms=2.8)
    c.annotate(label, (LAGS[-1], mean[-1]), textcoords="offset points",
               xytext=(4, 0), color=col, fontsize=6.5, va="center")
    print(label, "Tg vs lag:", " ".join(f"{L}ps {m:.1f}+/-{s:.1f}"
                                        for L, m, s in zip(LAGS, mean, sd)))
c.axvspan(2.5, 4.5, color="#bdbdbd", alpha=0.25, lw=0)
c.axvline(4, color="#555555", lw=0.7, ls=(0, (3, 2)))
c.text(4.15, 206, "4 ps", fontsize=6.5, color="#333333")
c.set_xlabel("probe window (ps)")
c.set_ylabel(r"apparent $T_g$ (K)")
c.set_xlim(0.6, 10.6)
c.set_ylim(115, 212)

for ax, lab in zip((a, b, c), "abc"):
    ax.text(-0.02, 1.02, lab, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="right")

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"Tg_lag_rationale.{ext}"), dpi=600)
print("written:", os.path.join(OUT, "Tg_lag_rationale.png"))
