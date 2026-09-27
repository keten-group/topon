"""Task 1 figure (run after task1_orientation.py and task1_followup.py).

Reads only files in NPJ_ANALYSIS_OUT/cg_structure/ (no raw data) and writes figures/task1_orientation.png there.
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
from common import OUT

df = pd.read_csv(os.path.join(OUT, "data", "task1_orientation_per_network.csv"))
ax = pd.read_csv(os.path.join(OUT, "data", "task1_per_axis.csv"))
S = pd.read_csv(os.path.join(OUT, "cache", "strand_level_memory_eq.csv"))
vz = np.load(os.path.join(OUT, "cache", "strand_vectors.npz"))
corr = pd.read_csv(os.path.join(OUT, "tables", "task1_mechanics_correlations.csv"))
wc = pd.read_csv(os.path.join(OUT, "tables", "task1_within_network_axis_correlations.csv"))

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "savefig.dpi": 300})
C1, C2, C3, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK2 = "#52514e"
fig, axs = plt.subplots(2, 3, figsize=(7.4, 5.3))

# (a) build progression
a = axs[0, 0]
lab = ["gen", "min2", "min3", "minsoft", "minfin", "minreal", "nvt", "first", "eq"]
short = ["generator", "min2", "min3", "soft min", "LJ min", "FENE min", "NVT", "MD start", "equilibrated"]
xs = np.arange(len(lab))
for col, key, nm in ((C1, "str_C4", r"$C_4=\langle u_x^4+u_y^4+u_z^4\rangle$"), (C2, "str_P2mem", r"memory $\langle P_2(u\cdot e_0)\rangle$")):
    v = np.array([df[f"{s}_{key}"].values for s in lab])
    a.fill_between(xs, np.quantile(v, .05, 1), np.quantile(v, .95, 1), color=col, alpha=0.2, lw=0)
    a.plot(xs, np.median(v, 1), color=col, marker="o", ms=3, lw=1.2, label=nm)
a.axhline(0.6, color=C1, ls=":", lw=0.8); a.axhline(0, color=C2, ls=":", lw=0.8)
a.text(0.1, 0.63, "isotropic 3/5", fontsize=6, color=C1)
a.text(0.1, 0.03, "no memory", fontsize=6, color=C2)
a.set_xticks(xs); a.set_xticklabels(short, rotation=55, ha="right", fontsize=6.3)
a.set_ylabel("Strands: median, 5-95% of networks")
a.set_ylim(-0.05, 1.12)
a.legend(frameon=False, fontsize=6.2, loc="lower left", bbox_to_anchor=(0.0, 0.12))
a.set_title("a  orientation through the build", fontsize=8, loc="left")

# (b) z-scores vs iid isotropic null (equilibrated)
b = axs[0, 1]
bins = np.linspace(-4, 6, 41); zz = np.linspace(-4, 4, 200)
for col, cz, nm, side in ((C1, "eq_str_zC4_iid", r"$C_4$ strands", "gt"), (C3, "eq_str_zS_iid", r"$S$ strands", "gt"),
                          (C5, "eq_bond_zC4_iid", r"$C_4$ bonds", "abs")):
    frac = np.mean(df[cz] > 2) if side == "gt" else np.mean(np.abs(df[cz]) > 2)
    b.hist(df[cz], bins=bins, density=True, histtype="step", color=col, lw=1.1,
           label=f"{nm} ({frac:.0%} z>2)" if side == "gt" else f"{nm} ({frac:.0%} |z|>2)")
b.plot(zz, stats.norm.pdf(zz), "k--", lw=0.8, label="N(0,1)")
b.set_ylim(0, 0.75)
b.set_xlabel("z vs isotropic null (same n), equilibrated"); b.set_ylabel("Density")
b.legend(frameon=False, fontsize=6, loc="upper left")
b.set_title("b  cubic and second-rank order", fontsize=8, loc="left")

# (c) per-strand memory, pooled
c = axs[0, 2]
edges = np.linspace(0, 1, 21); mid = 0.5 * (edges[1:] + edges[:-1])
for st, col, nm in (("first", C2, "MD start"), ("eq", C1, "equilibrated")):
    cc = []
    for n in df.folder_name:
        R = vz[f"{st}__{n}"].astype(float); u = R / np.linalg.norm(R, axis=1)[:, None]
        cc.append(np.abs(u[np.arange(len(u)), vz[f"axis__{n}"].astype(int)]))
    cc = np.concatenate(cc)
    h, _ = np.histogram(cc, bins=edges, density=True)
    c.step(mid, h, where="mid", color=col, lw=1.2, label=f"{nm}: " + r"$\langle P_2\rangle$" + f"={np.mean(1.5 * cc ** 2 - 0.5):.2f}")
c.axhline(1, color="k", ls="--", lw=0.8, label="no memory (uniform)")
c.set_xlabel(r"$|u\cdot e_0|$, strand vs its scaffold direction"); c.set_ylabel("Probability density (all strands)")
c.legend(frameon=False, fontsize=6.2, loc="upper left")
c.set_title("c  per-strand directional memory", fontsize=8, loc="left")

# (d) memory vs end-junction degrees
d = axs[1, 0]
g = S.groupby(["fmin", "fmax"]).agg(n=("P2", "size"), P2=("P2", "mean"), sd=("P2", "std")).reset_index()
cols = {1: "0.55", 2: C3, 3: C1, 4: C5, 5: C2, 6: "k"}
for fmin in range(1, 7):
    gg = g[(g.fmin == fmin) & (g.n >= 200)]
    if len(gg):
        d.errorbar(gg.fmax, gg.P2, yerr=gg.sd / np.sqrt(gg.n), color=cols[fmin], marker="o", ms=3, lw=1, capsize=0,
                   label=f"lower-degree end f={fmin}")
d.axhline(0, color="k", ls=":", lw=0.8)
d.set_xlabel("Degree of the higher-degree end junction"); d.set_ylabel(r"$\langle P_2(u\cdot e_0)\rangle$, equilibrated")
d.legend(frameon=False, fontsize=5.8, loc="upper left")
d.set_title("d  memory is set by junction pinning", fontsize=8, loc="left")

# (e) C4 vs UTS
e = axs[1, 1]
e.scatter(df.eq_str_C4, df.uts_mean, s=6, color=C1, alpha=0.6, lw=0)
r = corr[(corr.x == "C4 strands (eq)") & (corr.y == "UTS")].iloc[0]
e.axvline(0.6, color="k", ls=":", lw=0.8)
e.set_xlabel(r"$C_4$ strands, equilibrated"); e.set_ylabel("Network-mean UTS (12 pulls)")
e.set_title(f"e  rho = {r.rho:+.2f}; partial|degree counts -0.16", fontsize=7.2, loc="left")

# (f) within-network per-axis: 2-core weakest lattice plane vs UTS_a
f = axs[1, 2]
dd = ax.copy()
for col in ("core_plane_min", "uts", "eq_Qaa"):
    dd[col + "_c"] = dd[col] - dd.groupby("folder_name")[col].transform("mean")
jit = np.random.default_rng(1).uniform(-0.25, 0.25, len(dd))
f.scatter(dd.core_plane_min_c + jit, dd.uts_c, s=4, color=C5, alpha=0.45, lw=0)
w1 = wc[(wc.x == "2-core min plane count along a") & (wc.y == "UTS_a")].iloc[0]
w2 = wc[(wc.x == "Q_aa strands (eq)") & (wc.y == "UTS_a")].iloc[0]
f.set_xlabel("2-core strands crossing the weakest\nlattice plane normal to a (centred)")
f.set_ylabel(r"UTS$_a$, 4-seed mean (centred)")
f.set_title(f"f  per-axis: r = {w1.r_within:+.2f}; " + r"$Q_{aa}$" + f": r = {w2.r_within:+.2f}", fontsize=7.2, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figures", "task1_orientation.png"))
plt.close(fig)
print("saved")
