"""Task 1: lattice imprint on strand orientation.

Inputs: cache/per_network_raw.csv, cache/strand_vectors.npz (extract_per_network.py),
data/scaffold_per_axis_planes.csv (task2_counts.py), mechanics tables (data/derived/mechanics/mechanics_12pull.csv
and four_seed_metrics.csv of four_seed_analysis.py), block_matching_seed1.csv (s1_extract_box.py).
No raw file is read directly. Intermediate files are in NPJ_ANALYSIS_OUT/<folder>/ (see ../paths.py).

Metrics per network (strand = junction-to-junction end-to-end vector, unit u):
  S   = largest |eigenvalue| of Q = <1.5 uu - 0.5 I>   (second rank; blind to cubic order)
  C4  = <ux^4+uy^4+uz^4>   (isotropic 3/5, SC edges 1, body diagonals 1/3); for any distribution
        with axial symmetry about the lattice axes, C4 = 3/5 + (2/5) <P4>, i.e. C4 sees only l=4 content.
  P2mem = <P2(u.e0)>, e0 = the strand's own lattice direction in the generator output
        (direct per-strand directional memory; 0 if no memory).
Nulls: (i) iid isotropic unit vectors with the network's own n (Monte Carlo);
       (ii) rotation null: the same vector set under 400 random rotations (keeps all correlations,
            tests whether the lattice/box axes are special);
       (iii) for P2mem, permutation of the e0 labels among strands (500 shuffles).
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
from common import OUT, MECH12, PULLS

CACHE = os.path.join(OUT, "cache")
raw = pd.read_csv(os.path.join(CACHE, "per_network_raw.csv"))
mech = pd.read_csv(MECH12)[["folder_name", "uts_mean", "toughness_mean", "Quadrant"]]
pulls = pd.read_csv(PULLS)
vz = np.load(os.path.join(CACHE, "strand_vectors.npz"))
rng = np.random.default_rng(20260924)
df = raw.merge(mech, on="folder_name")
N = len(df)

# ------------------------------------------------------------------ iid Monte-Carlo nulls
def mc_null(n, ndraw, chunk):
    C4 = []; S = []
    for k in range(0, ndraw, chunk):
        m = min(chunk, ndraw - k)
        u = rng.normal(size=(m, n, 3)); u /= np.linalg.norm(u, axis=2, keepdims=True)
        C4.append((u ** 4).sum(2).mean(1))
        Qm = 1.5 * np.einsum("rni,rnj->rij", u, u) / n - 0.5 * np.eye(3)
        ev = np.linalg.eigvalsh(Qm)
        S.append(np.abs(ev).max(1))
    return np.concatenate(C4), np.concatenate(S)

null_str = {n: mc_null(n, 20000, 2000) for n in sorted(df.n_strands.unique())}
null_bond = {n: mc_null(n, 1500, 100) for n in sorted(df.n_bonds.unique())}
SD1 = np.sqrt(41 / 105 - 9 / 25)   # analytic single-vector SD of ux^4+uy^4+uz^4
chk_c4 = max(abs(v[0].std() / (SD1 / np.sqrt(n)) - 1) for n, v in null_str.items())
print(f"analytic SD of C4 per vector {SD1:.5f}; max rel. deviation of MC SD from analytic (strands): {chk_c4:.3f}")


def z_iid(vals, ns, table):
    zc, zs, ps = [], [], []
    for v_c4, v_s, n in zip(*vals, ns):
        c, s = table[n]
        zc.append((v_c4 - c.mean()) / c.std()); zs.append((v_s - s.mean()) / s.std())
        ps.append((np.sum(s >= v_s) + 1) / (len(s) + 1))
    return np.array(zc), np.array(zs), np.array(ps)

STATES = ["gen", "min2", "min3", "minsoft", "minfin", "minreal", "nvt", "first", "eq"]
for st in STATES:
    zc, zs, ps = z_iid((df[f"{st}_str_C4"].values, df[f"{st}_str_S"].values), df.n_strands.values, null_str)
    df[f"{st}_str_zC4_iid"] = zc; df[f"{st}_str_zS_iid"] = zs; df[f"{st}_str_pS_iid"] = ps
    df[f"{st}_str_zP2mem_iid"] = df[f"{st}_str_P2mem"] / np.sqrt(0.2 / df.n_strands)
for st in ["first", "eq"]:
    zc, zs, ps = z_iid((df[f"{st}_bond_C4"].values, df[f"{st}_bond_S"].values), df.n_bonds.values, null_bond)
    df[f"{st}_bond_zC4_iid"] = zc; df[f"{st}_bond_zS_iid"] = zs; df[f"{st}_bond_pS_iid"] = ps
    df[f"{st}_str_zC4_rot"] = (df[f"{st}_str_C4"] - df[f"{st}_str_C4_rotmean"]) / df[f"{st}_str_C4_rotsd"]
df["eq_bond_zC4_rot"] = (df["eq_bond_C4"] - df["eq_bond_C4_rotmean"]) / df["eq_bond_C4_rotsd"]

# permutation null for P2mem (e0 labels shuffled among strands)
for st in ["first", "eq"]:
    zp = []; zp_core = []; p2_core = []
    for net in df.folder_name:
        R = vz[f"{st}__{net}"].astype(float); u = R / np.linalg.norm(R, axis=1)[:, None]
        ax = vz[f"axis__{net}"].astype(int)
        c2 = u ** 2                                   # (n,3)
        obs = (1.5 * c2[np.arange(len(ax)), ax] - 0.5).mean()
        perm = np.array([(1.5 * c2[np.arange(len(ax)), rng.permutation(ax)] - 0.5).mean() for _ in range(500)])
        zp.append((obs - perm.mean()) / perm.std())
    df[f"{st}_str_zP2mem_perm"] = zp

# ------------------------------------------------------------------ ensemble summary table
def summarize(col_val, col_z, label, null_note):
    z = df[col_z].values
    v = df[col_val].values
    return dict(metric=label, median=np.median(v), p5=np.quantile(v, 0.05), p95=np.quantile(v, 0.95),
                mean_z=z.mean(), sd_z=z.std(ddof=1), frac_absz_gt2=np.mean(np.abs(z) > 2), frac_z_gt2=np.mean(z > 2),
                wilcoxon_p_meanz0=stats.wilcoxon(z).pvalue, null=null_note)

rows = []
for st, nm in (("gen", "generator output"), ("min3", "after soft min., junctions free"), ("minfin", "after LJ-ramp min."),
               ("nvt", "after 1000 NVT steps"), ("first", "MD start (firstloop, rho 0.11)"), ("eq", "equilibrated (pull start)")):
    rows.append(summarize(f"{st}_str_C4", f"{st}_str_zC4_iid", f"C4 strands, {nm}", "iid"))
    rows.append(summarize(f"{st}_str_S", f"{st}_str_zS_iid", f"S strands, {nm}", "iid"))
    rows.append(summarize(f"{st}_str_P2mem", f"{st}_str_zP2mem_iid", f"P2 memory strands, {nm}", "iid N(0,1/5n)"))
for st, nm in (("first", "MD start"), ("eq", "equilibrated")):
    rows.append(summarize(f"{st}_str_C4", f"{st}_str_zC4_rot", f"C4 strands, {nm}", "rotation"))
    rows.append(summarize(f"{st}_str_P2mem", f"{st}_str_zP2mem_perm", f"P2 memory strands, {nm}", "permutation"))
    rows.append(summarize(f"{st}_bond_C4", f"{st}_bond_zC4_iid", f"C4 bonds, {nm}", "iid"))
    rows.append(summarize(f"{st}_bond_S", f"{st}_bond_zS_iid", f"S bonds, {nm}", "iid"))
rows.append(summarize("eq_bond_C4", "eq_bond_zC4_rot", "C4 bonds, equilibrated", "rotation"))
summ = pd.DataFrame(rows)
# pooled C4 over all strands of all networks (equilibrated)
pooled = {}
for st in ["first", "eq"]:
    allu = np.concatenate([vz[f"{st}__{n}"] / np.linalg.norm(vz[f"{st}__{n}"], axis=1)[:, None] for n in df.folder_name])
    c4 = (allu ** 4).sum(1)
    pooled[st] = dict(n=len(allu), C4=c4.mean(), z=(c4.mean() - 0.6) / (SD1 / np.sqrt(len(allu))))
    ax_all = np.concatenate([vz[f"axis__{n}"] for n in df.folder_name]).astype(int)
    c = allu[np.arange(len(allu)), ax_all]
    pooled[st]["P2mem"] = (1.5 * c ** 2 - 0.5).mean()
    pooled[st]["P4mem"] = ((35 * c ** 4 - 30 * c ** 2 + 3) / 8).mean()
    # memory vs strand extension (eq only)
    if st == "eq":
        Rn = np.concatenate([np.linalg.norm(vz[f"eq__{n}"], axis=1) for n in df.folder_name])
        core = np.concatenate([vz[f"core__{n}"] for n in df.folder_name])
        q = np.quantile(Rn, [0, 0.25, 0.5, 0.75, 1])
        pooled[st]["P2mem_by_R_quartile"] = [float((1.5 * c[(Rn >= q[i]) & (Rn <= q[i + 1])] ** 2 - 0.5).mean()) for i in range(4)]
        pooled[st]["R_quartile_edges"] = [float(v) for v in q]
        pooled[st]["P2mem_core"] = float((1.5 * c[core] ** 2 - 0.5).mean())
        pooled[st]["P2mem_noncore"] = float((1.5 * c[~core] ** 2 - 0.5).mean()) if (~core).any() else np.nan
        pooled[st]["cos_hist"] = np.histogram(c, bins=20, range=(0, 1))[0]
    if st == "first":
        pooled[st]["cos_hist"] = np.histogram(c, bins=20, range=(0, 1))[0]

# ------------------------------------------------------------------ mechanics correlations
# per-axis 4-seed means
pa = pulls.groupby(["folder_name", "axis"])[["uts", "toughness"]].mean().reset_index()
aniso = pa.groupby("folder_name").agg(uts_axis_cv=("uts", lambda s: s.std(ddof=0) / s.mean()),
                                      T_axis_cv=("toughness", lambda s: s.std(ddof=0) / s.mean()),
                                      uts_axis_range=("uts", lambda s: (s.max() - s.min()) / s.mean())).reset_index()
df = df.merge(aniso, on="folder_name")
corr = []
for x, xl in (("eq_str_C4", "C4 strands (eq)"), ("eq_str_zC4_iid", "z C4 strands (eq)"), ("eq_str_S", "S strands (eq)"),
              ("eq_str_P2mem", "P2 memory (eq)"), ("eq_bond_C4", "C4 bonds (eq)"), ("first_str_C4", "C4 strands (MD start)")):
    for y, yl in (("uts_mean", "UTS"), ("toughness_mean", "toughness"), ("uts_axis_cv", "UTS axis-CV"), ("T_axis_cv", "toughness axis-CV")):
        r = stats.spearmanr(df[x], df[y])
        corr.append(dict(x=xl, y=yl, rho=r.statistic, p=r.pvalue))
corr = pd.DataFrame(corr)

# within-network per-axis analysis: does any per-axis structural quantity explain which axis is strongest?
planes = pd.read_csv(os.path.join(OUT, "data", "scaffold_per_axis_planes.csv"))
pl = planes[planes.set == "all"].rename(columns={"n_along": "scaf_n_along", "plane_min": "scaf_plane_min", "plane_cv": "scaf_plane_cv"})
plc = planes[planes.set == "core"].rename(columns={"n_along": "core_n_along", "plane_min": "core_plane_min", "plane_cv": "core_plane_cv"})
bm = pd.read_csv(os.path.join(OUT, "..", "stress_definition", "block_matching_seed1.csv"))[["folder_name", "axis", "L0"]]
axrows = []
for _, r in df.iterrows():
    Lm = (r.eq_Lx * r.eq_Ly * r.eq_Lz) ** (1 / 3)
    for a in "xyz":
        axrows.append(dict(folder_name=r.folder_name, axis=a, eq_Qaa=r[f"eq_str_Q{a}{a}"], eq_Lrel=r[f"eq_L{a}"] / Lm,
                           first_Qaa=r[f"first_str_Q{a}{a}"]))
ax = (pd.DataFrame(axrows).merge(pa, on=["folder_name", "axis"])
      .merge(pl[["folder_name", "axis", "scaf_n_along", "scaf_plane_min", "scaf_plane_cv"]], on=["folder_name", "axis"])
      .merge(plc[["folder_name", "axis", "core_n_along", "core_plane_min"]], on=["folder_name", "axis"])
      .merge(bm, on=["folder_name", "axis"], how="left"))
ax["pull_L0_rel"] = ax.L0 / ax.groupby("folder_name").L0.transform(lambda s: np.exp(np.log(s).mean()))
# box shape at pull start (seed 1) vs equilibrated snapshot
ax.to_csv(os.path.join(OUT, "data", "task1_per_axis.csv"), index=False)


def within_corr(xc, yc, nperm=5000):
    d = ax[[xc, yc, "folder_name"]].dropna().copy()
    for c in (xc, yc):
        d[c] = d[c] - d.groupby("folder_name")[c].transform("mean")
    r = np.corrcoef(d[xc], d[yc])[0, 1]
    # permutation: shuffle axis assignment of x within each network
    g = d.groupby("folder_name").indices
    xv = d[xc].values.copy(); yv = d[yc].values
    null = []
    idx = list(g.values())
    for _ in range(nperm):
        xp = xv.copy()
        for ii in idx:
            xp[ii] = xv[rng.permutation(ii)]
        null.append(np.corrcoef(xp, yv)[0, 1])
    null = np.array(null)
    return r, (np.sum(np.abs(null) >= abs(r)) + 1) / (nperm + 1), len(d)

wc = []
for x, xl in (("eq_Qaa", "Q_aa strands (eq)"), ("first_Qaa", "Q_aa strands (MD start)"), ("scaf_n_along", "scaffold strands along a"),
              ("core_n_along", "2-core strands along a"), ("scaf_plane_min", "min plane count along a"),
              ("core_plane_min", "2-core min plane count along a"), ("eq_Lrel", "box length L_a/L (eq)"),
              ("pull_L0_rel", "box length L_a/L at pull start (seed 1)")):
    for y, yl in (("uts", "UTS_a"), ("toughness", "T_a")):
        r, p, n = within_corr(x, y, nperm=2000)
        wc.append(dict(x=xl, y=yl, r_within=r, p_perm=p, n=n))
# structural links: box shape vs scaffold axis counts, eq box vs pull-start box
for x, y, xl, yl in (("scaf_n_along", "eq_Lrel", "scaffold strands along a", "L_a/L (eq)"),
                     ("scaf_n_along", "eq_Qaa", "scaffold strands along a", "Q_aa (eq)"),
                     ("eq_Lrel", "pull_L0_rel", "L_a/L (eq)", "L_a/L pull start"),
                     ("eq_Lrel", "eq_Qaa", "L_a/L (eq)", "Q_aa (eq)")):
    r, p, n = within_corr(x, y, nperm=2000)
    wc.append(dict(x=xl, y=yl, r_within=r, p_perm=p, n=n))
wc = pd.DataFrame(wc)
# axis effect vs seed noise (is there per-axis anisotropy at all?)
pp = pulls.copy()
cell = pp.groupby(["folder_name", "axis"])
ss_seed = ((pp.uts - cell.uts.transform("mean")) ** 2).sum(); df_seed = len(pp) - pp.groupby(["folder_name", "axis"]).ngroups
netm = pp.groupby("folder_name").uts.transform("mean")
ss_axis = 4 * ((cell.uts.mean() - cell.uts.mean().groupby(level=0).transform("mean")) ** 2).sum(); df_axis = 2 * N
F_uts = (ss_axis / df_axis) / (ss_seed / df_seed); p_F_uts = stats.f.sf(F_uts, df_axis, df_seed)
ss_seedT = ((pp.toughness - cell.toughness.transform("mean")) ** 2).sum()
ss_axisT = 4 * ((cell.toughness.mean() - cell.toughness.mean().groupby(level=0).transform("mean")) ** 2).sum()
F_T = (ss_axisT / df_axis) / (ss_seedT / df_seed); p_F_T = stats.f.sf(F_T, df_axis, df_seed)

# ------------------------------------------------------------------ outputs
keep = ["folder_name", "Quadrant", "uts_mean", "toughness_mean", "n_strands", "n_bonds"]
for st in STATES:
    keep += [f"{st}_str_C4", f"{st}_str_S", f"{st}_str_P2mem", f"{st}_str_zC4_iid", f"{st}_str_zS_iid", f"{st}_Sq100", f"{st}_R_mean"]
keep += ["first_str_zC4_rot", "eq_str_zC4_rot", "first_str_zP2mem_perm", "eq_str_zP2mem_perm", "eq_str_P4mem",
         "first_bond_C4", "first_bond_zC4_iid", "eq_bond_C4", "eq_bond_S", "eq_bond_zC4_iid", "eq_bond_zS_iid", "eq_bond_zC4_rot",
         "eq_core_str_C4", "eq_core_str_P2mem", "eq_Sq110", "eq_Sq111", "first_Sq110", "first_Sq111",
         "eq_Lx", "eq_Ly", "eq_Lz", "uts_axis_cv", "T_axis_cv"]
df[keep].to_csv(os.path.join(OUT, "data", "task1_orientation_per_network.csv"), index=False)
summ.to_csv(os.path.join(OUT, "tables", "task1_orientation_summary.csv"), index=False)
corr.to_csv(os.path.join(OUT, "tables", "task1_mechanics_correlations.csv"), index=False)
wc.to_csv(os.path.join(OUT, "tables", "task1_within_network_axis_correlations.csv"), index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(summ.to_string(float_format=lambda v: f"{v:.4g}"))
print("\npooled:", {k: {kk: (np.round(vv, 5) if np.ndim(vv) == 0 else vv) for kk, vv in v.items() if kk != "cos_hist"} for k, v in pooled.items()})
print("\nstructure factor (junctions) at lattice q, median [5-95%]:")
for st in STATES:
    for f in ("100", "110", "111"):
        v = df[f"{st}_Sq{f}"]
        print(f"  {st:8s} S({f}) {np.median(v):8.3f} [{np.quantile(v, .05):.3f}, {np.quantile(v, .95):.3f}]", end="")
    print()
g95 = {f: np.quantile(rng.gamma(k, 1 / k, 200000), 0.95) for f, k in (("100", 3), ("110", 6), ("111", 4))}
print("  ideal-gas null 95th pct of family-averaged S(q):", g95)
for st in ["first", "eq"]:
    print(f"  {st}: frac S100 > null95 = {np.mean(df[f'{st}_Sq100'] > g95['100']):.3f}")
print("\nmechanics correlations:\n", corr.to_string(float_format=lambda v: f"{v:.3g}"))
print("\nwithin-network per-axis correlations:\n", wc.to_string(float_format=lambda v: f"{v:.3g}"))
print(f"\naxis effect vs seed noise: UTS F({df_axis},{df_seed})={F_uts:.2f} p={p_F_uts:.2g}; toughness F={F_T:.2f} p={p_F_T:.2g}")
print("scaffold fraction of strands along x/y/z: mean", df[["gen_axis_frac_x", "gen_axis_frac_y", "gen_axis_frac_z"]].mean().round(4).tolist(),
      "per-network SD", df[["gen_axis_frac_x", "gen_axis_frac_y", "gen_axis_frac_z"]].std().round(4).tolist())
with open(os.path.join(OUT, "tables", "task1_extra.txt"), "w") as fh:
    fh.write(f"pooled={ {k: {kk: vv for kk, vv in v.items()} for k, v in pooled.items()} }\n")
    fh.write(f"axis effect: UTS F({df_axis},{df_seed})={F_uts:.3f} p={p_F_uts:.3g}; toughness F={F_T:.3f} p={p_F_T:.3g}\n")
    fh.write(f"ideal-gas S(q) null 95th pct: {g95}\n")
    fh.write(f"walk vs min-image mismatches (all states): {int(sum(df[c].sum() for c in df.columns if c.endswith('_n_walk_ne_minimage')))}\n")

# figure: see task1_figure.py (run after task1_followup.py)
