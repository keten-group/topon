"""Task 1 follow-up checks (run after task1_orientation.py and task2_counts.py).

1. Are the network-level correlations of the orientation measures (C4, P2 memory) with UTS/toughness
   independent of degree composition? Partial Spearman correlations controlling for n_f1, n_f2, n_f3plus.
2. What drives the per-strand memory? Pooled strand-level regression of P2(u.e0) on the degrees of
   the two end junctions and the strand's 2-core membership.
3. Within-network per-axis model: UTS_a and T_a on the 2-core minimum lattice-plane count, the
   equilibrated box length L_a/L and Q_aa jointly (network fixed effects, i.e. centred per network).
Inputs: the data/ and cache/ files of extract_per_network.py, task1_orientation.py and task2_counts.py,
data/derived/mechanics/mechanics_12pull.csv, and equilibrated_network.data of every network
(RAW_ROOT/crosslinker/sc_6x6x6/<net>/minimize_equilibrate/, not deposited).
Outputs go to cache/ and tables/ in NPJ_ANALYSIS_OUT/cg_structure/.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from common import OUT, read_lammps_data, me_path, strands_from_bonds, JUNCTION_TYPE

t1 = pd.read_csv(os.path.join(OUT, "data", "task1_orientation_per_network.csv"))
t2 = pd.read_csv(os.path.join(OUT, "data", "task2_counts_per_network.csv"))
ax = pd.read_csv(os.path.join(OUT, "data", "task1_per_axis.csv"))
vz = np.load(os.path.join(OUT, "cache", "strand_vectors.npz"))
df = t1.merge(t2[["folder_name", "n_f1", "n_f2", "n_f3plus", "junction_density_f3plus", "core_strands"]], on="folder_name")
out = []


def partial_spearman(x, y, Z):
    """Spearman partial correlation: Pearson on rank residuals."""
    R = lambda v: stats.rankdata(v)
    Zr = np.column_stack([np.ones(len(x))] + [R(z) for z in Z])
    rx = R(x) - Zr @ np.linalg.lstsq(Zr, R(x), rcond=None)[0]
    ry = R(y) - Zr @ np.linalg.lstsq(Zr, R(y), rcond=None)[0]
    r = np.corrcoef(rx, ry)[0, 1]
    n, k = len(x), len(Z)
    t = r * np.sqrt((n - 2 - k) / (1 - r ** 2))
    return r, 2 * stats.t.sf(abs(t), n - 2 - k)


Z = [df.n_f1.values, df.n_f2.values, df.n_f3plus.values]
out.append("1. network-level: raw vs partial Spearman (controls: n_f1, n_f2, n_f3plus)")
for x in ["eq_str_C4", "eq_str_P2mem", "eq_str_S", "eq_R_mean"]:
    for y in ["uts_mean", "toughness_mean"]:
        r0 = stats.spearmanr(df[x], df[y])
        rp, pp = partial_spearman(df[x].values, df[y].values, Z)
        out.append(f"   {x:14s} vs {y:15s}: raw rho {r0.statistic:+.3f} (p={r0.pvalue:.1g})   partial {rp:+.3f} (p={pp:.2g})")
from common import MECH12
m12 = pd.read_csv(MECH12)[["folder_name"] + [f"cnt_d{f}" for f in range(1, 7)] + ["deg_std"]]
df = df.merge(m12, on="folder_name")
Zfull = [df[f"cnt_d{f}"].values for f in range(1, 7)]
out.append("   --- controls: full degree counts cnt_d1..cnt_d6 (vacancies implied); and deg_std alone")
for x in ["eq_str_C4", "eq_str_P2mem"]:
    for y in ["uts_mean", "toughness_mean"]:
        rp, pp = partial_spearman(df[x].values, df[y].values, Zfull)
        rq, pq = partial_spearman(df[x].values, df[y].values, [df.deg_std.values])
        out.append(f"   {x:14s} vs {y:15s}: partial|cnt_d1..6 {rp:+.3f} (p={pp:.2g});  partial|deg_std {rq:+.3f} (p={pq:.2g})")
for x in ["eq_str_C4", "eq_str_P2mem"]:
    for z in ["n_f1", "n_f2", "n_f3plus", "eq_R_mean", "deg_std", "cnt_d6"]:
        r0 = stats.spearmanr(df[x], df[z])
        out.append(f"   rho({x}, {z}) = {r0.statistic:+.3f} (p={r0.pvalue:.1g})")

# 2. strand-level drivers of memory (eq); recompute end degrees from the LAMMPS topology
rows = []
for net in df.folder_name:
    d = read_lammps_data(me_path(net, "equilibrated_network.data"))
    st, _ = strands_from_bonds(d)
    deg = {}
    for a, b, _ in st:
        deg[a] = deg.get(a, 0) + 1; deg[b] = deg.get(b, 0) + 1
    R = vz[f"eq__{net}"].astype(float); u = R / np.linalg.norm(R, axis=1)[:, None]
    axis = vz[f"axis__{net}"].astype(int); core = vz[f"core__{net}"]
    c = np.abs(u[np.arange(len(u)), axis])
    for k, (a, b, _) in enumerate(st):
        rows.append((net, min(deg[a], deg[b]), max(deg[a], deg[b]), bool(core[k]), 1.5 * c[k] ** 2 - 0.5, np.linalg.norm(R[k])))
S = pd.DataFrame(rows, columns=["net", "fmin", "fmax", "core", "P2", "R"])
S.to_csv(os.path.join(OUT, "cache", "strand_level_memory_eq.csv"), index=False)
out.append("\n2. per-strand memory <P2(u.e0)> (equilibrated) by end-junction degrees (pooled, n strands)")
g = S.groupby(["fmin", "fmax"]).agg(n=("P2", "size"), P2=("P2", "mean"), R=("R", "mean")).reset_index()
for _, r in g[g.n >= 200].iterrows():
    out.append(f"   f_min={int(r.fmin)} f_max={int(r.fmax)}: n={int(r.n):6d}  <P2>={r.P2:+.3f}  <|R|>={r.R:.2f}")
out.append(f"   2-core strands <P2>={S[S.core].P2.mean():+.3f} (n={S.core.sum()}); non-core <P2>={S[~S.core].P2.mean():+.3f} (n={(~S.core).sum()})")
out.append(f"   Spearman(|R|, P2) pooled = {stats.spearmanr(S.R, S.P2).statistic:+.3f}")

# 3. within-network per-axis joint model
d = ax.copy()
cols = ["core_plane_min", "scaf_plane_min", "eq_Lrel", "eq_Qaa", "core_n_along", "uts", "toughness", "pull_L0_rel"]
for c in cols:
    d[c + "_c"] = d[c] - d.groupby("folder_name")[c].transform("mean")
out.append("\n3. within-network per-axis OLS (network fixed effects; standardized betas; n=981 cells, 327 networks)")
for y in ["uts", "toughness"]:
    for xs in (["core_plane_min"], ["core_plane_min", "eq_Lrel"], ["core_plane_min", "eq_Lrel", "eq_Qaa"],
               ["core_plane_min", "core_n_along", "eq_Lrel", "eq_Qaa"], ["eq_Qaa"], ["pull_L0_rel"], ["core_plane_min", "pull_L0_rel"]):
        dd = d.dropna(subset=[x + "_c" for x in xs] + [y + "_c"])
        X = np.column_stack([dd[x + "_c"] / dd[x + "_c"].std() for x in xs])
        Y = dd[y + "_c"].values / dd[y + "_c"].std()
        beta, res, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta
        dof = len(Y) - X.shape[1] - 327          # fixed effects consume one df per network
        s2 = (resid ** 2).sum() / dof
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        r2 = 1 - (resid ** 2).sum() / (Y ** 2).sum()
        terms = "  ".join(f"{x}={b:+.3f}(t={b / s:+.1f})" for x, b, s in zip(xs, beta, se))
        out.append(f"   {y:9s} R2_within={r2:.3f}  {terms}")
txt = "\n".join(out)
print(txt)
with open(os.path.join(OUT, "tables", "task1_followup.txt"), "w") as fh:
    fh.write(txt + "\n")
