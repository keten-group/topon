"""Task 2: counts and volumes per quadrant.

Graph counts from the deposited gpickle graphs (networkx MultiGraph on the 6x6x6 SC
scaffold; degree-0 nodes = vacancies), cross-checked against (a) mechanics_12pull.csv
cnt_d0..cnt_d6, (b) the folder name, (c) the LAMMPS topology traced in
extract_per_network.py (junction beads, strands, recursive 2-core).
Volume: time average over the final 2e6-step production block (thermo), plus the
equilibrated_network.data snapshot (which equals the last thermo row).
Also writes the per-axis lattice-plane edge counts used by task 1.
Inputs: cache/per_network_raw.csv and cache/thermo.npz (extract_per_network.py), the deposited graphs
data/mechanics/<net>/*.gpickle and data/derived/mechanics/mechanics_12pull.csv. No raw file is read directly.
Outputs go to data/ and tables/ in NPJ_ANALYSIS_OUT/cg_structure/.
"""
import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from common import OUT, GRAPHS, MECH12

CACHE = os.path.join(OUT, "cache")
raw = pd.read_csv(os.path.join(CACHE, "per_network_raw.csv"))
assert "error" not in raw.columns or raw["error"].isna().all(), raw.loc[raw["error"].notna(), ["folder_name", "error"]]
mech = pd.read_csv(MECH12)
th = np.load(os.path.join(CACHE, "thermo.npz"))
COLS = ["step", "temp", "pe", "ke", "etotal", "press", "pxx", "pyy", "pzz", "lx", "ly", "lz", "density"]
ci = {c: i for i, c in enumerate(COLS)}


def peel2(G):
    H = nx.MultiGraph(G)
    H.remove_nodes_from([n for n, d in H.degree() if d == 0])
    while True:
        low = [n for n, d in H.degree() if d < 2]
        if not low:
            return H
        H.remove_nodes_from(low)


rows = []; axrows = []
for net in raw["folder_name"]:
    gp = glob.glob(os.path.join(GRAPHS, net, "*.gpickle"))
    assert len(gp) == 1, (net, gp)
    with open(gp[0], "rb") as fh:
        G = pickle.load(fh)
    assert nx.number_of_selfloops(G) == 0
    deg = dict(G.degree())
    attr_ok = all(int(G.nodes[n]["degree"]) == deg[n] for n in G)
    cnt = [sum(1 for v in deg.values() if v == f) for f in range(7)]
    core = peel2(G)
    cdeg = dict(core.degree())
    r = dict(folder_name=net, g_nodes=G.number_of_nodes(), g_attr_degree_ok=attr_ok,
             vacancies=cnt[0], n_f1=cnt[1], n_f2=cnt[2], n_f3plus=sum(cnt[3:]),
             **{f"g_cnt_d{f}": cnt[f] for f in range(7)},
             strands=G.number_of_edges(), core_strands=core.number_of_edges(),
             core_junctions_f3plus=sum(1 for v in cdeg.values() if v >= 3),
             core_nodes=core.number_of_nodes(),
             mean_degree_occupied=2 * G.number_of_edges() / (G.number_of_nodes() - cnt[0]))
    # folder-name check: vacancies_f1_f2_..._f6
    fn = [int(v) for v in net.split("_")]
    r["foldername_ok"] = fn == cnt
    rows.append(r)
    # per-axis lattice plane counts (scaffold): edges along axis a crossing each of the 6 planes
    for a, an in enumerate("xyz"):
        for tag, H in (("all", G), ("core", core)):
            planes = np.zeros(6, int)
            for u, v, k in H.edges(keys=True):
                cu = np.array(u, float); cv = np.array(v, float)
                d = (cv - cu) % 6
                other = [i for i in range(3) if i != a]
                if d[a] in (1, 5) and np.all(np.minimum(d[other], 6 - d[other]) == 0):
                    lo = int(cu[a]) if d[a] == 1 else int(cv[a])
                    planes[lo] += 1
            axrows.append(dict(folder_name=net, axis=an, set=tag, n_along=int(planes.sum()),
                               plane_min=int(planes.min()), plane_max=int(planes.max()),
                               plane_cv=float(planes.std() / planes.mean()) if planes.mean() > 0 else np.nan))

g = pd.DataFrame(rows)
ax = pd.DataFrame(axrows)
df = raw.merge(g, on="folder_name").merge(mech[["folder_name", "cnt_d0", "cnt_d1", "cnt_d2", "cnt_d3", "cnt_d4", "cnt_d5", "cnt_d6",
                                                "uts_mean", "toughness_mean", "Quadrant"]], on="folder_name")
# ---- cross-checks -------------------------------------------------------------
chk = {}
chk["gpickle degree attr == graph degree"] = bool(df["g_attr_degree_ok"].all())
chk["gpickle counts == mechanics_12pull cnt_d*"] = bool(all((df[f"g_cnt_d{f}"] == df[f"cnt_d{f}"]).all() for f in range(7)))
chk["gpickle counts == folder name"] = bool(df["foldername_ok"].all())
chk["216 - vacancies == LAMMPS junction beads"] = bool((216 - df["vacancies"] == df["n_junction_beads"]).all())
chk["gpickle degrees (f=1..6) == LAMMPS junction degrees"] = bool(all((df[f"g_cnt_d{f}"] == df[f"lmp_cnt_d{f}"]).all() for f in range(1, 7)))
chk["gpickle edges == LAMMPS strands"] = bool((df["strands"] == df["n_strands"]).all())
chk["gpickle 2-core edges == LAMMPS 2-core strands"] = bool((df["core_strands"] == df["n_core_strands"]).all())
chk["beads == junction beads + 30*strands"] = bool((df["n_atoms"] == df["n_junction_beads"] + 30 * df["n_strands"]).all())
chk["all strands have 30 beads"] = bool(((df["beads_per_strand_min"] == 30) & (df["beads_per_strand_max"] == 30)).all())
chk["data-file box == last thermo row box"] = bool(np.allclose(df[["eq_Lx", "eq_Ly", "eq_Lz"]].values,
                                                              df[["logfinal_lx", "logfinal_ly", "logfinal_lz"]].values, rtol=1e-6))
chk["Ext_05 slurm output == log.lammps"] = bool(df["thermo_ok_log"].all())
chk["restart continuity (max rel diff)"] = float(df["thermo_cont_maxrel"].max())
chk["scaffold per-axis edge counts sum to strands"] = bool(
    (ax[ax.set == "all"].groupby("folder_name")["n_along"].sum().reindex(df.folder_name).values == df["strands"].values).all())

# ---- volumes -----------------------------------------------------------------
nets = list(th["nets"]); assert nets == list(df["folder_name"])
prod = th["prod"].astype(np.float64)       # (327, 2005, 13); block k = rows 401k..401k+400
last = prod[:, 4 * 401 + 1: 5 * 401, :]    # final block, first row (duplicate of previous block end) dropped
V = last[..., ci["lx"]] * last[..., ci["ly"]] * last[..., ci["lz"]]
df["V_lastblock_mean"] = V.mean(1)
df["V_lastblock_sd"] = V.std(1)
df["density_lastblock_mean_log"] = last[..., ci["density"]].mean(1)
df["V_snapshot"] = df["eq_Lx"] * df["eq_Ly"] * df["eq_Lz"]
df["density_snapshot"] = df["n_atoms"] / df["V_snapshot"]
df["number_density"] = df["n_atoms"] / df["V_lastblock_mean"]
chk["N/<V> vs <N/V> (log density), max rel diff"] = float(np.max(np.abs(df["number_density"] / df["density_lastblock_mean_log"] - 1)))
chk["snapshot density == logged final density"] = bool(np.allclose(df["density_snapshot"], df["logfinal_density"], rtol=1e-6))
df["junction_density_f3plus"] = df["n_f3plus"] / df["V_lastblock_mean"]
df["core_strand_density"] = df["core_strands"] / df["V_lastblock_mean"]
df["strand_density"] = df["strands"] / df["V_lastblock_mean"]
df["core_junction_density_f3plus"] = df["core_junctions_f3plus"] / df["V_lastblock_mean"]
df["beads"] = df["n_atoms"]

keep = ["folder_name", "Quadrant", "uts_mean", "toughness_mean", "vacancies", "n_f1", "n_f2", "n_f3plus", "strands",
        "core_strands", "core_junctions_f3plus", "mean_degree_occupied", "beads", "V_lastblock_mean", "V_lastblock_sd",
        "V_snapshot", "number_density", "density_snapshot", "junction_density_f3plus", "core_junction_density_f3plus",
        "strand_density", "core_strand_density"]
df[keep].to_csv(os.path.join(OUT, "data", "task2_counts_per_network.csv"), index=False)
ax.to_csv(os.path.join(OUT, "data", "scaffold_per_axis_planes.csv"), index=False)

# ---- per-quadrant table ----------------------------------------------------------
VARS = [("vacancies", "Vacancies (f=0 sites)", "{:.1f}"), ("n_f1", "Dangling ends (f=1)", "{:.1f}"),
        ("n_f2", "Chain extenders (f=2)", "{:.1f}"), ("n_f3plus", "Junctions (f>=3)", "{:.1f}"),
        ("strands", "Strands (edges)", "{:.1f}"), ("core_strands", "2-core strands", "{:.1f}"),
        ("core_junctions_f3plus", "2-core junctions (f>=3 in core)", "{:.1f}"),
        ("mean_degree_occupied", "Mean degree (occupied sites)", "{:.3f}"),
        ("beads", "Beads", "{:.0f}"), ("V_lastblock_mean", "Volume <V> (sigma^3)", "{:.0f}"),
        ("number_density", "Bead density (sigma^-3)", "{:.4f}"),
        ("junction_density_f3plus", "Junction density f>=3 (1e-3 sigma^-3)", "{:.3f}"),
        ("core_strand_density", "2-core strand density (1e-3 sigma^-3)", "{:.2f}")]
scale = {"junction_density_f3plus": 1e3, "core_strand_density": 1e3}
qs = ["Q1: Weak/Brittle", "Q2: Weak/Tough", "Q3: Strong/Brittle", "Q4: Strong/Tough"]
tab = []
for v, lab, fmt in VARS:
    x = df[v] * scale.get(v, 1.0)
    row = {"variable": lab}
    for q in qs:
        s = x[df.Quadrant == q]
        row[f"{q[:2]}_mean"] = s.mean(); row[f"{q[:2]}_sd"] = s.std(); row[f"{q[:2]}_min"] = s.min(); row[f"{q[:2]}_max"] = s.max()
        row[f"{q[:2]}_n"] = len(s)
    row["KW_p"] = stats.kruskal(*[x[df.Quadrant == q] for q in qs]).pvalue
    for m, mn in (("uts_mean", "UTS"), ("toughness_mean", "T")):
        rr = stats.spearmanr(x, df[m])
        row[f"rho_{mn}"] = rr.statistic; row[f"p_{mn}"] = rr.pvalue
    row["_fmt"] = fmt
    tab.append(row)
tab = pd.DataFrame(tab)
tab.drop(columns="_fmt").to_csv(os.path.join(OUT, "tables", "task2_quadrant_table.csv"), index=False)


def pfmt(p):
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.0e}"


md = ["| Quantity | Q1 weak/brittle (n=128) | Q2 weak/tough (n=35) | Q3 strong/brittle (n=35) | Q4 strong/tough (n=129) | KW p | rho(UTS) | rho(T) |",
      "|---|---|---|---|---|---|---|---|"]
for _, r in tab.iterrows():
    f = r["_fmt"]
    cells = []
    for q in ("Q1", "Q2", "Q3", "Q4"):
        cells.append(f"{f.format(r[q + '_mean'])} ± {f.format(r[q + '_sd'])} [{f.format(r[q + '_min'])}–{f.format(r[q + '_max'])}]")
    md.append(f"| {r['variable']} | " + " | ".join(cells) +
              f" | {pfmt(r['KW_p'])} | {r['rho_UTS']:+.2f} ({pfmt(r['p_UTS'])}) | {r['rho_T']:+.2f} ({pfmt(r['p_T'])}) |")
md_txt = "\n".join(md)
with open(os.path.join(OUT, "tables", "task2_quadrant_table.md"), "w", encoding="utf-8") as fh:
    fh.write("Task 2. Per-quadrant counts and volumes (mean ± SD [min–max]); Kruskal–Wallis p across the four quadrants; "
             "Spearman rho (p) with the 12-pull network-mean UTS and toughness (n=327).\n"
             "Counts from the deposited graphs (cross-checked against the LAMMPS topology); <V> is the time average over the "
             "final 2e6-step NPT block (P=0, T=1). Junction density uses f>=3 sites; 2-core = recursive removal of f<2 nodes.\n\n")
    fh.write(md_txt + "\n")
with open(os.path.join(OUT, "tables", "task2_crosschecks.txt"), "w") as fh:
    for k, v in chk.items():
        fh.write(f"{k}: {v}\n")
print("\n".join(f"{k}: {v}" for k, v in chk.items()))
print(md_txt)
# a few extra numbers for the report
print("\nensemble ranges:")
for v in ["vacancies", "n_f1", "n_f2", "n_f3plus", "strands", "core_strands", "beads", "V_lastblock_mean", "number_density",
          "junction_density_f3plus", "core_strand_density", "mean_degree_occupied"]:
    print(f"  {v}: mean {df[v].mean():.4g} sd {df[v].std():.3g} min {df[v].min():.4g} max {df[v].max():.4g} CV {df[v].std()/df[v].mean():.3%}")
# how much does the volume vary beyond bead count? V per bead
vpb = df["V_lastblock_mean"] / df["beads"]
print(f"  V/bead: mean {vpb.mean():.4f} CV {vpb.std()/vpb.mean():.3%}; rho(V/bead, UTS)={stats.spearmanr(vpb, df.uts_mean).statistic:+.2f}")
print("  mean relative SD of V within final block (thermal):", (df.V_lastblock_sd / df.V_lastblock_mean).mean())
