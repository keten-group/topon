"""One read-only pass over the raw CG data (327 networks, 2 worker processes).

Per network it computes
  * strand (junction-to-junction) end-to-end vectors in every stage of the build/
    equilibration pipeline, by walking bonds with the minimum image:
      gen      nodes.data                     generator output (straight strands, L=23.4)
      min2     min_2.data                     box x2, soft minimisation, junctions frozen
      min3     min_3.data                     soft minimisation, junctions free
      minsoft  after_minimization_soft.data
      minfin   after_final_minimization.data  (LJ ramp, harmonic bonds)
      minreal  after_minimization_real2.data  (LJ 2.5 + FENE minimisation)
      nvt      after_nvt_real.data            (1000 NVT steps)
      first    firstloop.data                 MD start of the anneal (dilute, rho~0.11)
      eq       equilibrated_network.data      start state of every tensile pull
    -> order tensor Q, S=max|eig|, C4=<ux^4+uy^4+uz^4>, <u_a^2>, memory of the
       original lattice direction <P2(u.e0)>, <P4(u.e0)>, mean |R|.
  * bond unit vectors (first, eq): C4, S.
  * rotation null for eq/first strands and eq bonds: C4 of the same vector set
    after 400 random rotations (keeps all correlations; tests whether the
    lattice/box axes are special).
  * junction structure factor at the lattice reciprocal vectors ([100],[110],[111]).
  * junction-level topology (degrees, 2-core) for the count table.
  * snapshot virial stress tensor of equilibrated_network.data (validated against
    the last thermo row of log.lammps) incl. the unlogged shear components.
  * thermo time series of anneal + 5 production blocks (slurm-*.out, verified).
Inputs (not deposited): the LAMMPS data files above, slurm-*.out and log.lammps in
RAW_ROOT/crosslinker/sc_6x6x6/<net>/minimize_equilibrate/ (see common.py).
Outputs go to cache/ in NPJ_ANALYSIS_OUT/cg_structure/.
"""
import os
import sys
import time
import numpy as np
import networkx as nx
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
from common import (networks, me_path, read_lammps_data, strands_from_bonds, min_image,
                    OUT, JUNCTION_TYPE)
from thermo_parse import load_chain

STATES = [("gen", "nodes.data"), ("min2", "min_2.data"), ("min3", "min_3.data"),
          ("minsoft", "after_minimization_soft.data"), ("minfin", "after_final_minimization.data"),
          ("minreal", "after_minimization_real2.data"), ("nvt", "after_nvt_real.data"),
          ("first", "firstloop.data"), ("eq", "equilibrated_network.data")]
NROT = 400
RC = 2.5


def rand_rotations(n, rng):
    q = rng.normal(size=(n, 4)); q /= np.linalg.norm(q, axis=1)[:, None]
    w, x, y, z = q.T
    R = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)
    return R


def orient_stats(u):
    """u: (n,3) unit vectors."""
    Q = 1.5 * np.einsum("ni,nj->ij", u, u) / len(u) - 0.5 * np.eye(3)
    ev = np.linalg.eigvalsh(Q)
    S = ev[np.argmax(np.abs(ev))]
    C4 = float((u ** 4).sum(1).mean())
    return dict(S=float(abs(S)), S_signed=float(S), C4=C4,
                u2x=float((u[:, 0] ** 2).mean()), u2y=float((u[:, 1] ** 2).mean()),
                u2z=float((u[:, 2] ** 2).mean()),
                Qxx=float(Q[0, 0]), Qyy=float(Q[1, 1]), Qzz=float(Q[2, 2]),
                Qxy=float(Q[0, 1]), Qxz=float(Q[0, 2]), Qyz=float(Q[1, 2]))


def rot_null_C4(u, Rs, chunk=50):
    out = []
    for k in range(0, len(Rs), chunk):
        ur = np.einsum("rij,nj->rni", Rs[k:k + chunk], u)
        out.append((ur ** 4).sum(2).mean(1))
    out = np.concatenate(out)
    return float(out.mean()), float(out.std(ddof=1))


def sq_lattice(xj, L):
    """|sum exp(i q.r)|^2 / N at q = 2*pi*6*(h/Lx,k/Ly,l/Lz) for the {100},{110},{111} families."""
    fam = {"100": [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
           "110": [(1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)],
           "111": [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)]}
    N = len(xj)
    res = {}
    for f, hkl in fam.items():
        vals = []
        for h in hkl:
            q = 2 * np.pi * 6 * np.array(h) / L
            ph = xj @ q
            vals.append((np.cos(ph).sum() ** 2 + np.sin(ph).sum() ** 2) / N)
        res[f] = float(np.mean(vals))
        if f == "100":
            res["100x"], res["100y"], res["100z"] = vals
    return res


def virial_snapshot(d):
    from scipy.spatial import cKDTree
    L = d["L"]; x = (d["x"] - d["lo"]) % L; N = len(x)
    tree = cKDTree(x, boxsize=L)
    P = np.sort(tree.query_pairs(RC, output_type="ndarray"), axis=1)
    b = np.sort(d["bonds"][:, 1:3] - 1, axis=1)
    key = lambda a: a[:, 0].astype(np.int64) * N + a[:, 1]
    P = P[~np.isin(key(P), key(b))]
    r = min_image(x[P[:, 1]] - x[P[:, 0]], L); r2 = (r * r).sum(1)
    ir6 = 1 / r2 ** 3
    W = np.einsum("i,ij,ik->jk", (48 * ir6 * ir6 - 24 * ir6) / r2, r, r)
    epair = (4 * (ir6 * ir6 - ir6)).sum()
    rb = min_image(x[b[:, 1]] - x[b[:, 0]], L); rb2 = (rb * rb).sum(1)
    K, R0 = 30.0, 1.5
    fb = -K / (1 - rb2 / R0 ** 2)
    eb = -0.5 * K * R0 ** 2 * np.log(1 - rb2 / R0 ** 2)
    w = rb2 < 2 ** (1 / 3)
    i6 = 1 / rb2[w] ** 3
    fb[w] += (48 * i6 * i6 - 24 * i6) / rb2[w]
    eb[w] += 4 * (i6 * i6 - i6) + 1
    W += np.einsum("i,ij,ik->jk", fb, rb, rb)
    v = d["v"]; KE = np.einsum("ij,ik->jk", v, v)
    V = np.prod(L)
    Pt = (KE + W) / V
    temp = (v * v).sum() / (3 * N - 3)
    press = (temp * (3 * N - 3) + np.trace(W)) / (3 * V)
    return dict(pxx=Pt[0, 0], pyy=Pt[1, 1], pzz=Pt[2, 2], pxy=Pt[0, 1], pxz=Pt[0, 2], pyz=Pt[1, 2],
                press=press, temp=temp, pe=(epair + eb.sum()) / N,
                bond_mean=float(np.sqrt(rb2).mean()), bond_max=float(np.sqrt(rb2).max()))


def work(net):
    t0 = time.time()
    import zlib; rng = np.random.default_rng(zlib.crc32(net.encode()))
    row = {"folder_name": net}
    deq = read_lammps_data(me_path(net, "equilibrated_network.data"), want_vel=True)
    strands, _ = strands_from_bonds(deq)
    ns = len(strands)
    A = np.array([a for a, _, _ in strands]); B = np.array([b for _, b, _ in strands])
    bkey_ref = np.sort(np.sort(deq["bonds"][:, 1:3], axis=1).view("i8,i8"), axis=0)
    jidx = np.where(deq["type"] == JUNCTION_TYPE)[0]
    row.update(n_atoms=len(deq["ids"]), n_junction_beads=len(jidx), n_strands=ns,
               n_bonds=len(deq["bonds"]),
               beads_per_strand_min=min(len(p) - 2 for _, _, p in strands),
               beads_per_strand_max=max(len(p) - 2 for _, _, p in strands))
    # junction graph
    G = nx.MultiGraph(); G.add_nodes_from(jidx)
    for k, (a, b, _) in enumerate(strands):
        G.add_edge(a, b, key=k)
    deg = dict(G.degree())
    row["n_selfloop_strands"] = int(sum(1 for a, b, _ in strands if a == b))
    for f in range(0, 7):
        row[f"lmp_cnt_d{f}"] = int(sum(1 for v in deg.values() if v == f))
    row["lmp_deg_max"] = int(max(deg.values()))
    # recursive 2-core (peel degree<2 nodes until none remain; degree counts multi-edges)
    core = nx.MultiGraph([(a, b) for a, b, _ in strands if a != b])
    while True:
        low = [n for n, dg in core.degree() if dg < 2]
        if not low:
            break
        core.remove_nodes_from(low)
    core_nodes = set(core.nodes())
    in_core = np.array([(a in core_nodes and b in core_nodes) for a, b, _ in strands])
    # recursive 2-core on multigraph: k_core uses degree incl. multiplicity; verify edge count
    row["n_core_strands"] = int(core.number_of_edges())
    row["n_core_strands_check"] = int(in_core.sum())
    row["n_core_nodes"] = len(core_nodes)
    row["n_core_nodes_f3plus"] = int(sum(1 for n in core_nodes if core.degree(n) >= 3))
    row["n_junction_f3plus"] = int(sum(1 for v in deg.values() if v >= 3))
    vecs = {}
    e0 = None
    for tag, fname in STATES:
        d = deq if tag == "eq" else read_lammps_data(me_path(net, fname), want_vel=False)
        assert np.array_equal(d["type"], deq["type"])
        bk = np.sort(np.sort(d["bonds"][:, 1:3], axis=1).view("i8,i8"), axis=0)
        assert np.array_equal(bk, bkey_ref), f"{net} {tag}: bond topology differs"
        L = d["L"]; x = d["x"]
        R = np.zeros((ns, 3)); maxb = 0.0
        for k, (_, _, p) in enumerate(strands):
            dd = min_image(np.diff(x[p], axis=0), L)
            maxb = max(maxb, float(np.sqrt((dd ** 2).sum(1)).max()))
            R[k] = dd.sum(0)
        Rn = np.linalg.norm(R, axis=1)
        u = R / Rn[:, None]
        # straight minimum-image junction difference (sanity: equals walked vector if no strand wraps)
        Rmi = min_image(x[B] - x[A], L)
        row[f"{tag}_n_walk_ne_minimage"] = int((np.abs(R - Rmi).max(1) > 1e-6).sum())
        if tag == "gen":
            ax = np.argmax(np.abs(u), axis=1)
            e0 = np.eye(3)[ax]
            row["gen_axis_frac_x"], row["gen_axis_frac_y"], row["gen_axis_frac_z"] = [float((ax == i).mean()) for i in range(3)]
            row["gen_max_offaxis"] = float(np.sort(np.abs(u), axis=1)[:, :2].max())
        c = np.abs((u * e0).sum(1))
        st = orient_stats(u)
        for k2, v in st.items():
            row[f"{tag}_str_{k2}"] = v
        row[f"{tag}_str_P2mem"] = float((1.5 * c ** 2 - 0.5).mean())
        row[f"{tag}_str_P4mem"] = float(((35 * c ** 4 - 30 * c ** 2 + 3) / 8).mean())
        row[f"{tag}_R_mean"] = float(Rn.mean()); row[f"{tag}_R_rms"] = float(np.sqrt((Rn ** 2).mean()))
        row[f"{tag}_R_cv"] = float(Rn.std() / Rn.mean())
        row[f"{tag}_maxbond"] = maxb
        row[f"{tag}_Lx"], row[f"{tag}_Ly"], row[f"{tag}_Lz"] = [float(v) for v in L]
        row[f"{tag}_density"] = len(x) / float(np.prod(L))
        sq = sq_lattice(x[jidx] - d["lo"], L)
        for k2, v in sq.items():
            row[f"{tag}_Sq{k2}"] = v
        if tag in ("first", "eq"):
            # 2-core strands only
            stc = orient_stats(u[in_core])
            row[f"{tag}_core_str_C4"] = stc["C4"]; row[f"{tag}_core_str_S"] = stc["S"]
            row[f"{tag}_core_n"] = int(in_core.sum())
            cc = c[in_core]
            row[f"{tag}_core_str_P2mem"] = float((1.5 * cc ** 2 - 0.5).mean())
            # bonds
            bv = min_image(x[d["bonds"][:, 2] - 1] - x[d["bonds"][:, 1] - 1], L)
            bu = bv / np.linalg.norm(bv, axis=1)[:, None]
            sb = orient_stats(bu)
            for k2, v in sb.items():
                row[f"{tag}_bond_{k2}"] = v
            row[f"{tag}_n_bonds"] = len(bu)
            Rs = rand_rotations(NROT, rng)
            m, s = rot_null_C4(u, Rs)
            row[f"{tag}_str_C4_rotmean"], row[f"{tag}_str_C4_rotsd"] = m, s
            if tag == "eq":
                m, s = rot_null_C4(bu, Rs[:100])
                row["eq_bond_C4_rotmean"], row["eq_bond_C4_rotsd"] = m, s
            vecs[tag] = R.astype(np.float32)
    # snapshot stress (eq)
    vs = virial_snapshot(deq)
    for k2, v in vs.items():
        row[f"snap_{k2}"] = float(v)
    # thermo
    ch = load_chain(net)
    row["thermo_ok_log"] = bool(ch["ok_log"]); row["thermo_cont_maxrel"] = ch["cont_maxrel"]
    row["thermo_n_slurm"] = ch["n_slurm"]; row["thermo_job_anneal"] = ch["jobs"]["anneal"]
    last = ch["prod"][-1][-1]
    for j, cname in enumerate(["step", "temp", "pe", "ke", "etotal", "press", "pxx", "pyy", "pzz", "lx", "ly", "lz", "density"]):
        row[f"logfinal_{cname}"] = float(last[j])
    thermo = dict(anneal=ch["anneal"].astype(np.float64), prod=np.vstack(ch["prod"]).astype(np.float64),
                  prod_len=np.array([len(p) for p in ch["prod"]]))
    row["t_sec"] = time.time() - t0
    return row, vecs, thermo, e0.argmax(1).astype(np.int8), in_core


def safe_work(net):
    try:
        return work(net)
    except Exception as e:  # keep going; report at the end
        import traceback
        return ({"folder_name": net, "error": repr(e) + traceback.format_exc()[-400:]}, None, None, None, None)


def main():
    import pandas as pd
    nets = networks()
    if len(sys.argv) > 1:
        nets = nets[: int(sys.argv[1])]
    rows = []; vec_eq = {}; vec_first = {}; th = {}; axes = {}; cores = {}
    with Pool(2) as pool:
        for k, (row, vecs, thermo, ax, inc) in enumerate(pool.imap(safe_work, nets, chunksize=2)):
            rows.append(row)
            n = row["folder_name"]
            if vecs is None:
                print("ERROR", n, row["error"], flush=True)
                continue
            vec_eq[n] = vecs["eq"]; vec_first[n] = vecs["first"]; axes[n] = ax; cores[n] = inc
            th[n] = thermo
            if k % 25 == 0:
                print(k, n, "%.1fs" % row["t_sec"], flush=True)
    df = pd.DataFrame(rows)
    cache = os.path.join(OUT, "cache")
    df.to_csv(os.path.join(cache, "per_network_raw.csv"), index=False)
    np.savez_compressed(os.path.join(cache, "strand_vectors.npz"),
                        nets=np.array(list(vec_eq)),
                        **{f"eq__{n}": v for n, v in vec_eq.items()},
                        **{f"first__{n}": v for n, v in vec_first.items()},
                        **{f"axis__{n}": v for n, v in axes.items()},
                        **{f"core__{n}": v for n, v in cores.items()})
    # thermo: full production (2005 rows) + anneal (821 rows), float32 to keep it small
    def pad(arrs):
        m = max(len(a) for a in arrs)
        out = np.full((len(arrs), m, arrs[0].shape[1]), np.nan, np.float32)
        for i, a in enumerate(arrs):
            out[i, :len(a)] = a
        return out
    np.savez_compressed(os.path.join(cache, "thermo.npz"),
                        nets=np.array(list(th)),
                        prod=pad([th[n]["prod"] for n in th]),
                        anneal=pad([th[n]["anneal"] for n in th]),
                        anneal_len=np.array([len(th[n]["anneal"]) for n in th]),
                        prod_len=np.stack([th[n]["prod_len"] for n in th]))
    print("done", len(df))


if __name__ == "__main__":
    main()
