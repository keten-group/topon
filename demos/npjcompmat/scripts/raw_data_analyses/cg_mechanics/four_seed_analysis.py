"""Add replicate 4 (seed 9184735) to the cube and quantify what the 4th seed buys.

Same metric definitions as three_seed_analysis.py (UTS = max stress; strain-at-break = first
post-peak point < 5% UTS; toughness = trapz to sab).  Writes four_seed_metrics.csv,
four_seed_cube.npz (327 x 3 axes x 4 seeds), four_seed_network_summary.csv, and prints:
  - variance decomposition with m = 4 and the measured k = 12 reliability vs k = 9
  - quadrant stability: 12-pull vs 9-pull reference; seed-to-seed disagreement over 6 pairs
  - seed-4 as an INDEPENDENT replication of the toughening descriptors (partial r per seed)
  - headline Cohen's d and partial r at k = 9 vs k = 12

Inputs: three_seed_metrics.csv, three_seed_network_summary.csv and three_seed_cube.npz from
three_seed_analysis.py; stress_strain_npt_{x,y,z}.dat of seed 4, deposited in data/raw/
(cg_stress_strain_seed4.tar.xz, unpacked to RAW_ROOT/seed4/<folder_name>/); the deposited data/dataset.pkl.
The descriptor section also reads maxflow_loadpaths.csv, D_core_all.csv, simple_slice_metrics.csv and
tearing_metrics.csv. These graph-descriptor tables come from a separate analysis and are not deposited.
They are read from the output folder, and the cube and metrics files are written before they are needed.
All files are read from and written to NPJ_ANALYSIS_OUT/cg_mechanics/ (see ../paths.py).
"""
import itertools, pickle, numpy as np, pandas as pd
import sys
from pathlib import Path
from scipy import stats
from scipy.stats import ttest_ind

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))         # raw_data_analyses/, for paths.py
from paths import DATA, RAW_ROOT, out_dir
OUT = out_dir('cg_mechanics')                # three_seed_* inputs, four_seed_* outputs, descriptor tables
trapz = getattr(np, 'trapezoid', None) or np.trapz
SEED4 = RAW_ROOT / 'seed4'; AX = 'xyz'


def metrics(fp):
    d = pd.read_csv(fp, sep=r'\s+', comment='#', names=['t', 'strain', 'stress'])
    i = d.stress.idxmax(); uts = d.stress[i]; post = d.loc[i:]
    br = post[post.stress < 0.05 * uts]; sab = br.iloc[0].strain if len(br) else d.iloc[-1].strain
    v = d[d.strain <= sab]; return uts, sab, trapz(v.stress, v.strain)


L3 = pd.read_csv(OUT / 'three_seed_metrics.csv'); nets = list(pd.read_csv(OUT / 'three_seed_network_summary.csv').folder_name)
cache = OUT / 'four_seed_metrics.csv'
if cache.exists():
    L = pd.read_csv(cache)
else:
    rows = []
    for fn in nets:
        for ax in AX:
            fp = SEED4 / fn / f'stress_strain_npt_{ax}.dat'
            u, s, t = metrics(fp); rows.append(dict(folder_name=fn, seed=4, axis=ax, uts=u, sab=s, toughness=t))
    L = pd.concat([L3, pd.DataFrame(rows)], ignore_index=True); L.to_csv(cache, index=False)
print(f'pulls: {len(L)}  seeds: {sorted(L.seed.unique())}')
cube = {}
for p in ('uts', 'sab', 'toughness'):
    a = np.full((len(nets), 3, 4), np.nan); piv = L.pivot_table(index='folder_name', columns=['axis', 'seed'], values=p)
    for ia, ax in enumerate(AX):
        for js, sd in enumerate((1, 2, 3, 4)):
            a[:, ia, js] = piv[(ax, sd)].reindex(nets).to_numpy()
    assert not np.isnan(a).any(); cube[p] = a
np.savez_compressed(OUT / 'four_seed_cube.npz', uts=cube['uts'], toughness=cube['toughness'], sab=cube['sab'], nets=np.array(nets))
z3 = np.load(OUT / 'three_seed_cube.npz', allow_pickle=True)
assert np.allclose(z3['uts'], cube['uts'][:, :, :3]) and np.allclose(z3['toughness'], cube['toughness'][:, :, :3])
print('seeds 1-3 identical to three_seed_cube.npz: yes')
# seed 4 sanity vs the other seeds
for p in ('uts', 'toughness'):
    m = cube[p].mean((0, 1)); print(f"  {p}: per-seed grand mean {np.round(m, 4)}   seed-4 vs mean(1-3) t-test p = {ttest_ind(cube[p][:, :, 3].ravel(), cube[p][:, :, :3].ravel()).pvalue:.2f}")


def decompose(Y):
    n, k, m = Y.shape; gm = Y.mean(); net = Y.mean((1, 2))[:, None, None]; axm = Y.mean((0, 2))[None, :, None]; cell = Y.mean(2)[:, :, None]
    SS_net = k * m * ((net - gm) ** 2).sum(); SS_ax = n * m * ((axm - gm) ** 2).sum(); SS_int = m * ((cell - net - axm + gm) ** 2).sum(); SS_err = ((Y - cell) ** 2).sum()
    MS_net, MS_ax, MS_int, MS_err = SS_net / (n - 1), SS_ax / (k - 1), SS_int / ((n - 1) * (k - 1)), SS_err / (n * k * (m - 1))
    return dict(v_net=max((MS_net - MS_int) / (k * m), 0), v_ax=max((MS_ax - MS_int) / (n * m), 0), v_int=max((MS_int - MS_err) / m, 0), v_err=MS_err)


def rel(d, k, m): return d['v_net'] / (d['v_net'] + d['v_int'] / k + d['v_err'] / (k * m))


print('\n=== VARIANCE DECOMPOSITION and RELIABILITY: 3 seeds vs 4 seeds ===')
for p in ('uts', 'toughness'):
    d3 = decompose(cube[p][:, :, :3]); d4 = decompose(cube[p])
    tot4 = sum(d4.values())
    print(f"{p:10s} m=4: network {100*d4['v_net']/tot4:.1f}%  axis {100*d4['v_ax']/tot4:.1f}%  network x axis {100*d4['v_int']/tot4:.1f}%  seed noise {100*d4['v_err']/tot4:.1f}%")
    print(f"{'':10s} reliability of the network mean:  k=9 (3 seeds, measured) {rel(d3,3,3):.3f}   k=12 (4 seeds, measured) {rel(d4,3,4):.3f}   "
          f"[3-seed extrapolation to k=12 was {rel(d3,3,4):.3f}]   k=1 pull {rel(d4,1,1):.3f}")


def quadrants(u, t): return 1 + (t >= np.median(t)).astype(int) + 2 * (u >= np.median(u)).astype(int)


u9, t9 = cube['uts'][:, :, :3].mean((1, 2)), cube['toughness'][:, :, :3].mean((1, 2))
u12, t12 = cube['uts'].mean((1, 2)), cube['toughness'].mean((1, 2))
Q9, Q12 = quadrants(u9, t9), quadrants(u12, t12)
print('\n=== QUADRANT STABILITY ===')
print(f"  networks changing quadrant when seed 4 is added (9-pull -> 12-pull reference): {100*(Q9 != Q12).mean():.1f}%  ({int((Q9 != Q12).sum())} of {len(nets)})")
pair = [(quadrants(cube['uts'][:, :, a].mean(1), cube['toughness'][:, :, a].mean(1)) != quadrants(cube['uts'][:, :, b].mean(1), cube['toughness'][:, :, b].mean(1))).mean() for a, b in itertools.combinations(range(4), 2)]
print(f"  two independent seeds (3 axes each) disagree on {100*np.mean(pair):.1f}% of networks (6 pairs, range {100*min(pair):.1f}-{100*max(pair):.1f}%)")
Q4 = quadrants(cube['uts'][:, :, 3].mean(1), cube['toughness'][:, :, 3].mean(1))
print(f"  seed 4 alone vs the 9-pull reference: {100*(Q4 != Q9).mean():.1f}% differ")

# ---------------------------------------------------------------- descriptors: seed-4 replication and k=9 vs k=12
with open(DATA / 'dataset.pkl', 'rb') as f:
    M = pickle.load(f)['mechanics'].copy().reindex(nets)
MF = pd.read_csv(OUT / 'maxflow_loadpaths.csv', index_col=0).reindex(nets); DC = pd.read_csv(OUT / 'D_core_all.csv', index_col=0).reindex(nets)
SS = pd.read_csv(OUT / 'simple_slice_metrics.csv', index_col=0).reindex(nets); TE = pd.read_csv(OUT / 'tearing_metrics.csv', index_col=0).reindex(nets)
dc = [f'cnt_d{i}' for i in range(7)]; tot = M[dc].sum(axis=1)
from scipy.stats import skew, kurtosis
DEG = pd.DataFrame({f'frac_d{i}': M[f'cnt_d{i}'] / tot for i in range(7)}, index=M.index)
for c in ('deg_mean', 'deg_std', 'deg_skew', 'deg_entropy'):
    DEG[c] = M[c]
DEG['deg_bimodality'] = M.apply(lambda r: ((skew(sum([[int(c.split('_d')[1])] * int(r[c]) for c in dc], [])) ** 2 + 1) / (kurtosis(sum([[int(c.split('_d')[1])] * int(r[c]) for c in dc], []), fisher=True) + 3)), axis=1)
Xdeg = np.column_stack([np.ones(len(nets)), DEG.to_numpy(float)])
resid = lambda y, X: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def par(x, y, zc):
    Z = np.column_stack([np.ones(len(y)), zc]); return stats.pearsonr(resid(x, Z), resid(y, Z))[0]


def cohen_d(a, b):
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)); return (b.mean() - a.mean()) / sp


feats = {'load paths G/n (raw)': MF.maxflow_frac.to_numpy(float), 'load paths | degree': resid(MF.maxflow_frac.to_numpy(float), Xdeg),
         'D_core | degree': resid(DC.D_core.to_numpy(float), Xdeg), 'weakest slice (load-bearing)': SS.weakest_slice_frac_core.to_numpy(float),
         'tearing number T2': TE.T2.to_numpy(float), 'lambda_2 (published)': M.lambda_2.to_numpy(float), 'deg_bimodality (published)': DEG.deg_bimodality.to_numpy(float),
         'max_betweenness (published)': M.max_betweenness.to_numpy(float)}
print('\n=== SEED-4 AS AN INDEPENDENT REPLICATION: partial r(toughness | UTS) per seed (3 axes each) ===')
print(f"{'descriptor':30s} {'seed1':>7s} {'seed2':>7s} {'seed3':>7s} {'seed4':>7s} | {'k=9':>7s} {'k=12':>7s}   r(U|T) k=12")
for name, x in feats.items():
    per = [par(x, cube['toughness'][:, :, s].mean(1), cube['uts'][:, :, s].mean(1)) for s in range(4)]
    print(f"{name:30s} " + ' '.join(f"{v:+7.3f}" for v in per) + f" | {par(x, t9, u9):+7.3f} {par(x, t12, u12):+7.3f}   {par(x, u12, t12):+.3f}")
print("\n=== Cohen's d (manuscript quadrants) at k=9 vs k=12 ===")
print(f"{'descriptor':30s} {'Q1>Q2 k9':>9s} {'k12':>6s} | {'Q3>Q4 k9':>9s} {'k12':>6s} | {'Q1>Q3 k9':>9s} {'k12':>6s} | {'Q2>Q4 k9':>9s} {'k12':>6s}")
for name, x in feats.items():
    row = []
    for a, b in ((1, 2), (3, 4), (1, 3), (2, 4)):
        row += [cohen_d(x[Q9 == a], x[Q9 == b]), cohen_d(x[Q12 == a], x[Q12 == b])]
    print(f"{name:30s} " + ' | '.join(f"{row[i]:+9.2f} {row[i+1]:+6.2f}" for i in range(0, 8, 2)))
pd.DataFrame({'folder_name': nets, 'uts_mean12': u12, 'toughness_mean12': t12, 'sab_mean12': cube['sab'].mean((1, 2)),
              'uts_sd12': cube['uts'].reshape(len(nets), -1).std(1, ddof=1), 'toughness_sd12': cube['toughness'].reshape(len(nets), -1).std(1, ddof=1),
              'quadrant9': Q9, 'quadrant12': Q12}).to_csv(OUT / 'four_seed_network_summary.csv', index=False)
print('\nwrote four_seed_metrics.csv, four_seed_cube.npz, four_seed_network_summary.csv')
