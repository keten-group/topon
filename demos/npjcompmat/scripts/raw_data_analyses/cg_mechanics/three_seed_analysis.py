"""Three-seed noise decomposition of the CG tensile ensemble.

Supersedes per_direction_analysis.py, which had ONE velocity seed and therefore
could not separate network x axis interaction (per-network box anisotropy) from
pull-to-pull noise, and had to *extrapolate* the k=9 reliability via
Spearman-Brown.  With 3 seeds x 3 axes = 9 pulls per network we measure both
directly.

Metric definitions are copied verbatim from per_direction_analysis.py so the
seed-1 column reproduces the deposited mechanics.csv exactly.

Read-only with respect to every simulation tree.  Writes only new files
(three_seed_*) - nothing existing is overwritten.

Inputs: stress_strain_npt_{x,y,z}.dat of velocity seeds 1-3, deposited in data/raw/
(cg_stress_strain_seed{1,2,3}.tar.xz, unpacked to RAW_ROOT/seed<k>/<folder_name>/), and the
deposited data/csv/mechanics.csv.  Outputs go to NPJ_ANALYSIS_OUT/cg_mechanics/ (see ../paths.py).
"""
import itertools, numpy as np, pandas as pd
import sys
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))         # raw_data_analyses/, for paths.py
from paths import DATA, RAW_ROOT, out_dir
OUT = out_dir('cg_mechanics')
CSV = DATA / 'csv' / 'mechanics.csv'
trapz = getattr(np, 'trapezoid', None) or np.trapz

SEEDS = {   # deposited layout: RAW_ROOT/seed<k>/<folder_name>/stress_strain_npt_<axis>.dat
    1: (RAW_ROOT / 'seed1', ''),
    2: (RAW_ROOT / 'seed2', ''),
    3: (RAW_ROOT / 'seed3', ''),
}
AX = 'xyz'


def metrics(fp):
    """UTS, strain-at-break, toughness - identical to per_direction_analysis.py."""
    d = pd.read_csv(fp, sep=r'\s+', comment='#', names=['t', 'strain', 'stress'])
    i = d.stress.idxmax()
    uts = d.stress[i]
    post = d.loc[i:]
    br = post[post.stress < 0.05 * uts]
    sab = br.iloc[0].strain if len(br) else d.iloc[-1].strain
    v = d[d.strain <= sab]
    return uts, sab, trapz(v.stress, v.strain)


# ---------------------------------------------------------------- load
mech = pd.read_csv(CSV)
M = mech.set_index('folder_name')
cache = OUT / 'three_seed_metrics.csv'
if cache.exists():
    L = pd.read_csv(cache)
    print(f'loaded cached metrics: {len(L)} pulls')
else:
    rows = []
    for seed, (root, sub) in SEEDS.items():
        for fn in mech.folder_name:
            base = root / fn / sub if sub else root / fn
            for ax in AX:
                fp = base / f'stress_strain_npt_{ax}.dat'
                if not fp.exists():
                    print('MISSING', seed, fn, ax)
                    continue
                u, s, t = metrics(fp)
                rows.append(dict(folder_name=fn, seed=seed, axis=ax, uts=u, sab=s, toughness=t))
        print(f'  seed {seed}: done')
    L = pd.DataFrame(rows)
    L.to_csv(cache, index=False)
    print(f'wrote {cache.name}: {len(L)} pulls')

nets = list(mech.folder_name)
print(f'networks={len(nets)}  seeds={L.seed.nunique()}  axes={L.axis.nunique()}  pulls={len(L)}')

# cube[prop] -> array (n_networks, n_axes, n_seeds)
cube = {}
for p in ('uts', 'sab', 'toughness'):
    a = np.full((len(nets), 3, 3), np.nan)
    piv = L.pivot_table(index='folder_name', columns=['axis', 'seed'], values=p)
    for ia, ax in enumerate(AX):
        for js, sd in enumerate((1, 2, 3)):
            a[:, ia, js] = piv[(ax, sd)].reindex(nets).to_numpy()
    cube[p] = a
    assert not np.isnan(a).any(), f'missing cells in {p}'

# ------------------------------------------- sanity: seed 1 == deposited CSV
u1 = cube['uts'][:, :, 0].mean(1)
t1 = cube['toughness'][:, :, 0].mean(1)
print('\n=== reproduction check (seed 1 vs deposited mechanics.csv) ===')
print(f"  max|d uts_mean|       = {np.abs(u1 - mech.uts_mean.to_numpy()).max():.3e}")
print(f"  max|d toughness_mean| = {np.abs(t1 - mech.toughness_mean.to_numpy()).max():.3e}")


# ------------------------------------------- two-way ANOVA WITH replication
def decompose(Y):
    """Y: (n, k, m) = network x axis x seed.  Random-effects variance components."""
    n, k, m = Y.shape
    gm = Y.mean()
    net = Y.mean((1, 2))[:, None, None]
    axm = Y.mean((0, 2))[None, :, None]
    cell = Y.mean(2)[:, :, None]
    SS_net = k * m * ((net - gm) ** 2).sum()
    SS_ax = n * m * ((axm - gm) ** 2).sum()
    SS_int = m * ((cell - net - axm + gm) ** 2).sum()
    SS_err = ((Y - cell) ** 2).sum()
    MS_net, MS_ax = SS_net / (n - 1), SS_ax / (k - 1)
    MS_int, MS_err = SS_int / ((n - 1) * (k - 1)), SS_err / (n * k * (m - 1))
    v_err = MS_err
    v_int = max((MS_int - MS_err) / m, 0.0)
    v_ax = max((MS_ax - MS_int) / (n * m), 0.0)
    v_net = max((MS_net - MS_int) / (k * m), 0.0)
    return dict(v_net=v_net, v_ax=v_ax, v_int=v_int, v_err=v_err,
                MS_net=MS_net, MS_ax=MS_ax, MS_int=MS_int, MS_err=MS_err,
                F_ax=MS_ax / MS_int, p_ax=stats.f.sf(MS_ax / MS_int, k - 1, (n - 1) * (k - 1)),
                F_int=MS_int / MS_err, p_int=stats.f.sf(MS_int / MS_err, (n - 1) * (k - 1), n * k * (m - 1)))


def reliability(d, k_ax, m_seed):
    """Reliability of a network mean over k axes x m seeds."""
    return d['v_net'] / (d['v_net'] + d['v_int'] / k_ax + d['v_err'] / (k_ax * m_seed))


print('\n=== THREE-WAY VARIANCE DECOMPOSITION (network x axis x seed) ===')
print('    the axis-interaction term is NEW: one seed cannot separate it from noise\n')
dec = {}
for p in ('uts', 'toughness'):
    d = decompose(cube[p])
    dec[p] = d
    tot = d['v_net'] + d['v_ax'] + d['v_int'] + d['v_err']
    print(f"{p}:")
    print(f"  between-network      {d['v_net']:10.4f}  ({100*d['v_net']/tot:5.1f}%)  SD={np.sqrt(d['v_net']):.4f}")
    print(f"  axis (systematic)    {d['v_ax']:10.4f}  ({100*d['v_ax']/tot:5.1f}%)  F={d['F_ax']:.2f} p={d['p_ax']:.3g}")
    print(f"  network x axis       {d['v_int']:10.4f}  ({100*d['v_int']/tot:5.1f}%)  F={d['F_int']:.2f} p={d['p_int']:.3g}")
    print(f"  pull-to-pull (seed)  {d['v_err']:10.4f}  ({100*d['v_err']/tot:5.1f}%)  SD={np.sqrt(d['v_err']):.4f}")
    print(f"  reliability: 1 pull={reliability(d,1,1):.3f}  3 axes x1 seed={reliability(d,3,1):.3f}  "
          f"3 axes x3 seeds (k=9)={reliability(d,3,3):.3f}")

# --------------------------- measured vs Spearman-Brown-predicted reliability
print('\n=== MEASURED k=9 reliability vs the single-seed Spearman-Brown prediction ===')
sb_rows = []
for p in ('uts', 'toughness'):
    Y1 = cube[p][:, :, 0]                      # seed 1 only - what the old analysis had
    n, k = Y1.shape
    gm = Y1.mean(); net = Y1.mean(1, keepdims=True); axm = Y1.mean(0, keepdims=True)
    E = Y1 - net - axm + gm
    MSe = (E ** 2).sum() / ((n - 1) * (k - 1))
    MSnet = k * ((net - gm) ** 2).sum() / (n - 1)
    sb2 = max((MSnet - MSe) / k, 0)
    old_rel = lambda mm: sb2 / (sb2 + MSe / mm)
    meas3 = reliability(dec[p], 3, 1)
    meas9 = reliability(dec[p], 3, 3)
    print(f"{p:10s} old(1 seed): k=3 -> {old_rel(3):.3f}, SB-extrapolated k=9 -> {old_rel(9):.3f}")
    print(f"{'':10s} NEW(3 seeds): k=3 -> {meas3:.3f}, MEASURED k=9 -> {meas9:.3f}"
          f"   (SB {'OVER' if old_rel(9) > meas9 else 'UNDER'}-estimated by {abs(old_rel(9)-meas9):.3f})")
    sb_rows.append((p, old_rel(3), old_rel(9), meas3, meas9))

# ------------------------------------------------- quadrant stability
def quadrants(u, t):
    return 1 + (t >= np.median(t)).astype(int) + 2 * (u >= np.median(u)).astype(int)


print('\n=== QUADRANT STABILITY ===')
Q_ref = quadrants(cube['uts'].mean((1, 2)), cube['toughness'].mean((1, 2)))   # k=9 reference
print('  reference = 9-pull mean (3 axes x 3 seeds)')
for lbl, sel in [('1 seed x 3 axes (the published basis)', [(slice(None), s) for s in range(3)]),
                 ('1 pull (single axis, single seed)', None)]:
    chs = []
    if sel is not None:
        for s in range(3):
            Q = quadrants(cube['uts'][:, :, s].mean(1), cube['toughness'][:, :, s].mean(1))
            chs.append((Q != Q_ref).mean())
    else:
        for s in range(3):
            for a in range(3):
                Q = quadrants(cube['uts'][:, a, s], cube['toughness'][:, a, s])
                chs.append((Q != Q_ref).mean())
    print(f"  {lbl:42s} changes quadrant: {np.mean(chs)*100:.1f}% (range {np.min(chs)*100:.1f}-{np.max(chs)*100:.1f}%)")

# seed-to-seed disagreement between two independent 3-axis classifications
pair = []
for s1, s2 in itertools.combinations(range(3), 2):
    Qa = quadrants(cube['uts'][:, :, s1].mean(1), cube['toughness'][:, :, s1].mean(1))
    Qb = quadrants(cube['uts'][:, :, s2].mean(1), cube['toughness'][:, :, s2].mean(1))
    pair.append((Qa != Qb).mean())
print(f"  two independent seeds (3 axes each) disagree on: {np.mean(pair)*100:.1f}% of networks")

# ------------------------------------------------- Cohen's d, measured at k=9
feats = ['lambda_2', 'avg_path_len', 'max_betweenness', 'avg_eigenvector_cent', 'lambda_2_core', 'deg_std']
heads = [('Q1->Q2', 'max_betweenness', 1, 2), ('Q3->Q4', 'avg_eigenvector_cent', 3, 4),
         ('Q1->Q2', 'lambda_2', 1, 2), ('Q3->Q4', 'lambda_2', 3, 4)]


def cohen_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (b.mean() - a.mean()) / sp


print("\n=== headline Cohen's d: published (1 seed, 3 axes) vs measured at k=9 ===")
d_rows = []
for tr, f, qa, qb in heads:
    x = M[f].reindex(nets).to_numpy()
    d1 = np.mean([cohen_d(x[quadrants(cube['uts'][:, :, s].mean(1), cube['toughness'][:, :, s].mean(1)) == qa],
                          x[quadrants(cube['uts'][:, :, s].mean(1), cube['toughness'][:, :, s].mean(1)) == qb])
                  for s in range(3)])
    d9 = cohen_d(x[Q_ref == qa], x[Q_ref == qb])
    print(f"  {tr} {f:22s} d(3 axes,1 seed)={d1:+.3f}  ->  d(k=9 measured)={d9:+.3f}   ratio {d9/d1 if d1 else float('nan'):.2f}")
    d_rows.append((f'{tr}\n{f}', d1, d9))

# ------------------------------------------------- correlations
print('\n=== r(descriptor, property): 3-axis single seed vs 9-pull mean ===')
corr_rows = []
for p in ('uts', 'toughness'):
    for f in feats:
        x = M[f].reindex(nets).to_numpy()
        r1 = np.mean([stats.pearsonr(x, cube[p][:, :, s].mean(1))[0] for s in range(3)])
        r9 = stats.pearsonr(x, cube[p].mean((1, 2)))[0]
        print(f"  {p:10s} {f:22s} r(1 seed)={r1:+.3f}  r(k=9)={r9:+.3f}   gain {abs(r9)-abs(r1):+.3f}")
        corr_rows.append((p, f, r1, r9))

L.to_csv(OUT / 'three_seed_per_pull_metrics.csv', index=False)
summary = pd.DataFrame({
    'folder_name': nets,
    'uts_mean9': cube['uts'].mean((1, 2)), 'uts_sd9': cube['uts'].reshape(len(nets), -1).std(1, ddof=1),
    'toughness_mean9': cube['toughness'].mean((1, 2)), 'toughness_sd9': cube['toughness'].reshape(len(nets), -1).std(1, ddof=1),
    'sab_mean9': cube['sab'].mean((1, 2)),
    'quadrant9': Q_ref,
})
summary.to_csv(OUT / 'three_seed_network_summary.csv', index=False)
print(f"\nwrote three_seed_per_pull_metrics.csv ({len(L)} rows) and three_seed_network_summary.csv ({len(summary)} rows)")

np.savez_compressed(OUT / 'three_seed_cube.npz', uts=cube['uts'], toughness=cube['toughness'],
                    sab=cube['sab'], nets=np.array(nets))
print('wrote three_seed_cube.npz')
