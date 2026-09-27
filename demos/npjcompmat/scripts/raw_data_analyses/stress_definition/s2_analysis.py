"""Step 2: true (Cauchy, -P_aa) vs nominal (engineering, F/A0) stress for the CG tensile ensemble.

Inputs (read-only): box_curves_seed1.npz and block_matching_seed1.csv from s1_extract_box.py; replicate stress
files stress_strain_npt_{x,y,z}.dat of seeds 2-4 (deposited in cg_stress_strain_seed{2,3,4}.tar.xz, unpacked to
RAW_ROOT/seed<k>/<net>/); existing tables for reproduction checks (the deposited data/csv/mechanics.csv and
data/derived/mechanics/mechanics_12pull.csv, four_seed_metrics.csv of four_seed_analysis.py, and
figure6_SI_stats_k12.csv and per_direction_metrics_v2.csv, which come from earlier analyses, are not deposited
and are read from NPJ_ANALYSIS_OUT/cg_mechanics/); the deposited data/dataset.pkl for the graph metrics (same
source as fig6_si_table_k12.py).

Metric conventions copied from three_seed_analysis.py / four_seed_analysis.py:
  UTS = max of the stress column (first occurrence); sab = strain of the first post-peak row with
  stress < 0.05*UTS (else last row); toughness = trapezoid of stress over the file strain for rows
  with strain <= sab.  No smoothing.
Nominal: sigma_N = sigma_T * A/A0 (A/A0 interpolated to the centre of each averaging window).
  UTS_nom = max sigma_N; toughness_nom = trapezoid of sigma_N over the SAME rows (strain <= sab_true),
  so only the integrand changes.  Sensitivity variants: sab re-derived from the nominal curve, and
  UTS_nom restricted to strain <= sab (the definition used in nominal_stress_v2.py).
Replicates (no box data): A/A0 from a calibration built on the 981 seed-1 pulls; candidates are
validated on seed 1 (leave-one-network-out) against the measured area before use.
Writes only into NPJ_ANALYSIS_OUT/stress_definition/ (see ../paths.py).  At most 2 worker processes.
"""
import pickle, numpy as np, pandas as pd
import sys
from pathlib import Path
from multiprocessing import Pool
from scipy import stats
from scipy.stats import ttest_ind, skew, kurtosis

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))         # raw_data_analyses/, for paths.py
from paths import ANALYSIS_OUT, DATA, RAW_ROOT, out_dir
OUT = out_dir('stress_definition')
ND = ANALYSIS_OUT / 'cg_mechanics'           # four_seed_metrics.csv; per_direction_metrics_v2.csv (not deposited)
UPD = ANALYSIS_OUT / 'cg_mechanics'          # figure6_SI_stats_k12.csv (not deposited)
MECH12 = DATA / 'derived' / 'mechanics' / 'mechanics_12pull.csv'
MECH = DATA / 'csv' / 'mechanics.csv'
REPS = {2: RAW_ROOT / 'seed2', 3: RAW_ROOT / 'seed3',
        4: RAW_ROOT / 'seed4'}
trapz = getattr(np, 'trapezoid', None) or np.trapz
AX = 'xyz'
pd.set_option('display.width', 200)


# ------------------------------------------------------------------ metric definitions
def true_metrics(e, s):
    i = int(np.argmax(s)); uts = s[i]
    post = np.where(s[i:] < 0.05 * uts)[0]; sab = e[i + post[0]] if len(post) else e[-1]
    m = e <= sab
    return dict(UTS_true=uts, strain_at_UTS_true=e[i], sab_true=sab, tough_true=trapz(s[m], e[m]))


def nominal_metrics(e, s, r, sab, tag='nom'):
    sn = s * r; j = int(np.argmax(sn)); un = sn[j]; m = e <= sab
    post = np.where(sn[j:] < 0.05 * un)[0]; sab_n = e[j + post[0]] if len(post) else e[-1]; mn = e <= sab_n
    return {f'UTS_{tag}': un, f'strain_at_UTS_{tag}': e[j], f'tough_{tag}': trapz(sn[m], e[m]),
            f'sab_{tag}_own': sab_n, f'tough_{tag}_ownsab': trapz(sn[mn], e[mn]), f'UTS_{tag}_presab': sn[m].max()}


def read_dat(fp):
    d = np.loadtxt(fp, comments='#'); return d[:, 1], d[:, 2]


def load_rep(args):
    seed, fn = args
    return [(seed, fn, ax) + read_dat(REPS[seed] / fn / f'stress_strain_npt_{ax}.dat') for ax in AX]


def quad(u, t): return 1 + (t >= np.median(t)).astype(int) + 2 * (u >= np.median(u)).astype(int)


def cd(xa, xb):
    sp = np.sqrt(((len(xa) - 1) * xa.var(ddof=1) + (len(xb) - 1) * xb.var(ddof=1)) / (len(xa) + len(xb) - 2)); return (xb.mean() - xa.mean()) / sp if sp else 0.0


def bh(p):
    m = len(p); order = np.argsort(p); ranks = np.empty(m, int); ranks[order] = np.arange(1, m + 1)
    q = p * m / ranks; qs = np.minimum.accumulate(q[order][::-1])[::-1]; out = np.empty(m); out[order] = np.minimum(qs, 1); return out


if __name__ == '__main__':
    out = open(OUT / 's2_analysis_output.txt', 'w', encoding='utf-8')
    def P(*a):
        print(*a); print(*a, file=out)

    mech = pd.read_csv(MECH); nets = list(mech.folder_name); N = len(nets)
    z = np.load(OUT / 'box_curves_seed1.npz')
    C1 = {tuple(k.split('|')): z[k] for k in z.files}
    assert len(C1) == 3 * N
    BM = pd.read_csv(OUT / 'block_matching_seed1.csv').set_index(['folder_name', 'axis'])

    # ============================================================ 1. area geometry, seed 1
    egrid = C1[(nets[0], 'x')][:, 1]                      # common strain schedule (files agree to 1e-4)
    Agrid = np.array([np.interp(egrid, C1[(fn, ax)][:, 1], C1[(fn, ax)][:, 3]) for fn in nets for ax in AX])   # A/A0, (981, 800)
    Vgrid = np.array([np.interp(egrid, C1[(fn, ax)][:, 1], C1[(fn, ax)][:, 4]) for fn in nets for ax in AX])
    keys = [(fn, ax) for fn in nets for ax in AX]
    T1 = {k: true_metrics(C1[k][:, 1], C1[k][:, 2]) for k in keys}
    sab1 = np.array([T1[k]['sab_true'] for k in keys]); eu1 = np.array([T1[k]['strain_at_UTS_true'] for k in keys])
    P('=== 1. GEOMETRY (seed 1, 981 pulls): V/V0 = (A/A0)(1+e); identity checked to 1.7e-7 in s1 ===')
    P(f"A0 at step 1M vs stage-1 mean (500k-1M): ratio SD {np.std(BM.A0 / BM.A0_stage1_avg):.4f}, range {np.min(BM.A0 / BM.A0_stage1_avg):.3f}-{np.max(BM.A0 / BM.A0_stage1_avg):.3f}")
    rows = []
    P(f"{'strain':>6s} | V/V0 all pulls: mean  SD   p5    p95  | intact pulls (strain < own sab): n  V/V0 mean SD p5 p95 | A/A0 (1+e) incompress. ratio mean")
    for et in (0.5, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15):
        j = int(np.argmin(abs(egrid - et))); v = Vgrid[:, j]; it = sab1 > egrid[j]
        r = dict(strain=round(egrid[j], 3), n_all=len(v), VV0_mean=v.mean(), VV0_sd=v.std(ddof=1), VV0_p5=np.percentile(v, 5), VV0_p95=np.percentile(v, 95),
                 n_intact=int(it.sum()), VV0_intact_mean=v[it].mean(), VV0_intact_sd=v[it].std(ddof=1), VV0_intact_p5=np.percentile(v[it], 5), VV0_intact_p95=np.percentile(v[it], 95),
                 AA0_mean=Agrid[:, j].mean(), AA0_sd=Agrid[:, j].std(ddof=1), AA0_incompressible=1 / (1 + egrid[j]))
        rows.append(r)
        P(f"{r['strain']:6.2f} | {r['VV0_mean']:.4f} {r['VV0_sd']:.4f} {r['VV0_p5']:.4f} {r['VV0_p95']:.4f} | {r['n_intact']:4d} {r['VV0_intact_mean']:.4f} {r['VV0_intact_sd']:.4f} {r['VV0_intact_p5']:.4f} {r['VV0_intact_p95']:.4f}")
    pd.DataFrame(rows).to_csv(OUT / 'area_universality_seed1.csv', index=False)
    for de in (-4, -2, -1, 0, 1):
        v = np.array([np.interp(eu + de, egrid, vv) for eu, vv in zip(eu1, Vgrid)])
        P(f"  V/V0 at (strain_at_UTS {de:+d}): median {np.median(v):.4f}  p5 {np.percentile(v, 5):.4f}  p95 {np.percentile(v, 95):.4f}")
    # is the deviation network-specific?  ICC(1) across the 3 axes of a network
    def icc(x):
        X = x.reshape(N, 3); msb = 3 * X.mean(1).var(ddof=1); msw = ((X - X.mean(1, keepdims=True)) ** 2).sum() / (N * 2); return (msb - msw) / (msb + 2 * msw)
    vuts = np.array([np.interp(eu, egrid, vv) for eu, vv in zip(eu1, Vgrid)])
    P(f"ICC(1) across the 3 axes of a network: V/V0 at own UTS {icc(vuts):.3f};  V/V0 at strain 8 {icc(Vgrid[:, np.argmin(abs(egrid - 8))]):.3f};  V/V0 at strain 5 {icc(Vgrid[:, np.argmin(abs(egrid - 5))]):.3f}")

    # ============================================================ 2. calibrations of A/A0 for pulls without box data
    # Only pre-break states (strain < the pull's own sab) are used to build ensemble curves: UTS and toughness never use
    # the area after sab, and post-break boxes (A/A0 recovers to ~0.35 as the broken slab retracts) would bias the mean.
    # (a) incompressible 1/(1+e); (b) ensemble-mean V/V0(e)/(1+e); (c) mean V/V0 aligned on the pull's own
    # strain_at_UTS_true; (d) mean V/V0 aligned on the pull's own sab_true.  (b)-(d) validated leave-one-network-out.
    # (e) 'netaxes' (diagnostic only): the same network's OTHER two axes, aligned on strain_at_UTS_true.
    Vint = np.where(egrid[None, :] < sab1[:, None], Vgrid, np.nan)
    dgrid = np.round(np.arange(-20, 20.0001, 0.025), 4)
    def aligned(ref):
        Y = np.full((len(keys), len(dgrid)), np.nan)
        for i in range(len(keys)):
            m = (dgrid >= egrid[0] - ref[i]) & (dgrid <= egrid[-1] - ref[i]); Y[i, m] = np.interp(dgrid[m] + ref[i], egrid, Vint[i], right=np.nan)
        return Y
    AL = {'uts': aligned(eu1), 'sab': aligned(sab1)}
    def curve_from(Y, grid, excl=None, min_n=20):
        w = np.ones(len(Y), bool)
        if excl is not None: w[excl] = False
        cnt = np.sum(~np.isnan(Y[w]), 0)
        s = np.nansum(Y[w], 0); mu = np.where(cnt >= min_n, s / np.maximum(cnt, 1), np.nan)
        ok = ~np.isnan(mu); return np.interp(grid, grid[ok], mu[ok])   # edge-hold outside the supported range
    CAL = {'univ': curve_from(Vint, egrid), 'uts': curve_from(AL['uts'], dgrid), 'sab': curve_from(AL['sab'], dgrid)}
    pd.DataFrame({'delta_strain': dgrid, 'VV0_aligned_on_strain_at_UTS': CAL['uts'], 'VV0_aligned_on_sab': CAL['sab']}).to_csv(OUT / 'calibration_curves_aligned.csv', index=False)
    pd.DataFrame({'strain': egrid, 'VV0_ensemble_mean_prebreak': CAL['univ'], 'n_prebreak': np.sum(~np.isnan(Vint), 0)}).to_csv(OUT / 'calibration_curve_ensemble.csv', index=False)

    def ratio(method, e, eu, sab, cal=CAL):
        if method == 'incomp': return 1 / (1 + e)
        if method == 'univ': return np.interp(e, egrid, cal['univ']) / (1 + e)
        ref = eu if method == 'uts' else sab
        return np.interp(e - ref, dgrid, cal[method]) / (1 + e)

    # ============================================================ 3. per-pull metrics, seed 1 (measured area + calibration validation)
    rows = []
    for i, (fn, ax) in enumerate(keys):
        c = C1[(fn, ax)]; e, s, r = c[:, 1], c[:, 2], c[:, 3]; t = T1[(fn, ax)]
        row = dict(folder_name=fn, axis=ax, seed=1, area_source='measured_box', **t, **nominal_metrics(e, s, r, t['sab_true']),
                   VV0_at_UTS_true=np.interp(t['strain_at_UTS_true'], e, c[:, 4]), A0=BM.loc[(fn, ax), 'A0'], A0_stage1_avg=BM.loc[(fn, ax), 'A0_stage1_avg'])
        own = [3 * nets.index(fn) + a for a in range(3)]            # leave this network's 3 pulls out
        cal_lono = {'univ': curve_from(Vint, egrid, own), 'uts': curve_from(AL['uts'], dgrid, own), 'sab': curve_from(AL['sab'], dgrid, own)}
        for mth in ('incomp', 'univ', 'uts', 'sab'):
            nm = nominal_metrics(e, s, ratio(mth, e, t['strain_at_UTS_true'], t['sab_true'], cal_lono), t['sab_true'], 'x')
            row[f'UTS_nom_cal_{mth}'] = nm['UTS_x']; row[f'tough_nom_cal_{mth}'] = nm['tough_x']
        # network-specific calibration: mean aligned V/V0 of the OTHER two axes of the same network
        oth = [3 * nets.index(fn) + a for a in range(3) if AX[a] != ax]
        cnet = curve_from(AL['uts'][oth], dgrid, min_n=1)
        nm = nominal_metrics(e, s, np.interp(e - t['strain_at_UTS_true'], dgrid, cnet) / (1 + e), t['sab_true'], 'x')
        row['UTS_nom_cal_netaxes'] = nm['UTS_x']; row['tough_nom_cal_netaxes'] = nm['tough_x']
        rows.append(row)
    S1 = pd.DataFrame(rows)

    # ---- reproduction of the existing true-stress tables
    F4 = pd.read_csv(ND / 'four_seed_metrics.csv')
    chk = S1.merge(F4[F4.seed == 1], on=['folder_name', 'axis'])
    P('\n=== 2. REPRODUCTION OF EXISTING TRUE-STRESS NUMBERS ===')
    P(f"seed 1 per pull vs four_seed_metrics.csv (n={len(chk)}): max|dUTS| {np.abs(chk.UTS_true - chk.uts).max():.2e}  max|dsab| {np.abs(chk.sab_true - chk.sab).max():.2e}  max|dtough| {np.abs(chk.tough_true - chk.toughness).max():.2e}")
    g1 = S1.groupby('folder_name').agg(u=('UTS_true', 'mean'), t=('tough_true', 'mean'), sb=('sab_true', 'mean'), us=('UTS_true', lambda x: np.std(x, ddof=0)), ts=('tough_true', lambda x: np.std(x, ddof=0))).reindex(nets)
    P(f"seed-1 network means vs deposited mechanics.csv: max|d uts_mean| {np.abs(g1.u.values - mech.uts_mean).max():.2e}  max|d toughness_mean| {np.abs(g1.t.values - mech.toughness_mean).max():.2e}  "
      f"max|d strain_break_mean| {np.abs(g1.sb.values - mech.strain_break_mean).max():.2e}  max|d uts_std(ddof0)| {np.abs(g1.us.values - mech.uts_std).max():.2e}  max|d toughness_std| {np.abs(g1.ts.values - mech.toughness_std).max():.2e}")

    # ---- calibration validation against the measured area (seed 1)
    P('\n=== 3. CALIBRATION VALIDATION (seed 1, leave-one-network-out; error = calibrated - measured-area nominal) ===')
    wsd = {}
    for mt in ('UTS', 'tough'):
        X = S1.pivot(index='folder_name', columns='axis', values=f'{mt}_nom').reindex(nets).to_numpy()
        wsd[mt] = np.sqrt(((X - X.mean(1, keepdims=True)) ** 2).sum() / (N * 2))    # within-network (across-axis) SD
    P(f"pre-registered criterion: per-pull error SD <= 1/3 of the within-network pull-to-pull SD of the measured nominal metric (UTS_nom {wsd['UTS']:.4f}, tough_nom {wsd['tough']:.4f})")
    valrows = []
    for mth in ('incomp', 'univ', 'uts', 'sab', 'netaxes'):
        r = dict(method=mth)
        for mt in ('UTS', 'tough'):
            err = S1[f'{mt}_nom_cal_{mth}'] - S1[f'{mt}_nom']; rel = err / S1[f'{mt}_nom']
            gm = S1.assign(c=S1[f'{mt}_nom_cal_{mth}']).groupby('folder_name')[['c', f'{mt}_nom']].mean().reindex(nets)
            r.update({f'{mt}_bias_rel': rel.mean(), f'{mt}_err_sd': err.std(ddof=1), f'{mt}_err_sd_over_within': err.std(ddof=1) / wsd[mt],
                      f'{mt}_relerr_p95abs': np.percentile(abs(rel), 95), f'{mt}_netmean_spearman': stats.spearmanr(gm.c, gm[f'{mt}_nom'])[0]})
        u_c = S1.groupby('folder_name')[f'UTS_nom_cal_{mth}'].mean().reindex(nets).values; t_c = S1.groupby('folder_name')[f'tough_nom_cal_{mth}'].mean().reindex(nets).values
        u_m = S1.groupby('folder_name').UTS_nom.mean().reindex(nets).values; t_m = S1.groupby('folder_name').tough_nom.mean().reindex(nets).values
        r['quadrant_agreement_3pull'] = (quad(u_c, t_c) == quad(u_m, t_m)).mean()
        r['passes'] = (r['UTS_err_sd_over_within'] <= 1 / 3) and (r['tough_err_sd_over_within'] <= 1 / 3)
        valrows.append(r)
    VAL = pd.DataFrame(valrows); VAL.to_csv(OUT / 'calibration_validation_seed1.csv', index=False)
    P(VAL.round(4).to_string(index=False))

    # ============================================================ 4. replicates
    with Pool(2) as pool:
        reps = sum(pool.map(load_rep, [(sd, fn) for sd in REPS for fn in nets], chunksize=20), [])
    best = VAL[VAL.passes & (VAL.method != 'netaxes')].sort_values('tough_err_sd').method.iloc[0] if VAL[VAL.passes & (VAL.method != 'netaxes')].shape[0] else 'uts'
    P(f"\nreplicate calibration used: '{best}'  (lowest toughness error among passing ensemble calibrations; 'netaxes' is diagnostic only)")
    rows = []
    for sd, fn, ax, e, s in reps:
        t = true_metrics(e, s); row = dict(folder_name=fn, axis=ax, seed=sd, area_source=f'calibrated_{best}', **t)
        row.update(nominal_metrics(e, s, ratio(best, e, t['strain_at_UTS_true'], t['sab_true']), t['sab_true']))
        for mth in ('incomp', 'univ', 'uts', 'sab'):
            nm = nominal_metrics(e, s, ratio(mth, e, t['strain_at_UTS_true'], t['sab_true']), t['sab_true'], 'x')
            row[f'UTS_nom_cal_{mth}'] = nm['UTS_x']; row[f'tough_nom_cal_{mth}'] = nm['tough_x']
        rows.append(row)
    SR = pd.DataFrame(rows)
    chk = SR.merge(F4, on=['folder_name', 'axis', 'seed'])
    P(f"replicates vs four_seed_metrics.csv (n={len(chk)}): max|dUTS| {np.abs(chk.UTS_true - chk.uts).max():.2e}  max|dsab| {np.abs(chk.sab_true - chk.sab).max():.2e}  max|dtough| {np.abs(chk.tough_true - chk.toughness).max():.2e}")
    ALL = pd.concat([S1, SR], ignore_index=True)
    keep = ['folder_name', 'axis', 'seed', 'area_source', 'UTS_true', 'UTS_nom', 'tough_true', 'tough_nom', 'strain_at_UTS_true', 'strain_at_UTS_nom', 'sab_true',
            'sab_nom_own', 'tough_nom_ownsab', 'UTS_nom_presab', 'VV0_at_UTS_true', 'A0', 'A0_stage1_avg'] + [c for c in ALL.columns if c.startswith(('UTS_nom_cal', 'tough_nom_cal'))]
    ALL = ALL[keep].sort_values(['seed', 'folder_name', 'axis']); ALL.to_csv(OUT / 'per_pull_true_vs_nominal.csv', index=False)
    P(f"per-pull table: {len(ALL)} pulls")
    P(f"strain at UTS, median [5-95%]: true {np.median(ALL.strain_at_UTS_true):.2f} [{np.percentile(ALL.strain_at_UTS_true, 5):.2f}-{np.percentile(ALL.strain_at_UTS_true, 95):.2f}]   "
      f"nominal {np.median(ALL.strain_at_UTS_nom):.2f} [{np.percentile(ALL.strain_at_UTS_nom, 5):.2f}-{np.percentile(ALL.strain_at_UTS_nom, 95):.2f}]; min nominal {ALL.strain_at_UTS_nom.min():.2f}")
    P(f"per pull: nominal/true UTS ratio median {np.median(ALL.UTS_nom / ALL.UTS_true):.4f}; nominal/true toughness ratio median {np.median(ALL.tough_nom / ALL.tough_true):.4f}")

    # ============================================================ 5. network level
    def netmeans(D, seeds, ucol, tcol):
        d = D[D.seed.isin(seeds)].groupby('folder_name')[[ucol, tcol]].mean().reindex(nets); return d[ucol].to_numpy(), d[tcol].to_numpy()
    u12, t12 = netmeans(ALL, [1, 2, 3, 4], 'UTS_true', 'tough_true')
    k12 = pd.read_csv(MECH12).set_index('folder_name').reindex(nets)
    P(f"\n12-pull true means vs mechanics_12pull.csv: max|d uts| {np.abs(u12 - k12.uts_mean.values).max():.2e}  max|d toughness| {np.abs(t12 - k12.toughness_mean.values).max():.2e}")
    SETS = {'seed1_3pull': dict(seeds=[1], nom=('UTS_nom', 'tough_nom')),
            '12pull': dict(seeds=[1, 2, 3, 4], nom=('UTS_nom', 'tough_nom')),
            f'12pull_all_{best}': dict(seeds=[1, 2, 3, 4], nom=(f'UTS_nom_cal_{best}', f'tough_nom_cal_{best}'))}
    for alt in ('incomp', 'uts'):
        if alt != best: SETS[f'12pull_all_{alt}'] = dict(seeds=[1, 2, 3, 4], nom=(f'UTS_nom_cal_{alt}', f'tough_nom_cal_{alt}'))
    QN = {}
    NET = pd.DataFrame({'folder_name': nets})
    with open(DATA / 'dataset.pkl', 'rb') as f:
        M = pickle.load(f)['mechanics'].copy().reindex(nets)
    dc = [f'cnt_d{i}' for i in range(7)]
    M['cycle_rank'] = sum(M[f'cnt_d{i}'] * i for i in range(7)) / 2.0 - M[dc].sum(axis=1) + 1
    M['deg_bimodality'] = M.apply(lambda r: ((skew(sum([[int(c.split('_d')[1])] * int(r[c]) for c in dc], [])) ** 2 + 1) /
                                             (kurtosis(sum([[int(c.split('_d')[1])] * int(r[c]) for c in dc], []), fisher=True) + 3)), axis=1)
    FEATS = ['lambda_2', 'lambda_2_core', 'avg_path_len', 'cycle_rank', 'max_betweenness', 'num_bridges', 'deg_bimodality', 'deg_std', 'deg_entropy', 'avg_eigenvector_cent', 'assortativity']
    TR = {'Q1->Q2': (1, 2), 'Q3->Q4': (3, 4), 'Q1->Q3': (1, 3), 'Q2->Q4': (2, 4), 'Q1->Q4': (1, 4)}
    X = M[FEATS].to_numpy(float)

    def fig6(u, t):
        Q = quad(u, t); rr = []
        for j, f in enumerate(FEATS):
            for name, (a, b) in TR.items():
                xa, xb = X[Q == a, j], X[Q == b, j]; rr.append(dict(metric=f, transition=name, n_from=len(xa), n_to=len(xb), d=cd(xa, xb), p=ttest_ind(xa, xb, equal_var=False).pvalue))
        D = pd.DataFrame(rr); D['q'] = bh(D.p.to_numpy()); D['sig'] = D.q < 0.05; return D

    F6 = []
    for name, cfg in SETS.items():
        ut, tt = netmeans(ALL, cfg['seeds'], 'UTS_true', 'tough_true'); un, tn = netmeans(ALL, cfg['seeds'], *cfg['nom'])
        Qt, Qn = quad(ut, tt), quad(un, tn); QN[name] = Qn
        P(f"\n=== 4. NETWORK LEVEL: {name} ===")
        P(f"Pearson / Spearman true vs nominal:  UTS {stats.pearsonr(ut, un)[0]:.3f} / {stats.spearmanr(ut, un)[0]:.3f}   toughness {stats.pearsonr(tt, tn)[0]:.3f} / {stats.spearmanr(tt, tn)[0]:.3f}")
        P(f"Spearman(UTS, toughness): true {stats.spearmanr(ut, tt)[0]:.3f}   nominal {stats.spearmanr(un, tn)[0]:.3f}")
        P(f"quadrant sizes Q1..Q4: true {[int((Qt == q).sum()) for q in (1, 2, 3, 4)]}   nominal {[int((Qn == q).sum()) for q in (1, 2, 3, 4)]}")
        P(f"networks changing quadrant: {int((Qt != Qn).sum())} / {N} = {100 * (Qt != Qn).mean():.1f}%   (UTS half changes {int(((ut >= np.median(ut)) != (un >= np.median(un))).sum())}, toughness half changes {int(((tt >= np.median(tt)) != (tn >= np.median(tn))).sum())})")
        P('transition matrix (rows true Q1..Q4, cols nominal Q1..Q4):'); P(pd.crosstab(Qt, Qn).to_string())
        if name in ('seed1_3pull', '12pull'):
            NET[f'UTS_true_{name}'] = ut; NET[f'UTS_nom_{name}'] = un; NET[f'tough_true_{name}'] = tt; NET[f'tough_nom_{name}'] = tn
            NET[f'Q_true_{name}'] = Qt; NET[f'Q_nom_{name}'] = Qn
        Dt, Dn = fig6(ut, tt), fig6(un, tn)
        G = Dt[['metric', 'transition', 'n_from', 'n_to', 'd', 'p', 'q', 'sig']].merge(Dn[['metric', 'transition', 'n_from', 'n_to', 'd', 'p', 'q', 'sig']], on=['metric', 'transition'], suffixes=('_true', '_nom'))
        G.insert(0, 'set', name); G['sign_flip'] = np.sign(G.d_true) != np.sign(G.d_nom); G['sig_change'] = G.sig_true != G.sig_nom
        F6.append(G)
        P(f"Figure-6 cells: BH q<.05 true {int(G.sig_true.sum())}/55, nominal {int(G.sig_nom.sum())}/55; both {int((G.sig_true & G.sig_nom).sum())}; sign flips {int(G.sign_flip.sum())} "
          f"(of which significant in either {int((G.sign_flip & (G.sig_true | G.sig_nom)).sum())}); r(d_true, d_nom) = {stats.pearsonr(G.d_true, G.d_nom)[0]:.3f}; median |d_nom - d_true| {np.median(abs(G.d_nom - G.d_true)):.3f}")
        ch = G[G.sig_change | G.sign_flip]
        if len(ch): P(ch[['metric', 'transition', 'd_true', 'q_true', 'd_nom', 'q_nom', 'sign_flip']].round(4).to_string(index=False))
        if name == '12pull':
            ref = pd.read_csv(UPD / 'figure6_SI_stats_k12.csv')
            assert list(ref.n_from) == list(Dt.n_from) and list(ref.n_to) == list(Dt.n_to)       # same row order (11 metrics x 5 transitions)
            P(f"reproduction of figure6_SI_stats_k12.csv (true, 12-pull, cell by cell): max|d diff| {np.abs(Dt.d.values - ref.cohens_d.values).max():.2e}  "
              f"max|p diff| {np.abs(Dt.p.values - ref.welch_p.values).max():.2e}  max|q diff| {np.abs(Dt.q.values - ref.bh_q.values).max():.2e}  BH-significant {int(Dt.sig.sum())} vs {int(ref.sig_bh.sum())}")
            # split-half reliability of network means: seeds {1,2} vs {3,4} (6 pulls each, same axes), Spearman-Brown to 12 pulls;
            # and the network x axis interaction relative to the network effect (two-way ANOVA mean squares, 3 axes x 4 seeds)
            for col in ('UTS_true', 'UTS_nom', 'tough_true', 'tough_nom'):
                Y = np.stack([ALL[ALL.seed == sd].pivot(index='folder_name', columns='axis', values=col).reindex(nets).to_numpy() for sd in (1, 2, 3, 4)], 2)
                r = stats.pearsonr(Y[:, :, :2].mean((1, 2)), Y[:, :, 2:].mean((1, 2)))[0]
                n_, k_, m_ = Y.shape; gm = Y.mean(); net = Y.mean((1, 2))[:, None, None]; axm = Y.mean((0, 2))[None, :, None]; cell = Y.mean(2)[:, :, None]
                MSn = k_ * m_ * ((net - gm) ** 2).sum() / (n_ - 1); MSi = m_ * ((cell - net - axm + gm) ** 2).sum() / ((n_ - 1) * (k_ - 1)); MSe = ((Y - cell) ** 2).sum() / (n_ * k_ * (m_ - 1))
                P(f"  {col:10s}: split-half r(6 vs 6 pulls) {r:.3f} -> Spearman-Brown 12-pull {2 * r / (1 + r):.3f};  MS(network x axis)/MS(network) {MSi / MSn:.2f};  MS(seed)/MS(network x axis) {MSe / MSi:.2f}")
    for k in [k for k in QN if k.startswith('12pull_all')]:
        P(f"\nnominal 12-pull quadrants, seed-1 measured area + calibrated replicates vs {k}: agree for {100 * (QN['12pull'] == QN[k]).mean():.1f}% of networks")
    # seed 1: does the axis dependence follow the box aspect ratio?  deviations from the network mean over its 3 axes
    s1 = ALL[ALL.seed == 1].merge(BM.reset_index()[['folder_name', 'axis', 'L0']], on=['folder_name', 'axis'])
    dl = np.log(s1.L0) - s1.groupby('folder_name').L0.transform(lambda x: np.log(x).mean())
    for c in ('UTS_true', 'UTS_nom', 'tough_true', 'tough_nom', 'strain_at_UTS_true'):
        dv = np.log(s1[c]) - s1.groupby('folder_name')[c].transform(lambda x: np.log(x).mean())
        P(f"seed 1, within-network axis deviations: r(log {c}, log L0_axis) = {stats.pearsonr(dv, dl)[0]:+.3f}")
    F6 = pd.concat(F6, ignore_index=True); F6.to_csv(OUT / 'fig6_cells_true_vs_nominal.csv', index=False)
    NET.to_csv(OUT / 'network_true_vs_nominal.csv', index=False)

    # ============================================================ 6. the '27 %' audit claim (nominal_stress_v2.py definitions)
    P('\n=== 5. AUDIT CLAIM "27% change quadrant, n ~ 98/65/65/99" ===')
    ut, tt = netmeans(ALL, [1], 'UTS_true', 'tough_true'); un2, tn2 = netmeans(ALL, [1], 'UTS_nom_presab', 'tough_nom')
    Qt, Qn = quad(ut, tt), quad(un2, tn2)
    P(f"v2 definitions (seed 1 only, 3-pull means, measured box area, UTS_nom = max over strain <= sab): {100 * (Qt != Qn).mean():.1f}% change; nominal sizes {[int((Qn == q).sum()) for q in (1, 2, 3, 4)]}; true sizes {[int((Qt == q).sum()) for q in (1, 2, 3, 4)]}")
    v2 = pd.read_csv(ND / 'per_direction_metrics_v2.csv')
    mm = ALL[ALL.seed == 1].merge(v2, on=['folder_name', 'axis'])
    P(f"per-pull agreement with per_direction_metrics_v2.csv: max|dUTS_nom| {np.abs(mm.UTS_nom_presab - mm.uts_nominal).max():.2e}  max|dtough_nom| {np.abs(mm.tough_nom - mm.work_to_break).max():.2e}")
    for lab, (a, b) in {'incompressible A/A0 = 1/(1+e), seed 1': ('UTS_nom_cal_incomp', 'tough_nom_cal_incomp')}.items():
        un3, tn3 = netmeans(ALL, [1], a, b); P(f"{lab}: {100 * (Qt != quad(un3, tn3)).mean():.1f}% change")
    # context: how often do quadrants change between two independent 3-pull measurements of the SAME (true) quantity?
    pairs = []
    for a in (1, 2, 3, 4):
        for b in (1, 2, 3, 4):
            if a < b:
                ua, ta = netmeans(ALL, [a], 'UTS_true', 'tough_true'); ub, tb = netmeans(ALL, [b], 'UTS_true', 'tough_true'); pairs.append((quad(ua, ta) != quad(ub, tb)).mean())
    P(f"context: two independent seeds (true stress, 3-pull means each) disagree on {100 * np.mean(pairs):.1f}% of networks (6 pairs, {100 * min(pairs):.1f}-{100 * max(pairs):.1f}%)")
    # A0 sensitivity (stage-1 mean area instead of the step-1M snapshot)
    fac = ALL.A0 / ALL.A0_stage1_avg; s1 = ALL.seed == 1
    un4, tn4 = netmeans(ALL.assign(u=ALL.UTS_nom * fac, t=ALL.tough_nom * fac), [1], 'u', 't'); un1, tn1 = netmeans(ALL, [1], 'UTS_nom', 'tough_nom')
    P(f"A0 sensitivity (seed 1): using the stage-1 mean area instead of the step-1M snapshot moves {100 * (quad(un1, tn1) != quad(un4, tn4)).mean():.1f}% of networks between nominal quadrants")
    un5, tn5 = netmeans(ALL, [1], 'UTS_nom', 'tough_nom_ownsab')
    P(f"sab sensitivity (seed 1): nominal toughness integrated to the nominal-curve sab instead of the true-curve sab moves {100 * (quad(un1, tn1) != quad(un5, tn5)).mean():.1f}% ; Spearman {stats.spearmanr(tn1, tn5)[0]:.4f}")
    out.close()
