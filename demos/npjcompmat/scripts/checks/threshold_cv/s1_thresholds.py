"""Step 1. How the 55-cell quadrant analysis depends on the threshold.

Every scheme uses the Cohen's d, Welch test and BH correction of the published analysis.
  P50        median split of both properties (the published analysis, 31 of 55 cells significant)
  P40        both properties split at the 40th percentile (>= rule)
  P60        both properties split at the 60th percentile
  tercile    top third vs bottom third of each property, with the networks in the middle third of either one dropped.
             UTS and toughness correlate (r = 0.78), so the off-diagonal groups hold 1 and 2 networks and only Q1->Q4 can
             be tested (a cell needs at least 5 networks per group, and BH runs over the tested cells).
  tercile within halves   each transition split into thirds along its own property (middle third dropped) inside the
             relevant half of the other property (common.tercile_within_halves), and Q1->Q4 as in 'tercile'. All 55
             cells can be tested.
  mixed      UTS at P40 and toughness at P60, and the reverse
  sweep      a common percentile of 30, 35, ..., 70 (headline cells)
It also checks the Q3->Q4 and Q1->Q2 cells with a continuous measure (the partial correlation with toughness at fixed
UTS inside the strong half, UTS >= median, and inside the weak half), because Q4 is also stronger than Q3 on average.
Writes out/thresholds_*.csv and out/s1_thresholds.txt.
"""
import numpy as np, pandas as pd
from scipy import stats
from common import load, quad, quad_pct, quad_tercile, tercile_within_halves, cells, partial_r, FEATS, SHORT, save_csv, write_log

M, u, t, nets, cube = load()
X = M[FEATS].to_numpy(float)
out = []; P = out.append
NMIN = 5

SCHEMES = {'P50 (median, published)': dict(Q=quad(u, t)), 'P40': dict(Q=quad_pct(u, t, 40)), 'P60': dict(Q=quad_pct(u, t, 60)),
           'tercile (both axes, middle dropped)': dict(Q=quad_tercile(u, t)),
           'tercile within halves': dict(Q=None, groups=tercile_within_halves(u, t)),
           'mixed: UTS P40, T P60': dict(Q=quad_pct(u, t, 40, 60)), 'mixed: UTS P60, T P40': dict(Q=quad_pct(u, t, 60, 40))}
assert (SCHEMES['P50 (median, published)']['Q'] == quad_pct(u, t, 50)).all()

ref = cells(X, SCHEMES['P50 (median, published)']['Q'])
assert int(ref.sig_bh.sum()) == 31
sig0 = ref.sig_bh.to_numpy(); sgn0 = np.sign(ref.cohens_d.to_numpy())
long, summ = [], []
for name, sc in SCHEMES.items():
    S = cells(X, sc['Q'], nmin=NMIN, groups=sc.get('groups')); S.insert(0, 'scheme', name); long.append(S)
    sig = S.sig_bh.to_numpy(); sgn = np.sign(S.cohens_d.to_numpy()); tested = S.welch_p.notna().to_numpy()
    keep = sig0 & sig & (sgn == sgn0)
    ns = S.groupby('tr')[['n_from', 'n_to']].first()
    summ.append(dict(scheme=name, n_Q1toQ2=f"{ns.loc['Q1->Q2', 'n_from']}/{ns.loc['Q1->Q2', 'n_to']}", n_Q3toQ4=f"{ns.loc['Q3->Q4', 'n_from']}/{ns.loc['Q3->Q4', 'n_to']}",
                     n_Q1toQ3=f"{ns.loc['Q1->Q3', 'n_from']}/{ns.loc['Q1->Q3', 'n_to']}", n_Q2toQ4=f"{ns.loc['Q2->Q4', 'n_from']}/{ns.loc['Q2->Q4', 'n_to']}",
                     n_Q1toQ4=f"{ns.loc['Q1->Q4', 'n_from']}/{ns.loc['Q1->Q4', 'n_to']}",
                     cells_tested=int(tested.sum()), bh_sig=int(sig.sum()),
                     orig_sig_tested=int((sig0 & tested).sum()), orig_sig_kept_same_sign=int(keep.sum()),
                     orig_sig_lost=int((sig0 & tested & ~sig).sum()), orig_sig_flipped=int((sig0 & sig & (sgn != sgn0)).sum()),
                     new_sig=int((~sig0 & sig).sum()), same_sign_tested=int(((sgn == sgn0) & tested).sum()),
                     r_d_vs_published=float(np.corrcoef(S.cohens_d[tested], ref.cohens_d[tested])[0, 1])))
L = pd.concat(long, ignore_index=True); save_csv(L, 'thresholds_cells_long.csv', index=False)
SU = pd.DataFrame(summ); save_csv(SU, 'thresholds_summary.csv', index=False)
P("Summary (group sizes from/to per transition; 'kept' = BH-significant under the published split and still BH-significant with the same sign)")
P(SU.to_string(index=False))

for name in ('P40', 'P60', 'tercile (both axes, middle dropped)', 'tercile within halves'):
    S = L[L.scheme == name].reset_index(drop=True); sig = S.sig_bh.to_numpy(); tested = S.welch_p.notna().to_numpy()
    lost = S[sig0 & tested & ~sig]; new = S[~sig0 & sig]
    for lab, D in (('originally significant cells that lose BH significance', lost), ('newly significant cells', new)):
        P(f"\n{name}: {lab} ({len(D)})")
        if len(D):
            tmp = D[['key', 'tr', 'n_from', 'n_to', 'cohens_d', 'bh_q']].copy(); tmp['d_published'] = ref.cohens_d.to_numpy()[D.index]
            P(tmp.round(3).to_string(index=False))

HEAD = [('max_betweenness', 'Q1->Q3'), ('max_betweenness', 'Q2->Q4'), ('max_betweenness', 'Q1->Q4'),
        ('avg_eigenvector_cent', 'Q1->Q3'), ('avg_eigenvector_cent', 'Q2->Q4'), ('avg_eigenvector_cent', 'Q1->Q4'),
        ('lambda_2', 'Q1->Q3'), ('lambda_2', 'Q2->Q4'), ('lambda_2', 'Q1->Q4'),
        ('num_bridges', 'Q3->Q4'), ('deg_bimodality', 'Q3->Q4'),
        ('max_betweenness', 'Q1->Q2'), ('num_bridges', 'Q1->Q2'), ('deg_bimodality', 'Q1->Q2')]
H = []
for k, tr in HEAD:
    row = dict(metric=SHORT[k], transition=tr)
    for name in SCHEMES:
        c = L[(L.scheme == name) & (L.key == k) & (L.tr == tr)].iloc[0]
        row[f'd | {name}'] = c.cohens_d; row[f'q | {name}'] = c.bh_q
    H.append(row)
H = pd.DataFrame(H); save_csv(H, 'thresholds_headline.csv', index=False)
P("\nHeadline cells, d (BH q) per scheme: " + " | ".join(SCHEMES))
for _, r in H.iterrows():
    s = f"{r.metric:16s} {r.transition:7s}"
    for name in SCHEMES:
        d, q = r[f'd | {name}'], r[f'q | {name}']
        s += " |   untestable   " if np.isnan(q) else f" | {d:+.2f} ({q:.1e})"
    P(s)

SW, SWS = [], []
for pc in range(30, 71, 5):
    Qp = quad_pct(u, t, pc); S = cells(X, Qp)
    sig = S.sig_bh.to_numpy(); sgn = np.sign(S.cohens_d.to_numpy())
    SWS.append(dict(percentile=pc, sizes='/'.join(str(int((Qp == q).sum())) for q in (1, 2, 3, 4)), bh_sig=int(sig.sum()),
                    orig_sig_kept_same_sign=int((sig0 & sig & (sgn == sgn0)).sum()), orig_sig_sign_flipped=int((sig0 & (sgn != sgn0)).sum()),
                    lost='; '.join(f"{SHORT[k]} {tr} (q={q:.3f})" for k, tr, q in S[sig0 & ~sig][['key', 'tr', 'bh_q']].values)))
    for k, tr in HEAD[:11]:
        c = S[(S.key == k) & (S.tr == tr)].iloc[0]
        SW.append(dict(percentile=pc, metric=SHORT[k], transition=tr, n_from=c.n_from, n_to=c.n_to, d=c.cohens_d, bh_q=c.bh_q, n_sig_55=int(S.sig_bh.sum())))
SW = pd.DataFrame(SW); save_csv(SW, 'thresholds_sweep.csv', index=False)
SWS = pd.DataFrame(SWS); save_csv(SWS, 'thresholds_sweep_summary.csv', index=False)
P("\nSweep of a common percentile (30-70): quadrant sizes, BH-significant cells of 55, originally significant cells kept (same sign), sign flips, cells lost")
P(SWS.to_string(index=False))
P("headline cells significant (q < 0.05) at each percentile (1 = yes; the sign never changes)")
assert ((np.sign(SW.d) == np.sign(SW.merge(ref.assign(metric=ref.key.map(SHORT), transition=ref.tr)[['metric', 'transition', 'cohens_d']], on=['metric', 'transition']).cohens_d))).all()
P(SW.assign(ok=(SW.bh_q < .05)).pivot_table(index=['metric', 'transition'], columns='percentile', values='ok', aggfunc='first').astype(int).to_string())
P("d at each percentile")
P(SW.pivot_table(index=['metric', 'transition'], columns='percentile', values='d', aggfunc='first').round(2).to_string())

# continuous check of the toughening cells (Q4 is also stronger than Q3, and Q2 than Q1)
Q = quad(u, t)
P(f"\nMean UTS by quadrant: " + ", ".join(f"Q{q} {u[Q == q].mean():.3f}" for q in (1, 2, 3, 4)) +
  f"; Welch p (Q3 vs Q4 UTS) {stats.ttest_ind(u[Q == 3], u[Q == 4], equal_var=False).pvalue:.1e}, (Q1 vs Q2 UTS) {stats.ttest_ind(u[Q == 1], u[Q == 2], equal_var=False).pvalue:.1e}")
CC = []
for half, m in (('strong half (UTS >= median; Q3+Q4)', u >= np.median(u)), ('weak half (UTS < median; Q1+Q2)', u < np.median(u)), ('all networks', np.ones_like(u, bool))):
    for j, f in enumerate(FEATS):
        x = X[m, j]; n = int(m.sum())
        r_raw = stats.pearsonr(x, t[m])[0]; r_p = partial_r(x, t[m], u[m])
        tt = r_p * np.sqrt((n - 3) / (1 - r_p ** 2)); pv = 2 * stats.t.sf(abs(tt), n - 3)
        CC.append(dict(subset=half, n=n, metric=SHORT[f], r_toughness=r_raw, partial_r_toughness_given_UTS=r_p, p=pv))
CC = pd.DataFrame(CC); save_csv(CC, 'thresholds_toughness_within_halves.csv', index=False)
P("Correlation with toughness, raw and at fixed UTS, inside each half (continuous version of the toughening cells)")
P(CC.pivot_table(index='metric', columns='subset', values=['r_toughness', 'partial_r_toughness_given_UTS'], sort=False).round(3).to_string())
P("p of the partial correlations"); P(CC.pivot_table(index='metric', columns='subset', values='p', sort=False).map(lambda v: f"{v:.3f}").to_string())

write_log('s1_thresholds.txt', out)
print("\n".join(out))
