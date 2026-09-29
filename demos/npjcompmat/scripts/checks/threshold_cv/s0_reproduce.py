"""Step 0. Reproduce the published numbers before anything new.

Checks the 55 cells against data/derived/figure_stats/figure6_stats.csv (the values behind Fig. 7 and
tables_checks/table_cells.tex) and the 44 correlations against figure6_continuous.csv (tables_checks/table_continuous.tex).
Also prints the dangling-end numbers of the paper (0 to 22% of the sites, partial correlation 0.56 of lambda_2 and
lambda_2,core with UTS at fixed dangling-end fraction) and the reliabilities of the network means of
tables_checks/table_variance.tex, with the decomposition code of the variance cell of generate_checks.ipynb.
Writes out/s0_reproduce.txt and out/reliability.csv (read by s2_cv_ridge.py).
"""
import numpy as np, pandas as pd
from scipy import stats
from common import load, quad, cells, FEATS, NAMES, derived, read_csv_exact, resid, partial_r, save_csv, write_log

M, u, t, nets, cube = load()
X = M[FEATS].to_numpy(float); fd = M['frac_dangling'].to_numpy(float)
out = []; P = out.append

# 1) the 55 cells
Q = quad(u, t)
S = cells(X, Q)
ref = read_csv_exact(derived('figure_stats', 'figure6_stats.csv'))
assert (ref.metric.values == S.metric.values).all() and (ref.transition.values == S.transition.values).all()
dd = np.abs(ref.cohens_d - S.cohens_d).max(); dp = np.abs(ref.welch_p - S.welch_p).max(); dq = np.abs(ref.bh_q - S.bh_q).max()
P(f"quadrant sizes {[int((Q == q).sum()) for q in (1, 2, 3, 4)]} (reference 128/35/35/129)")
P(f"55 cells vs figure6_stats.csv: max|d diff| {dd:.1e}  max|p diff| {dp:.1e}  max|q diff| {dq:.1e}")
P(f"BH-significant cells: {int(S.sig_bh.sum())} of 55 (reference {int(ref.sig_bh.sum())})")
assert dd < 1e-12 and dp < 1e-12 and dq < 1e-12 and S.sig_bh.sum() == ref.sig_bh.sum() == 31

# 2) Pearson and partial correlations of each descriptor with UTS and toughness
C = read_csv_exact(derived('figure_stats', 'figure6_continuous.csv'))
mx = 0.0
for j, f in enumerate(FEATS):
    x = X[:, j]; row = C[C.metric == NAMES[f]].iloc[0]
    vals = [stats.pearsonr(x, t)[0], stats.pearsonr(x, u)[0], stats.pearsonr(resid(x, u), resid(t, u))[0], stats.pearsonr(resid(x, t), resid(u, t))[0]]
    mx = max(mx, *np.abs(np.array(vals) - row[['r_toughness', 'r_uts', 'partial_r_T_given_U', 'partial_r_U_given_T']].to_numpy(float)))
P(f"44 correlations vs figure6_continuous.csv: max |difference| {mx:.1e}")
assert mx < 1e-12
i_bc = FEATS.index('max_betweenness'); i_ev = FEATS.index('avg_eigenvector_cent')
P(f"  e.g. max betweenness r(UTS) {stats.pearsonr(X[:, i_bc], u)[0]:+.3f}, mean eigenvector centrality r(UTS) {stats.pearsonr(X[:, i_ev], u)[0]:+.3f}")

# 3) dangling-end numbers of the paper
P(f"dangling-end fraction range {fd.min():.3f}-{fd.max():.3f} (0 to 22% in the paper)")
P(f"r(dangling-end fraction, bridges) = {np.corrcoef(fd, M['num_bridges'])[0, 1]:.4f}")
for f in ('lambda_2', 'lambda_2_core'):
    P(f"partial r({f}, UTS | dangling-end fraction) = {partial_r(M[f].to_numpy(float), u, fd):.4f} (0.56 in the paper)")
P(f"r(UTS, toughness) = {np.corrcoef(u, t)[0, 1]:.3f}")


# 4) reliability of the network means (decomposition code of the variance cell of generate_checks.ipynb)
def decompose(Y):
    n, k, m = Y.shape; gm = Y.mean(); net = Y.mean((1, 2))[:, None, None]; axm = Y.mean((0, 2))[None, :, None]; cell = Y.mean(2)[:, :, None]
    SS_net = k * m * ((net - gm) ** 2).sum(); SS_ax = n * m * ((axm - gm) ** 2).sum(); SS_int = m * ((cell - net - axm + gm) ** 2).sum(); SS_err = ((Y - cell) ** 2).sum()
    MS_net, MS_ax, MS_int, MS_err = SS_net / (n - 1), SS_ax / (k - 1), SS_int / ((n - 1) * (k - 1)), SS_err / (n * k * (m - 1))
    return dict(v_net=max((MS_net - MS_int) / (k * m), 0), v_ax=max((MS_ax - MS_int) / (n * m), 0), v_int=max((MS_int - MS_err) / m, 0), v_err=MS_err,
                F_int=MS_int / MS_err, F_ax=MS_ax / MS_int, F_net=MS_net / MS_int)


def rel(d, k, m): return d['v_net'] / (d['v_net'] + d['v_int'] / k + d['v_err'] / (k * m))


def rel_fixed(d, k, m): return (d['v_net'] + d['v_int'] / k) / (d['v_net'] + d['v_int'] / k + d['v_err'] / (k * m))


R = []
for p in ('uts', 'toughness'):
    d4 = decompose(cube[p])
    R.append(dict(response=p, rel_fixed_axes_12pull=rel_fixed(d4, 3, 4), rel_random_axes_12pull=rel(d4, 3, 4)))
    P(f"reliability of the 12-pull mean, {p}: axes fixed (new seeds) {R[-1]['rel_fixed_axes_12pull']:.4f}; axes random (new loading directions) {R[-1]['rel_random_axes_12pull']:.4f}")
save_csv(pd.DataFrame(R), 'reliability.csv', index=False)
P("(table_variance.tex: axes fixed 0.93 / 0.80, axes random 0.50 / 0.21)")

write_log('s0_reproduce.txt', out)
print("\n".join(out))
