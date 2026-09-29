"""Step 3. Collinearity of the descriptors and partial correlations at fixed dangling-end fraction.

VIF_j = 1 / (1 - R^2_j) from the inverse correlation matrix, for the 11 descriptors and for the 11 with the dangling-end
fraction added, and the condition number of the correlation matrix. Partial correlation of each descriptor with UTS and
with toughness at fixed dangling-end fraction (linear adjustment, as for the partial correlation of 0.56 in the paper),
with a t-test p (n - 3 df) and a 95% percentile bootstrap interval over networks (5,000 resamples, numpy
default_rng(0)). Also toughness at fixed dangling-end fraction and UTS together.
Writes out/collinearity_vif.csv, out/collinearity_corr_matrix.csv, out/partial_corr_dangling.csv and
out/s3_collinearity.txt.
"""
import numpy as np, pandas as pd
from scipy import stats
from common import load, bh, FEATS, SHORT, save_csv, write_log

M, u, t, nets, cube = load()
n = len(u)
G = M[FEATS].to_numpy(float); fd = M['frac_dangling'].to_numpy(float)
out = []; P = out.append


def vif(X):
    R = np.corrcoef(X, rowvar=False); return np.diag(np.linalg.inv(R)), np.linalg.cond(R)


v11, c11 = vif(G); v12, c12 = vif(np.column_stack([fd, G]))
V = pd.DataFrame(dict(metric=[SHORT[f] for f in FEATS], r_with_dangling_fraction=[np.corrcoef(G[:, j], fd)[0, 1] for j in range(11)],
                      VIF_11_descriptors=v11, VIF_with_dangling_fraction=v12[1:]))
V = pd.concat([V, pd.DataFrame([dict(metric='dangling frac.', r_with_dangling_fraction=1.0, VIF_11_descriptors=np.nan, VIF_with_dangling_fraction=v12[0])])], ignore_index=True)
save_csv(V, 'collinearity_vif.csv', index=False)
C = pd.DataFrame(np.corrcoef(np.column_stack([fd, G, u, t]), rowvar=False), index=['dangling frac.'] + [SHORT[f] for f in FEATS] + ['UTS', 'toughness'],
                 columns=['dangling frac.'] + [SHORT[f] for f in FEATS] + ['UTS', 'toughness'])
save_csv(C, 'collinearity_corr_matrix.csv')
P(f"Variance inflation factors (condition number of the correlation matrix: 11 descriptors {c11:.0f}; with dangling-end fraction {c12:.0f})")
P(V.round(2).to_string(index=False))
P(f"descriptors with VIF > 10 (11 descriptors): {int((v11 > 10).sum())}; > 5: {int((v11 > 5).sum())}")
cm = C.iloc[1:12, 1:12].to_numpy(); iu = np.triu_indices(11, 1)
P(f"pairwise |r| among the 11 descriptors: median {np.median(np.abs(cm[iu])):.2f}; pairs with |r| > 0.8: {int((np.abs(cm[iu]) > 0.8).sum())} of 55")
P("correlation matrix (dangling fraction, 11 descriptors, UTS, toughness)"); P(C.round(2).to_string())


def presid(x, Z):
    Z = np.column_stack([np.ones(len(x)), Z]); return x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]


def pr(x, y, Z):
    return np.corrcoef(presid(x, Z), presid(y, Z))[0, 1]


def ptest(r, k):
    df = n - 2 - k; tt = r * np.sqrt(df / (1 - r ** 2)); return 2 * stats.t.sf(abs(tt), df)


NB = 5000
rng = np.random.default_rng(0)
B = np.empty((NB, 11, 3))
for b in range(NB):
    i = rng.integers(0, n, n); fb, ub, tb = fd[i], u[i], t[i]
    for j in range(11):
        x = G[i, j]
        B[b, j] = pr(x, ub, fb), pr(x, tb, fb), pr(x, tb, np.column_stack([fb, ub]))
rows = []
for j, f in enumerate(FEATS):
    x = G[:, j]
    rU, rT, rTU = pr(x, u, fd), pr(x, t, fd), pr(x, t, np.column_stack([fd, u]))
    (lu, hu), (lt, ht), (ltu, htu) = [np.percentile(B[:, j, k], [2.5, 97.5]) for k in range(3)]
    rows.append(dict(metric=SHORT[f], r_UTS=stats.pearsonr(x, u)[0], partial_r_UTS_given_D=rU, ci_low_UTS=lu, ci_high_UTS=hu, p_UTS=ptest(rU, 1),
                     r_toughness=stats.pearsonr(x, t)[0], partial_r_T_given_D=rT, ci_low_T=lt, ci_high_T=ht, p_T=ptest(rT, 1),
                     partial_r_T_given_D_UTS=rTU, ci_low_T_DU=ltu, ci_high_T_DU=htu, p_T_DU=ptest(rTU, 2)))
PC = pd.DataFrame(rows)
p_all = np.concatenate([PC.p_UTS, PC.p_T])   # BH over the 22 partial correlations with UTS and toughness
q = bh(p_all); PC['bh_q_UTS'] = q[:11]; PC['bh_q_T'] = q[11:]
save_csv(PC, 'partial_corr_dangling.csv', index=False)
P(f"\nPartial correlations at fixed dangling-end fraction D (n = {n}; 95% bootstrap CI over networks, {NB} resamples; BH over the 22 UTS and toughness tests)")
P(f"check: lambda_2 {PC.partial_r_UTS_given_D[0]:.4f}, lambda_2,core {PC.partial_r_UTS_given_D[1]:.4f} (0.56 in the paper)")
show = pd.DataFrame({'metric': PC.metric})
fp = lambda v: f"{v:.1e}" if v < 1e-3 else f"{v:.3f}"
for a, r0, r1, lo, hi, pc in (('UTS', 'r_UTS', 'partial_r_UTS_given_D', 'ci_low_UTS', 'ci_high_UTS', 'p_UTS'),
                               ('T', 'r_toughness', 'partial_r_T_given_D', 'ci_low_T', 'ci_high_T', 'p_T'),
                               ('T|D,UTS', None, 'partial_r_T_given_D_UTS', 'ci_low_T_DU', 'ci_high_T_DU', 'p_T_DU')):
    if r0: show[f'r {a}'] = PC[r0].map(lambda v: f"{v:+.3f}")
    show[f'partial {a}|D' if a != 'T|D,UTS' else 'partial T|D,UTS'] = PC[r1].map(lambda v: f"{v:+.3f}")
    show[f'95% CI {a}'] = [f"[{l:+.2f}, {h:+.2f}]" for l, h in zip(PC[lo], PC[hi])]
    show[f'p {a}'] = PC[pc].map(fp)
P(show.to_string(index=False))
write_log('s3_collinearity.txt', out)
print("\n".join(out))
