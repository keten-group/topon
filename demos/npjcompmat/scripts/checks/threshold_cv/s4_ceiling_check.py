"""Step 4. Which reliability bounds the CV R^2 of the graph descriptors?

The reliability of the 12-pull network means (tables_checks/table_variance.tex) is 0.93 for UTS and 0.80 for toughness
with respect to new velocity seeds (the three cube axes fixed), and 0.50 and 0.21 when the three axes are treated as a
random sample of loading directions. The second equals the mean covariance between networks of two different axes
divided by the variance of the 3-axis mean. s2_cv_ridge.py finds a CV R^2 of 0.68 for UTS from the 11 descriptors,
above 0.50. This script shows why. After the descriptor prediction is removed, the axis-specific residuals of a network
are negatively correlated across axes (a network strong along x is weaker along y and z), so the covariance between
axes understates the variance that network structure explains, and 0.50 and 0.21 are not upper bounds. The bound for
predicting the 12-pull means is the reliability with the axes fixed (0.93 and 0.80).
Writes out/s4_ceiling_check.txt and out/ceiling_check.csv.
"""
import numpy as np, pandas as pd
from common import load, FEATS, derived, read_csv_exact, save_csv, write_log
from ridge_cv import make_splits, cv

M, u, t, nets, cube = load()
n = len(u); G = M[FEATS].to_numpy(float)
SPL = make_splits(n, 20, 10)
out = []; P = out.append
rows = []
for p in ('uts', 'toughness'):
    Y = cube[p]; A = Y.mean(2); ybar = A.mean(1)                       # axis means (4 seeds), 3-axis means
    C = np.cov(A, rowvar=False); off = C[np.triu_indices(3, 1)].mean()
    v_err = ((Y - Y.mean(2, keepdims=True)) ** 2).sum() / (n * 3 * 3)   # seed variance of one pull (MS_err)
    pooled, _, _, preds = cv(G, ybar, SPL, return_pred=True)
    yhat = preds.mean(0)                                                 # out-of-fold prediction averaged over the 20 repeats
    E = A - yhat[:, None]; Ce = np.cov(E, rowvar=False); off_e = Ce[np.triu_indices(3, 1)].mean()
    r_e = np.corrcoef(E, rowvar=False)[np.triu_indices(3, 1)]
    single = [cv(G, A[:, k], SPL)[0].mean() for k in range(3)]
    row = dict(response=p, var_3axis_mean=ybar.var(ddof=1), mean_cov_between_axes=off,
               rel_random_axes=off / ybar.var(ddof=1), rel_fixed_axes=1 - (v_err / 12) / ybar.var(ddof=1),
               cv_R2_3axis_mean=pooled.mean(), resid_cov_between_axes=off_e, resid_corr_between_axes_mean=r_e.mean(),
               cv_R2_single_axis_mean=np.mean(single),
               single_axis_rel_random=off / np.diag(C).mean(), single_axis_rel_fixed=1 - (v_err / 4) / np.diag(C).mean())
    rows.append(row)
R = pd.DataFrame(rows); save_csv(R, 'ceiling_check.csv', index=False)
for _, r in R.iterrows():
    P(f"{r.response}: var(3-axis mean) {r.var_3axis_mean:.4g}; mean covariance between two axes {r.mean_cov_between_axes:.4g} "
      f"-> axes-random reliability {r.rel_random_axes:.3f}; fixed-axes reliability {r.rel_fixed_axes:.3f}")
    P(f"   CV R^2 of the 11 descriptors for the 3-axis mean {r.cv_R2_3axis_mean:.3f}; after removing the prediction, residual covariance between axes "
      f"{r.resid_cov_between_axes:+.4g} (mean correlation {r.resid_corr_between_axes_mean:+.3f})")
    P(f"   single axis (4-seed mean): CV R^2 {r.cv_R2_single_axis_mean:.3f}; axes-random reliability {r.single_axis_rel_random:.3f}; fixed {r.single_axis_rel_fixed:.3f}")
W = read_csv_exact(derived('cg_structure', 'task1_within_network_axis_correlations.csv'))
P("\nwithin-network axis correlations (data/derived/cg_structure/task1_within_network_axis_correlations.csv, panel f of the orientation check):")
P(W.round(3).to_string(index=False))
write_log('s4_ceiling_check.txt', out)
print("\n".join(out))
