"""Step 2. Cross-validated ridge regression of the network means.

The unit is the network (327). The outer loop is 10-fold cross-validation repeated 20 times (partition r uses numpy
default_rng(r)), and every model sees the same 200 splits, so the differences between models are paired. Inside each
training set the predictors are standardized with the training mean and SD, and the ridge penalty alpha is chosen from
91 values (1e-4 to 1e5) by the exact leave-one-out error on the training set (the closed form of scikit-learn's RidgeCV),
with an unpenalized intercept. Test networks never enter the standardization, the choice of alpha or the fit.

Reported are the pooled out-of-fold R^2 of each repeat (1 - SSE/SST over all 327 networks), its mean, SD and range over
the 20 repeats, and the mean and SD of the 200 per-fold R^2 values (each against the mean of its test fold).

Predictor sets are D (the dangling-end fraction), G (the 11 descriptors of Fig. 7), D + G, Pf (the site degree
fractions f = 0..6, i.e., the composition the generator prescribes) and Pf + G. Toughness at fixed UTS is treated in two
ways, (a) with UTS as a predictor (UTS alone vs UTS + D, UTS + G and UTS + D + G) and (b) as the toughness residual after
an OLS fit on UTS made inside each training set, predicted from D, G and D + G. For the models whose gain is small a
permutation null (200 permutations of the descriptor rows, which keeps the pairing of UTS and toughness) gives a p value.
Also reported are the add-one (D + one descriptor) and drop-one (D + G minus one descriptor) changes in CV R^2.
Reads out/reliability.csv from s0_reproduce.py. Writes out/cv_*.csv and out/s2_cv_ridge.txt. Takes a few minutes.
"""
import os, time
import numpy as np, pandas as pd
from common import load, FEATS, SHORT, OUT, save_csv, write_log
from ridge_cv import ALPHAS, ridge_fit, make_splits, cv as _cv

M, u, t, nets, cube = load()
n = len(u)
G = M[FEATS].to_numpy(float); D = M[['frac_dangling']].to_numpy(float)
PF = (M[[f'cnt_d{i}' for i in range(7)]].to_numpy(float) / 216)
U1 = u[:, None]
rel_path = os.path.join(OUT, 'reliability.csv')
if not os.path.exists(rel_path):
    raise FileNotFoundError(f"{rel_path} not found. Run s0_reproduce.py first.")
REL = pd.read_csv(rel_path).set_index('response')
NREP, K = 20, 10
SPLITS = make_splits(n, NREP, K)
out = []; P = out.append


def cv(X, y, resid_on=None):
    return _cv(X, y, SPLITS, resid_on)


MODELS = [  # (response, label, X, resid_on)
    ('UTS', 'D (dangling-end fraction)', D, None), ('UTS', 'G (11 descriptors)', G, None), ('UTS', 'D + G', np.hstack([D, G]), None),
    ('UTS', 'Pf (degree fractions f=0..6)', PF, None), ('UTS', 'Pf + G', np.hstack([PF, G]), None),
    ('toughness', 'D (dangling-end fraction)', D, None), ('toughness', 'G (11 descriptors)', G, None), ('toughness', 'D + G', np.hstack([D, G]), None),
    ('toughness', 'Pf (degree fractions f=0..6)', PF, None), ('toughness', 'Pf + G', np.hstack([PF, G]), None),
    ('toughness', 'UTS', U1, None), ('toughness', 'UTS + D', np.hstack([U1, D]), None), ('toughness', 'UTS + G', np.hstack([U1, G]), None),
    ('toughness', 'UTS + D + G', np.hstack([U1, D, G]), None),
    ('toughness | UTS (residual)', 'D (dangling-end fraction)', D, u), ('toughness | UTS (residual)', 'G (11 descriptors)', G, u),
    ('toughness | UTS (residual)', 'D + G', np.hstack([D, G]), u)]
Y = {'UTS': u, 'toughness': t, 'toughness | UTS (residual)': t}

t0 = time.time()
rows, per_rep = [], {}
for resp, lab, X, ro in MODELS:
    pooled, folds, al = cv(X, Y[resp], ro)
    per_rep[(resp, lab)] = pooled
    rel_key = 'uts' if resp == 'UTS' else 'toughness'
    rows.append(dict(response=resp, predictors=lab, n_predictors=X.shape[1], R2_mean=pooled.mean(), R2_sd_repeats=pooled.std(ddof=1),
                     R2_min=pooled.min(), R2_max=pooled.max(), fold_R2_mean=folds.mean(), fold_R2_sd=folds.std(ddof=1),
                     fold_R2_p2_5=np.percentile(folds, 2.5), fold_R2_p97_5=np.percentile(folds, 97.5),
                     alpha_median=np.median(al),
                     R2_over_rel_fixed=pooled.mean() / REL.loc[rel_key, 'rel_fixed_axes_12pull'] if resp != 'toughness | UTS (residual)' else np.nan,
                     R2_over_rel_random=pooled.mean() / REL.loc[rel_key, 'rel_random_axes_12pull'] if resp != 'toughness | UTS (residual)' else np.nan))
R = pd.DataFrame(rows); save_csv(R, 'cv_r2_summary.csv', index=False)
save_csv(pd.DataFrame({f"{k[0]} ~ {k[1]}": v for k, v in per_rep.items()}), 'cv_r2_per_repeat.csv', index_label='repeat')
print(f"cross-validation of the {len(MODELS)} models took {time.time() - t0:.0f} s")
P(f"Repeated {K}-fold CV x {NREP} repeats, ridge on standardized predictors, alpha by LOO inside each training set")
P(R[['response', 'predictors', 'n_predictors', 'R2_mean', 'R2_sd_repeats', 'R2_min', 'R2_max', 'fold_R2_mean', 'fold_R2_sd', 'alpha_median',
     'R2_over_rel_fixed', 'R2_over_rel_random']].round(3).to_string(index=False))
P("Reliability of the 12-pull means (s0): " + "; ".join(f"{k}: axes fixed {r.rel_fixed_axes_12pull:.3f}, axes random {r.rel_random_axes_12pull:.3f}" for k, r in REL.iterrows()))

# paired gains
GAINS = [('UTS', 'D + G', 'D (dangling-end fraction)'), ('UTS', 'G (11 descriptors)', 'D (dangling-end fraction)'),
         ('UTS', 'Pf + G', 'Pf (degree fractions f=0..6)'),
         ('toughness', 'D + G', 'D (dangling-end fraction)'), ('toughness', 'Pf + G', 'Pf (degree fractions f=0..6)'),
         ('toughness', 'UTS + G', 'UTS'), ('toughness', 'UTS + D', 'UTS'), ('toughness', 'UTS + D + G', 'UTS + D')]
GR = []
for resp, a, b in GAINS:
    dlt = per_rep[(resp, a)] - per_rep[(resp, b)]
    GR.append(dict(response=resp, model=a, baseline=b, dR2_mean=dlt.mean(), dR2_min=dlt.min(), dR2_max=dlt.max(), repeats_positive=int((dlt > 0).sum())))

# permutation null for the small toughness gains (the rows of D and G are permuted together, UTS-toughness pairing kept)
NPERM = 200
rng = np.random.default_rng(12345)
null = {k: [] for k in ('UTS + G vs UTS', 'UTS + D + G vs UTS + D', 'resid ~ G', 'resid ~ D + G')}
base_u = per_rep[('toughness', 'UTS')].mean()
t0 = time.time()
for i in range(NPERM):
    pi = rng.permutation(n); Dp, Gp = D[pi], G[pi]
    null['UTS + G vs UTS'].append(cv(np.hstack([U1, Gp]), t, None)[0].mean() - base_u)
    null['UTS + D + G vs UTS + D'].append(cv(np.hstack([U1, Dp, Gp]), t, None)[0].mean() - cv(np.hstack([U1, Dp]), t, None)[0].mean())
    null['resid ~ G'].append(cv(Gp, t, u)[0].mean())
    null['resid ~ D + G'].append(cv(np.hstack([Dp, Gp]), t, u)[0].mean())
obs = {'UTS + G vs UTS': per_rep[('toughness', 'UTS + G')].mean() - base_u,
       'UTS + D + G vs UTS + D': per_rep[('toughness', 'UTS + D + G')].mean() - per_rep[('toughness', 'UTS + D')].mean(),
       'resid ~ G': per_rep[('toughness | UTS (residual)', 'G (11 descriptors)')].mean(),
       'resid ~ D + G': per_rep[('toughness | UTS (residual)', 'D + G')].mean()}
PN = pd.DataFrame([dict(statistic=k, observed=obs[k], null_mean=np.mean(v), null_p95=np.percentile(v, 95),
                        p_perm=(1 + np.sum(np.array(v) >= obs[k])) / (NPERM + 1)) for k, v in null.items()])
save_csv(PN, 'cv_permutation_toughness_fixed_uts.csv', index=False)
GR = pd.DataFrame(GR); save_csv(GR, 'cv_gains.csv', index=False)
print(f"permutation null took {time.time() - t0:.0f} s")
P("\nPaired gains in pooled CV R^2 (same 20 x 10 splits)"); P(GR.round(4).to_string(index=False))
P(f"\nPermutation null ({NPERM} permutations of the descriptor rows): toughness at fixed UTS"); P(PN.round(4).to_string(index=False))

# add-one (D + one descriptor vs D) and drop-one (D + G minus one vs D + G), UTS and toughness
AO = []
for resp, y in (('UTS', u), ('toughness', t)):
    base_D = per_rep[(resp, 'D (dangling-end fraction)')]; full = per_rep[(resp, 'D + G')]
    for j, f in enumerate(FEATS):
        add = cv(np.hstack([D, G[:, [j]]]), y)[0]
        drop = cv(np.hstack([D, np.delete(G, j, axis=1)]), y)[0]
        AO.append(dict(response=resp, metric=SHORT[f], R2_D_plus_this=add.mean(), gain_over_D=(add - base_D).mean(),
                       loss_when_dropped_from_DG=(full - drop).mean(), drop_loss_min=(full - drop).min(), drop_loss_max=(full - drop).max()))
AO = pd.DataFrame(AO); save_csv(AO, 'cv_add_one_drop_one.csv', index=False)
P("\nAdd-one (D + one descriptor) and drop-one (from D + G) changes in pooled CV R^2, mean over 20 repeats")
P(AO.round(4).to_string(index=False))

# full-data ridge coefficients of D + G (standardized), for reference only (collinear, not interpreted one by one)
CO = []
for resp, y in (('UTS', u), ('toughness', t)):
    f, a, beta = ridge_fit(np.hstack([D, G]), y)
    for nm, b in zip(['frac_dangling'] + FEATS, beta):
        CO.append(dict(response=resp, alpha=a, predictor=SHORT[nm], std_coef=b))
save_csv(pd.DataFrame(CO), 'cv_fulldata_ridge_coefficients.csv', index=False)

write_log('s2_cv_ridge.txt', out)
print("\n".join(out))

# optional check of the ridge and leave-one-out code against scikit-learn (printed only, not written to out/)
try:
    import warnings; warnings.filterwarnings('ignore')
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    te = SPLITS[0][0]; tr = np.setdiff1d(np.arange(n), te); X = np.hstack([D, G])
    f, a, _ = ridge_fit(X[tr], u[tr])
    sk = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(X[tr], u[tr])
    print(f"\nscikit-learn check (fold 1, UTS ~ D + G): alpha {a:.4g} vs {sk[-1].alpha_:.4g}; max |prediction difference| {np.abs(f(X[te]) - sk.predict(X[te])).max():.1e}")
except ImportError as e:
    print(f"\nscikit-learn check skipped ({e})")
