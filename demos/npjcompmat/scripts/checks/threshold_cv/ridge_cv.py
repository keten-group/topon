"""Ridge regression with the penalty chosen inside the training set, and repeated K-fold cross-validation over networks
(numpy only).

ridge_fit standardizes the predictors with the training mean and SD, leaves the intercept unpenalized and chooses alpha
from ALPHAS by the exact leave-one-out error on the training set (the closed form of scikit-learn's RidgeCV, compared
with it in s2_cv_ridge.py). cv runs repeated K-fold, where partition r is numpy default_rng(r).permutation split into K
nearly equal folds.
"""
import numpy as np

ALPHAS = np.logspace(-4, 5, 91)


def ridge_fit(Xtr, ytr):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0
    Z = (Xtr - mu) / sd; ym = ytr.mean(); yc = ytr - ym; nt = len(ytr)
    Uu, s, Vt = np.linalg.svd(Z, full_matrices=False)
    Uty = Uu.T @ yc; s2 = s ** 2
    F = s2[None, :] / (s2[None, :] + ALPHAS[:, None])                  # (n_alpha, p) shrinkage factors
    yhat = Uu @ (F * Uty).T; h = 1.0 / nt + (Uu ** 2) @ F.T              # (n_train, n_alpha)
    loo = np.mean(((yc[:, None] - yhat) / (1 - h)) ** 2, axis=0)         # exact leave-one-out MSE per alpha
    ba = ALPHAS[int(np.argmin(loo))]
    beta = Vt.T @ (s / (s2 + ba) * Uty)
    return (lambda Xte: ym + ((Xte - mu) / sd) @ beta), ba, beta


def make_splits(n, nrep=20, k=10):
    return [np.array_split(np.random.default_rng(r).permutation(n), k) for r in range(nrep)]


def cv(X, y, splits, resid_on=None, return_pred=False):
    """Pooled out-of-fold R^2 per repeat, per-fold R^2 and the chosen alphas (and the out-of-fold predictions per repeat).
    If resid_on is given, the target is y minus an OLS fit on resid_on made inside each training set."""
    n = len(y); pooled, folds, alphas, preds = [], [], [], []
    for sp in splits:
        pred = np.empty(n); targ = np.empty(n)
        for te in sp:
            tr = np.setdiff1d(np.arange(n), te)
            ytr, yte = y[tr].copy(), y[te].copy()
            if resid_on is not None:
                A = np.column_stack([np.ones(len(tr)), resid_on[tr]]); c = np.linalg.lstsq(A, ytr, rcond=None)[0]
                ytr = ytr - A @ c; yte = yte - np.column_stack([np.ones(len(te)), resid_on[te]]) @ c
            f, a, _ = ridge_fit(X[tr], ytr); alphas.append(a)
            pred[te] = f(X[te]); targ[te] = yte
            folds.append(1 - np.sum((yte - pred[te]) ** 2) / np.sum((yte - yte.mean()) ** 2))
        pooled.append(1 - np.sum((targ - pred) ** 2) / np.sum((targ - targ.mean()) ** 2)); preds.append(pred)
    res = (np.array(pooled), np.array(folds), np.array(alphas))
    return res + (np.array(preds),) if return_pred else res
