"""Shared inputs and definitions for the threshold, cross-validation and collinearity checks of the quadrant analysis.

Inputs (read only, from data/derived/ of this companion):
  mechanics/mechanics_pulls_4seeds.csv   3,924 pulls (327 networks x 3 axes x 4 seeds)
  mechanics/mechanics_12pull.csv         graph descriptors of the 327 networks
  figure_stats/figure6_stats.csv and figure6_continuous.csv   reference values of the published analysis

The definitions are those of the statistics cell of generate_checks.ipynb (the code behind figure6_stats.csv and
figure6_continuous.csv).
  - Network UTS and toughness are the means of the 12 pulls (3 axes x 4 velocity seeds).
  - Quadrants are 1 + (t >= median t) + 2 (u >= median u), i.e., Q1 weak/brittle, Q2 weak/tough, Q3 strong/brittle and
    Q4 strong/tough.
  - Cycle rank is taken on the active graph, E - N_active + 1 (the deposited cyclomatic_idx).
  - Degree bimodality is (skew^2 + 1) / (excess kurtosis + 3) of the site degree sequence (216 sites, f = 0..6).
  - Cohen's d is (mean_to - mean_from) / pooled SD, with a Welch t test and Benjamini-Hochberg over the 55 cells.
  - The dangling-end fraction is the number of f = 1 sites divided by 216 (the deposited frac_dangling).

Outputs go to out/ next to this file. CSV and text files are written with LF line endings on every platform.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, ttest_ind

HERE = os.path.dirname(os.path.abspath(__file__))
COMPANION = os.path.normpath(os.path.join(HERE, '..', '..', '..'))      # demos/npjcompmat
DERIVED = os.path.join(COMPANION, 'data', 'derived')
OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)

FEATS = ['lambda_2', 'lambda_2_core', 'avg_path_len', 'cycle_rank', 'max_betweenness', 'num_bridges', 'deg_bimodality',
         'deg_std', 'deg_entropy', 'avg_eigenvector_cent', 'assortativity']
NAMES = {'lambda_2': 'Algebraic connectivity (lambda_2)', 'lambda_2_core': 'Algebraic connectivity excl. dangling ends (lambda_2,core)',
         'avg_path_len': 'Mean shortest path length (L)', 'cycle_rank': 'Cycle rank', 'max_betweenness': 'Max. betweenness centrality',
         'num_bridges': 'Bridges', 'deg_bimodality': 'Degree bimodality (D_bi)', 'deg_std': 'Degree heterogeneity (sigma_k)',
         'deg_entropy': 'Degree entropy (H)', 'avg_eigenvector_cent': 'Mean eigenvector centrality', 'assortativity': 'Assortativity (r)'}
SHORT = {'lambda_2': 'lambda_2', 'lambda_2_core': 'lambda_2,core', 'avg_path_len': 'L', 'cycle_rank': 'cycle rank',
         'max_betweenness': 'max betweenness', 'num_bridges': 'bridges', 'deg_bimodality': 'D_bi', 'deg_std': 'sigma_k',
         'deg_entropy': 'H', 'avg_eigenvector_cent': 'eigenvector', 'assortativity': 'assortativity', 'frac_dangling': 'dangling frac.'}
TR = {'Toughening (weak, Q1->Q2)': (1, 2), 'Toughening (strong, Q3->Q4)': (3, 4), 'Strengthening (brittle, Q1->Q3)': (1, 3),
      'Strengthening (tough, Q2->Q4)': (2, 4), 'Total (Q1->Q4)': (1, 4)}
TRS = {'Toughening (weak, Q1->Q2)': 'Q1->Q2', 'Toughening (strong, Q3->Q4)': 'Q3->Q4', 'Strengthening (brittle, Q1->Q3)': 'Q1->Q3',
       'Strengthening (tough, Q2->Q4)': 'Q2->Q4', 'Total (Q1->Q4)': 'Q1->Q4'}


def derived(*parts):
    """Path of an input file in data/derived/. A missing file raises FileNotFoundError with its name."""
    path = os.path.join(DERIVED, *parts)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required input not found: {path}")
    return path


def read_csv_exact(path, **kw):
    return pd.read_csv(path, float_precision='round_trip', **kw)


def save_csv(df, name, **kw):
    """Write a table to out/ with LF line endings."""
    df.to_csv(os.path.join(OUT, name), lineterminator='\n', **kw)


def write_log(name, lines):
    """Write the printed lines of a step to out/ with LF line endings."""
    with open(os.path.join(OUT, name), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write("\n".join(lines) + "\n")


def load():
    """Return (M, u, t, nets, cube), the descriptor table in pull-file network order, the 12-pull mean UTS and toughness,
    the network names and the pull cube (network x axis x seed)."""
    P = read_csv_exact(derived('mechanics', 'mechanics_pulls_4seeds.csv'))
    nets = list(dict.fromkeys(P['folder_name']))
    assert len(P) == 12 * len(nets) == 3924
    P = P.assign(_i=P['folder_name'].map({n: i for i, n in enumerate(nets)})).sort_values(['_i', 'axis', 'seed'])
    cube = {p: P[p].to_numpy().reshape(len(nets), 3, 4) for p in ('uts', 'toughness')}
    M = read_csv_exact(derived('mechanics', 'mechanics_12pull.csv')).set_index('folder_name').reindex(nets)
    dc = [f'cnt_d{i}' for i in range(7)]
    M['cycle_rank'] = sum(M[f'cnt_d{i}'] * i for i in range(7)) / 2.0 - (M[dc].sum(axis=1) - M['cnt_d0']) + 1   # active graph
    assert np.allclose(M['cycle_rank'], M['cyclomatic_idx'])
    M['deg_bimodality'] = M.apply(lambda r: ((skew(sum([[int(c.split('_d')[1])] * int(r[c]) for c in dc], [])) ** 2 + 1) /
        (kurtosis(sum([[int(c.split('_d')[1])] * int(r[c]) for c in dc], []), fisher=True) + 3)), axis=1)
    assert np.allclose(M['frac_dangling'], M['cnt_d1'] / 216)
    u = cube['uts'].mean((1, 2)); t = cube['toughness'].mean((1, 2))
    assert np.allclose(u, M['uts_mean']) and np.allclose(t, M['toughness_mean'])
    return M, u, t, nets, cube


def quad(u, t):
    """Median split of both properties (the published analysis)."""
    return 1 + (t >= np.median(t)).astype(int) + 2 * (u >= np.median(u)).astype(int)


def quad_pct(u, t, pu, pt=None):
    """Split at the pu-th percentile of UTS and the pt-th of toughness (same >= rule, pu = pt = 50 is the median split)."""
    pt = pu if pt is None else pt
    return 1 + (t >= np.percentile(t, pt)).astype(int) + 2 * (u >= np.percentile(u, pu)).astype(int)


def quad_tercile(u, t):
    """Top third vs bottom third on each axis. Networks in the middle third of either axis get 0 (dropped)."""
    lo_u, hi_u = np.quantile(u, [1 / 3, 2 / 3]); lo_t, hi_t = np.quantile(t, [1 / 3, 2 / 3])
    U = np.where(u < lo_u, 0, np.where(u > hi_u, 1, -1)); T = np.where(t < lo_t, 0, np.where(t > hi_t, 1, -1))
    Q = 1 + T + 2 * U
    Q[(U < 0) | (T < 0)] = 0
    return Q


def cd(xa, xb):
    if len(xa) < 2 or len(xb) < 2: return np.nan
    sp = np.sqrt(((len(xa) - 1) * xa.var(ddof=1) + (len(xb) - 1) * xb.var(ddof=1)) / (len(xa) + len(xb) - 2))
    return (xb.mean() - xa.mean()) / sp if sp else 0.0


def bh(p):
    """Benjamini-Hochberg q values, as in generate_checks.ipynb."""
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); ranks = np.empty(m, int); ranks[order] = np.arange(1, m + 1)
    q = p * m / ranks; q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]; out = np.empty(m); out[order] = np.minimum(q_sorted, 1)
    return out


def cells(X, Q, nmin=2, groups=None):
    """The 55 cells (11 metrics x 5 transitions) for a quadrant labelling Q (0 = dropped), in the order of figure6_stats.csv.
    A cell is tested only if both groups hold at least nmin networks, and BH runs over the tested cells (all 55 when every
    group is large enough, which reproduces the published analysis). If `groups` is given, it maps each transition name to
    a pair of boolean masks (from, to) and overrides Q (used for the within-half tercile scheme)."""
    rows = []
    for j, f in enumerate(FEATS):
        for name, (a, b) in TR.items():
            ma, mb = groups[name] if groups is not None else (Q == a, Q == b)
            xa, xb = X[ma, j], X[mb, j]
            ok = min(len(xa), len(xb)) >= nmin
            d = cd(xa, xb) if ok else np.nan; p = ttest_ind(xa, xb, equal_var=False).pvalue if ok else np.nan
            rows.append(dict(key=f, metric=NAMES[f], transition=name, tr=TRS[name], n_from=len(xa), n_to=len(xb), cohens_d=d, welch_p=p))
    S = pd.DataFrame(rows)
    q = np.full(len(S), np.nan); v = S.welch_p.notna().to_numpy()
    if v.any():
        q[v] = bh(S.welch_p.to_numpy()[v])
    S['bh_q'] = q
    S['sig_bh'] = S.bh_q < .05
    return S


def tercile_within_halves(u, t):
    """Each transition with its own axis split into thirds (middle third dropped) inside the relevant half of the other axis.
    Q1->Q2 is the bottom vs the top third of toughness among networks below the UTS median, and Q3->Q4 the same above the
    UTS median. Q1->Q3 is the bottom vs the top third of UTS among networks below the toughness median, and Q2->Q4 the same
    above it. Q1->Q4 is the bottom third of both vs the top third of both (global terciles)."""
    weak, strong = u < np.median(u), u >= np.median(u); brittle, tough = t < np.median(t), t >= np.median(t)

    def thirds(x, within):
        lo, hi = np.quantile(x[within], [1 / 3, 2 / 3]); return within & (x < lo), within & (x > hi)
    G = {}
    G['Toughening (weak, Q1->Q2)'] = thirds(t, weak)
    G['Toughening (strong, Q3->Q4)'] = thirds(t, strong)
    G['Strengthening (brittle, Q1->Q3)'] = thirds(u, brittle)
    G['Strengthening (tough, Q2->Q4)'] = thirds(u, tough)
    Qt = quad_tercile(u, t)
    G['Total (Q1->Q4)'] = (Qt == 1, Qt == 4)
    return G


def resid(y, Z):
    Z = np.column_stack([np.ones(len(y)), Z]); return y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]


def partial_r(x, y, Z):
    """Pearson r between x and y after linear adjustment for Z (column or matrix)."""
    return np.corrcoef(resid(x, Z), resid(y, Z))[0, 1]
