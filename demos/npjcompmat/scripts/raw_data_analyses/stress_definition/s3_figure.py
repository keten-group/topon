"""Step 3: comparison figure + Figure-6 robustness summary (reads only the outputs of s1/s2).

Writes true_vs_nominal_comparison.png and fig6_cells_summary_12pull.csv.  Reads and writes only
NPJ_ANALYSIS_OUT/stress_definition/ (see ../paths.py), no raw data.
"""
import numpy as np, pandas as pd
import sys
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))         # raw_data_analyses/, for paths.py
from paths import out_dir
OUT = out_dir('stress_definition')
BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#9a9993'
INK, INK2 = '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})

A = pd.read_csv(OUT / 'per_pull_true_vs_nominal.csv'); NET = pd.read_csv(OUT / 'network_true_vs_nominal.csv')
F6 = pd.read_csv(OUT / 'fig6_cells_true_vs_nominal.csv'); U = pd.read_csv(OUT / 'area_universality_seed1.csv')
z = np.load(OUT / 'box_curves_seed1.npz')

# ---------------------------------------------------------------- Figure-6 robustness summary (12-pull)
p = F6[F6.set == '12pull'].set_index(['metric', 'transition'])
S = p[['d_true', 'q_true', 'sig_true', 'd_nom', 'q_nom', 'sig_nom']].copy()
for v in ('12pull_all_incomp', '12pull_all_uts'):
    q = F6[F6.set == v].set_index(['metric', 'transition']); tag = v.replace('12pull_', '')
    S[f'd_nom_{tag}'] = q.d_nom; S[f'q_nom_{tag}'] = q.q_nom; S[f'sig_nom_{tag}'] = q.sig_nom
nomsig = S[['sig_nom', 'sig_nom_all_incomp', 'sig_nom_all_uts']]
S['class'] = np.select([S.sig_true & nomsig.all(axis=1), S.sig_true & ~nomsig.any(axis=1), ~S.sig_true & nomsig.all(axis=1), ~S.sig_true & ~nomsig.any(axis=1)],
                       ['significant under both', 'true only', 'nominal only', 'neither'], 'depends on replicate area treatment')
S.reset_index().to_csv(OUT / 'fig6_cells_summary_12pull.csv', index=False)
print(S['class'].value_counts().to_string())
for c in ('true only', 'nominal only', 'depends on replicate area treatment'):
    print(f'\n{c}:'); print(S[S['class'] == c][['d_true', 'q_true', 'd_nom', 'q_nom', 'q_nom_all_incomp', 'q_nom_all_uts']].round(3).to_string())

# ---------------------------------------------------------------- example networks (12-pull quadrants)
pr = lambda x: stats.rankdata(x) / len(x)
NET['dUTSrank'] = pr(NET.UTS_nom_12pull) - pr(NET.UTS_true_12pull)
exA = NET[(NET.Q_true_12pull == 4) & (NET.Q_nom_12pull == 2)].sort_values('dUTSrank').folder_name.iloc[0]
exB = NET[(NET.Q_true_12pull == 1) & (NET.Q_nom_12pull == 3)].sort_values('dUTSrank').folder_name.iloc[-1]
print('\nexamples:', exA, '(Q4 -> Q2)', exB, '(Q1 -> Q3)')

fig, ax = plt.subplots(2, 4, figsize=(18, 8.4)); ax = ax.ravel()
for k, (col, lab) in enumerate(((2, 'True stress $-P_{aa}$ (current area)'), (None, 'Nominal stress $F/A_0$'))):
    a = ax[k]
    for fn, c, tag in ((exA, BLUE, 'A: Q4 true -> Q2 nominal'), (exB, ORANGE, 'B: Q1 true -> Q3 nominal')):
        for i, axn in enumerate('xyz'):
            arr = z[f'{fn}|{axn}']; e = arr[:, 1]; s = arr[:, 2] if col else arr[:, 2] * arr[:, 3]
            a.plot(e, s, color=c, lw=1.2, alpha=0.85, label=tag if i == 0 else None)
    a.set_xlabel('Engineering strain'); a.set_ylabel(lab + ', LJ units'); a.set_xlim(0, 18)
    a.set_title(('a  ' if k == 0 else 'b  ') + ('True stress, seed-1 pulls (3 axes each)' if k == 0 else 'Same pulls, nominal stress (measured area)'))
    a.axhline(0, color=GRAY, lw=0.6); a.legend(frameon=False, loc='upper left')

# c: volume ratio of pre-break states
a = ax[2]
a.fill_between(U.strain, U.VV0_intact_p5, U.VV0_intact_p95, color=BLUE, alpha=0.18, lw=0, label='5-95% of pulls not yet broken')
a.plot(U.strain, U.VV0_intact_mean, color=BLUE, lw=2, label='mean, pulls not yet broken')
a.plot(U.strain, U.VV0_p95, color=GRAY, lw=1, ls='--', label='95th pct, all pulls (incl. broken)')
a.axhline(1, color=INK2, lw=0.8, ls=':'); a.set_ylim(0.95, 1.8); a.set_xlabel('Engineering strain'); a.set_ylabel('$V/V_0 = (A/A_0)(1+\\varepsilon)$')
a.set_title('c  Volume is conserved to 0.5% up to strain 6'); a.legend(frameon=False, loc='upper left')
for _, r in U[U.strain.round(0).isin([10, 13])].iterrows():
    a.annotate(f"n={int(r.n_intact)}", (r.strain, r.VV0_intact_p95), textcoords='offset points', xytext=(0, 4), ha='center', color=INK2, fontsize=8)

# d, e: network means (12 pulls)
chg = (NET.Q_true_12pull != NET.Q_nom_12pull).to_numpy()
for k, (m, lab) in enumerate((('UTS', 'UTS'), ('tough', 'Toughness'))):
    a = ax[3 + k]; x = NET[f'{m}_true_12pull']; y = NET[f'{m}_nom_12pull']
    a.scatter(x[~chg], y[~chg], s=16, color=GRAY, alpha=0.7, lw=0, label=f'same quadrant (n={int((~chg).sum())})')
    a.scatter(x[chg], y[chg], s=16, color=ORANGE, alpha=0.9, lw=0, label=f'changes quadrant (n={int(chg.sum())})')
    a.axvline(np.median(x), color=INK2, lw=0.7, ls=':'); a.axhline(np.median(y), color=INK2, lw=0.7, ls=':')
    a.set_xlabel(f'{lab}, true stress (12-pull mean)'); a.set_ylabel(f'{lab}, nominal stress (12-pull mean)')
    a.set_title(f"{'d' if k == 0 else 'e'}  {lab}: Spearman {stats.spearmanr(x, y)[0]:.2f}; dotted = medians"); a.legend(frameon=False, loc='upper left', fontsize=8)

# f, h: quadrant transition matrices
def mat(a, qt, qn, title):
    M = pd.crosstab(qt, qn).reindex(index=[1, 2, 3, 4], columns=[1, 2, 3, 4], fill_value=0).to_numpy()
    a.imshow(M, cmap=matplotlib.colors.LinearSegmentedColormap.from_list('b', ['#f4f8fd', '#86b6ef', '#184f95']), vmin=0)
    for i in range(4):
        for j in range(4):
            a.text(j, i, M[i, j], ha='center', va='center', color='white' if M[i, j] > 0.55 * M.max() else INK, fontsize=10)
    names = ['Q1 weak/brittle', 'Q2 weak/tough', 'Q3 strong/brittle', 'Q4 strong/tough']
    a.set_xticks(range(4)); a.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=8); a.set_yticks(range(4)); a.set_yticklabels(names, fontsize=8)
    a.set_xlabel('Nominal-stress quadrant'); a.set_ylabel('True-stress quadrant'); a.set_title(title)
    for s in a.spines.values(): s.set_visible(False)
mat(ax[5], NET.Q_true_12pull, NET.Q_nom_12pull, f"f  12-pull means: {int(chg.sum())}/327 change ({100 * chg.mean():.0f}%)")
c3 = (NET.Q_true_seed1_3pull != NET.Q_nom_seed1_3pull)
mat(ax[7], NET.Q_true_seed1_3pull, NET.Q_nom_seed1_3pull, f"h  Seed 1 only, 3-pull means: {int(c3.sum())}/327 ({100 * c3.mean():.0f}%)")

# g: Figure-6 Cohen's d, true vs nominal (12-pull, primary nominal)
a = ax[6]; G = F6[F6.set == '12pull']
cats = [('BH-significant under both', G.sig_true & G.sig_nom, BLUE), ('true only', G.sig_true & ~G.sig_nom, ORANGE),
        ('nominal only', ~G.sig_true & G.sig_nom, AQUA), ('neither', ~G.sig_true & ~G.sig_nom, GRAY)]
lim = float(np.ceil(10 * (max(abs(G.d_true).max(), abs(G.d_nom).max()) + 0.05)) / 10)
a.plot([-lim, lim], [-lim, lim], color=INK2, lw=0.7, ls=':'); a.axhline(0, color=GRAY, lw=0.5); a.axvline(0, color=GRAY, lw=0.5)
for lab, m, c in cats:
    a.scatter(G.d_true[m], G.d_nom[m], s=30, color=c, lw=0.8, edgecolor='white', label=f'{lab} ({int(m.sum())})')
a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect('equal')
a.set_xlabel("Cohen's d, true-stress quadrants"); a.set_ylabel("Cohen's d, nominal-stress quadrants")
a.set_title(f"g  Figure-6 cells (55): r = {stats.pearsonr(G.d_true, G.d_nom)[0]:.2f}"); a.legend(frameon=False, loc='upper left', fontsize=8)

fig.suptitle('True (Cauchy) vs nominal (engineering) stress, 327 CG networks: seed 1 uses the measured box area; replicate seeds use A/A0 = 1/(1+strain)',
             x=0.01, ha='left', fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / 'true_vs_nominal_comparison.png', dpi=170); print('wrote true_vs_nominal_comparison.png')
