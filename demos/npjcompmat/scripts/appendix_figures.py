# Appendix figures (Fig. 8, entanglement range, Appendix B. Fig. 9, bond/create comparison, Appendix C) in the npj_style_v1 style
# of the other manuscript figures. Run from demos/npjcompmat/ (same code as the notebook cells).
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath('data'))
from npj_style_v1 import set_npj_style

output_dir = 'figs'
os.makedirs(output_dir, exist_ok=True)
SAVE_DPI = 600
REV_PATH = os.path.join('data', 'derived')


def require(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required input not found: {path}. Run from demos/npjcompmat/.")
    return path


def align_label_to_ylabel(fig, ax, label_text):
    # panel letter centred over the y-axis label, as in Figure 1
    ylabel = ax.yaxis.label
    bbox = ylabel.get_window_extent(renderer=fig.canvas.get_renderer())
    ax_bbox = ax.get_window_extent()
    x_rel = ((bbox.x0 + bbox.x1) / 2 - ax_bbox.x0) / ax_bbox.width
    ax.text(x_rel, 1.0, label_text, transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='bottom', ha='center')


REF_C, TOP_C = '#7f7f7f', '#0072B2'

# =====================================================================================================
# Figure 9 (Appendix C): bond/create references versus Topon networks built from their statistics
# =====================================================================================================
BC = os.path.join(REV_PATH, 'bond_create')
with open(require(os.path.join(BC, 'four_distributions.json')), encoding='utf-8') as fh:
    four = json.load(fh)


def cycle_frac(d, kmax):
    ks = np.arange(3, kmax + 1)
    return ks, np.array([d.get(str(k), 0) for k in ks], float) / sum(d.values())


def load_ss(path, w=100):
    d = np.loadtxt(require(path), skiprows=1, ndmin=2)
    e, s = d[:, 1], d[:, 2]
    ss = np.convolve(s, np.ones(w) / w, 'valid')      # 100 rows = 0.1 strain
    return e[w // 2: w // 2 + len(ss)], ss


width, _ = set_npj_style(column_type='double')
CASES = (('N20', r'$DP=20$'), ('N100', r'$DP=100$'))


def binned_fraction(values, edges):
    h, _ = np.histogram(np.asarray(values, float), bins=edges)
    return 0.5 * (edges[1:] + edges[:-1]), h / len(values)


def draw_bondcreate(first, fname):
    # first column: 'ring' (smallest ring size), 'distance' (junction separation), 'rank' (neighbor rank)
    fig, axes = plt.subplots(2, 3, figsize=(width, width * 0.55), constrained_layout=True)
    for r, (case, tag) in enumerate(CASES):
        ref, top = four[case]['reference'], four[case]['topon']

        ax = axes[r, 0]
        if first == 'ring':
            kmax = 12 if case == 'N20' else 10
            x, fr = cycle_frac(ref['cycle'], kmax)
            _, ft = cycle_frac(top['cycle'], kmax)
            ax.set_xticks(x[::2] if case == 'N20' else x)
            ax.set_xlabel('Smallest ring size (strands)')
        elif first == 'reach':
            # reach = distance between the two junctions of a load-bearing strand / mean junction spacing
            rd = reach_data[case]
            edges = np.arange(0, 3.31, 0.15) if case == 'N20' else np.arange(0, 4.51, 0.25)
            x, fr = binned_fraction(rd['reference']['reach'], edges)
            _, ft = binned_fraction(rd['topon_relaxed']['reach'], edges)
            _, fb = binned_fraction(rd['topon_built']['reach'], edges)
            cut = max(rd['topon_built']['reach'])        # largest lattice distance the cutoff admits
            ax.plot(x, fb, '-', color='#56B4E9', lw=1.0, label='Topon, as built')
            ax.axvline(cut, color='gray', ls='--', lw=0.5)
            ax.set_xlabel(r'Strand reach ($r/a$)')
        elif first == 'distance':
            edges = np.arange(0, 16.01, 1.0) if case == 'N20' else np.arange(0, 40.01, 2.5)
            x, fr = binned_fraction(ref['span'], edges)
            _, ft = binned_fraction(top['span'], edges)
            ax.set_xlabel(r'Distance between connected junctions ($\sigma$)')
        else:
            # ranks were counted up to the 80th neighbor; strands beyond it (8-11% at DP 100) are left out
            edges = np.arange(0.5, 60.6, 4) if case == 'N20' else np.arange(0.5, 80.6, 8)
            x, fr = binned_fraction(ref['rank'], edges)
            _, ft = binned_fraction(top['rank'], edges)
            ax.set_xlabel('Neighbor rank of connected junction')
        ax.plot(x, fr, 's-', color=REF_C, lw=1.5, markersize=4, label='bond/create')
        ax.plot(x, ft, 'o-', color=TOP_C, lw=1.5, markersize=4, label='Topon')
        ax.set_ylim(0, 1.25 * max(fr.max(), ft.max(), fb.max() if first == 'reach' else 0))
        ax.set_ylabel('Fraction of strands')
        ax.text(0.05, 0.88, tag, transform=ax.transAxes, fontsize=8)
        if first == 'reach' and r == 0:
            ax.legend(frameon=False, loc='upper right')

        ax = axes[r, 1]
        zr = np.asarray(ref['Z_bridge_hist'], float); zt = np.asarray(top['Z_bridge_hist'], float)
        n = max(len(zr), len(zt))
        zr = np.pad(zr / zr.sum(), (0, n - len(zr))); zt = np.pad(zt / zt.sum(), (0, n - len(zt)))
        xz = np.arange(n)
        ax.plot(xz, zr, 's-', color=REF_C, lw=1.5, markersize=4)
        ax.plot(xz, zt, 'o-', color=TOP_C, lw=1.5, markersize=4)
        ax.set_xticks(xz)
        ax.set_ylim(0, 1.1 * max(zr.max(), zt.max()))
        ax.set_xlabel(r'Entanglements per strand ($Z$)')
        ax.set_ylabel('Fraction of strands')

        ax = axes[r, 2]
        er, sr = load_ss(os.path.join(BC, f'{case}_reference_stress_strain_high.txt'))
        et, st = load_ss(os.path.join(BC, f'{case}_topon_stress_strain_high.txt'))
        ax.plot(er, sr, '-', color=REF_C, lw=1.0, label='bond/create')
        ax.plot(et, st, '-', color=TOP_C, lw=1.0, label='Topon')
        ax.set_xlim(0, 10 if case == 'N20' else 25)
        ax.set_ylim(bottom=-0.02)
        ax.set_xlabel('Engineering strain')
        ax.set_ylabel(r'Stress ($\epsilon/\sigma^3$)')

    axes[0, 2].legend(frameon=False, loc='upper right')
    fig.canvas.draw()
    for ax, lab in zip(axes.flat, 'abcdef'):
        align_label_to_ylabel(fig, ax, lab)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f'{fname}.{ext}'), dpi=SAVE_DPI)
    plt.close(fig)
    print(f'{fname} written')


with open(require(os.path.join(BC, 'reach_distributions.json')), encoding='utf-8') as fh:
    reach_data = json.load(fh)
draw_bondcreate('reach', 'Figure_A1_bondcreate')                 # Fig. 9 in the manuscript (strand reach r/a)
draw_bondcreate('ring', 'Figure_A1_bondcreate_ring')             # alternative first column (smallest ring)
draw_bondcreate('distance', 'Figure_A1_bondcreate_distance')     # alternative first column
draw_bondcreate('rank', 'Figure_A1_bondcreate_rank')             # alternative first column

# =====================================================================================================
# Figure 8 (Appendix B): requested versus realized entanglement, and the realizable range per strand length
# =====================================================================================================
with open(require(os.path.join(REV_PATH, 'entanglement', 'target_vs_realized.json')), encoding='utf-8') as fh:
    tvr = [r for r in json.load(fh) if 'error' not in r]
with open(require(os.path.join(REV_PATH, 'entanglement', 'figure_completion.json')), encoding='utf-8') as fh:
    fill = [r for r in json.load(fh) if 'error' not in r]

DPS = (20, 40, 80, 120)
DP_COLORS = {20: '#E69F00', 40: '#CC79A7', 80: '#0072B2', 120: '#009E73'}


def mean_sd(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


cells = {}
for dp in DPS:
    rows = [(0.0, *mean_sd([r['Zmean'] for r in fill if r['dp'] == dp and r['kind'] == 'zero']), False)]
    for tg in (1.0, 2.0, 4.0, 8.0):
        sel = [r for r in tvr if r['dp'] == dp and r['target'] == tg]
        if sel:
            rows.append((tg, *mean_sd([r['Zmean'] for r in sel]), any(r.get('saturated') for r in sel)))
    if dp == 120:
        zc = [r['Zmean'] for r in fill if r['dp'] == 120 and r['kind'] == 'ceiling']
        rows.append((12.0, *mean_sd(zc), True))
    cells[dp] = rows
floor = {dp: cells[dp][0] for dp in DPS}
ceil = {dp: max((c for c in cells[dp] if c[3]), key=lambda c: c[1]) for dp in DPS}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(width, width * 0.4), constrained_layout=True)

axA.plot([0, 12.5], [0, 12.5], '--', color='gray', lw=0.5, alpha=0.6)
for k, dp in enumerate(DPS):
    t = np.array([c[0] for c in cells[dp]]) + (k - 1.5) * 0.12
    m = np.array([c[1] for c in cells[dp]]); s = np.array([c[2] for c in cells[dp]])
    clamped = np.array([c[3] for c in cells[dp]])
    col = DP_COLORS[dp]
    axA.errorbar(t, m, yerr=s, fmt='-', color=col, lw=1.5, capsize=2, elinewidth=0.8, label=fr'$DP={dp}$')
    axA.plot(t[~clamped], m[~clamped], 'o', color=col, markersize=4)
    axA.plot(t[clamped], m[clamped], 'o', mfc='white', mec=col, markersize=4)
axA.set_xlim(-0.5, 12.5); axA.set_ylim(-0.5, 12.5)
axA.set_xlabel('Requested entanglements per strand')
axA.set_ylabel(r'Realized entanglements per strand ($Z$)')
axA.legend(frameon=False, loc='upper left')

xs = np.array(DPS, float)
fl = np.array([floor[d][1] for d in DPS]); ce = np.array([ceil[d][1] for d in DPS])
axB.fill_between(xs, fl, ce, color='#0072B2', alpha=0.12, linewidth=0)
axB.plot(xs, ce, 'o-', color='#0072B2', lw=1.5, markersize=4, label='Densest build')
axB.plot(xs, fl, 's-', color=REF_C, lw=1.5, markersize=4, label='Straight strands')
bc_z = [four['N20']['reference']['summary']['Zmean_bridge'], four['N100']['reference']['summary']['Zmean_bridge']]
axB.plot([20, 100], bc_z, 'D', mfc='white', mec='black', markersize=4.5, linestyle='none', label='bond/create')
# CG ensemble of this work: mean of three networks traced with the same junction-to-junction export
axB.plot([30], [np.mean([0.326, 0.352, 0.394])], '^', color='#D55E00', markersize=5, linestyle='none',
         label='CG ensemble (this work)')
axB.set_xlim(0, 130); axB.set_ylim(-0.5, 12.5)
axB.set_xticks([0, 20, 40, 60, 80, 100, 120])
axB.set_xlabel(r'Degree of polymerization ($DP$)')
axB.set_ylabel(r'Entanglements per strand ($Z$)')
axB.legend(frameon=False, loc='upper left')

fig.canvas.draw()
align_label_to_ylabel(fig, axA, 'a')
align_label_to_ylabel(fig, axB, 'b')
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(output_dir, f'Figure_A2_entanglement.{ext}'), dpi=SAVE_DPI)
plt.close(fig)

with open(os.path.join(output_dir, 'Figure_A2_entanglement_data.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['DP', 'Z_requested', 'Z_realized_mean', 'Z_realized_sd', 'clamped'])
    for dp in DPS:
        for t, m, s, c in cells[dp]:
            w.writerow([dp, t, round(m, 4), round(s, 4), int(c)])
print('Figure 8 (Figure_A2_entanglement) written')
for dp in DPS:
    print(f'DP {dp}: straight strands {floor[dp][1]:.3f}, densest build {ceil[dp][1]:.2f}')
