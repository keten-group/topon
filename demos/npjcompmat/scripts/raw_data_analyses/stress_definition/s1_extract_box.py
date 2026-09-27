"""Step 1: recover the transverse box area for every ORIGINAL-seed pull (327 networks x 3 axes).

Read-only on the raw data.  Writes only into NPJ_ANALYSIS_OUT/stress_definition/ (see ../paths.py):
  box_curves_seed1.npz        per pull: file step/strain/stress + A/A0, V/V0, rho interpolated onto the file rows
  block_matching_seed1.csv    one row per pull: which thermo block was used and how well it matches the stress file

Inputs: the thermo output of the seed-1 pulls (slurm-*.out and log.lammps in
RAW_ROOT/crosslinker/sc_6x6x6/<net>/tensile_test_attractive_xyz/, not deposited), the seed-1 stress files
stress_strain_npt_{x,y,z}.dat (deposited in cg_stress_strain_seed1.tar.xz, unpacked to RAW_ROOT/seed1/<net>/),
and the network list of the deposited data/csv/mechanics.csv.

Thermo blocks are parsed from every slurm-*.out and log.lammps in <net>/tensile_test_attractive_xyz.
A stage-2 block starts at step 1,000,000 (end of the 1M-step NPT pre-equilibration).  The loading
axis of a block is the box length that grows most.  A block is accepted for (net, axis) only if
  (a) strain schedule: its thermo strain L(t)/L0 - 1, evaluated at the centre of each fix ave/time
      window (step S-4500, because each file row averages steps S-9000..S), reproduces the file's
      v_strain column (max |diff| < 1e-3), and
  (b) stress trajectory: the instantaneous thermo -P_aa at step S (one of the 10 samples averaged
      into file row S) tracks the file stress; the block with the lowest RMS difference wins.
Several files can contain the same run (log.lammps duplicates the last slurm run); identical
blocks are collapsed.  A0 = product of the two transverse lengths at step 1,000,000, i.e. the same
snapshot that defines L0 in the input deck.  A0_avg (mean over steps 500k-1M of stage 1) is kept
for a sensitivity check.
"""
import re, sys, glob, numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))         # raw_data_analyses/, for paths.py
from paths import CG, DATA, RAW_ROOT, out_dir
ROOT = CG / 'sc_6x6x6'                       # thermo output of the pulls (not deposited)
SEED1 = RAW_ROOT / 'seed1'                   # stress files of the pulls (deposited)
OUT = out_dir('stress_definition')
MECH = DATA / 'csv' / 'mechanics.csv'
COLS = ['step', 'temp', 'press', 'pxx', 'pyy', 'pzz', 'lx', 'ly', 'lz', 'vol', 'dens', 'bonds']
RATE, DT = 5e-4, 0.005


def parse_blocks(fp):
    """Return list of DataFrames, one per thermo block with the custom 12-column style."""
    blocks, cur = [], None
    with open(fp, errors='ignore') as fh:
        for line in fh:
            s = line.split()
            if s[:2] == ['Step', 'Temp'] and 'Lx' in s:
                cur = []; blocks.append(cur); continue
            if cur is None: continue
            if len(s) == 12:
                try: cur.append([float(v) for v in s]); continue
                except ValueError: pass
            if s[:2] == ['Loop', 'time'] or (s and s[0] == 'ERROR'): cur = None   # block ends
            # other lines (warnings) inside a block are skipped without closing it
    return [pd.DataFrame(b, columns=COLS) for b in blocks if len(b) > 0]


def read_dat(fp):
    return pd.read_csv(fp, sep=r'\s+', comment='#', names=['step', 'strain', 'stress'])


def process(fn):
    tdir = ROOT / fn / 'tensile_test_attractive_xyz'
    files = sorted(glob.glob(str(tdir / 'slurm-*.out'))) + [str(tdir / 'log.lammps')]
    stage2, stage1_before = [], {}
    for fp in files:
        if not Path(fp).exists(): continue
        bl = parse_blocks(fp)
        for i, b in enumerate(bl):
            if int(b.step.iloc[0]) == 1000000 and len(b) > 5:
                prev = bl[i - 1] if i > 0 and int(bl[i - 1].step.iloc[0]) == 0 else None
                stage2.append((Path(fp).name, i, b, prev))
    # collapse identical blocks (same run printed in two files)
    uniq = []
    for name, i, b, prev in stage2:
        dup = [u for u in uniq if len(u['b']) == len(b) and np.allclose(u['b'].to_numpy(), b.to_numpy())]
        if dup: dup[0]['files'].append(name); continue
        uniq.append(dict(files=[name], b=b, prev=prev))
    out_rows, curves = [], {}
    for ax in 'xyz':
        d = read_dat(SEED1 / fn / f'stress_strain_npt_{ax}.dat')
        cands = []
        for u in uniq:
            b = u['b']; ratios = {a: b[f'l{a}'].iloc[-1] / b[f'l{a}'].iloc[0] for a in 'xyz'}
            if max(ratios, key=ratios.get) != ax: continue
            L0 = b[f'l{ax}'].iloc[0]; e_th = b[f'l{ax}'].to_numpy() / L0 - 1; st = b.step.to_numpy()
            cover = d.step.to_numpy() <= st[-1]
            if cover.sum() < 10: cands.append(dict(u=u, ok=False, why='short', n=len(b))); continue
            e_mid = np.interp(d.step.to_numpy()[cover] - 4500, st, e_th)
            smis = np.abs(e_mid - d.strain.to_numpy()[cover]).max()
            # stress: thermo sample at step S vs file row S (common steps only)
            m = pd.merge(d, b[['step', f'p{ax}{ax}']], on='step')
            rms = float(np.sqrt(np.mean((m.stress + m[f'p{ax}{ax}']) ** 2)))
            r = float(np.corrcoef(m.stress, -m[f'p{ax}{ax}'])[0, 1])
            full = bool(cover.all())
            cands.append(dict(u=u, ok=(smis < 1e-3) and full, why='' if full else 'truncated', smis=smis, rms=rms, r=r, n=len(b), L0=L0))
        good = sorted([c for c in cands if c['ok']], key=lambda c: c['rms'])
        row = dict(folder_name=fn, axis=ax, n_candidate_blocks=len(cands), n_schedule_matched=len(good),
                   candidates=';'.join(f"{'+'.join(c['u']['files'])}:{c['n']}rows:" + (f"smis={c['smis']:.1e},rms={c['rms']:.3f},r={c['r']:.3f}" if 'smis' in c else c['why']) for c in cands))
        if not good:
            row['status'] = 'NO_MATCH'; out_rows.append(row); continue
        best = good[0]; b = best['u']['b']; prev = best['u']['prev']
        perp = [a for a in 'xyz' if a != ax]
        st = b.step.to_numpy(); A = (b[f'l{perp[0]}'] * b[f'l{perp[1]}']).to_numpy(); V = b.vol.to_numpy()
        A0, V0, L0 = A[0], V[0], b[f'l{ax}'].iloc[0]
        A0_avg, s1_cont = np.nan, np.nan
        if prev is not None:
            s1_cont = bool(np.allclose(prev.iloc[-1].to_numpy(), b.iloc[0].to_numpy()))   # stage-1 end == stage-2 start
            late = prev[prev.step >= 500000]; A0_avg = float((late[f'l{perp[0]}'] * late[f'l{perp[1]}']).mean())
        smid = d.step.to_numpy() - 4500.0                      # centre of each averaging window
        arr = np.column_stack([d.step, d.strain, d.stress, np.interp(smid, st, A / A0), np.interp(smid, st, V / V0),
                               np.interp(smid, st, b.dens.to_numpy()), np.interp(smid, st, b.bonds.to_numpy())])
        curves[f'{fn}|{ax}'] = arr
        row.update(status='OK', chosen=('+'.join(best['u']['files'])), strain_mismatch=best['smis'], stress_rms=best['rms'], stress_r=best['r'],
                   runner_up_rms=(good[1]['rms'] if len(good) > 1 else np.nan), L0=L0, A0=A0, V0=V0, A0_stage1_avg=A0_avg, stage1_continuous=s1_cont,
                   L0_other=(b[f'l{perp[0]}'].iloc[0], b[f'l{perp[1]}'].iloc[0]), rows=len(d),
                   # geometric identity check: V = Lx*Ly*Lz at every thermo row
                   vol_identity_maxrel=float(np.abs(b.lx * b.ly * b.lz / b.vol - 1).max()))
        out_rows.append(row)
    return out_rows, curves


if __name__ == '__main__':
    nets = list(pd.read_csv(MECH).folder_name)
    rows, curves = [], {}
    with Pool(2) as pool:
        for k, (r, c) in enumerate(pool.imap(process, nets, chunksize=4)):
            rows += r; curves.update(c)
            if k % 50 == 0: print(k, flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT / 'block_matching_seed1.csv', index=False)
    np.savez_compressed(OUT / 'box_curves_seed1.npz', **curves)
    print(R.status.value_counts().to_string())
    print('schedule-matched candidates per pull:', R.n_schedule_matched.value_counts().sort_index().to_dict())
    print('candidate blocks per pull:', R.n_candidate_blocks.value_counts().sort_index().to_dict())
    ok = R[R.status == 'OK']
    print('max strain mismatch %.2e; stress rms median %.3f max %.3f; corr min %.3f; vol identity max rel %.1e' %
          (ok.strain_mismatch.max(), ok.stress_rms.median(), ok.stress_rms.max(), ok.stress_r.min(), ok.vol_identity_maxrel.max()))
    amb = ok[ok.runner_up_rms.notna()]
    print(f'pulls with >1 schedule-matched block: {len(amb)}; runner-up/best rms ratio min {np.nanmin(amb.runner_up_rms / amb.stress_rms) if len(amb) else np.nan:.2f}')
    print('chosen file by axis:'); print(ok.groupby('axis').chosen.apply(lambda s: s.str.contains('log.lammps').mean()).rename('frac_also_in_log.lammps').to_string())
