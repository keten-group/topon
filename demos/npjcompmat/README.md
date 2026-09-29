# Paper companion

Data and notebooks behind the figures of *A generative framework for precise topology engineering of polymer
networks* (npj Computational Materials). No simulation runs here. Every figure and table is made from the files in
`data/`.

## Contents

```
generate_figures.ipynb          data figures of the manuscript (Figs. 2-9)        -> figs/
generate_checks.ipynb           supporting checks, tables and figures             -> tables_checks/, figs_checks/
generate_*_executed.ipynb       the same notebooks with their outputs, for reading
requirements.txt                the environment used to execute and verify them
data/
  dataset.pkl, csv/, mechanics/ the data deposited with v0.1.0 (csv/ is a CSV export of dataset.pkl, mechanics/
                                holds the network graphs)
  npj_style_v1.py               plotting style
  convert_pkl_to_csv.py         builds csv/ from dataset.pkl
  derived/                      derived data read by the notebooks (data dictionary in data/derived/README.md)
  raw/                          raw simulation output and the atomistic systems of the cooling runs, as built
                                and equilibrated (see data/raw/README.md)
scripts/
  appendix_figures.py           Figs. 8 and 9 as a standalone script (same code as the notebook cell)
  make_timing_table.py          Table 1 (Appendix B) from data/derived/generator_benchmark/ -> tables_checks/
  checks/threshold_cv/          threshold and cross-validation checks of the quadrant analysis (Figs. 6 and 7),
                                with its own README and reference outputs in out/
  build_raw_data.py             how data/raw/ was packed from the simulation folders (a record, not needed to run)
  raw_data_analyses/            the scripts that made the derived data from raw simulation output
figs/, figs_checks/, tables_checks/   outputs of the notebooks
```

## Running

```
pip install -r requirements.txt
python -m nbconvert --to notebook --execute generate_figures.ipynb --output generate_figures_executed.ipynb
python -m nbconvert --to notebook --execute generate_checks.ipynb --output generate_checks_executed.ipynb
```

Run the notebooks from this folder, since every path is relative to it. A missing input raises `FileNotFoundError`
with the file name (there are no fallbacks to mock data). Each notebook runs in well under a minute. The note in
`requirements.txt` explains how the verified numpy and scipy versions were installed. The figures use the Arial font,
and without it matplotlib falls back to DejaVu Sans and the images differ.

The Tg cell has a variable `TG_LAG` (default 4 ps) that redraws Fig. 5 at another MSD lag (written as
`Figure_4_Tg_<lag>ps`).

## Figure numbers

The file names follow the notebook cells, which number the figures differently from the manuscript.

| File | Manuscript | Data |
|---|---|---|
| `Figure_1_RDF` | Fig. 2 | `data/dataset.pkl` |
| `Figure_2_Scaling_and_Distributions` | Fig. 3 | `data/dataset.pkl` |
| `Figure_3_Characteristic_Ratio` | Fig. 4 | `data/dataset.pkl` and the Brownian-bridge model with seed 20260924 |
| `Figure_4_Tg` | Fig. 5 | `data/derived/tg/tg_msd_histories.csv` (three cooling histories) |
| `Figure_5_Mechanical_State` | Fig. 6 | graphs in `data/dataset.pkl` and `data/derived/mechanics/mechanics_pulls_4seeds.csv` (12 pulls per network) |
| `Figure_6_Contrast_Maps` | Fig. 7 | as Fig. 6 |
| `Figure_A2_entanglement` | Fig. 8 (Appendix B) | `data/derived/entanglement/` |
| `Figure_A1_bondcreate` | Fig. 9 (Appendix C) | `data/derived/bond_create/` |
| `Figure_A1_bondcreate_{distance,rank,ring}` | not in the manuscript (variants of Fig. 9) | as Fig. 9 |
| `pearson_correlation_matrix.png` | not in the manuscript | as Fig. 6 |

Fig. 1 is a workflow schematic and is not made here.

## Changes since v0.1.0

The plotting code, layout, colours and fonts are those of the v0.1.0 notebook, and every change to its cells is
marked `# CHANGE (v0.2.0)`.

- Figs. 2 to 4 use the same code and data. The PDFs are now saved as well, and Fig. 4 has a fixed seed
  (`np.random.seed(20260924)`). Its model curves lie within Monte Carlo noise of the unseeded v0.1.0 curves.
- Fig. 5 shows the mean and SD of three independent cooling histories. The fit is of the history-averaged curve, and
  the labels give Tg ± SD of the three per-history values.
- Figs. 6 and 7 and the Pearson matrix use network means over 12 pulls (3 axes and 4 velocity seeds) instead of 3.
  Fig. 6a labels its x axis "Toughness", and Fig. 6b adds 95% network-bootstrap bands and the number of networks per
  quadrant. Fig. 7 uses neutral metric names and the cycle rank of the active graph (`cyclomatic_idx`), marks cells with
  Benjamini-Hochberg q < 0.05 over the 55 Welch tests, and shows the same 10 descriptors in panels b and c.
- The mock-data branches are removed.
- Figs. 8 and 9 are new.
- Since 0.2.1 the panel titles of Fig. 7 read "Signatures of Toughening" and "Signatures of Strengthening" (they read
  "Drivers of ..." before).

Each changed cell checks its numbers against the values quoted in the manuscript (quadrant sizes 128/35/35/129, 31 of
55 marked cells, the Tg labels, the reference statistics tables in `data/derived/figure_stats/`) and stops if they
differ.

## Supporting checks

`generate_checks.ipynb` regenerates the supporting checks from `data/derived/`. They cover equilibration stationarity,
the lattice imprint on strand orientation, true versus nominal stress, the four-seed variance decomposition, the
statistics behind Figs. 6 and 7, Tg per cooling history (with the MSD lag scan), the chain statistics of the atomistic
networks and the generator benchmark tables. The table at the top of the notebook lists every cell and its output
files.

The scripts in `scripts/checks/threshold_cv/` repeat the quadrant contrasts of Fig. 7 with other thresholds (e.g., the
40th and 60th percentiles) and predict the UTS and toughness of held-out networks from the 11 descriptors with
cross-validated ridge regression. Their README gives the commands and the results.

## Data provenance

`data/dataset.pkl`, `data/csv/` and `data/mechanics/` are the data deposited with v0.1.0, and their data files are
unchanged. `data/derived/` holds
derived data only (per-pull mechanics, MSD per temperature and cooling history, statistics tables and compact subsets
of the structure-analysis cache). `data/derived/README.md` gives the columns, units and origin of every file, and
`data/derived/MANIFEST.csv` its source and SHA-256. The raw simulation output behind them is in `data/raw/` (the
stress-strain files of all 3,924 pulls, the MSD files and inputs of the three cooling histories with the as-built and
equilibrated atomistic systems they start from, and the LAMMPS inputs and generator settings of every coarse-grained
network). Trajectories, restart files and logs are not included.

## What is not regenerated here

- Quantities that need raw simulation output enter as derived data. These are the per-pull mechanical metrics, the
  MSD averages, the strand vectors and thermo time series behind the stationarity and orientation checks, the box areas
  behind the stress comparison, the Z1+ entanglement counts behind Figs. 8 and 9, and the generator benchmark runs.
  The scripts that made them are in `scripts/raw_data_analyses/` or named in `data/derived/README.md`.
- The stationarity and orientation figures are drawn from compact derived data (block means, drift statistics and the
  used subset of the 55 MB analysis cache, 4.9 MB in total) rather than from the raw thermo and data files. The
  plotting code is the same.
- The software table is a record, not a computation. It is written from `data/derived/software/software_seeds.csv`.

## Verification

Both notebooks were executed with `nbconvert` on 26 Sep 2026 in the environment of `requirements.txt`. Figs. 2, 3 and
5 to 9 render pixel-identical (300 dpi) to the figure files of the manuscript. Fig. 4 differs from the manuscript
figure by the Monte Carlo noise of the seeded model curves described above.

## Notes

- `data/derived/mechanics/mechanics_12pull.csv` carries a column `cycle_rank` in an older convention (vacancies
  counted as nodes). The analyses use `cyclomatic_idx` (E - N_active + 1).
- Save order matters for byte-identical files. A second `savefig` redraws the figure and constrained layout moves by
  a fraction of a point, so the cells save with `fig.savefig` in a fixed order.
