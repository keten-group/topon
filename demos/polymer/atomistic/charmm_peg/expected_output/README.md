# Expected output of the CHARMM PEG demo

Made on 5 Oct 2026 from `config.json` with the atomistic defaults of topon 0.4.5. The
backbones are drawn as meanders and settled in at most 1,500 rounds (no two backbone bonds
closer than 1.5 A, every bond and angle at its equilibrium, no bond moved through another),
the relaxation runs the hard-backbone stages (stage 1 5,000 steps, the ramp 2,500 steps, and
a minimization capped at 1,000 iterations in stage 3), and the crossing detector reads every
stage's backbone dump. Both global random streams are seeded with 20260929, and the
placement draws from a stream keyed on the study name (`run` in the recipe below), so the
recipe writes the same build files on every run.

The topology comes from the demo's own generator settings.

3,648 atoms with CHARMM parameters, 128 strands drawn as meanders. The settling pass parted
874 pairs of backbone bonds that were closer than 1.5 A in 91 rounds, left none, and moved
no bond through another.

## Stages

LAMMPS 2 Apr 2025 with 8 OpenMP threads took 3.6 minutes (stage 1 12 s, stage 2 30 s, stage 3
173 s), and every gate passed.

| Checkpoint | g/cm³ | T (K) | longest backbone bond (× r0) | backbone passages in its stage | Z1+ per bridge (4 seeds) |
|---|---|---|---|---|---|
| build | 1.000 | - | 1.019 | - | 0.016 ± 0.006 |
| stage 1 | 1.000 | 272 | 1.067 | 0 | 0.020 ± 0.012 |
| ramp | 1.000 | 298 | 1.076 | 0 | 0.023 ± 0.020 |
| minimized | 1.000 | 298 | 1.025 | 0 | 0.016 ± 0.011 |
| NVT | 1.000 | 301 | 1.074 | 0 | 0.016 ± 0.012 |
| NPT | 0.962 | 302 | 1.087 | 0 | 0.018 ± 0.006 |

Passages are read from each stage's backbone dump, and the one dump of stage 3 covers its
minimization, NVT and NPT. Z1+ is reported and not gated. It follows the junctions as they
move, so it changes even where nothing crosses.
The table gives Z1+ over four seeds, as the gates read it, and the tracker page over eight,
so the two differ a little.

## Files

This folder holds the small text files of the run. The data files are not included (they
are rebuilt by the commands below).

- `system.in.settings`, `system.in.settings.soft`, `system.in.settings.lj` and
  `system.groups`, written by the chemistry stage.
- The stage scripts `minimize_1_serial.in`, `minimize_2_parallel.in` and
  `minimize_3_parallel.in`, and their LAMMPS logs `log.stage1.lammps` to
  `log.stage3.lammps` (the plugin path is replaced by `<LAMMPS install>`).
- `manifest.json`, the run manifest with the strand record, the CHARMM files read and the
  record of the placement and the settling pass. `topon analyze` and `topon track` read it.
- `relaxation_tracker.html`, the `topon track` page of the run. Open it in a browser to see
  the network at each checkpoint and the numbers through the stages.

The commands below rebuild the files these runs started from. The dynamics still make a
rerun of the stages differ a little from the numbers above.

## Reproducing

From the repository root,

```bash
python - <<'PY'
import random, numpy as np
from pathlib import Path
from topon.config import load_config_full
from topon.pipeline import Pipeline
cfg, raw = load_config_full(Path('demos/polymer/atomistic/charmm_peg/config.json'))
cfg.study.output_dir = 'runs/charmm_peg_demo'; cfg.study.name = 'run'
random.seed(20260929); np.random.seed(20260929)
Pipeline(cfg, raw_config=raw).run()
PY

cd runs/charmm_peg_demo/run/04_Simulation
lmp -sf omp -pk omp 8 -in minimize_1_serial.in
lmp -sf omp -pk omp 8 -in minimize_2_parallel.in
lmp -sf omp -pk omp 8 -in minimize_3_parallel.in
cd ../../..
topon track runs/charmm_peg_demo/run --omp 8
```

`topon.simulation.protocols.atomistic.AtomisticRun` runs the three stages and applies the
gates after each one in a single call. `topon track` needs Z1+ for the Z columns (see
`docs/USAGE.md`).
