# Expected output of the atomistic basic demo

Made on 6 Oct 2026 from `config.json` with the atomistic defaults of topon 0.4.5. The
backbones are drawn as meanders and settled in at most 1,500 rounds (no two backbone bonds
closer than 1.5 A, every bond and angle at its equilibrium, no bond moved through another),
the relaxation runs the hard-backbone stages (stage 1 5,000 steps, the ramp 2,500 steps, and
a minimization capped at 1,000 iterations in stage 3), and the crossing detector reads every
stage's backbone dump. The DREIDING parameters carry the corrections of 0.4.5 (geometric
mixing with the tail correction, the dihedral sign and umbrella impropers). Both global
random streams are seeded with 20260929, and the placement draws from a stream keyed on the
study name (`run` in the recipe below), so the recipe writes the same build files on every
run.

The topology is the 5x5x5 SC network in `demos/showcase/network_5x5x5/`, loaded in place of
the demo's generator settings, so every DREIDING demo starts from the same graph.

11,087 atoms with DREIDING parameters, 210 strands drawn as meanders. No two backbone bonds
were closer than 1.5 A as drawn, and the settling pass converged in 154 rounds and moved no
bond through another.

## Stages

LAMMPS 2 Apr 2025 with 8 OpenMP threads took 1.9 minutes (stage 1 18 s, stage 2 41 s, stage
3 53 s), and every gate passed.

| Checkpoint | g/cm³ | T (K) | longest backbone bond (× r0) | backbone passages in its stage | Z1+ per bridge (4 seeds) |
|---|---|---|---|---|---|
| build | 0.900 | - | 1.018 | - | 0.010 ± 0.009 |
| stage 1 | 0.900 | 276 | 1.059 | 0 | 0.007 ± 0.004 |
| ramp | 0.900 | 304 | 1.080 | 0 | 0.006 ± 0.008 |
| minimized | 0.900 | 304 | 1.032 | 0 | 0.004 ± 0.004 |
| NVT | 0.900 | 311 | 1.083 | 0 | 0.004 ± 0.002 |
| NPT | 0.904 | 301 | 1.067 | 0 | 0.005 ± 0.003 |

Passages are read from each stage's backbone dump, and the one dump of stage 3 covers its
minimization, NVT and NPT. Z1+ is reported and not gated. It follows the junctions as they
move, so it changes even where nothing crosses.
The table gives Z1+ over four seeds, as the gates read it, and the tracker page over eight,
so the two differ a little.

## Files

This folder holds the small text files of the run. The data files and the displacement
files are not included (they are rebuilt by the commands below).

- `system.in.settings` and `system.groups`, written by the chemistry stage.
- The stage scripts `minimize_1_serial.in`, `minimize_2_parallel.in` and
  `minimize_3_parallel.in`, and their LAMMPS logs `log.stage1.lammps` to
  `log.stage3.lammps` (the plugin path is replaced by `<LAMMPS install>`).
- `manifest.json`, the run manifest with the strand record and the record of the placement
  and the settling pass. `topon analyze` and `topon track` read it.
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
from topon.config.schema import TopologyConfig, ExistingFilesConfig
from topon.pipeline import Pipeline
cfg, raw = load_config_full(Path('demos/polymer/atomistic/basic/config.json'))
cfg.topology = TopologyConfig(source='load', existing_files=ExistingFilesConfig(
    nodes_file='demos/showcase/network_5x5x5/network.nodes',
    edges_file='demos/showcase/network_5x5x5/network.edges'))
cfg.study.output_dir = 'runs/basic_demo'; cfg.study.name = 'run'
random.seed(20260929); np.random.seed(20260929)
Pipeline(cfg, raw_config=raw).run()
PY

cd runs/basic_demo/run/04_Simulation
lmp -sf omp -pk omp 8 -in minimize_1_serial.in
lmp -sf omp -pk omp 8 -in minimize_2_parallel.in
lmp -sf omp -pk omp 8 -in minimize_3_parallel.in
cd ../../..
topon track runs/basic_demo/run --omp 8
```

`topon.simulation.protocols.atomistic.AtomisticRun` runs the three stages and applies the
gates after each one in a single call. `topon track` needs Z1+ for the Z columns (see
`docs/USAGE.md`).
