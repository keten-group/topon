# Expected output of the atomistic graft demo

Made on 29 Sep 2026 from `config.json` with the atomistic defaults of topon 0.4.0. The
backbones are drawn as meanders and settled (no two backbone bonds closer than 1.5 A, every
bond and angle at its equilibrium, no bond moved through another), the relaxation runs the
hard-backbone stages (stage 1 and the ramp 5,000 steps each, and a capped minimization in
stage 3), and the crossing detector reads every stage's backbone dump. The random streams
are seeded with 20260929.

The topology is the 5x5x5 SC network in `demos/showcase/network_5x5x5/`, loaded in place of
the demo's generator settings, so every DREIDING demo starts from the same graph.

41,186 atoms with DREIDING parameters, 210 strands drawn as meanders. The settling pass
parted 162 pairs of backbone bonds that were closer than 1.5 A, left none, and moved no bond
through another. The grafts are drawn at random (seeded here), so a rebuild with another
seed has another atom count.

## Stages

LAMMPS 2 Apr 2025 with 4 OpenMP threads took 9.4 minutes (stage 1 30 s, stage 2 263 s, stage
3 270 s), and every gate passed.

| Checkpoint | g/cm³ | T (K) | longest backbone bond (× r0) | backbone passages in its stage | Z1+ per bridge (4 seeds) |
|---|---|---|---|---|---|
| build | 0.900 | - | 1.019 | - | 0.001 ± 0.002 |
| stage 1 | 0.900 | 275 | 1.074 | 0 | 0.010 ± 0.003 |
| ramp | 0.900 | 302 | 1.079 | 0 | 0.007 ± 0.010 |
| minimized | 0.900 | 302 | 1.028 | 0 | 0.013 ± 0.007 |
| NVT | 0.900 | 289 | 1.079 | 0 | 0.010 ± 0.003 |
| NPT | 0.868 | 301 | 1.090 | 0 | 0.017 ± 0.007 |

Passages are read from each stage's backbone dump, and the one dump of stage 3 covers its
minimization, NVT and NPT. Z1+ is reported and not gated. It follows the junctions as they
move, so it changes even where nothing crosses.

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

The LAMMPS runs started from coordinates made before topon 0.4.0 drew the conformation noise
from a stream of its own. A rebuild with the commands below starts from coordinates at most
0.003 A away, so its numbers differ a little.

## Reproducing

From the repository root,

```bash
python - <<'PY'
import random, numpy as np
from pathlib import Path
from topon.config import load_config_full
from topon.config.schema import TopologyConfig, ExistingFilesConfig
from topon.pipeline import Pipeline
cfg, raw = load_config_full(Path('demos/polymer/atomistic/graft/config.json'))
cfg.topology = TopologyConfig(source='load', existing_files=ExistingFilesConfig(
    nodes_file='demos/showcase/network_5x5x5/network.nodes',
    edges_file='demos/showcase/network_5x5x5/network.edges'))
cfg.study.output_dir = 'runs/graft_demo'; cfg.study.name = 'run'
random.seed(20260929); np.random.seed(20260929)
Pipeline(cfg, raw_config=raw).run()
PY

cd runs/graft_demo/run/04_Simulation
lmp -sf omp -pk omp 4 -in minimize_1_serial.in
lmp -sf omp -pk omp 4 -in minimize_2_parallel.in
lmp -sf omp -pk omp 4 -in minimize_3_parallel.in
cd ../../..
topon track runs/graft_demo/run --omp 4
```

`topon.simulation.protocols.atomistic.AtomisticRun` runs the three stages and applies the
gates after each one in a single call. `topon track` needs Z1+ for the Z columns (see
`docs/USAGE.md`).
