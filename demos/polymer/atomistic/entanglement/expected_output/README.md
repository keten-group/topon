# Expected output of the atomistic entanglement demo

Made on 29 Sep 2026 from `config.json` with the atomistic defaults of topon 0.4.0. The
backbones are drawn as meanders and settled (no two backbone bonds closer than 1.5 A, every
bond and angle at its equilibrium, no bond moved through another), the relaxation runs the
hard-backbone stages (stage 1 and the ramp 5,000 steps each, and a capped minimization in
stage 3), and the crossing detector reads every stage's backbone dump. The random streams
are seeded with 20260929.

The topology is the 5x5x5 SC network in `demos/showcase/network_5x5x5/`, loaded in place of
the demo's generator settings, so every DREIDING demo starts from the same graph.

21,587 atoms with DREIDING parameters, 200 strands drawn as meanders and 10 drawn as
designed entangled pairs. The settling pass parted 672 pairs of backbone bonds that were
closer than 1.5 A, left none, and moved no bond through another.

## Stages

LAMMPS 2 Apr 2025 with 4 OpenMP threads took 4.0 minutes (stage 1 17 s, stage 2 127 s, stage
3 95 s), and every gate passed.

| Checkpoint | g/cm³ | T (K) | longest backbone bond (× r0) | backbone passages in its stage | Z1+ per bridge (4 seeds) |
|---|---|---|---|---|---|
| build | 0.900 | - | 1.007 | - | 0.077 ± 0.009 |
| stage 1 | 0.900 | 275 | 1.062 | 0 | 0.064 ± 0.008 |
| ramp | 0.900 | 298 | 1.083 | 0 | 0.076 ± 0.016 |
| minimized | 0.900 | 298 | 1.025 | 0 | 0.070 ± 0.014 |
| NVT | 0.900 | 299 | 1.082 | 0 | 0.072 ± 0.015 |
| NPT | 0.863 | 301 | 1.080 | 0 | 0.070 ± 0.019 |

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
cfg, raw = load_config_full(Path('demos/polymer/atomistic/entanglement/config.json'))
cfg.topology = TopologyConfig(source='load', existing_files=ExistingFilesConfig(
    nodes_file='demos/showcase/network_5x5x5/network.nodes',
    edges_file='demos/showcase/network_5x5x5/network.edges'))
cfg.study.output_dir = 'runs/entanglement_demo'; cfg.study.name = 'run'
random.seed(20260929); np.random.seed(20260929)
Pipeline(cfg, raw_config=raw).run()
PY

cd runs/entanglement_demo/run/04_Simulation
lmp -sf omp -pk omp 4 -in minimize_1_serial.in
lmp -sf omp -pk omp 4 -in minimize_2_parallel.in
lmp -sf omp -pk omp 4 -in minimize_3_parallel.in
cd ../../..
topon track runs/entanglement_demo/run --omp 4
```

`topon.simulation.protocols.atomistic.AtomisticRun` runs the three stages and applies the
gates after each one in a single call. `topon track` needs Z1+ for the Z columns (see
`docs/USAGE.md`).
