# Expected output of the POSS demo

Made on 7 Oct 2026 from `config.json` with the atomistic defaults of topon 0.4.5. LAMMPS has
not been run on this build, so this folder holds the small text files of the build and its
stage scripts, and no stage results (no LAMMPS logs and no tracker page). Both global random
streams are seeded with 20260929, as for the atomistic demos, and the placement and the
conformation noise draw from streams keyed on the study name (`run` in the recipe below), so
the recipe writes the same build files on every run.

The topology is the 5x5x5 SC network in `demos/showcase/network_5x5x5/`, loaded in place of
the demo's generator settings, as the atomistic demos load it. Its 4 vacant sites are
dropped, which leaves 121 junctions and 210 strands of DP 10 (204 bridges and 6 dangling
strands).

The config maps the 6 degree-1 nodes to `POSS_AM0270` caps and the other 115 to a bare Si
junction (the 46 trifunctional ones carry a methyl). 22,731 atoms, every one with a DREIDING
type (Si3 2,263, O_3 2,380, C_3 4,600, H_ 13,488) and a Gasteiger charge (net 2.5e-9 e). Each
cage's 8 Si are Si3 and its 12 O are O_3. Two caps bond to their strand's head Si and four
to its tail O (see `../README.md`).

## Placement

The strands are drawn as meanders at the force field's bond lengths and settled (no two
backbone bonds closer than 1.5 A, every bond and angle at its equilibrium, no bond moved
through another), and each POSS cage is placed whole, as a rigid body.

- Each cap's shape is its own fragment of the network molecule (the Si8O12 cage, the propyl
  arm, the seven isooctyl arms and their hydrogens, 204 atoms), embedded (ETKDG, then MMFF,
  seed 42) and minimized with every bond held at its DREIDING r0. There are two templates,
  one for a strand that bonds to the propyl carbon through its Si and one through its O, and
  every bond of both is within 0.56 % of r0. Both shapes are stored with topon
  (`topon/data/poss_templates.json`, as RDKit 2025.09.6 embedded them) and read from there,
  so the build is the same whatever RDKit is installed.
- The cage's center sits on the junction site, and the cage is turned so the atom the strand
  bonds to points at the strand's other junction. Of 36 turns about that axis it takes the
  one that keeps the cage furthest from the other strands' chords, the other junctions and
  the other cages (two pairs of caps sit on neighboring sites, 12.7 A apart, and their arms
  reach 9 A).
- Seven strands were turned about their chords before the settling pass, one drawn into a
  cage's core and six drawn within 1.5 A of the arm bonds of a cap on a neighboring site,
  which the settling pass cannot move. The pass parted the 1,231 pairs of backbone bonds
  drawn closer than 1.5 A in 334 rounds and converged, and no bond moved through another
  backbone bond or through a cage bond or face.
- The cage atoms are held by the overlap pass of stage 5, which resolved 68 overlaps in one
  round.

Bond lengths of `system_conformed.data` (as placed) and `system_relaxed.data` (after stage
5), read against `system.in.settings`, in A.

| Bond | n | r0 | as placed, median (range) | after stage 5, median (range) |
|---|---|---|---|---|
| Si-O | 4,756 | 1.587 | 1.587 (1.568 to 1.602) | 1.587 (1.568 to 1.602) |
| C-Si | 4,296 | 1.697 | 1.697 (1.697 to 1.706) | 1.697 (1.616 to 1.792) |
| C-H | 13,488 | 1.090 | 1.090 (1.090 to 1.091) | 1.090 (0.961 to 1.236) |
| C-C | 306 | 1.530 | 1.531 (1.530 to 1.532) | 1.531 (1.530 to 1.532) |
| C-O | 4 | 1.420 | 1.420 | 1.420 |
| all, bond / r0 | 22,850 | | 0.988 to 1.010 | 0.882 to 1.134 |

No bond is more than 15 % off its r0 in either file. The spread after stage 5 is the overlap
pass, which moves an atom closer than 0.2 A to another by up to 0.25 A. Both files hold 6
whole T8 cages, with no bond through a cage face.

## Stage scripts

The stage scripts are the hard-backbone stages (stage 1 5,000 steps, the ramp 2,500 steps,
and a minimization capped at 1,000 iterations in stage 3). Their backbone types are those of
the strands' backbones and end atoms, and a cap's strand ends on the propyl carbon, so C_3 is
a backbone type here with Si3 and O_3. Every non-bonded pair of carbons beyond the angle
neighbors (the PDMS methyls and the isooctyl arms with them) is hard from the first step,
only hydrogen ramps, and every carbon is dumped for the crossing detector. On the other
atomistic demos the methyl carbons ramp. DREIDING weighs 1-4 pairs in full, so the gauche
C...C pairs of an arm, at 2.5 to 3.0 A, sit inside the 3 A soft core from the first step.

## Files

This folder holds the small text files of the build. The data files and the displacement
files are not included (they are rebuilt by the commands below), and there are no LAMMPS
logs and no tracker page.

- `system.in.settings` and `system.groups`, written by the chemistry stage.
- The stage scripts `minimize_1_serial.in`, `minimize_2_parallel.in` and
  `minimize_3_parallel.in`.
- `manifest.json`, the run manifest with the strand record and the record of the placement,
  the cages and the settling pass.

## Reproducing

From the repository root,

```bash
python - <<'PY'
import random, numpy as np
from pathlib import Path
from topon.config import load_config_full
from topon.config.schema import TopologyConfig, ExistingFilesConfig
from topon.pipeline import Pipeline
cfg, raw = load_config_full(Path('demos/poss/config.json'))
cfg.topology = TopologyConfig(source='load', existing_files=ExistingFilesConfig(
    nodes_file='demos/showcase/network_5x5x5/network.nodes',
    edges_file='demos/showcase/network_5x5x5/network.edges'))
cfg.study.output_dir = 'runs/poss_demo'; cfg.study.name = 'run'
random.seed(20260929); np.random.seed(20260929)
Pipeline(cfg, raw_config=raw).run()
PY
```

The build takes one to three minutes, most of it the settling pass (both cage shapes are read
from the stored file, not embedded). The stages have not been run on this build. They run as

```bash
cd runs/poss_demo/run/04_Simulation
lmp -in minimize_1_serial.in
lmp -in minimize_2_parallel.in
lmp -in minimize_3_parallel.in
```
