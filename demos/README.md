# Demos

Configs and scripts that show what topon builds. Run them from the repository root after `pip install -e .`.

```bash
topon generate demos/polymer/coarse_grained/basic/config.json --output ./runs
```

Every polymer and POSS config generates its own network with the Python generator (a 5x5x5 simple cubic lattice), so
it also runs from any other folder. `topon generate` writes the LAMMPS data file and input scripts. It does not run
LAMMPS.

## Layout

```
demos/
  templates/        minimal.json (loads the showcase network) and full.json (every section written out)
  defaults/         assignment fragments (node types by degree, one uniform edge type)
  showcase/         a small 5x5x5 network in the .nodes/.edges format
  run_via_api.py    runs a config through topon.pipeline.Pipeline, the way topon generate does
  polymer/          atomistic (DREIDING) and coarse-grained (Kremer-Grest) networks, one folder per feature
  poss/             an atomistic network with POSS chain caps
  topology/         topology generation only, with the Python and the C generator
  workflows/        a batch of topologies exported as .nodes/.edges, GraphML and NPZ
  npjcompmat/       data and notebooks of the npj Computational Materials paper
```

- [polymer/](polymer/README.md) lists the twelve polymer demos and the config section behind each feature.
- [poss/](poss/README.md) explains the POSS chain caps.
- [topology/](topology/README.md) builds a network without any chemistry.
- [workflows/](workflows/README.md) scripts a batch of networks.
- [showcase/](showcase/README.md) describes the `.nodes`/`.edges` format.
- [npjcompmat/](npjcompmat/README.md) regenerates the data figures of the paper.

`topon init --preset atomistic_pdms`, `--preset cg_kg` and `--preset poss` copy `polymer/atomistic/basic/`,
`polymer/coarse_grained/basic/` and `poss/` respectively. The full CLI and the config schema are in
[`docs/USAGE.md`](../docs/USAGE.md).
