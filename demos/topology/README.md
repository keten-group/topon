# Topology demos

Topology generation on its own, without chemistry, conformation or LAMMPS files. This is the place to look at a graph
(degree distribution, connectivity, cycles) before a full build, or to time the generators.

[`end_linking/`](end_linking/) builds the same kind of 6x6x6 simple cubic network in two ways.

- [`end_linking/python/run.py`](end_linking/python/run.py) uses the Python generator (`topon.topology.generator_python`),
  which is the pipeline default and needs no compiler.
- [`end_linking/c/run.py`](end_linking/c/run.py) calls the C generator. Build it first from
  `topon/topology/csrc/generator.c` (`gcc -O2 -o generator.exe generator.c -lm`) and set `TOPON_GENERATOR_EXE` to the
  binary.

Both write into an `output/` folder next to the script (the Python one as `network.nodes` and `network.edges`, the C
one as `output/network_N6x6x6_trial<k>.nodes` and `.edges`, since the binary makes its own `output/` folder). Any
polymer demo can load them by setting `topology.source` to `"load"`.
