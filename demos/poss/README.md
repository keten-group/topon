# POSS chain caps

An atomistic PDMS network whose dangling ends are capped with POSS (polyhedral oligomeric silsesquioxane, an Si₈O₁₂
cage). The config maps degree-1 sites to `POSS_AM0270` and every junction to a bare Si atom.

## Run

```bash
topon generate demos/poss/config.json --output ./runs
```

or through the Python API

```bash
python demos/run_via_api.py poss/config.json
```

## What the config encodes

```json
"degree_distribution": "0:0,1:25"
"node_types.degree.mapping": { "1": "POSS", "2": "A", "3": "A", "4": "A" }
"node_type_map": {
  "POSS": { "molecule": "POSS_AM0270", "is_end_cap": true },
  "A":    { "molecule": "Si",          "is_end_cap": false }
}
```

The generator places 25 dangling ends on the 5x5x5 lattice, and each one gets a POSS cage. `POSS_AM0270` is a
built-in molecule placed by `ChemistryBuilder._place_poss_am0270()`, a Si₈O₁₂ cage with seven isooctyl arms and one
propyl linker that ties it to the strand, with explicit hydrogens (204 atoms per cap). The built system has about
24,000 atoms.

The amine of AM0270 is not built. The real AM0270 carries an aminopropyl arm (-CH₂CH₂CH₂-NH₂), whose N bonds to an
opened epoxide in a cure. Here the propyl arm is tethered straight to the strand's end atom, which follows the strand's
direction in the graph. At the strand's first end that is its head Si (a carbosilane, POSS-(CH₂)₃-SiMe₂-O-), and at its
last end its tail O (an alkoxysilane, POSS-(CH₂)₃-O-SiMe₂-). The `topon simbox` route below builds the amine and its
reaction.

POSS at an internal junction (degree 2 or more) is not supported, and `topon doctor` reports it. Since 0.4.5 each
cage is placed whole at its chain end by the settled atomistic placement, with every bond near its r0, and the
strands are settled clear of it.

## Output

The LAMMPS data file and input scripts go to `<output>/atomistic_poss/`. For a different POSS workflow (packed
AM0270-POSS molecules with epoxy and amine crosslinking templates), see `topon simbox` in
[`docs/USAGE.md`](../../docs/USAGE.md).

## Expected output

[`expected_output/`](expected_output/README.md) holds this config built on the 5x5x5 SC network in
`demos/showcase/network_5x5x5/` instead of its generator, with its recipe. The build has 121 nodes (115 Si junctions
and 6 POSS caps), 210 strands and 22,731 atoms, and every cage is placed whole with every bond near its r0. LAMMPS has
not been run on it, so the folder has the settings, groups, stage scripts and run manifest and no stage results.
