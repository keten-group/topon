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
"degree_distribution": "0:0,1:25",
"node_types.degree.mapping": { "1": "POSS", "2": "A", "3": "A", "4": "A" },
"node_type_map": {
  "POSS": { "molecule": "POSS_AM0270", "is_end_cap": true },
  "A":    { "molecule": "Si",          "is_end_cap": false }
}
```

The generator places 25 dangling ends on the 5x5x5 lattice, and each one gets a POSS cage. `POSS_AM0270` is a
built-in molecule placed by `ChemistryBuilder._place_poss_am0270()`, a Si₈O₁₂ cage with seven isooctyl arms and one
propyl linker that ties it to the strand, with explicit hydrogens. The built system has about 24,000 atoms.

POSS at an internal junction (degree 2 or more) is not supported, and `topon doctor` reports it.

## Output

The LAMMPS data file and input scripts go to `<output>/atomistic_poss/`. For a different POSS workflow (packed
AM0270-POSS molecules with epoxy and amine crosslinking templates), see `topon simbox` in
[`docs/USAGE.md`](../../docs/USAGE.md).
