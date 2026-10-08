"""
Command-line interface for Topon.
"""

import json
import sys
from pathlib import Path

import click

from topon import __version__


_BANNER = r"""
   +==================================================================+
   |     ########   ########   ########    ########   ###     ##      |
   |        ##      ##    ##   ##     ##   ##    ##   ####    ##      |
   |        ##      ##    ##   ########    ##    ##   ## ##   ##      |
   |        ##      ##    ##   ##          ##    ##   ##  ##  ##      |
   |        ##      ########   ##          ########   ##   #####      |
   |                                                                  |
   |   Topological polymer & protein network generator for LAMMPS     |
   |                                                       v{version:<7s}    |
   +==================================================================+

   Get started in 3 commands:
     init                  Make a starter config that works out of the box
     doctor <config>       Lint it for common mistakes
     generate <config>     Run the 6-stage pipeline -> LAMMPS data files

   All commands (type `help <cmd>` for details):
     init       validate    doctor      generate    inspect
     analyze    fit         track       simbox      chain
     protein    recipes

   Pipeline (`generate`):
      Topology -> Analysis -> Assignment -> Chemistry -> Conformation -> Output

   Quick start:
     topon> init                       # writes config.json (atomistic PDMS)
     topon> doctor config.json         # checks for footguns
     topon> generate config.json       # builds LAMMPS files
     topon> inspect <output_dir>       # summarises what landed
     topon> recipes                    # "I want X -> run Y" cheatsheet

   Shell built-ins:
     help [cmd]            Top-level command list, or detail on one command
     exit | quit | Ctrl-D  Leave the shell

   Docs: README.md  |  docs/USAGE.md  |  demos/  |  demos/workflows/
"""


_BANNER_ONE_SHOT_FOOTER = r"""
   Tips:
     `python -m topon`            drop into the interactive shell
     `python -m topon <command>`  run a single command and exit
     `python -m topon <cmd> --help`  flags for one command
"""


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="topon")
@click.option(
    "--no-shell", is_flag=True,
    help="Print the banner and exit (don't drop into the interactive shell).",
)
@click.pass_context
def main(ctx, no_shell: bool):
    """Topon: topological polymer and protein network generator for LAMMPS.

    With no subcommand on a TTY, drops into an interactive shell where
    you can type `help`, `init`, `doctor`, etc. directly. Pipe input or
    pass --no-shell to get the old banner-and-exit behaviour.
    """
    if ctx.invoked_subcommand is not None:
        return

    banner = _BANNER.format(version=__version__)
    interactive = sys.stdin.isatty() and not no_shell

    if not interactive:
        click.echo(banner)
        click.echo(_BANNER_ONE_SHOT_FOOTER)
        return

    # Interactive shell. Banner first, then drop into the REPL.
    from topon.shell import run_shell
    intro = banner + "\n   Type `help` for the command list, or `exit` to leave.\n"
    sys.exit(run_shell(main, intro=intro))


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Override output directory")
@click.option("--dry-run", is_flag=True, help="Validate config without running pipeline")
@click.option(
    "--export-graphml",
    is_flag=True,
    help="Also export the dual-graph (chains + entanglement edges) as <name>.graphml",
)
@click.option(
    "--export-npz",
    is_flag=True,
    help="Also export the graph as <name>.npz for downstream GNN pipelines",
)
@click.option("--verify", "verify_ref", type=click.Path(exists=True), default=None,
              help="Reference to hold the build against (LAMMPS data file, NPZ "
                   "dual graph or strand graph): requested against achieved "
                   "P(f) and loops, descriptor composite, reach. Writes "
                   "verify.json into the run directory")
@click.option("--verify-seeds", type=int, default=1, show_default=True,
              help="Seeds --verify regenerates, from topology.generator.seed up "
                   "(topology.crosslinking.seed for a crosslinked melt)")
@click.option("--verify-only", is_flag=True,
              help="With --verify: build nothing, only regenerate the graphs "
                   "(stages 1-3, in a scratch directory) and verify them")
@click.option("--relaxed", type=click.Path(exists=True), default=None,
              help="With --verify: a relaxed end-linked data file of this "
                   "build, or the run directory it is in (the last MD "
                   "checkpoint is taken), for Z1+ per strand class, the "
                   "per-bridge histogram and partners against the reference "
                   "(making it needs MD, which generate does not run)")
@click.option("--junction-type", type=int, default=None,
              help="With --verify: junction atom type of a reference data "
                   "file that is not typed 1 end / 2 interior / 3 junction")
@click.option("--verify-replicate", "verify_replicates", multiple=True,
              type=click.Path(exists=True),
              help="With --verify of a build crosslinked along its chains: a "
                   "replicate of the reference (the same process, another "
                   "seed); repeat it. The report says which measures of each "
                   "build sit inside the replicates' scatter")
def generate(
    config_path: str,
    output: str,
    dry_run: bool,
    export_graphml: bool,
    export_npz: bool,
    verify_ref: str,
    verify_seeds: int,
    verify_only: bool,
    relaxed: str,
    junction_type: int,
    verify_replicates: tuple,
):
    """
    Run the full pipeline from a configuration file.

    CONFIG_PATH: Path to the JSON configuration file.

    With --verify REFERENCE the build is regenerated (and, with
    --verify-seeds N, N seeds from the config's own) and held against the
    reference, which is what a config written by `topon fit` is checked
    with:

        topon generate ref_config.json --verify ref.data --verify-seeds 3
    """
    if (verify_only or relaxed or junction_type is not None
            or verify_replicates) and not verify_ref:
        click.echo("Error: --verify-only, --relaxed, --junction-type and "
                   "--verify-replicate go with --verify REFERENCE", err=True)
        sys.exit(2)
    if relaxed:
        # before the build, not after it: a run with no MD has nothing to read
        from topon.inverse.verify import resolve_relaxed
        try:
            resolve_relaxed(relaxed)
        except ValueError as e:
            click.echo(f"Error: --relaxed: {e}", err=True)
            sys.exit(2)
    from topon.config import load_config_full, validate_config
    from topon.pipeline import Pipeline

    click.echo(f"Loading configuration from: {config_path}")

    try:
        config, raw_cfg = load_config_full(config_path)
    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        sys.exit(1)

    # Override output directory if specified
    if output:
        config.study.output_dir = output

    # CLI flags override config.output.*
    if export_graphml:
        config.output.export_graphml = True
    if export_npz:
        config.output.export_npz = True

    # Validate configuration
    errors = validate_config(config)
    if errors:
        click.echo("Configuration validation errors:", err=True)
        for error in errors:
            click.echo(f"  - {error}", err=True)
        sys.exit(1)

    click.echo("Configuration is valid.")
    if raw_cfg:
        click.echo(
            f"  Raw extras forwarded to Pipeline: {sorted(raw_cfg.keys())}"
        )

    if dry_run:
        click.echo("Dry run - not executing pipeline.")
        return

    built = {}
    if not verify_only:
        # Run pipeline (raw_config carries conformation / simulation /
        # execution / experimental sections that aren't in ToponConfig).
        click.echo("Running pipeline...")
        pipeline = Pipeline(config, raw_config=raw_cfg)
        from topon.chemistry.charmm import CharmmTypingError, MissingCharmmParameters
        from topon.forcefield.dreiding import ChargeError, UntypedAtomError
        try:
            pipeline.run()
        except (CharmmTypingError, MissingCharmmParameters) as e:
            # a config or force-field problem the message fully explains
            click.echo(f"CHARMM: {e}", err=True)
            sys.exit(1)
        except (UntypedAtomError, ChargeError) as e:
            # the same for an atom DREIDING has no type for and a
            # molecule the chemistry stage could not charge
            click.echo(f"DREIDING: {e}", err=True)
            sys.exit(1)

        click.echo(f"Pipeline complete. Output written to: {config.study.output_dir}")
        if _graph_seed(config)[0] is not None:
            built[int(_graph_seed(config)[0])] = pipeline.graph

    if verify_ref:
        _verify_build(config, raw_cfg, verify_ref, verify_seeds, built,
                      relaxed, junction_type, verify_replicates)


def _graph_seed(config):
    """The seed that pins the graph, and the key it is set by.

    ``topology.crosslinking.seed`` for a crosslinked melt,
    ``topology.generator.seed`` otherwise.
    """
    if config.topology.source == "crosslink":
        return config.topology.crosslinking.seed, "topology.crosslinking.seed"
    return config.topology.generator.seed, "topology.generator.seed"


def _verify_build(config, raw_cfg, reference, n_seeds, built, relaxed,
                  junction_type, replicates=()):
    """The --verify half of `topon generate`.

    The config is taken as the pipeline took it, validated, with its legacy
    keys renamed and its defaults filled, not re-read from the file.
    """
    from topon.inverse.verify import format_verify, verify, write_verify

    cfg = {**(raw_cfg or {}), **config.model_dump(mode="json")}
    seed0, key = _graph_seed(config)
    if seed0 is None:
        click.echo(f"  [note] {key} is not set, so the build "
                   "is unpinned and --verify regenerates seeds 1 and up "
                   "instead of the graph just built.")
        seed0 = 1
    seeds = [int(seed0) + i for i in range(max(1, int(n_seeds)))]
    click.echo(f"Verifying against {reference} on seed(s) "
               f"{', '.join(str(s) for s in seeds)}...")
    try:
        report = verify(cfg, reference, seeds=seeds, built=built,
                        relaxed=relaxed, junction_type=junction_type,
                        z1_config=config.analysis.z1plus,
                        log=click.echo, replicates=list(replicates) or None)
    except Exception as e:
        # Whatever stopped it, the build (when there was one) is on disk;
        # say what failed rather than leave a traceback after it.
        click.echo(f"Error: the verification did not run: "
                   f"{type(e).__name__}: {e}", err=True)
        sys.exit(1)
    run_dir = Path(config.study.output_dir) / config.study.name
    path = write_verify(report, run_dir / "verify.json")
    click.echo(format_verify(report))
    click.echo(f"\nwrote {path}")


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
def validate(config_path: str):
    """
    Validate a configuration file without running the pipeline.

    CONFIG_PATH: Path to the JSON configuration file.
    """
    from topon.config import validate_config
    from topon.utils.errors import load_config_or_die

    config, _raw = load_config_or_die(config_path)

    errors = validate_config(config)

    if errors:
        click.echo("Configuration validation errors:", err=True)
        for error in errors:
            click.echo(f"  - {error}", err=True)
        sys.exit(1)
    else:
        click.echo("Configuration is valid!")


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--strict", is_flag=True,
    help="Treat WARNs as errors (exit 1 if any warn-or-worse fires).",
)
def doctor(config_path: str, strict: bool):
    """Lint a config for known footguns beyond what `validate` checks.

    Where `topon validate` is a Pydantic schema check, `topon doctor` runs
    a small registry of semantic rules for known issues and things we've
    watched new users trip on. Each rule prints one of:

    \b
        [ok]    ... informational
        [warn]  ... likely surprise; recommended fix included
        [error] ... will crash or silently produce wrong output

    Exit code: 0 if no errors (and no warns when --strict), 1 otherwise.
    """
    from topon.diagnostics import run_all_rules
    from topon.utils.errors import load_config_or_die

    cfg, raw = load_config_or_die(config_path)

    issues = run_all_rules(cfg, raw)
    if not issues:
        click.echo("No issues found.")
        return

    n_err = sum(1 for i in issues if i.level == "error")
    n_warn = sum(1 for i in issues if i.level == "warn")
    n_ok = sum(1 for i in issues if i.level == "ok")

    icon = {"ok": "[ok]   ", "warn": "[warn] ", "error": "[error]"}
    for issue in issues:
        click.echo(f"{icon[issue.level]} {issue.rule}: {issue.message}")
        if issue.fix:
            click.echo(f"        fix: {issue.fix}")

    click.echo()
    click.echo(f"Summary: {n_err} error / {n_warn} warn / {n_ok} ok")
    if n_err or (strict and n_warn):
        sys.exit(1)


def _z1_config(config_path, z1_exe, z1_distro):
    """``analysis.z1plus`` from a config file, with the flags on top."""
    from topon.config.schema import Z1PlusConfig

    base = {}
    if config_path:
        from topon.utils.errors import load_config_or_die
        cfg, _raw = load_config_or_die(config_path)
        base = cfg.analysis.z1plus.model_dump()
    if z1_exe:
        base["executable"] = z1_exe
    if z1_distro:
        base["wsl_distro"] = z1_distro
    return Z1PlusConfig(**base)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text",
              help="Print the report as text (default) or as JSON on stdout")
@click.option("--nodes", type=click.Path(exists=True), default=None,
              help="Companion .nodes file (when PATH is a .edges file)")
@click.option("--json", "json_out", type=click.Path(), default=None,
              help="Also write the report to this JSON file, with the "
                   "distributions beside it as .npz")
@click.option("--compare", "compare_to", type=click.Path(exists=True), default=None,
              help="Reference to compare against: a graph, an end-linked data "
                   "file, or a JSON report written by --json")
@click.option("--z1", is_flag=True,
              help="Run Z1+ on PATH (an end-linked LAMMPS data file)")
@click.option("--z1-exe", default=None,
              help="Z1+ binary (overrides analysis.z1plus.executable)")
@click.option("--z1-distro", default=None,
              help="WSL distribution Z1+ is installed in")
@click.option("--config", "config_path", type=click.Path(exists=True), default=None,
              help="topon config to read analysis.z1plus from")
@click.option("--fast", is_flag=True,
              help="Skip the cycle spectrum, effective resistance and edge "
                   "betweenness (the slow part on a large network)")
@click.option("--seed", type=int, default=0, show_default=True,
              help="Seed of the sampled betweenness and path lengths")
@click.option("--strands", "strands", type=click.Path(exists=True), default=None,
              help="Run manifest (or run directory) holding the strand record "
                   "of an atomistic data file; found beside the file when omitted")
def analyze(path: str, format: str, nodes: str, json_out: str, compare_to: str,
            z1: bool, z1_exe: str, z1_distro: str, config_path: str,
            fast: bool, seed: int, strands: str):
    """Connectivity descriptors of a network, Z1+, and a comparison.

    PATH is a strand graph (.gpickle, .nodes with its .edges, .edges with
    --nodes, .graphml), a LAMMPS data file in the end-linked convention
    (type 1 chain end, 2 interior, 3 junction; one molecule per chain), which
    is read into its strand graph, or an atomistic data file of a topon run,
    read through the strand record in the run's manifest.json (one Z1+ point
    per repeat unit). Every descriptor is defined in docs/USAGE.md section 3.6.

    Examples:

        topon analyze network.gpickle

        topon analyze final.data --z1 --json final_desc.json

        topon analyze network.gpickle --compare reference.data

        topon analyze run/03_Conformation/system_relaxed.data --z1
    """
    from topon.analysis.analyze import (
        analyze_network, format_report, write_report)
    from topon.analysis.endlinked import NotEndLinked

    try:
        z1_cfg = _z1_config(config_path, z1_exe, z1_distro) if z1 else None
        report, dists = analyze_network(
            path, nodes=nodes, heavy=not fast, seed=seed, z1=z1,
            z1_config=z1_cfg, compare_to=compare_to, strands=strands)
    except (NotEndLinked, FileNotFoundError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if json_out:
        j, npz = write_report(report, dists, json_out)
        click.echo(f"wrote {j} and {npz.name}", err=(format == "json"))
    if format == "json":
        from topon.analysis.descriptors import to_jsonable
        click.echo(json.dumps(report, indent=1, default=to_jsonable))
    else:
        click.echo(format_report(report))


def _floats(text):
    return [float(x) for x in text.split(",") if x.strip()] if text else None


@main.command(name="fit")
@click.argument("reference", type=click.Path(exists=True))
@click.option("--out", "-o", type=click.Path(), default=None,
              help="Config to write (default <reference stem>_config.json in "
                   "the current directory); the report goes beside it as "
                   "<stem>.fit.json")
@click.option("--junction-type", type=int, default=None,
              help="Junction atom type of a data file not typed 1 end / "
                   "2 interior / 3 junction; chains are then read from the bonds")
@click.option("--nodes", type=click.Path(exists=True), default=None,
              help="Companion .nodes file (when REFERENCE is a .edges file)")
@click.option("--lattice", type=click.Choice(["SC", "BCC", "FCC", "Diamond", "MIX"]),
              default="SC", show_default=True, help="Lattice of the fitted cell")
@click.option("--mix", default=None,
              help="MIX fractions SC,BCC,FCC, e.g. 0.9,0.05,0.05 (with --lattice MIX)")
@click.option("--sweep-cutoffs", default=None,
              help="Cutoffs to sweep, comma-separated, in place of the ones the "
                   "rule of thumb gives, e.g. 1.74,2.01,2.24")
@click.option("--seeds", type=int, default=2, show_default=True,
              help="Builds per candidate cutoff")
@click.option("--seed", type=int, default=1, show_default=True,
              help="First sweep seed, and the seed the config is pinned to")
@click.option("--max-functionality", type=int, default=None,
              help="Junction ceiling (default: the reference's highest degree)")
@click.option("--dp", type=float, default=None,
              help="Strand DP, when the reference does not carry it")
@click.option("--density", type=float, default=None,
              help="Bead density, when the reference does not carry it")
@click.option("--no-z1", is_flag=True,
              help="Do not run Z1+ on the reference (no entanglement target)")
@click.option("--z1-exe", default=None, help="Z1+ binary")
@click.option("--z1-distro", default=None, help="WSL distribution Z1+ is installed in")
@click.option("--name", default=None, help="study.name of the config")
@click.option("--no-control", is_flag=True,
              help="Skip the nearest-neighbour control row of the sweep")
@click.option("--crosslinked", is_flag=True,
              help="Read REFERENCE as crosslinked along its chains (two chain "
                   "beads bonded); otherwise decided from the file")
@click.option("--route", type=click.Choice(["crosslink", "lattice"]), default=None,
              help="Crosslinked reference: the generator the config is for, the "
                   "crosslink generator (default) or the lattice route "
                   "(architecture random_crosslinked)")
@click.option("--crosslink-bond-type", "crosslink_bond_types", type=int,
              multiple=True,
              help="Crosslinked data file: a bond type of its crosslinks "
                   "(repeat it), so a chain with a crosslink within itself is "
                   "walked")
@click.option("--sequence", default=None,
              help="Crosslinked reference: one-letter sequence of one repeat, "
                   "one bead per residue, its crosslink residues reactive")
@click.option("--repeats", type=int, default=1, show_default=True,
              help="With --sequence: repeats per chain")
@click.option("--crosslink-residue", default="Y", show_default=True,
              help="With --sequence: the residue that crosslinks")
@click.option("--reactive-every", type=int, default=None,
              help="Crosslinked reference: every so many beads reactive, in "
                   "place of the period read off the reference")
@click.option("--reactive-start", type=int, default=None,
              help="With --reactive-every: the first reactive bead")
@click.option("--packing", type=float, default=None,
              help="Crosslinked reference: lattice packing, in place of the "
                   "one read or swept")
@click.option("--contact-radius", type=float, default=None,
              help="Crosslinked reference: contact radius, in place of the one "
                   "the gaps within chains point at")
@click.option("--quiet", "-q", is_flag=True, help="Print the summary only")
def fit_cmd(reference, out, junction_type, nodes, lattice, mix, sweep_cutoffs,
            seeds, seed, max_functionality, dp, density, no_z1, z1_exe,
            z1_distro, name, no_control, crosslinked, route,
            crosslink_bond_types, sequence, repeats, crosslink_residue,
            reactive_every, reactive_start, packing, contact_radius, quiet):
    """Measure a network and write a config that regenerates it.

    REFERENCE is a LAMMPS data file of an end-linked network, an NPZ dual
    graph or a strand graph. The config holds the cubic cell that fits its
    sites, the neighbour cutoff a short sweep chose (candidates from the
    rule cutoff ~ p95 junction separation / site spacing), the exact P(f),
    the loops and sol with the degrees of the junctions they sit on, DP,
    density, and the entanglement target from the reference's Z1+.
    Everything that cannot be matched is flagged. Check the result with
    `topon generate CONFIG --verify REFERENCE`.

    A reference crosslinked along its chains (a vulcanised melt, a protein
    network crosslinked at fixed residues) is fitted to the crosslink
    generator (topology.source "crosslink"): its chains, the reactive beads
    read off where the crosslinks sit, the crosslink count, the contact rule
    and the packing. --route lattice writes the lattice route (architecture
    random_crosslinked) instead.

    Examples:

        topon fit reference.data --out reference_config.json

        topon fit ref.data --junction-type 3 --sweep-cutoffs 1.74,2.01,2.24 --seeds 2

        topon fit vulcanised.data --out vulc_config.json

        topon fit resilin.gpickle --sequence GGRPSDSYGAPGGGN --repeats 12
    """
    from topon.inverse.fit import FitError, fit, format_fit, write_fit

    mix_d = None
    if bool(mix) != (lattice == "MIX"):
        click.echo("Error: --mix and --lattice MIX go together", err=True)
        sys.exit(2)
    try:
        if mix:
            f = _floats(mix)
            if len(f) != 3:
                raise ValueError
            mix_d = {"SC": f[0], "BCC": f[1], "FCC": f[2]}
        cutoffs = _floats(sweep_cutoffs)
    except ValueError:
        click.echo("Error: --mix takes three fractions SC,BCC,FCC and "
                   "--sweep-cutoffs a comma-separated list of numbers", err=True)
        sys.exit(2)
    z1_cfg = None if no_z1 else _z1_config(None, z1_exe, z1_distro)
    try:
        result = fit(reference, junction_type=junction_type, nodes=nodes,
                     lattice=lattice, mix=mix_d,
                     cutoffs=cutoffs, seeds=seeds, seed=seed,
                     max_functionality=max_functionality, dp=dp,
                     density=density, z1=False if no_z1 else None,
                     z1_config=z1_cfg, name=name, control=not no_control,
                     log=None if quiet else click.echo,
                     crosslinked=True if crosslinked else None, route=route,
                     crosslink_bond_types=list(crosslink_bond_types) or None,
                     sequence=sequence, repeats=repeats,
                     crosslink_residue=crosslink_residue,
                     reactive_every=reactive_every,
                     reactive_start=reactive_start, packing=packing,
                     contact_radius=contact_radius)
    except FitError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    out = out or f"{Path(reference).stem}_config.json"
    cfg_path, rep_path = write_fit(result, out)
    click.echo(format_fit(result))
    click.echo(f"\nwrote {cfg_path} and {rep_path.name}")


# Preset name -> the demo config it copies. The copies ship inside the
# package (topon/presets/<name>.json), so `topon init --preset` also works
# from a regular (non-editable) install.
_PRESET_MAP = {
    "atomistic_pdms": "demos/polymer/atomistic/basic/config.json",
    "cg_kg":         "demos/polymer/coarse_grained/basic/config.json",
    "poss":          "demos/poss/config.json",
}


def _resolve_preset_path(preset: str) -> Path:
    """The bundled copy of a preset's config.json."""
    path = Path(__file__).resolve().parent / "presets" / f"{preset}.json"
    if not path.exists():
        raise FileNotFoundError(f"Could not find bundled preset '{preset}' ({path}).")
    return path


def _interactive_config() -> dict:
    """Prompt the user for the 5-6 knobs that actually vary; return a raw dict.

    The dict is in the on-disk JSON shape (not a Pydantic object); demos already
    use this shape so the result is a drop-in replacement for any
    demos/.../config.json.
    """
    click.echo()
    click.echo("topon init — interactive config builder")
    click.echo("Press Enter to accept the default in [brackets].")
    click.echo()

    name = click.prompt("Study name", default="my_run").strip()
    out_dir = click.prompt("Output directory", default=f"output_{name}").strip()

    model = click.prompt(
        "Chemistry model",
        type=click.Choice(["atomistic", "coarse_grained"]),
        default="atomistic",
    )
    # MIX is deliberately absent: it needs sublattice fractions, which is
    # more than this prompt-driven starter should ask for. Set
    # `lattice_type: "MIX"` and `mix_fractions` in the JSON instead.
    lattice_type = click.prompt(
        "Lattice type",
        type=click.Choice(["SC", "BCC", "FCC", "Diamond"]),
        default="SC",
    )
    lattice_size = click.prompt("Lattice size (NxNxN)", default="5x5x5").strip()
    max_func = click.prompt("Max functionality", type=int, default=4)
    dp_mean = click.prompt("Degree of polymerization (mean)", type=int, default=10)
    density = click.prompt(
        "Target density",
        type=float,
        default=(1.1 if model == "atomistic" else 0.85),
    )

    return {
        "study": {"name": name, "output_dir": out_dir},
        "topology": {
            "source": "generate",
            "generator": {
                "lattice_size": lattice_size,
                "lattice_type": lattice_type,
                "max_functionality": max_func,
                "degree_distribution": "0:0,1:25",
            },
        },
        "chemistry": {
            "model_type": model,
            "target_density": density,
        },
        "assignment": {
            "dp_distribution": {"default": {"mean": float(dp_mean), "pdi": 1.0}},
        },
        "conformation": {"overlap_cutoff": 0.2, "overlap_max_iters": 20},
        "simulation": {"run_steps": 2000},
        "execution": {"auto_run": False, "executable": "lmp", "n_procs": 1},
    }


@main.command()
@click.option("--output", "-o", type=click.Path(), default="config.json",
              show_default=True, help="Path for the new config file")
@click.option(
    "--preset",
    type=click.Choice(list(_PRESET_MAP.keys())),
    default=None,
    help="Start from a bundled demo (atomistic_pdms / cg_kg / poss). "
         "Without --preset and without "
         "--interactive, writes a working atomistic_pdms-style starter.",
)
@click.option(
    "--interactive", "-i",
    is_flag=True,
    help="Prompt for the 5-6 knobs that actually vary and write the result.",
)
def init(output: str, preset: str, interactive: bool):
    """Create a starter config.json that runs as-is.

    Three modes:

      \b
      topon init                    fast path - copy the atomistic_pdms preset
      topon init --preset cg_kg     start from a different bundled demo
      topon init --interactive      prompt-driven, walks through 5-6 knobs
    """
    import json as _json
    import shutil

    out_path = Path(output)
    if out_path.exists():
        if not click.confirm(f"{output} already exists. Overwrite?", default=False):
            click.echo("Aborted; no file written.")
            sys.exit(1)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if interactive:
        data = _interactive_config()
        out_path.write_text(_json.dumps(data, indent=2), encoding="utf-8")
        click.echo()
        click.echo(f"Wrote {out_path}")
    else:
        chosen = preset or "atomistic_pdms"
        src = _resolve_preset_path(chosen)
        shutil.copy2(src, out_path)
        click.echo(f"Wrote {out_path} (preset: {chosen}, copied from {src.name})")

    click.echo()
    click.echo("Next steps:")
    click.echo(f"  topon doctor {out_path}        # lint for known footguns")
    click.echo(f"  topon validate {out_path}      # schema check")
    click.echo(f"  topon generate {out_path}      # run the 6-stage pipeline")


@main.command()
@click.option("--output", "-o", type=click.Path(), default="simbox_output",
              show_default=True, help="Output directory for LAMMPS files")
@click.option("--n-epoxy", type=int, default=50, show_default=True,
              help="Number of Epoxy-PDMS molecules")
@click.option("--n-amino", type=int, default=25, show_default=True,
              help="Number of Amino-PDMS molecules")
@click.option("--n-poss", type=int, default=10, show_default=True,
              help="Number of AM0270-POSS molecules")
@click.option("--density", type=float, default=0.85, show_default=True,
              help="Target packing density (g/cm³)")
@click.option("--seed", type=int, default=42, show_default=True,
              help="Random seed for reproducible packing")
def simbox(output: str, n_epoxy: int, n_amino: int, n_poss: int,
           density: float, seed: int):
    """
    Pack a crosslink simulation box and write LAMMPS input files.

    Builds Epoxy-PDMS, Amino-PDMS, and AM0270-POSS molecules, packs them
    into a periodic box at the target density, and writes DREIDING-
    parameterised LAMMPS data + input scripts ready to run.

    Example:

        topon simbox --output my_system --n-epoxy 600 --n-amino 300

    Then run LAMMPS:

        cd my_system && lmp -in 1_minimize.in
    """
    from topon.simbox.workflow import run_workflow

    try:
        run_workflow(
            output_dir=output,
            n_epoxy=n_epoxy,
            n_amino=n_amino,
            n_poss=n_poss,
            density=density,
            seed=seed,
            verbose=True,
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option("--output", "-o", type=click.Path(), default="chain_output",
              show_default=True, help="Output directory for LAMMPS files")
@click.option("--chain-smiles", required=True,
              help="SMILES for the polymer repeat unit (e.g. \"[Si](C)(C)O\" for PDMS)")
@click.option("--dp", type=int, required=True,
              help="Degree of polymerization (number of repeat units)")
@click.option("--solvent-smiles", default=None,
              help="SMILES for single solvent (default: toluene). Ignored if --solvent-mixture is set.")
@click.option("--n-solvent", type=int, default=None,
              help="Number of solvent molecules (auto if omitted)")
@click.option("--solvent-mixture", default=None,
              help='Multi-solvent JSON: \'{"smiles":"...","weight_fraction":0.5}\'')
@click.option("--graft-density", type=float, default=0.0, show_default=True,
              help="Graft density: probability of side-chain attachment per backbone unit (0–1)")
@click.option("--graft-smiles", default=None,
              help="SMILES for graft repeat unit (required if --graft-density > 0)")
@click.option("--graft-dp", type=int, default=5, show_default=True,
              help="Number of repeat units per side chain")
@click.option("--density", type=float, default=0.85, show_default=True,
              help="Target packing density (g/cm³)")
@click.option("--seed", type=int, default=42, show_default=True,
              help="Random seed")
def chain(
    output, chain_smiles, dp, solvent_smiles, n_solvent,
    solvent_mixture, graft_density, graft_smiles, graft_dp, density, seed,
):
    """
    Build a single polymer chain in solvent and write DREIDING LAMMPS files.

    The chain is built as a linear atomistic polymer from the given repeat
    unit SMILES and packed with the specified solvent in a periodic box.
    Optional side-chain grafts are supported.

    Examples:

        # PDMS chain in toluene
        topon chain --chain-smiles "[Si](C)(C)O" --dp 20 \\
                    --solvent-smiles "Cc1ccccc1" --n-solvent 200

        # Grafted chain
        topon chain --chain-smiles "[Si](C)(C)O" --dp 30 \\
                    --graft-density 0.1 --graft-smiles "[Si](C)(C)O" --graft-dp 5 \\
                    --solvent-smiles "Cc1ccccc1" --n-solvent 150

    Then run LAMMPS:

        cd chain_output && lmp -in 1_minimize.in
    """
    from topon.singlechain.workflow import run_workflow

    if graft_density > 0 and not graft_smiles:
        click.echo("Error: --graft-smiles is required when --graft-density > 0", err=True)
        sys.exit(1)

    # Parse --solvent-mixture JSON if provided
    parsed_mixture = None
    if solvent_mixture:
        try:
            parsed_mixture = json.loads(solvent_mixture)
            if isinstance(parsed_mixture, dict):
                parsed_mixture = [parsed_mixture]  # wrap single entry
        except json.JSONDecodeError as e:
            click.echo(f"Error parsing --solvent-mixture JSON: {e}", err=True)
            sys.exit(1)

    try:
        result = run_workflow(
            output_dir=output,
            chain_smiles=chain_smiles,
            dp=dp,
            solvent_smiles=solvent_smiles,
            n_solvent=n_solvent,
            solvent_mixture=parsed_mixture,
            graft_density=graft_density,
            graft_smiles=graft_smiles,
            graft_dp=graft_dp,
            density=density,
            seed=seed,
            verbose=True,
        )
        click.echo(f"Chain atoms  : {result['chain_atoms']}")
        click.echo(f"Box length   : {result['box_length_ang']:.2f} Å")
        click.echo(f"Data file    : {result.get('data', result.get('data_file', ''))}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--z1/--no-z1", default=True, show_default=True,
              help="Measure Z1+ on the most relaxed end-linked data file, "
                   "when Z1+ is installed")
@click.option("--z1-exe", default=None,
              help="Z1+ binary (overrides analysis.z1plus.executable)")
@click.option("--config", "config_path", type=click.Path(exists=True), default=None,
              help="topon config to read analysis.z1plus from")
def inspect(run_dir: str, z1: bool, z1_exe: str, config_path: str):
    """Summarise what's inside a `topon generate` output directory.

    RUN_DIR: path either to the study folder (containing 02_Chemistry/,
    03_Conformation/, 04_Simulation/) or to its parent. Prints atom counts,
    box dimensions, which displacement files landed, the topology and
    defects requested against achieved, and the next LAMMPS commands to run.
    When the most relaxed data file is in the end-linked convention it also
    reads the strands, loops and P(f) off it, and Z1+ when installed.
    """
    from topon.analysis.run_summary import summarise, format_summary

    z1_cfg = _z1_config(config_path, z1_exe, None) if z1 else None
    summary = summarise(Path(run_dir), network=True, z1=z1, z1_config=z1_cfg)
    click.echo(format_summary(summary))


def _study_dir(path: Path) -> Path:
    """The study folder (with manifest.json) at or directly under ``path``."""
    if (path / "manifest.json").exists():
        return path
    found = [p for p in sorted(path.iterdir()) if (p / "manifest.json").exists()]
    if len(found) == 1:
        return found[0]
    raise click.BadParameter(f"{path} is not a topon study folder (no manifest.json)")


@main.command()
@click.argument("run_dirs", nargs=-1, required=True,
                type=click.Path(exists=True, file_okay=False))
@click.option("-o", "--output", default="relaxation_tracker.html", show_default=True,
              help="The page to write")
@click.option("--seeds", type=int, default=8, show_default=True,
              help="Z1+ seeds per checkpoint (0 leaves Z1+ out)")
@click.option("--energies/--no-energies", default=True, show_default=True,
              help="Evaluate every checkpoint under the full force field (one "
                   "zero-step LAMMPS run each)")
@click.option("--lmp", default="lmp", show_default=True,
              help="LAMMPS executable for the energies")
@click.option("--omp", type=int, default=1, show_default=True,
              help="OpenMP threads for those runs")
@click.option("--title", default="topon Relaxation Tracker", show_default=True,
              help="The page's title")
@click.option("--label", "labels", multiple=True,
              help="A label per run, in the order given (default: the folder name)")
@click.option("--z1-exe", default=None,
              help="Z1+ binary (overrides analysis.z1plus.executable)")
@click.option("--config", "config_path", type=click.Path(exists=True), default=None,
              help="topon config to read analysis.z1plus from")
def track(run_dirs, output, seeds, energies, lmp, omp, title, labels, z1_exe,
          config_path):
    """A tracker page for atomistic relaxation runs.

    RUN_DIRS: one or more study folders of the atomistic route (each with
    manifest.json, 03_Conformation/ and 04_Simulation/), or their parents.
    Writes one self-contained HTML page: per run and per checkpoint (the
    build, stage 1, the ramp, and the minimised, NVT and NPT states of stage
    3), the network drawn as curves with its crosslinks and Z1+ kinks, and
    through the stages Z per bridge over several Z1+ seeds, the build's
    robust partner pairs still seen, the backbone passages from the stage
    dumps, the energy density under the full force field, temperature,
    density and the longest backbone bond. A run that stopped early shows
    the checkpoints it wrote.

    Examples:

        topon track output/pdms_dp30

        topon track runs/z1 runs/z2 --label "Z = 1" --label "Z = 2" -o z.html

        topon track run --seeds 4 --omp 4
    """
    from topon.analysis.tracker import track_run, write_tracker
    from topon.analysis.z1plus import z1plus_available

    z1_cfg = _z1_config(config_path, z1_exe, None) if seeds > 0 else None
    if seeds > 0 and not z1plus_available(z1_cfg):
        click.echo("Z1+ is not available: Z and the kinks are left out", err=True)
    runs = []
    for i, d in enumerate(run_dirs):
        study = _study_dir(Path(d))
        label = labels[i] if i < len(labels) else None
        click.echo(f"reading {study}" + (" (with energies)" if energies else ""), err=True)
        try:
            runs.append(track_run(study, label=label, seeds=seeds, z1_config=z1_cfg,
                                  energies=energies, lmp=lmp, omp=omp))
        except (FileNotFoundError, KeyError, ValueError) as e:
            click.echo(f"Error in {study}: {e}", err=True)
            sys.exit(1)
    path = write_tracker(runs, output, title=title)
    click.echo(f"wrote {path} ({len(runs)} run{'s' if len(runs) != 1 else ''})")


@main.command()
def shell():
    """Drop into the interactive `topon>` shell explicitly.

    Same as running `python -m topon` on a real terminal; use this
    when you want to force the REPL even if stdin isn't a TTY (e.g.
    inside a script or from an IDE terminal that confuses isatty).
    """
    from topon.shell import run_shell
    banner = _BANNER.format(version=__version__)
    intro = banner + "\n   Type `help` for the command list, or `exit` to leave.\n"
    sys.exit(run_shell(main, intro=intro))


@main.command(
    name="protein",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],  # let the argparse CLI print its --help
    },
)
@click.argument("protein_args", nargs=-1, type=click.UNPROCESSED)
def protein(protein_args):
    """Protein network from a sequence, CHARMM36m or Martini 3.

    \b
        topon protein --sequence GGRPSDSYGAPGGGN --repeats 18 --chains 8 \\
                      --model martini --output runs/resilin_martini
        topon protein --sequence GGRPSDSYGAPGGGN --repeats 12 --chains 8 \\
                      --model charmm --water-content 35 --output runs/resilin_charmm
        topon protein --config protein.json

    Same as `python -m topon.protein_network build`; run
    `topon protein --help` for every flag.
    """
    from topon.protein_network.cli import main as protein_main
    sys.exit(protein_main(["build", *protein_args]))


@main.command()
def recipes():
    """Print a 'I want X -> run Y' table for the most common use cases.

    Quick orientation for new users: which subcommand or script handles
    which kind of network. Add rows to the list below.
    """
    rows = [
        ("Polymer network from a JSON config",
         "topon init && topon doctor config.json && topon generate config.json"),
        ("Atomistic PDMS network",
         "topon init --preset atomistic_pdms"),
        ("Coarse-grained Kremer-Grest network",
         "topon init --preset cg_kg"),
        ("POSS chain-cap demo",
         "topon init --preset poss"),
        ("Protein network from a sequence (Martini 3)",
         "topon protein --sequence GGRPSDSYGAPGGGN --repeats 18 --chains 8 "
         "--model martini --output runs/resilin_martini"),
        ("Protein network from a sequence (CHARMM36m)",
         "topon protein --sequence GGRPSDSYGAPGGGN --repeats 12 --chains 8 "
         "--model charmm --output runs/resilin_charmm"),
        ("Crosslink simulation box (no graph)",
         "topon simbox --n-epoxy 50 --n-amino 25 --n-poss 10"),
        ("Single polymer chain in solvent",
         "topon chain --chain-smiles \"[Si](C)(C)O\" --dp 50"),
        ("Batch of 25 lattice graphs + CSV summary",
         "python demos/workflows/batch_polymer_topology/run.py"),
        ("Inspect a finished run directory",
         "topon inspect <run_dir>"),
        ("Graph statistics (no chemistry build)",
         "topon analyze graph.gpickle"),
        ("A config that regenerates an existing network, and its check",
         "topon fit ref.data --out ref_config.json\n"
         "topon generate ref_config.json --verify ref.data --verify-seeds 3"),
        ("The same for a network crosslinked along its chains",
         "topon fit ref.gpickle --out ref_config.json\n"
         "topon generate ref_config.json --verify ref.gpickle --verify-only "
         "--verify-seeds 8 --verify-replicate rep2.gpickle ... "
         "--verify-replicate rep9.gpickle   (4 or more a side)"),
    ]
    click.echo()
    click.echo("topon recipes — common use cases")
    click.echo("=" * 70)
    for goal, cmd in rows:
        click.echo()
        click.echo(f"  {goal}")
        for line in cmd.split("\n"):
            click.echo(f"    {line}")
    click.echo()
    click.echo("More: docs/USAGE.md, demos/, demos/workflows/")


if __name__ == "__main__":
    main()
