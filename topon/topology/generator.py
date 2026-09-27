"""
C generator wrapper for Topon.

Handles invoking the C-based topology generator and SLURM script generation.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Union

from topon.config.schema import GeneratorConfig
from topon.topology.degree_matching import (
    DEFAULT_ATTEMPTS,
    DEFAULT_MIN_GIANT_FRACTION,
    parse_degree_distribution,
    resolve_search,
)
from topon.topology.shells import DEFAULT_CUTOFF


def resolve_config_search(config: GeneratorConfig) -> str:
    """The search a generator config asks for, by the Python generator's rule.

    ``search`` when the config names one; otherwise ``"exact"`` exactly
    when ``degree_distribution`` pins every degree from 0 to
    ``max_functionality``, ``"strict"`` otherwise. The same
    :func:`~topon.topology.degree_matching.resolve_search` the Python
    generator calls, so one config means the same search on either side.

    Raises:
        ValueError: if ``"exact"`` is asked for with a request that does
            not pin every degree, or an ``e:N`` term contradicts the
            per-degree counts.
    """
    counts, edge_count = parse_degree_distribution(config.degree_distribution)
    return resolve_search(
        getattr(config, "search", None), counts,
        config.max_functionality, edge_count,
    )


def format_lattice_arg(config: GeneratorConfig) -> str:
    """Render the ``<lattice_type>`` argument the C generator expects.

    SC, BCC, FCC and Diamond pass through unchanged. MIX carries its
    sublattice fractions and cutoff inside the same argument, as
    ``MIX:<sc>,<bcc>,<fcc>,<cutoff>``, so the executable keeps the
    eight-positional-argument CLI that existing callers and SLURM scripts
    already write. The pure lattices get their cutoff from
    :func:`format_generator_args` instead.

    Args:
        config: Generator configuration.

    Returns:
        The argument string.
    """
    if config.lattice_type != "MIX":
        return config.lattice_type
    f = config.mix_fractions
    return (
        f"MIX:{f.get('SC', 0.0):g},{f.get('BCC', 0.0):g},"
        f"{f.get('FCC', 0.0):g},{config.neighbour_cutoff:g}"
    )


def format_generator_args(config: GeneratorConfig) -> list[str]:
    """The positional arguments the C generator takes, in order.

    Eight positions are the CLI every existing caller and SLURM script
    writes. A ninth, the neighbour cutoff in cell units, is appended only
    for a pure lattice at a non-default cutoff: MIX carries its cutoff
    inside the lattice argument, and at the default the binary builds the
    canonical lattice without being told, so an older build keeps working
    for every config that does not use the new range.

    The search rides as a named flag, ``--search=exact``, added only when
    the config resolves to the exact search (see
    :func:`resolve_config_search`), with ``--min-giant-fraction=F`` when
    that floor is not the default. A strict request gets no flag, which is
    the binary's default, so it runs on a build older than the flag too.

    For the exact search one trial is one attempt. A config that leaves
    ``max_trials`` at its default (a million, sized for the strict
    sculptor) gets the Python search's budget instead,
    ``DEFAULT_ATTEMPTS`` attempts per network, so an infeasible request
    gives up after seconds on either route rather than retrying for hours.
    A ``max_trials`` the config sets explicitly is passed as it is.
    """
    search = resolve_config_search(config)
    trials = config.max_trials
    if search == "exact" and "max_trials" not in getattr(config, "model_fields_set",
                                                          {"max_trials"}):
        trials = DEFAULT_ATTEMPTS * max(1, int(config.max_saves))
    args = [
        config.lattice_size,
        config.periodicity,
        str(config.max_functionality),
        str(trials),
        str(config.max_saves),
        str(config.degree_distribution),
        "0",  # extensive_logging
        format_lattice_arg(config),
    ]
    if config.lattice_type != "MIX" and config.neighbour_cutoff != DEFAULT_CUTOFF:
        args.append(f"{config.neighbour_cutoff:g}")
    if search == "exact":
        args.append("--search=exact")
        floor = getattr(config, "min_giant_fraction", None)
        if floor is not None and float(floor) != DEFAULT_MIN_GIANT_FRACTION:
            # repr is the shortest string that reads back as the same
            # double, so the binary compares against exactly this floor.
            args.append(f"--min-giant-fraction={float(floor)!r}")
    return args


def run_generator(
    config: GeneratorConfig,
    output_dir: Union[str, Path],
    exe_path: Optional[Union[str, Path]] = None,
    seed: Optional[int] = None,
) -> tuple[Path, Path]:
    """
    Run the C topology generator.

    Args:
        config: Generator configuration.
        output_dir: Directory for output files.
        exe_path: Path to generator executable (overrides config).
        seed: Handed to the binary as ``TOPON_SEED``, which fixes its
            whole random stream. ``None`` leaves the binary to seed itself
            from the clock (or from a ``TOPON_SEED`` already set).

    Returns:
        Tuple of (nodes_file_path, edges_file_path).

    The search follows the config the way the Python generator's does
    (:func:`resolve_config_search`): an exact request runs the C port of
    the exact search, where one trial is one attempt (see
    :func:`format_generator_args` for the budget).

    Raises:
        FileNotFoundError: If generator executable not found.
        ValueError: If the config asks for the exact search without
            pinning every degree.
        RuntimeError: If generator fails.
    """
    exe = Path(exe_path or config.exe_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not exe.exists():
        raise FileNotFoundError(f"Generator executable not found: {exe}")
    
    # Build command
    cmd = [str(exe), *format_generator_args(config)]
    
    print(f"Running generator: {' '.join(cmd)}")
    
    # Run generator. The child gets a copy of this process's environment,
    # plus the seed when there is one.
    env = None
    if seed is not None:
        env = dict(os.environ, TOPON_SEED=str(int(seed)))
    result = subprocess.run(
        cmd,
        cwd=output_dir,
        capture_output=True,
        text=True,
        env=env,
    )

    if result.returncode != 0:
        print(f"Generator stdout: {result.stdout}")
        print(f"Generator stderr: {result.stderr}")
        # A binary older than the flag either counts it as a tenth
        # positional (Usage) or, with no ninth argument, reads it as the
        # neighbour cutoff and refuses that.
        if "--search=exact" in cmd and ("Usage" in result.stderr
                                        or "got '--search=exact'" in result.stderr):
            raise RuntimeError(
                f"Generator failed with return code {result.returncode}: "
                f"{exe} does not take --search=exact, so it predates the C "
                f"exact search. Rebuild it from "
                f"topon/topology/csrc/generator.c."
            )
        raise RuntimeError(
            f"Generator failed with return code {result.returncode}"
            + (f": {result.stderr.strip()}" if result.stderr.strip() else "")
        )
    
    # Find output files
    nodes_files = list(output_dir.glob("output/*.nodes"))
    edges_files = list(output_dir.glob("output/*.edges"))
    
    if not nodes_files or not edges_files:
        raise RuntimeError("Generator did not produce output files")
    
    return nodes_files[0], edges_files[0]


def generate_slurm_script(
    config: GeneratorConfig,
    output_path: Union[str, Path],
    slurm_config: Optional[dict] = None,
) -> str:
    """
    Generate a SLURM batch script for running the generator on HPC.
    
    Args:
        config: Generator configuration.
        output_path: Path to write the SLURM script.
        slurm_config: SLURM-specific configuration (account, partition, etc.).
        
    Returns:
        Path to the generated script.
    """
    output_path = Path(output_path)
    
    # Default SLURM config
    slurm = slurm_config or {}
    account = slurm.get("account", "default_account")
    partition = slurm.get("partition", "short")
    nodes = slurm.get("nodes", 1)
    tasks = slurm.get("tasks_per_node", 1)
    time = slurm.get("time", "03:59:59")
    exe_path = slurm.get("generator_exe_path", "./generator.exe")
    module_loads = slurm.get("module_loads", [])
    
    # Build job name from lattice config
    job_name = f"gen-{config.lattice_size}-{config.lattice_type}"

    # Build command. The degree distribution and the lattice argument are
    # quoted (they can hold commas and colons); the rest are bare tokens.
    args = format_generator_args(config)
    cmd = " ".join(
        [f'"{exe_path}"']
        + [f'"{a}"' if i in (5, 7) else a for i, a in enumerate(args)]
    )
    
    # Build script
    script_lines = [
        "#!/bin/bash",
        f"#SBATCH -A {account}",
        f"#SBATCH -p {partition}",
        f"#SBATCH -N {nodes}",
        f"#SBATCH --ntasks-per-node={tasks}",
        f"#SBATCH -t {time}",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --export=ALL",
        "",
        "module purge all",
    ]
    
    for module in module_loads:
        script_lines.append(f"module load {module}")
    
    script_lines.extend([
        "",
        f'echo "Running generator with config: {config.lattice_size} {config.lattice_type}"',
        "",
        cmd,
        "",
        'echo "Generator complete"',
    ])
    
    script_content = "\n".join(script_lines)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(script_content)
    
    print(f"Generated SLURM script: {output_path}")
    
    return str(output_path)
