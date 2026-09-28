"""Martini 3 chain topology for any amino-acid sequence.

The protein-network builder replicates one chain topology (a
:class:`~topon.protein_network.itp_template.ChainTemplate`) per chain. This
module finds that topology for a sequence, in order:

1. a user ITP (``itp_path``), checked against the sequence;
2. a vendored polyply ITP whose sequence is exactly this one (the resilin
   reference, no polyply needed);
3. ``polyply gen_params -lib martini3`` run on the sequence, the same command
   that produced the vendored ITPs (Martini 3 with the Martini3-IDP bonded
   terms). The ITP is kept next to the output so the build can be rerun
   without polyply.

polyply is an optional dependency (``pip install polyply cgsmiles``, since
polyply 1.8 imports cgsmiles without declaring it). It runs in a
subprocess with ``PYTHONUTF8=1``, because vermouth reads its citation file
with the platform encoding and fails on Windows otherwise.

A chain topology with virtual sites is refused: LAMMPS has no virtual sites,
and Martini 3 tryptophan carries one (its SC3 bead, mass 0).
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

from .itp_template import ChainTemplate, load_chain_template, _VENDORED
from .residues import THREE_TO_ONE
from .sequence import validate_sequence


class MartiniTopologyError(RuntimeError):
    """The sequence has no usable Martini 3 chain topology."""


def template_sequence(template: ChainTemplate) -> str:
    """One-letter sequence of a chain template, in residue order."""
    seen: dict[int, str] = {}
    for a in template.atoms:
        seen.setdefault(a.resnr, a.resname)
    try:
        return "".join(THREE_TO_ONE[seen[r]] for r in sorted(seen))
    except KeyError as exc:
        raise MartiniTopologyError(f"residue {exc} in the ITP is not a standard amino acid")


def _itp_has_virtual_sites(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    return "virtual_sites" in text


def _check(template: ChainTemplate, path: Path, sequence: str) -> ChainTemplate:
    got = template_sequence(template)
    if got != sequence:
        raise MartiniTopologyError(
            f"{path.name} holds a {len(got)}-residue chain that is not the requested "
            f"{len(sequence)}-residue sequence")
    if _itp_has_virtual_sites(path):
        raise MartiniTopologyError(
            f"{path.name} has virtual sites (Martini 3 tryptophan), which LAMMPS "
            f"cannot represent; use a sequence without W or the CHARMM model")
    return template


def vendored_template_for(sequence: str) -> tuple[ChainTemplate, Path] | None:
    """The vendored ITP whose chain is exactly ``sequence``, if any."""
    from importlib import resources
    with resources.as_file(resources.files("topon.protein_network.data")) as d:
        for fname in sorted(set(_VENDORED.values())):
            p = Path(d) / fname
            if not p.exists():
                continue
            t = load_chain_template(p)
            if template_sequence(t) == sequence:
                return t, p
    return None


def polyply_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("polyply") is not None


def run_polyply(sequence: str, out_itp: Path, name: str = "chain") -> Path:
    """Write the Martini 3 ITP of ``sequence`` with polyply; return its path."""
    if not polyply_available():
        raise MartiniTopologyError(
            "no Martini 3 topology is bundled for this sequence and polyply is not "
            "installed. Install it (pip install polyply cgsmiles) or pass a chain ITP made "
            "with 'polyply gen_params -lib martini3 -seqf <fasta>'.")
    out_itp = Path(out_itp).resolve()
    out_itp.parent.mkdir(parents=True, exist_ok=True)
    fasta = out_itp.with_suffix(".fasta")
    # polyply reads the sequence type from the FASTA comment line.
    fasta.write_text(f">{name} PROTEIN\n" + "\n".join(textwrap.wrap(sequence, 60)) + "\n",
                     encoding="ascii")
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from polyply import gen_itp\n"
        "gen_itp(name=sys.argv[1], outpath=Path(sys.argv[2]), inpath=[],\n"
        "        lib=['martini3'], seq=None, seq_file=Path(sys.argv[3]))\n"
    )
    env = {**os.environ, "PYTHONUTF8": "1"}
    proc = subprocess.run([sys.executable, "-c", code, name, str(out_itp), str(fasta)],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env)
    if proc.returncode != 0 or not out_itp.exists():
        raise MartiniTopologyError(f"polyply gen_params failed:\n{proc.stderr[-3000:]}")
    # polyply records its command line, which here is the subprocess with
    # absolute paths; keep the equivalent command without them
    text = out_itp.read_text(encoding="utf-8")
    first, sep, rest = text.partition("\n")
    if first.startswith(";"):
        first = (f"; polyply gen_params -lib martini3 -seqf {fasta.name} -name {name} "
                 f"-o {out_itp.name}  (run by topon)")
        out_itp.write_text(first + sep + rest, encoding="utf-8")
    return out_itp


def chain_template_for_sequence(
    sequence: str,
    *,
    itp_path: str | Path | None = None,
    work_dir: str | Path | None = None,
    name: str = "chain",
) -> tuple[ChainTemplate, Path, str]:
    """Return ``(template, itp_path, source)`` for ``sequence``.

    ``source`` is ``"user"``, ``"vendored"`` or ``"polyply"``. With polyply
    the ITP is written to ``work_dir`` (required in that case).
    """
    seq = validate_sequence(sequence)
    if itp_path is not None:
        p = Path(itp_path)
        return _check(load_chain_template(p), p, seq), p, "user"
    hit = vendored_template_for(seq)
    if hit is not None:
        t, p = hit
        return _check(t, p, seq), p, "vendored"
    if "W" in seq:
        raise MartiniTopologyError(
            "tryptophan (W) carries a virtual site in Martini 3, which LAMMPS cannot "
            "represent; use a sequence without W or the CHARMM model")
    if work_dir is None:
        raise MartiniTopologyError("work_dir is needed to write the polyply ITP")
    p = run_polyply(seq, Path(work_dir) / f"{name}.itp", name=name)
    return _check(load_chain_template(p), p, seq), p, "polyply"
