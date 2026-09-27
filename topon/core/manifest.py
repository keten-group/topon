"""The run manifest: what a pipeline run asked for and what it got.

One JSON file per run directory, ``manifest.json``, written stage by
stage as the pipeline goes. It is the machine-readable half of
``topon inspect``: the stage outputs say what landed on disk, the
manifest says what was requested, what came back and how long it took.

The first section is stage 1's. An exact sculpt records the degree counts
requested against the degree counts achieved, the seconds, the seeds, the
augmentation and repair work and the giant-component fraction, which is
the precision / convergence / success-rate evidence the topology claims
rest on. A strict sculpt records the trial it succeeded on and the same
before-and-after counts. Later stages add their own sections under
``stages``; nothing here is specific to topology except the helper that
formats a sculpt record.

The file is advisory. Nothing in the pipeline reads it back to make a
decision, so a missing or stale manifest never changes what a run does --
it only leaves ``topon inspect`` with less to say.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

#: Bumped when the layout changes in a way a reader has to know about.
MANIFEST_VERSION = 1

#: File name inside the run directory.
MANIFEST_NAME = "manifest.json"


def manifest_path(run_dir) -> Path:
    """Where the manifest of ``run_dir`` lives."""
    return Path(run_dir) / MANIFEST_NAME


def read_manifest(run_dir) -> Optional[dict]:
    """The manifest of ``run_dir``, or ``None`` if there is none to read.

    A manifest that is missing, unreadable or not valid JSON reads as
    ``None`` rather than raising: it is a record of a run, not part of
    one, and a half-written file from an interrupted run should not stop
    an inspection of everything else.
    """
    path = manifest_path(run_dir)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_manifest(run_dir, manifest: dict) -> Path:
    """Write ``manifest`` into ``run_dir``, replacing what was there."""
    path = manifest_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=_plain)
        fh.write("\n")
    return path


def record_stage(run_dir, stage: str, entry: dict, study: Optional[str] = None) -> Path:
    """Merge one stage's entry into the manifest of ``run_dir``.

    Creates the manifest on the first call of a run and updates the named
    stage on later ones, so each stage writes its own section and a run
    that stops early still leaves the sections it finished.
    """
    manifest = read_manifest(run_dir) or {
        "manifest_version": MANIFEST_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stages": {},
    }
    if study is not None:
        manifest["study"] = study
    manifest.setdefault("stages", {})[stage] = entry
    manifest["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return write_manifest(run_dir, manifest)


def _plain(value: Any):
    """Make NumPy scalars and sets JSON-serialisable.

    Sculpt records carry NumPy integers (node ids on a lattice the search
    indexed with NumPy) and a sorted list of tuples; ``json`` handles
    neither on its own.
    """
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)
