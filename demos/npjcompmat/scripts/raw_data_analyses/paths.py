"""Folders used by the raw-data analysis scripts. Set them with two environment variables.

NPJ_RAW_ROOT (RAW_ROOT): raw simulation output. The default is data/raw/ of this companion. Unpacked there, the
deposited tarballs give the layout the scripts read:
    seed<k>/<folder_name>/stress_strain_npt_{x,y,z}.dat    from cg_stress_strain_seed<k>.tar.xz (k = 1 to 4)
    <PDMS|PMTFPS>/DP<n>/<history>/msd.dat                  from tg_cooling.tar.xz
Inputs that are not deposited are read from the original simulation trees below the same folder:
    crosslinker/sc_6x6x6/<folder_name>/                     LAMMPS data files of every stage, slurm-*.out, log.lammps
    bond_create_validation/                                 bond/create reference and Topon data files, scripts/refnet.py

NPJ_ANALYSIS_OUT (ANALYSIS_OUT): intermediate and output files. The default is output/ next to this file. Each script
writes into the subfolder named after its own folder (cg_mechanics/, cg_structure/, stress_definition/, tg/), and
reach_distributions.py into bond_create/. The subfolders are created on demand.

The deposited companion data (data/dataset.pkl, data/csv/, data/mechanics/, data/derived/) are found relative to this
file.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPANION = HERE.parents[1]                   # demos/npjcompmat
DATA = COMPANION / "data"
RAW_ROOT = Path(os.environ.get("NPJ_RAW_ROOT", DATA / "raw"))
ANALYSIS_OUT = Path(os.environ.get("NPJ_ANALYSIS_OUT", HERE / "output"))
CG = RAW_ROOT / "crosslinker"                 # original CG simulation tree (not deposited)

# names used in tg_cooling.tar.xz for the original folder names and history tags of the Tg scripts
TG_LABEL = {"PDMS": "PDMS", "FPDMS": "PMTFPS"}
TG_HISTORY = {"orig": "original", "rep2": "replicate2", "rep3": "replicate3"}


def out_dir(name):
    """ANALYSIS_OUT/<name>, created if missing."""
    d = ANALYSIS_OUT / name
    d.mkdir(parents=True, exist_ok=True)
    return d
