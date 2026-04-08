import os
import re
import tempfile
import numpy as np
from glob import glob
from collections import defaultdict

from ase.io import read, write

from . import rdf_calc
from . import energydist_calc


_NATOMS_TO_SUPERCELL = {8 * n**3: f"{n}{n}{n}" for n in range(1, 10)}


def notexist(path):
    return not os.path.exists(path)


def normalize_system(name):
    # Already supercell notation: Si111, SiGe222, etc.
    if re.match(r'^(SiGe|Si|Ge)(111|222|333)$', name):
        return name
    # Pure Si with atom count: Si8, Si64, Si216
    m = re.match(r'^Si(\d+)$', name)
    if m:
        sc = _NATOMS_TO_SUPERCELL.get(int(m.group(1)))
        if sc:
            return f"Si{sc}"
    # SiGe with atom counts: Si4Ge4, Si32Ge32, Si108Ge108
    m = re.match(r'^Si(\d+)Ge(\d+)$', name)
    if m:
        sc = _NATOMS_TO_SUPERCELL.get(int(m.group(1)) + int(m.group(2)))
        if sc:
            return f"SiGe{sc}"
    return name


def parse_traj(folder, keep_nsteps=False):
    # Groups trajectory files by supercell notation, combining all files for the same system.
    # If keep_nsteps=True, groups by {system}_{nsteps} to keep step counts separate.
    files = glob(folder + "/*.traj")

    groups = defaultdict(list)
    for f in files:
        basename = os.path.basename(f)
        stem = os.path.splitext(basename)[0]
        parts = stem.split("_", 1)
        system = parts[0]
        category = normalize_system(system)
        if keep_nsteps and len(parts) > 1:
            category = f"{category}_{parts[1]}"
        groups[category].append(f)

    category_names = sorted(groups.keys())
    atoms_list = []
    for name in category_names:
        atoms = []
        for fn in sorted(groups[name]):
            atoms += read(fn, index=":")
        atoms_list.append(atoms)
    return atoms_list, category_names


def calc_rdf(atoms_or_files, out_file, recalc=False, n_bins=500, rcut=10):
    if not recalc and os.path.exists(out_file):
        return
    # Accept file paths directly — avoids loading all frames into memory
    if isinstance(atoms_or_files[0], (str, os.PathLike)):
        rdf_calc.partial_rdf(list(atoms_or_files), n_bins, rcut, out_file)
        return
    atoms = atoms_or_files
    if isinstance(atoms[0], list):
        atoms = [a for sublist in atoms for a in sublist]
    with tempfile.NamedTemporaryFile(suffix=".traj", delete=False) as f:
        tmp_path = f.name
    try:
        write(tmp_path, atoms)
        rdf_calc.partial_rdf([tmp_path], n_bins, rcut, out_file)
    finally:
        os.unlink(tmp_path)


def calc_edist(atoms_or_files, out_file, category=None, folder_label=None, recalc=False):
    if not recalc and os.path.exists(out_file):
        return
    # Accept file paths directly — reads one file at a time to avoid memory spike
    if isinstance(atoms_or_files[0], (str, os.PathLike)):
        energies = []
        for f in atoms_or_files:
            print(f"Processing {f}")
            atoms = read(f, index=":")
            e = energydist_calc.evaluate_energies(atoms, category=category, folder_label=folder_label)
            if e is None:
                print(f"WARNING: no energies found for {f} (folder_label={folder_label!r}, category={category!r})")
            else:
                energies.extend(e)
        if not energies:
            print(f"WARNING: no energies collected; {out_file} will be empty")
        np.savetxt(out_file, np.array(energies))
        return
    atoms = atoms_or_files
    if isinstance(atoms[0], list):
        atoms = [a for sublist in atoms for a in sublist]
    energydist_calc.edist(atoms, out_file, category=category, folder_label=folder_label)
