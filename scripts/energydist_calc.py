from __future__ import annotations
import shutil
import os
from pathlib import Path
import re
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase import Atoms
from ase.io.lammpsdata import write_lammps_data
from namespace import FOLDER_COLORS, LINESTYLES, NSTEPS_LINESTYLES

def edist(atoms, name, category=None, folder_label=None, recalc=False):
    energies = evaluate_energies(atoms, category=category, folder_label=folder_label)
    if energies is None:
        return False
    np.savetxt(name, energies)
    return True


def plot_edist(folder_data, nsteps=None, plot_debug=False, energy_spread=1.5):
    """
    folder_data: list of (folder_label, category_names)
    nsteps: list of int, e.g. [250, 1000], to restrict which nsteps are plotted (None = all)

    One figure per {elems}{supercell}, with a single histogram subplot.
    Each histogram corresponds to a (folder_label, nsteps) combination.
    nbins = int(nstruct / 5) where nstruct is the number of energies in the file.
    Figures saved to figures/{elems}{supercell}_edist.pdf.
    """
    allowed = {str(n) for n in nsteps} if nsteps is not None else None
    entries = []
    for folder_label, category_names in folder_data:
        for name in category_names:
            m = re.match(r'^([A-Za-z]+\d+)(?:_(\d+))?$', name)
            if m:
                elems_supercell = m.group(1)
                nsteps_str = m.group(2)
                if allowed is not None and nsteps_str is not None and nsteps_str not in allowed:
                    continue
                display_label = f"{folder_label}_{nsteps_str}" if nsteps_str else folder_label
                entries.append((folder_label, elems_supercell, f"output/{folder_label}/{name}_edist.dat", display_label))

    groups = defaultdict(list)
    for entry in entries:
        groups[entry[1]].append(entry)

    for elems_supercell, group_entries in sorted(groups.items()):
        fig, ax = plt.subplots(figsize=(7, 5))

        sorted_entries = sorted(group_entries, key=lambda e: 0 if e[0] == "ref" else 1)
        n_atoms = _natoms(elems_supercell)

        # Determine plot range from ref (±50% of ref half-width around its mean)
        ref_entry = next((e for e in sorted_entries if e[0] == "ref"), None)
        spread = energy_spread.get(elems_supercell, 1.5) if isinstance(energy_spread, dict) else energy_spread
        if ref_entry is not None:
            ref_e = np.loadtxt(ref_entry[2]) / n_atoms
            mean_r = ref_e.mean()
            lo = mean_r - spread * (mean_r - ref_e.min())
            hi = mean_r + spread * (ref_e.max() - mean_r)
            nbins = int(len(ref_e) / 5)
        else:
            all_e = np.concatenate([np.loadtxt(e[2]) for e in sorted_entries]) / n_atoms
            lo, hi = all_e.min(), all_e.max()
            nbins = int(len(all_e) / 5)

        for folder_label, _, dat_file, display_label in sorted_entries:
            energies = np.loadtxt(dat_file) / n_atoms
            nsteps = re.search(r'_(\d+)$', display_label)
            nsteps = nsteps.group(1) if nsteps else None
            color = FOLDER_COLORS.get(folder_label, "gray")
            ls = NSTEPS_LINESTYLES.get(nsteps, "-") if nsteps else "-"
            weights = np.ones(len(energies)) / len(energies) * 100
            ax.hist(energies, bins=nbins, range=(lo, hi), weights=weights,
                    histtype='stepfilled', alpha=0.4, color=color, linestyle=ls, label=display_label)
        ax.set_xlim(lo, hi)

        ax.set_xlabel("Energy per atom (eV/atom)", fontsize=13)
        ax.set_ylabel("Percentage of configurations (%)", fontsize=13)
        ax.legend(fontsize=11)
        fig.suptitle(f"Energy distribution — {elems_supercell}", fontsize=15)
        fig.tight_layout()
        fig.savefig(f"figures/{elems_supercell}_edist.pdf")
        plt.close(fig)

    if plot_debug:
        os.makedirs("figures/debug", exist_ok=True)

        ref_lookup = {elems_supercell: dat_file
                      for folder_label, elems_supercell, dat_file, _ in entries
                      if folder_label == "ref"}

        for folder_label, category_names in folder_data:
            for name in category_names:
                m = re.match(r'^([A-Za-z0-9]+)_([A-Za-z]+\d+)(?:_(\d+))?$', name)
                #m = re.match(r'^([A-Za-z]+\d*)_([A-Za-z]+\d+)(?:_(\d+))?$', name)
                if not m:
                    continue
                prefix, elems_supercell, nsteps_str = m.group(1), m.group(2), m.group(3)
                if allowed is not None and nsteps_str is not None and nsteps_str not in allowed:
                    continue
                dat_file = f"output/{folder_label}/{name}_edist.dat"
                variant_label = f"{prefix}_{folder_label}_{nsteps_str}" if nsteps_str else f"{prefix}_{folder_label}"

                fig, ax = plt.subplots(figsize=(7, 5))
                n_atoms = _natoms(elems_supercell)

                ref_file = ref_lookup.get(elems_supercell)
                ref_e = np.loadtxt(ref_file) / n_atoms if ref_file and os.path.exists(ref_file) else None

                spread = energy_spread.get(elems_supercell, 1.5) if isinstance(energy_spread, dict) else energy_spread
                if ref_e is not None:
                    mean_r = ref_e.mean()
                    lo = mean_r - spread * (mean_r - ref_e.min())
                    hi = mean_r + spread * (ref_e.max() - mean_r)
                    nbins = int(len(ref_e) / 5)
                else:
                    finite_e = np.loadtxt(dat_file) / n_atoms
                    finite_e = finite_e[np.isfinite(finite_e)]
                    lo, hi = finite_e.min(), finite_e.max()
                    nbins = int(len(finite_e) / 5)

                if ref_e is not None:
                    weights_ref = np.ones(len(ref_e)) / len(ref_e) * 100
                    ax.hist(ref_e, bins=nbins, range=(lo, hi), weights=weights_ref,
                            histtype='stepfilled', alpha=0.4, color=FOLDER_COLORS["ref"], linestyle="-", label="ref")

                energies = np.loadtxt(dat_file) / n_atoms
                energies = energies[np.isfinite(energies)]
                color = FOLDER_COLORS.get(folder_label, "gray")
                ls = NSTEPS_LINESTYLES.get(nsteps_str, "-") if nsteps_str else "-"
                weights = np.ones(len(energies)) / len(energies) * 100
                ax.hist(energies, bins=nbins, range=(lo, hi), weights=weights,
                        histtype='stepfilled', alpha=0.4, color=color, linestyle=ls, label=variant_label)

                ax.set_xlim(lo, hi)
                ax.set_xlabel("Energy per atom (eV/atom)", fontsize=13)
                ax.set_ylabel("Percentage of configurations (%)", fontsize=13)
                ax.legend(fontsize=11)
                fig_stem = f"{prefix}_{elems_supercell}_{nsteps_str}" if nsteps_str else f"{prefix}_{elems_supercell}"
                fig.suptitle(f"Energy distribution — {fig_stem} (debug)", fontsize=15)
                fig.tight_layout()
                fig.savefig(f"figures/debug/{fig_stem}_edist.pdf")
                plt.close(fig)


ELEMENT_MASSES = {"Si": 28.085, "Ge": 72.630}


LAMMPS_REF_BASE = Path("/home/olivi/projects/02-Simodels-ZBL/lammps_ref")

_SUPERCELL_TO_NATOMS = {"111": 8, "222": 64, "333": 216}


def _natoms(elems_supercell):
    import re
    m = re.match(r'[A-Za-z]+(111|222|333)$', elems_supercell)
    return _SUPERCELL_TO_NATOMS[m.group(1)] if m else 1


def _original_prefix(category):
    """Convert supercell notation back to atom-count filename prefix.
    E.g. Si333 -> Si216, SiGe222 -> Si32Ge32."""
    import re
    m = re.match(r'^(SiGe|Si)(111|222|333)$', category)
    if not m:
        return category
    elem, sc = m.group(1), m.group(2)
    n = _SUPERCELL_TO_NATOMS[sc]
    if elem == "SiGe":
        return f"Si{n // 2}Ge{n // 2}"
    return f"Si{n}"


def _lammps_log_dir(category):
    is_sige = category is not None and category.startswith("SiGe")
    return LAMMPS_REF_BASE / ("SiGe_lammps_ref" if is_sige else "Si_lammps_ref")


def _lammps_log_files(category, folder_label):
    """Return sorted list of energies.log paths for noZBL/ZBL.
    Logs live in calc_energies/{folder_label}/{category}_{nsteps}/energies.log."""
    from glob import glob
    return sorted(glob(f"calc_energies/{folder_label}/{category}/energies.log"))


def energy_ready(category, folder_label):
    """Return True if energies are available without running anything."""
    if folder_label == "ref":
        return True
    return len(_lammps_log_files(category, folder_label)) > 0


def prepare_lammps(atoms, category=None, folder_label=None):
    """Create LAMMPS input files so the user can run LAMMPS to get energies.
    Returns True (pending) if LAMMPS still needs to be run.
    Must not be called for ref — ref energies come from the traj via get_potential_energy()."""
    if folder_label == "ref":
        raise ValueError("prepare_lammps must not be called for ref; use get_potential_energy()")
    if len(_lammps_log_files(category, folder_label)) > 0:
        return False

    is_sige = category is not None and category.startswith("SiGe")
    elements = ["Si", "Ge"] if is_sige else ["Si"]
    calc_folder = Path("calc_energies") / folder_label / category
    calc_folder.mkdir(parents=True, exist_ok=True)
    is_sige = category is not None and category.startswith("SiGe")
    potential = Path("SiGe_lammps_ref/SiGe.sw") if is_sige else Path("Si_lammps_ref/Si.sw")
    lammps_output = calc_folder / "energies.log"
    create_atoms_in(atoms, calc_folder=calc_folder, prefix=category, specorder=elements)
    create_lammps_input(len(atoms), calc_folder=calc_folder, potential=potential,
                        atoms_prefix=category, log_file=lammps_output.resolve(),
                        elements=elements)
    shutil.copy(potential, calc_folder)
    print(f"You'll need to run lammps in {calc_folder} to get energies")
    return True


def evaluate_energies(atoms, category=None, folder_label=None):
    """Read energies from ASE (ref) or existing LAMMPS logs. Returns None if logs are missing."""
    if folder_label == "ref":
        return np.array([at.get_potential_energy() for at in atoms])

    log_files = _lammps_log_files(category, folder_label)
    if not log_files:
        return None

    energies = []
    for log_file in log_files:
        energies.extend(read_lammps_energies(log_file))
    nstruct = len(atoms)
    if len(energies) != nstruct:
        raise ValueError(f"Expected {nstruct} energies, got {len(energies)} "
                         f"(from {len(log_files)} log files including {log_files[0]})")
    return np.array(energies)

def create_atoms_in(atoms, calc_folder="tmp", prefix=None, specorder=None):
    calc_folder = Path(calc_folder)
    calc_folder.mkdir(parents=True, exist_ok=True)

    if specorder is None:
        specorder = ["Si"]
    if isinstance(atoms, Atoms):
        atoms = [atoms]

    out_files = []
    for i, at in enumerate(atoms, start=1):
        out_name = calc_folder / f"{prefix}_{i}.in"
        write_lammps_data(
            out_name,
            at,
            velocities=True,
            atom_style="atomic",
            specorder=specorder,
        )
        out_files.append(out_name)
    return out_files


def create_lammps_input(
    nstruct,
    calc_folder="tmp",
    potential=Path("Si.sw"),
    atoms_prefix="atoms",
    out_name="lammps_input.in",
    log_file=Path("energies.log"),
    elements=None,
):
    if elements is None:
        elements = ["Si"]
    calc_folder = Path(calc_folder)
    calc_folder.mkdir(parents=True, exist_ok=True)

    mass_lines = [f'mass {i+1} {ELEMENT_MASSES[el]}' for i, el in enumerate(elements)]
    pair_coeff = f'pair_coeff * * {potential.name} {" ".join(elements)}'

    lines = [
        'variable nstruct equal {}'.format(nstruct),
        'variable i loop ${nstruct}',
        '',
        'label loop_struct',
        'clear',
        f'log {log_file.name} append',
        'units metal',
        'boundary p p p',
        'atom_style atomic',
        f'read_data {atoms_prefix}' + '_${i}.in',
        *mass_lines,
        'pair_style sw',
        pair_coeff,
        'run 0',
        'next i',
        'jump SELF loop_struct',
        '',
    ]

    text = "\n".join(lines)
    (calc_folder / out_name).write_text(text)


def read_lammps_energies(lammps_output):
    energies = []
    with open(lammps_output) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        cols = line.split()
        if cols and cols[0] == "Step" and "E_pair" in cols:
            epair_idx = cols.index("E_pair")
            next_line = lines[i + 1]
            if next_line.startswith("WARNING:"):
                next_line = lines[i + 2]
            data = next_line.split()
            energies.append(float(data[epair_idx]))

    return energies
