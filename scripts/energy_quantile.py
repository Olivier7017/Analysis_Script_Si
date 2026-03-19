from __future__ import annotations
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from namespace import FOLDER_COLORS, NSTEPS_LINESTYLES
from energydist_calc import _natoms


def plot_energy_quantile(folder_data, nsteps=None, plot_debug=False, energy_spread=1.5):
    """
    folder_data: list of (folder_label, category_names)
    nsteps: list of int, e.g. [250, 1000], to restrict which nsteps are plotted (None = all)

    One figure per {elems}{supercell}. X-axis: percentile of configuration by energy.
    Y-axis: energy (eV). Y range set to ±50% of ref half-width around ref mean.
    Figures saved to figures/{elems}{supercell}_equantile.pdf.
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

        # Y range from ref
        ref_entry = next((e for e in sorted_entries if e[0] == "ref"), None)
        spread = energy_spread.get(elems_supercell, 1.5) if isinstance(energy_spread, dict) else energy_spread
        if ref_entry is not None:
            ref_e = np.loadtxt(ref_entry[2]) / n_atoms
            mean_r = ref_e.mean()
            lo = mean_r - spread * (mean_r - ref_e.min())
            hi = mean_r + spread * (ref_e.max() - mean_r)
        else:
            all_e = np.concatenate([np.loadtxt(e[2]) for e in sorted_entries]) / n_atoms
            lo, hi = all_e.min(), all_e.max()

        for folder_label, _, dat_file, display_label in sorted_entries:
            energies = np.loadtxt(dat_file) / n_atoms
            sorted_e = np.sort(energies)
            percentiles = np.linspace(0, 100, len(sorted_e))
            nsteps = re.search(r'_(\d+)$', display_label)
            nsteps = nsteps.group(1) if nsteps else None
            color = FOLDER_COLORS.get(folder_label, "gray")
            ls = NSTEPS_LINESTYLES.get(nsteps, "-") if nsteps else "-"
            ax.plot(percentiles, sorted_e, color=color, linestyle=ls, label=display_label)

        ax.set_ylim(lo, hi)
        ax.set_xlabel("Percentile", fontsize=13)
        ax.set_ylabel("Energy per atom (eV/atom)", fontsize=13)
        ax.legend(fontsize=11)
        fig.suptitle(f"Energy quantile — {elems_supercell}", fontsize=15)
        fig.tight_layout()
        fig.savefig(f"figures/{elems_supercell}_equantile.pdf")
        plt.close(fig)

    if plot_debug:
        os.makedirs("figures/debug", exist_ok=True)

        ref_lookup = {elems_supercell: dat_file
                      for folder_label, elems_supercell, dat_file, _ in entries
                      if folder_label == "ref"}

        for folder_label, category_names in folder_data:
            for name in category_names:
                m = re.match(r'^([A-Za-z0-9]+)_([A-Za-z]+\d+)(?:_(\d+))?$', name)
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
                    sorted_ref = np.sort(ref_e)
                    ax.plot(np.linspace(0, 100, len(sorted_ref)), sorted_ref,
                            color=FOLDER_COLORS["ref"], linestyle="-", label="ref")
                else:
                    finite_e = np.loadtxt(dat_file) / n_atoms
                    finite_e = finite_e[np.isfinite(finite_e)]
                    lo, hi = finite_e.min(), finite_e.max()

                energies = np.loadtxt(dat_file) / n_atoms
                sorted_e = np.sort(energies)
                color = FOLDER_COLORS.get(folder_label, "gray")
                ls = NSTEPS_LINESTYLES.get(nsteps_str, "-") if nsteps_str else "-"
                ax.plot(np.linspace(0, 100, len(sorted_e)), sorted_e,
                        color=color, linestyle=ls, label=variant_label)

                ax.set_ylim(lo, hi)
                ax.set_xlabel("Percentile", fontsize=13)
                ax.set_ylabel("Energy per atom (eV/atom)", fontsize=13)
                ax.legend(fontsize=11)
                fig_stem = f"{prefix}_{elems_supercell}_{nsteps_str}" if nsteps_str else f"{prefix}_{elems_supercell}"
                fig.suptitle(f"Energy quantile — {fig_stem} (debug)", fontsize=15)
                fig.tight_layout()
                fig.savefig(f"figures/debug/{fig_stem}_equantile.pdf")
                plt.close(fig)
