import os
import re
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")   # must be before pyplot import
import matplotlib.pyplot as plt
from collections import defaultdict

from ase.io import read, write
from ase.units import _Nav, J, kB, GPa

from ovito.data import NearestNeighborFinder
from ovito.modifiers import CoordinationAnalysisModifier
from ovito.io import import_file, export_file
from namespace import FOLDER_COLORS, NSTEPS_LINESTYLES


def partial_rdf(traj_files, n_bins, rcut, out_file):
    """Compute the partial RDF averaged over all frames from all traj_files and save to out_file."""
    all_frames = []
    header = None
    for traj_fn in traj_files:
        pipeline = import_file(traj_fn)
        mod = CoordinationAnalysisModifier(cutoff=rcut, number_of_bins=n_bins, partial=True)
        pipeline.modifiers.append(mod)
        for frame in range(pipeline.source.num_frames):
            rdf_table = pipeline.compute(frame).tables['coordination-rdf']
            if header is None:
                header = "r " + " ".join(rdf_table.y.component_names)
            all_frames.append(rdf_table.xy())
    avg = np.mean(np.array(all_frames), axis=0)
    np.savetxt(out_file, avg, header=header)


def plot_rdf(folder_data, smooth_sigma=1, nsteps=None, plot_debug=False):
    """
    folder_data: list of (folder_label, category_names)
    nsteps: list of int, e.g. [250, 1000], to restrict which nsteps are plotted (None = all)

    One figure per {elems}{supercell}, with one subplot per partial RDF pair.
    Each line corresponds to a (folder_label, nsteps) combination.
    Figures saved to figures/{elems}{supercell}_rdf.pdf.
    """
    allowed = {str(n) for n in nsteps} if nsteps is not None else None
    # Build flat list of (folder_label, elems_supercell, dat_file)
    entries = []
    for folder_label, category_names in folder_data:
        for name in category_names:
            m = re.match(r'^([A-Za-z]+\d+)(?:_(\d+))?$', name)
            if m:
                elems_supercell = m.group(1)
                nsteps_str = m.group(2)
                if allowed is not None and nsteps_str is not None and nsteps_str not in allowed:
                    continue
                entries.append((folder_label, elems_supercell, f"output/{folder_label}/{name}_rdf.dat", nsteps_str))

    # Group by elems_supercell
    groups = defaultdict(list)
    for entry in entries:
        groups[entry[1]].append(entry)

    for elems_supercell, group_entries in sorted(groups.items()):
        # Read header from first file to get partial pair names
        with open(group_entries[0][2]) as f:
            header_line = f.readline().strip().lstrip('#').strip()
        pair_names = header_line.split()[1:]  # drop 'r' column
        n_pairs = len(pair_names)

        fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 5), squeeze=False)

        # ref plotted first (solid, underneath); others on top
        sorted_entries = sorted(group_entries, key=lambda e: 0 if e[0] == "ref" else 1)

        # Compute per-pair y limits from ref (mirrors edist x-limit logic)
        ref_entry = next((e for e in sorted_entries if e[0] == "ref"), None)
        if ref_entry is not None:
            ref_data = np.loadtxt(ref_entry[2])
            ylims = []
            for j in range(n_pairs):
                g_ref = gaussian_filter1d(ref_data[:, j + 1], sigma=smooth_sigma) if smooth_sigma else ref_data[:, j + 1]
                mean_g = g_ref.mean()
                ylims.append(mean_g + 1.5 * (g_ref.max() - mean_g))
        else:
            ylims = [None] * n_pairs

        for folder_label, _, dat_file, nsteps in sorted_entries:
            data = np.loadtxt(dat_file)
            r = data[:, 0]
            color = FOLDER_COLORS.get(folder_label, "gray")
            ls = NSTEPS_LINESTYLES.get(nsteps, "-") if nsteps else "-"
            label = f"{folder_label}_{nsteps}" if nsteps else folder_label
            for j, pair_name in enumerate(pair_names):
                g = gaussian_filter1d(data[:, j + 1], sigma=smooth_sigma) if smooth_sigma else data[:, j + 1]
                axes[0, j].plot(r, g, color=color, linestyle=ls, label=label)
                axes[0, j].set_title(pair_name, fontsize=14)
                axes[0, j].set_xlabel("r (Å)", fontsize=13)
                axes[0, j].set_ylabel("g(r)", fontsize=13)
                axes[0, j].set_xlim(left=0)
                axes[0, j].set_ylim(0, ylims[j])

        axes[0, 0].legend(fontsize=11)
        fig.suptitle(f"Partial RDF — {elems_supercell}", fontsize=15)
        fig.tight_layout()
        fig.savefig(f"figures/{elems_supercell}_rdf.pdf")
        plt.close(fig)

    if plot_debug:
        os.makedirs("figures/debug", exist_ok=True)

        # Build ref lookup: elems_supercell -> dat_file
        ref_lookup = {elems_supercell: dat_file
                      for folder_label, elems_supercell, dat_file, nsteps_str in entries
                      if folder_label == "ref"}

        for folder_label, category_names in folder_data:
            for name in category_names:
                m = re.match(r'^([A-Za-z0-9]+)_([A-Za-z]+\d+)(?:_(\d+))?$', name)
                if not m:
                    continue
                prefix, elems_supercell, nsteps_str = m.group(1), m.group(2), m.group(3)
                if allowed is not None and nsteps_str is not None and nsteps_str not in allowed:
                    continue
                dat_file = f"output/{folder_label}/{name}_rdf.dat"

                with open(dat_file) as f:
                    header_line = f.readline().strip().lstrip('#').strip()
                pair_names = header_line.split()[1:]
                n_pairs = len(pair_names)

                fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 5), squeeze=False)

                # Plot ref and compute y limits from it
                ref_file = ref_lookup.get(elems_supercell)
                ylims = [None] * n_pairs
                if ref_file and os.path.exists(ref_file):
                    ref_data = np.loadtxt(ref_file)
                    r_ref = ref_data[:, 0]
                    for j in range(n_pairs):
                        g_ref = gaussian_filter1d(ref_data[:, j + 1], sigma=smooth_sigma) if smooth_sigma else ref_data[:, j + 1]
                        axes[0, j].plot(r_ref, g_ref, color=FOLDER_COLORS["ref"], linestyle="-", label="ref")
                        mean_g = g_ref.mean()
                        ylims[j] = mean_g + 1.5 * (g_ref.max() - mean_g)

                # Plot prefixed variant
                data = np.loadtxt(dat_file)
                r = data[:, 0]
                color = FOLDER_COLORS.get(folder_label, "gray")
                ls = NSTEPS_LINESTYLES.get(nsteps_str, "-") if nsteps_str else "-"
                variant_label = f"{prefix}_{folder_label}_{nsteps_str}" if nsteps_str else f"{prefix}_{folder_label}"
                for j in range(n_pairs):
                    g = gaussian_filter1d(data[:, j + 1], sigma=smooth_sigma) if smooth_sigma else data[:, j + 1]
                    axes[0, j].plot(r, g, color=color, linestyle=ls, label=variant_label)

                fig_stem = f"{prefix}_{elems_supercell}_{nsteps_str}" if nsteps_str else f"{prefix}_{elems_supercell}"
                for j, pair_name in enumerate(pair_names):
                    axes[0, j].set_title(pair_name, fontsize=14)
                    axes[0, j].set_xlabel("r (Å)", fontsize=13)
                    axes[0, j].set_ylabel("g(r)", fontsize=13)
                    axes[0, j].set_xlim(left=0)
                    axes[0, j].set_ylim(0, ylims[j])
                axes[0, 0].legend(fontsize=11)
                fig.suptitle(f"Partial RDF — {fig_stem} (debug)", fontsize=15)
                fig.tight_layout()
                fig.savefig(f"figures/debug/{fig_stem}_rdf.pdf")
                plt.close(fig)
