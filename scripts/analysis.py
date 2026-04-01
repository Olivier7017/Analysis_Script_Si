import os
import re
import tempfile
import numpy as np
from glob import glob
from collections import defaultdict

from ase.io import read, write

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import rdf_calc
import energydist_calc
import energy_quantile
from namespace import LINESTYLES, LINESTYLES_DICT
from energydist_calc import _natoms


def notexist(path):
    return not os.path.exists(path)


def main():
    if True:
        recalc_e, recalc_rdf = False, False
        recalc_e, recalc_rdf = True, False
        plot_debug = True
        #plot_debug = False
        folders = ["/home/olivi/projects/02-Simodels-ZBL/samples/ref",
                   "/home/olivi/projects/02-Simodels-ZBL/samples/noZBL",
                   "/home/olivi/projects/02-Simodels-ZBL/samples/ZBL",]
        #folders = ["/home/olivi/projects/02-Simodels-ZBL/samples/ref"]
        #nsteps = [250, 1000]
        nsteps = [1000]

        make_classic_graph(nsteps, folders, plot_debug, recalc_e, recalc_rdf)

    #if True:
    if False:
        #special_datas = ["/home/olivi/projects/02-Simodels-ZBL/scripts/output/ref/Si333_edist.dat",
        #                 "/home/olivi/projects/02-Simodels-ZBL/scripts/output/ZBL/lowestforce_Si333_1000_edist.dat",  # 1
        #                 "/home/olivi/projects/02-Simodels-ZBL/scripts/output/ZBL/lowerforce_Si333_1000_edist.dat",  # 10
        #                 "/home/olivi/projects/02-Simodels-ZBL/scripts/output/ZBL/lowforce_Si333_1000_edist.dat", # 100
        #                 #"/home/olivi/projects/02-Simodels-ZBL/scripts/output/ZBL/Si333_1000_edist.dat",  # 1000
        #                 "/home/olivi/projects/02-Simodels-ZBL/scripts/output/ZBL/highforce_Si333_1000_edist.dat",
        #                 "/home/olivi/projects/02-Simodels-ZBL/scripts/output/noZBL/Si333_1000_edist.dat"] #5000
        #special_datas = ["/home/olivi/projects/02-Simodels-ZBL/scripts/output/ref/Si111_rdf.dat",
        #                 "/home/olivi/projects/02-Simodels-ZBL/scripts/output/noZBL/Si111_1000_rdf.dat",
        #                 "/home/olivi/projects/02-Simodels-ZBL/scripts/output/noZBL/epoch60_Si111_1000_rdf.dat"]
        special_datas = ["/home/olivi/projects/02-Simodels-ZBL/scripts/output/ref/Si111_edist.dat",
                         "/home/olivi/projects/02-Simodels-ZBL/scripts/output/noZBL/Si111_1000_edist.dat",
                         "/home/olivi/projects/02-Simodels-ZBL/scripts/output/noZBL/epoch60_Si111_1000_edist.dat"]
        #special_datas = ["/home/olivi/projects/02-Simodels-ZBL/scripts/output/ref/Si333_rdf.dat",
        #                 "/home/olivi/projects/02-Simodels-ZBL/scripts/output/ZBL/lowestforce_Si333_1000_rdf.dat",  # 1
        #                 "/home/olivi/projects/02-Simodels-ZBL/scripts/output/ZBL/lowerforce_Si333_1000_rdf.dat", # 10
        #                 "/home/olivi/projects/02-Simodels-ZBL/scripts/output/ZBL/lowforce_Si333_1000_rdf.dat", # 100
        #                 #"/home/olivi/projects/02-Simodels-ZBL/scripts/output/ZBL/Si333_1000_rdf.dat",  # 500
        #                 "/home/olivi/projects/02-Simodels-ZBL/scripts/output/ZBL/highforce_Si333_1000_rdf.dat"] #5000
        special_legends = ["ref", "lowest_ZBL", "lower_ZBL", "low_ZBL", "high_ZBL", "no_ZBL"]
        special_legends = ["ref", "Old_potential", "Retrained"] 
        special_title = "epoch60_potential.pdf"
        special_figname = "epoch60_potential.pdf"
        make_special_graph(special_datas, special_legends, special_title, special_figname)


def make_classic_graph(nsteps, folders, plot_debug, recalc_e, recalc_rdf):
    setup_lammps = False   # set True to write LAMMPS input files for missing energy logs
    n_bins = 500
    rcut = 10
    os.makedirs("figures", exist_ok=True)

    any_e_pending = False
    folder_data = []
    for folder in folders:
        folder_label = os.path.basename(folder)
        os.makedirs(f"output/{folder_label}", exist_ok=True)
        atoms_list, category_names = parse_traj(folder, keep_nsteps=(folder_label != "ref"))
        folder_data.append((folder_label, category_names))

        # RDF
        if recalc_rdf or any([notexist(f"output/{folder_label}/{name}_rdf.dat") for name in category_names]):
            print("Calculating RDF")
            for atoms, name in zip(atoms_list, category_names):
                calc_rdf(atoms, name, folder_label=folder_label, recalc=recalc_rdf, n_bins=n_bins, rcut=rcut)

        # Energy distribution — level 1: ensure lammps energy evaluation
        for atoms, name in zip(atoms_list, category_names):
            if energydist_calc.energy_ready(name, folder_label):
                continue
            if setup_lammps:
                energydist_calc.prepare_lammps(atoms, name, folder_label=folder_label)
            else:
                print(f"Energy log missing for {folder_label}/{name} (set setup_lammps=True to prepare)")
            any_e_pending = True

        # Energy distribution — level 2: read energies and write edist.dat
        if not any_e_pending:
            for atoms, name in zip(atoms_list, category_names):
                if recalc_e or notexist(f"output/{folder_label}/{name}_edist.dat"):
                    calc_edist(atoms, name, folder_label=folder_label, recalc=recalc_e)

    rdf_calc.plot_rdf(folder_data, nsteps=nsteps, plot_debug=plot_debug)
    if not any_e_pending:
        energy_spread = {"Si333": 5}
        energydist_calc.plot_edist(folder_data, nsteps=nsteps, plot_debug=plot_debug, energy_spread=energy_spread)
        energy_quantile.plot_energy_quantile(folder_data, nsteps=nsteps, plot_debug=plot_debug, energy_spread=energy_spread)


def calc_rdf(atoms, name, folder_label=None, recalc=False, n_bins=500, rcut=10):
    out_file = f"output/{folder_label}/{name}_rdf.dat" if folder_label else f"output/{name}_rdf.dat"
    if not recalc and os.path.exists(out_file):
        return
    with tempfile.NamedTemporaryFile(suffix=".traj", delete=False) as f:
        tmp_path = f.name
    try:
        write(tmp_path, atoms)
        rdf_calc.partial_rdf([tmp_path], n_bins, rcut, out_file)
    finally:
        os.unlink(tmp_path)


def calc_edist(atoms, name, folder_label=None, recalc=False):
    out_file = f"output/{folder_label}/{name}_edist.dat" if folder_label else f"output/{name}_edist.dat"
    if not recalc and os.path.exists(out_file):
        return
    energydist_calc.edist(atoms, out_file, category=name, folder_label=folder_label)


_NATOMS_TO_SUPERCELL = {8: "111", 64: "222", 216: "333"}


def normalize_system(name):
    """Convert atom-count names (Si216, Si4Ge4) to supercell notation (Si333, SiGe111)."""
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
    """
    Groups trajectory files by supercell notation (e.g. Si333, SiGe222), combining
    all files for the same system. Ref files with atom-count names (Si216, Si4Ge4)
    are normalized to supercell notation before grouping.
    If keep_nsteps=True, groups by {system}_{nsteps} (e.g. Si111_1000) so files with
    different step counts are kept separate.
    Returns (atoms_list, category_names) where each entry aggregates all frames.
    """
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

def make_special_graph(data, legends=None, title=None, figname=None, smooth_sigma=1):
    """Plot all files in data on the same figure.

    File type is inferred from the filename suffix:
      - *_rdf.dat   → partial RDF (one subplot per pair, all files overlaid)
      - *_edist.dat → energy histogram + energy quantile (two figures)

    legends:  list of labels (one per file); falls back to filename stem if None.
    title:    suptitle for the figure(s); falls back to a generic title if None.
    figname:  output filename (e.g. "ncorrector_step.pdf"); saved under figures/.
              For edist, "_edist" and "_equantile" are inserted before the extension.
              Falls back to figures/special_{type}.pdf if None.
    """
    if not data:
        return

    labels = legends if legends is not None else [_file_label(p) for p in data]
    first = data[0]
    if first.endswith("_rdf.dat"):
        _special_rdf(data, labels, title, figname, smooth_sigma)
    elif first.endswith("_edist.dat"):
        _special_edist(data, labels, title, figname)
        _special_equantile(data, labels, title, figname)
    else:
        raise ValueError(f"Cannot infer plot type from filename: {first}")


def _special_outfile(figname, suffix):
    """Build output path from figname (e.g. 'ncorrector_step.pdf') and a suffix tag."""
    if figname is None:
        return f"figures/special_{suffix}.pdf"
    base, ext = os.path.splitext(figname)
    return os.path.join("figures", f"{base}_{suffix}{ext}")


def _file_label(path):
    """Return a short label from a dat file path: stem minus the trailing _type."""
    stem = os.path.basename(path)
    stem = re.sub(r'_(rdf|edist)\.dat$', '', stem)
    return stem


def _special_rdf(data, labels, title, figname, smooth_sigma):
    with open(data[0]) as f:
        header_line = f.readline().strip().lstrip('#').strip()
    pair_names = header_line.split()[1:]   # drop 'r'
    n_pairs = len(pair_names)

    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 5), squeeze=False)

    for i, path in enumerate(data):
        d = np.loadtxt(path)
        r = d[:, 0]
        color = f"C{i}"
        #ls = LINESTYLES[i % len(LINESTYLES)]
        linestyles = ["-"] + [LINESTYLES_DICT['dashed'] for i in range(len(labels)-1)]
        for j, pair_name in enumerate(pair_names):
            g = gaussian_filter1d(d[:, j + 1], sigma=smooth_sigma) if smooth_sigma else d[:, j + 1]
            axes[0, j].plot(r, g, color=color, linestyle=linestyles[i], label=labels[i])
            axes[0, j].set_title(pair_name, fontsize=14)
            axes[0, j].set_xlabel("r (Å)", fontsize=13)
            axes[0, j].set_ylabel("g(r)", fontsize=13)
            axes[0, j].set_xlim(left=0)

    axes[0, 0].legend(fontsize=11)
    fig.suptitle(title or "Partial RDF", fontsize=15)
    fig.tight_layout()
    out = _special_outfile(figname, "rdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def _special_edist(data, labels, title, figname):
    # Determine n_atoms from first file's stem
    stem0 = _file_label(data[0])
    m = re.search(r'(SiGe|Si|Ge)(111|222|333)', stem0)
    n_atoms = _natoms(m.group(0)) if m else 1

    # Compute plot range from ref (first file) only
    ref_e = np.loadtxt(data[0]) / n_atoms
    ref_e = ref_e[np.isfinite(ref_e)]
    mean_r = ref_e.mean()
    half_width = max(ref_e.max() - mean_r, mean_r - ref_e.min())
    lo = mean_r - 1.5 * half_width
    hi = mean_r + 1.5 * half_width
    nbins = max(10, int(len(ref_e) / 5))

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, path in enumerate(data):
        energies = np.loadtxt(path) / n_atoms
        energies = energies[np.isfinite(energies)]
        weights = np.ones(len(energies)) / len(energies) * 100
        ax.hist(energies, bins=nbins, range=(lo, hi), weights=weights,
                histtype='stepfilled', alpha=0.5, color=f"C{i}",
                linestyle=LINESTYLES[i % len(LINESTYLES)], label=labels[i])
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Energy per atom (eV/atom)", fontsize=13)
    ax.set_ylabel("Percentage of configurations (%)", fontsize=13)
    ax.legend(fontsize=11)
    fig.suptitle(title or "Energy distribution", fontsize=15)
    fig.tight_layout()
    out = _special_outfile(figname, "edist")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def _special_equantile(data, labels, title, figname):
    stem0 = _file_label(data[0])
    m = re.search(r'(SiGe|Si|Ge)(111|222|333)', stem0)
    n_atoms = _natoms(m.group(0)) if m else 1

    ref_e = np.loadtxt(data[0]) / n_atoms
    ref_e = ref_e[np.isfinite(ref_e)]
    ref_mean = np.mean(ref_e)
    half_width = max(ref_e.max() - ref_mean, ref_mean - ref_e.min())
    lo = -5 * half_width
    hi = 5 * half_width
    linestyles = ["-"] + [LINESTYLES_DICT['dashed'] for i in range(len(labels)-1)]

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, path in enumerate(data):
        energies = np.loadtxt(path) / n_atoms
        energies = energies[np.isfinite(energies)]
        sorted_e = np.sort(energies) - ref_mean
        percentiles = np.linspace(0, 100, len(sorted_e))
        ax.plot(percentiles, sorted_e, color=f"C{i}",
                linestyle=linestyles[i], label=labels[i])
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Percentile", fontsize=13)
    ax.set_ylabel("Energy per atom relative to ref mean (eV/atom)", fontsize=13)
    ax.legend(fontsize=11)
    fig.suptitle(title or "Energy quantile", fontsize=15)
    fig.tight_layout()
    out = _special_outfile(figname, "equantile")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__=="__main__":
    main()
