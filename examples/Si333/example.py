import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from atoms_analysis import rdf_calc, energydist_calc, energy_quantile
from atoms_analysis.analysis import calc_rdf, calc_edist, parse_traj
from atoms_analysis.namespace import LINESTYLES, LINESTYLES_DICT, FOLDER_COLORS, NSTEPS_LINESTYLES
from atoms_analysis.energydist_calc import _natoms


def main():
    if True:
        recalc_e, recalc_rdf = False, False
        recalc_e, recalc_rdf = True, False
        plot_debug = True
        #plot_debug = False
        folders = ["/home/olivi/projects/02-Simodels-ZBL/examples/samples/ref",
                   "/home/olivi/projects/02-Simodels-ZBL/examples/samples/noZBL",
                   "/home/olivi/projects/02-Simodels-ZBL/examples/samples/ZBL",]
        #folders = ["/home/olivi/projects/02-Simodels-ZBL/samples/ref"]
        #nsteps = [250, 1000]
        nsteps = [1000, 5000]

        make_classic_graph(nsteps, folders, plot_debug, recalc_e, recalc_rdf)

    #if True:
    if False:
        special_datas = ["/home/olivi/projects/02-Simodels-ZBL/scripts/output/ref/Si111_edist.dat",
                         "/home/olivi/projects/02-Simodels-ZBL/scripts/output/noZBL/Si111_1000_edist.dat",
                         "/home/olivi/projects/02-Simodels-ZBL/scripts/output/noZBL/epoch60_Si111_1000_edist.dat"]
        special_legends = ["ref", "Old_potential", "Retrained"]
        special_title = "epoch60_potential.pdf"
        special_figname = "epoch60_potential.pdf"
        make_special_graph(special_datas, special_legends, special_title, special_figname)


def make_classic_graph(nsteps, folders, plot_debug, recalc_e, recalc_rdf):
    setup_lammps = False   # set True to write LAMMPS input files for missing energy logs
    n_bins = 500
    rcut = 10
    energy_spread = {"Si333": 5}
    os.makedirs("figures", exist_ok=True)

    allowed = {str(n) for n in nsteps} if nsteps is not None else None
    any_e_pending = False
    folder_data = []

    for folder in folders:
        folder_label = os.path.basename(folder)
        os.makedirs(f"output/{folder_label}", exist_ok=True)
        atoms_list, category_names = parse_traj(folder, keep_nsteps=(folder_label != "ref"))
        folder_data.append((folder_label, category_names))

        for atoms, name in zip(atoms_list, category_names):
            calc_rdf(atoms, f"output/{folder_label}/{name}_rdf.dat", recalc=recalc_rdf, n_bins=n_bins, rcut=rcut)

        for atoms, name in zip(atoms_list, category_names):
            if energydist_calc.energy_ready(name, folder_label):
                continue
            if setup_lammps:
                energydist_calc.prepare_lammps(atoms, name, folder_label=folder_label)
            else:
                print(f"Energy log missing for {folder_label}/{name} (set setup_lammps=True to prepare)")
            any_e_pending = True

        if not any_e_pending:
            for atoms, name in zip(atoms_list, category_names):
                calc_edist(atoms, f"output/{folder_label}/{name}_edist.dat",
                           category=name, folder_label=folder_label, recalc=recalc_e)

    # Build entries grouped by elems_supercell for main plots
    entries = []
    for folder_label, category_names in folder_data:
        for name in category_names:
            m = re.match(r'^([A-Za-z]+\d+)(?:_(\d+))?$', name)
            if not m:
                continue
            elems_supercell, nsteps_str = m.group(1), m.group(2)
            if allowed is not None and nsteps_str is not None and nsteps_str not in allowed:
                continue
            display_label = f"{folder_label}_{nsteps_str}" if nsteps_str else folder_label
            color = FOLDER_COLORS.get(folder_label, "gray")
            ls = NSTEPS_LINESTYLES.get(nsteps_str, "-") if nsteps_str else "-"
            entries.append((folder_label, elems_supercell,
                            f"output/{folder_label}/{name}_rdf.dat",
                            f"output/{folder_label}/{name}_edist.dat",
                            display_label, color, ls))

    groups = defaultdict(list)
    for e in entries:
        groups[e[1]].append(e)

    for elems_supercell, group_entries in sorted(groups.items()):
        sorted_entries = sorted(group_entries, key=lambda e: 0 if e[0] == "ref" else 1)
        rdf_files   = [e[2] for e in sorted_entries]
        edist_files = [e[3] for e in sorted_entries]
        labels      = [e[4] for e in sorted_entries]
        colors      = [e[5] for e in sorted_entries]
        linestyles  = [e[6] for e in sorted_entries]
        n_atoms = _natoms(elems_supercell)
        spread = energy_spread.get(elems_supercell, 1.5) if isinstance(energy_spread, dict) else energy_spread

        rdf_calc.plot_rdf(rdf_files, f"figures/{elems_supercell}_rdf.pdf",
                          labels=labels, colors=colors, linestyles=linestyles,
                          title=f"Partial RDF — {elems_supercell}")
        if not any_e_pending:
            energydist_calc.plot_edist(edist_files, f"figures/{elems_supercell}_edist.pdf",
                                       labels=labels, colors=colors, linestyles=linestyles,
                                       n_atoms=n_atoms, energy_spread=spread,
                                       title=f"Energy distribution — {elems_supercell}")
            energy_quantile.plot_equantile(edist_files, f"figures/{elems_supercell}_equantile.pdf",
                                           labels=labels, colors=colors, linestyles=linestyles,
                                           n_atoms=n_atoms, energy_spread=spread,
                                           title=f"Energy quantile — {elems_supercell}")

    if plot_debug:
        os.makedirs("figures/debug", exist_ok=True)
        ref_rdf   = {es: rdf   for fl, es, rdf,   _,     _, _, _ in entries if fl == "ref"}
        ref_edist = {es: edist for fl, es, _,   edist,   _, _, _ in entries if fl == "ref"}

        for folder_label, category_names in folder_data:
            for name in category_names:
                m = re.match(r'^([A-Za-z0-9]+)_([A-Za-z]+\d+)(?:_(\d+))?$', name)
                if not m:
                    continue
                prefix, elems_supercell, nsteps_str = m.group(1), m.group(2), m.group(3)
                if allowed is not None and nsteps_str is not None and nsteps_str not in allowed:
                    continue
                fig_stem = f"{prefix}_{elems_supercell}_{nsteps_str}" if nsteps_str else f"{prefix}_{elems_supercell}"
                color = FOLDER_COLORS.get(folder_label, "gray")
                ls = NSTEPS_LINESTYLES.get(nsteps_str, "-") if nsteps_str else "-"
                variant_label = f"{prefix}_{folder_label}_{nsteps_str}" if nsteps_str else f"{prefix}_{folder_label}"
                n_atoms = _natoms(elems_supercell)
                spread = energy_spread.get(elems_supercell, 1.5) if isinstance(energy_spread, dict) else energy_spread

                rdf_ref = ref_rdf.get(elems_supercell)
                d_rdf = ([rdf_ref, f"output/{folder_label}/{name}_rdf.dat"] if rdf_ref and os.path.exists(rdf_ref)
                         else [f"output/{folder_label}/{name}_rdf.dat"])
                d_rdf_labels = (["ref", variant_label] if len(d_rdf) == 2 else [variant_label])
                d_rdf_colors = ([FOLDER_COLORS["ref"], color] if len(d_rdf) == 2 else [color])
                d_rdf_ls     = (["-", ls] if len(d_rdf) == 2 else [ls])
                rdf_calc.plot_rdf(d_rdf, f"figures/debug/{fig_stem}_rdf.pdf",
                                  labels=d_rdf_labels, colors=d_rdf_colors, linestyles=d_rdf_ls,
                                  title=f"Partial RDF — {fig_stem} (debug)")

                if not any_e_pending:
                    edist_ref = ref_edist.get(elems_supercell)
                    d_edist = ([edist_ref, f"output/{folder_label}/{name}_edist.dat"] if edist_ref and os.path.exists(edist_ref)
                               else [f"output/{folder_label}/{name}_edist.dat"])
                    d_edist_labels = (["ref", variant_label] if len(d_edist) == 2 else [variant_label])
                    d_edist_colors = ([FOLDER_COLORS["ref"], color] if len(d_edist) == 2 else [color])
                    d_edist_ls     = (["-", ls] if len(d_edist) == 2 else [ls])
                    energydist_calc.plot_edist(d_edist, f"figures/debug/{fig_stem}_edist.pdf",
                                               labels=d_edist_labels, colors=d_edist_colors, linestyles=d_edist_ls,
                                               n_atoms=n_atoms, energy_spread=spread,
                                               title=f"Energy distribution — {fig_stem} (debug)")
                    energy_quantile.plot_equantile(d_edist, f"figures/debug/{fig_stem}_equantile.pdf",
                                                   labels=d_edist_labels, colors=d_edist_colors, linestyles=d_edist_ls,
                                                   n_atoms=n_atoms, energy_spread=spread,
                                                   title=f"Energy quantile — {fig_stem} (debug)")


def make_special_graph(data, legends=None, title=None, figname=None, smooth_sigma=1):
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
    if figname is None:
        return f"figures/special_{suffix}.pdf"
    base, ext = os.path.splitext(figname)
    return os.path.join("figures", f"{base}_{suffix}{ext}")


def _file_label(path):
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
    stem0 = _file_label(data[0])
    m = re.search(r'(SiGe|Si|Ge)(111|222|333)', stem0)
    n_atoms = _natoms(m.group(0)) if m else 1

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


if __name__ == "__main__":
    main()
