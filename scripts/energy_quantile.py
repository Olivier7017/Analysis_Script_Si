import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_equantile(dat_files, out_file, labels=None, colors=None, linestyles=None, n_atoms=1, energy_spread=1.5, title=None):
    """Plot energy quantile curves from dat_files, save to out_file.
    First file sets the y range."""
    if not dat_files:
        return
    if labels is None:
        labels = [os.path.splitext(os.path.basename(f))[0] for f in dat_files]

    ref_e = np.loadtxt(dat_files[0]) / n_atoms
    ref_e = ref_e[np.isfinite(ref_e)]
    mean_r = ref_e.mean()
    lo = mean_r - energy_spread * (mean_r - ref_e.min())
    hi = mean_r + energy_spread * (ref_e.max() - mean_r)

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (path, label) in enumerate(zip(dat_files, labels)):
        energies = np.loadtxt(path) / n_atoms
        energies = energies[np.isfinite(energies)]
        sorted_e = np.sort(energies)
        percentiles = np.linspace(0, 100, len(sorted_e))
        color = colors[i] if colors else f"C{i}"
        ls = linestyles[i] if linestyles else "-"
        ax.plot(percentiles, sorted_e, color=color, linestyle=ls, label=label)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Percentile", fontsize=13)
    ax.set_ylabel("Energy per atom (eV/atom)", fontsize=13)
    ax.legend(fontsize=11)
    fig.suptitle(title or "Energy quantile", fontsize=15)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")
