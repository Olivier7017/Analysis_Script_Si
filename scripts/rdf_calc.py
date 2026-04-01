import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ovito.modifiers import CoordinationAnalysisModifier
from ovito.io import import_file


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


def plot_rdf(dat_files, out_file, labels=None, colors=None, linestyles=None, rcut=None, smooth_sigma=1, title=None):
    """Plot partial RDF from dat_files onto a single figure saved to out_file.
    First file is treated as reference for y-limits."""
    if not dat_files:
        return
    if labels is None:
        labels = [os.path.splitext(os.path.basename(f))[0] for f in dat_files]

    with open(dat_files[0]) as f:
        header_line = f.readline().strip().lstrip('#').strip()
    pair_names = header_line.split()[1:]
    n_pairs = len(pair_names)

    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 5), squeeze=False)

    ref_data = np.loadtxt(dat_files[0])
    ylims = []
    for j in range(n_pairs):
        g_ref = gaussian_filter1d(ref_data[:, j + 1], sigma=smooth_sigma) if smooth_sigma else ref_data[:, j + 1]
        mean_g = g_ref.mean()
        ylims.append(mean_g + 1.5 * (g_ref.max() - mean_g))

    for i, (path, label) in enumerate(zip(dat_files, labels)):
        data = np.loadtxt(path)
        r = data[:, 0]
        color = colors[i] if colors else f"C{i}"
        ls = linestyles[i] if linestyles else "-"
        for j, pair_name in enumerate(pair_names):
            g = gaussian_filter1d(data[:, j + 1], sigma=smooth_sigma) if smooth_sigma else data[:, j + 1]
            axes[0, j].plot(r, g, color=color, linestyle=ls, label=label)
            axes[0, j].set_title(pair_name, fontsize=14)
            axes[0, j].set_xlabel("r (Å)", fontsize=13)
            axes[0, j].set_ylabel("g(r)", fontsize=13)
            axes[0, j].set_xlim(0, rcut) if rcut else axes[0, j].set_xlim(left=0)
            axes[0, j].set_ylim(0, ylims[j])

    axes[0, 0].legend(fontsize=11)
    fig.suptitle(title or "Partial RDF", fontsize=15)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")
