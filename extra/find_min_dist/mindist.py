from ase.io import read
from ase.geometry import get_distances
import numpy as np

filenames = [
    "ZBL/lowforce_Si333_1000",
    "ZBL/new_Si333_5000",
    "noZBL/Si333_1000",
]

traj_dir = "/home/olivi/projects/02-Simodels-ZBL/examples/samples/"
output_dir = "/home/olivi/projects/02-Simodels-ZBL/examples/output/"

summary = []

for filename in filenames:
    print(f"LOOKING AT {filename}")

    traj_path = traj_dir + filename + ".traj"
    output_path = output_dir + filename + "_edist.dat"

    images = read(traj_path, ":")
    energies = np.loadtxt(output_path)

    n_incorrect = 0
    min_dist_overall = np.inf
    min_dist_frame = None

    for i, atoms in enumerate(images):
        if energies[i] < -934.75:
            continue

        positions = atoms.get_positions()
        cell = atoms.get_cell()
        scaled = atoms.get_scaled_positions()

        # Compute all pair distances with PBC
        _, dists = get_distances(positions, positions, cell=cell, pbc=True)

        # Ignore self-distances
        np.fill_diagonal(dists, np.inf)

        # Closest pair
        i1, i2 = np.unravel_index(np.argmin(dists), dists.shape)
        dist = dists[i1, i2]

        r1 = scaled[i1]
        r2 = scaled[i2]

        print(
            f"Atoms {i} ({energies[i]:.2f} eV): "
            f"{dist:.2f} Ang between "
            f"at{i1} ({r1[0]:.2f}, {r1[1]:.2f}, {r1[2]:.2f}) "
            f"and "
            f"at{i2} ({r2[0]:.2f}, {r2[1]:.2f}, {r2[2]:.2f})"
        )

        n_incorrect += 1
        if dist < min_dist_overall:
            min_dist_overall = dist
            min_dist_frame = i

    if n_incorrect == 0:
        summary.append((filename, 0, None, None))
    else:
        summary.append((filename, n_incorrect, min_dist_overall, min_dist_frame))

    print("")

print("RECAP")
for filename, n_incorrect, min_dist, frame in summary:
    if n_incorrect == 0:
        print(f"{filename}: 0 incorrect configurations.")
    else:
        print(
            f"{filename}: {n_incorrect} incorrect configurations. "
            f"Min dist = {min_dist:.3f} Ang (frame {frame})"
        )
