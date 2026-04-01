import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from ase.io import read, write
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.repulsive_force.zbl_force import (ZBLForce, ZBLForceParameters)
from namespace import SW_POTENTIAL, SW_CACHE, ZBL_CACHE

def main():
    dist = [None, 1.454, 0.969, 0.576]
    force_activation_scales = [1, 10, 100, 1000]

    dist_list = np.linspace(0.00719, 3, 5000)
    for si_struct in ["/home/pomax/Data/Structures/Si/Si216.cif",
                      "/home/olivi/Data/Structures/Si216.cif"]:
        if os.path.exists(si_struct):
            break

    atoms = create_atoms(dist_list, si_struct)

    make_graph1(atoms, dist_list)
    make_graph2(atoms, dist_list, dist, force_activation_scales)


def make_graph1(atoms, dist_list):
    sw_energy, sw_forces = sw_calculate_traj(atoms)
    zbl_forces = calc_zbl_forces(atoms)
    zbl_forces0_norm = torch.norm(zbl_forces[:, 0, :], dim=-1).numpy()
    probs = calc_probs(sw_energy)

    plot_forces_and_probs(dist_list, sw_forces, zbl_forces0_norm, probs)


def make_graph2(atoms, dist_list, min_dists, force_activation_scales):
    t = torch.zeros(len(atoms))
    zbl_forces = calc_zbl_forces(atoms)
    sw_energy, _ = sw_calculate_traj(atoms)
    probs = calc_probs(sw_energy)

    zbl_percents = {}
    for scale in force_activation_scales:
        zbl_percent, _ = zbl_egnn_fractions(zbl_forces, atoms, t, scale)
        zbl_percents[scale] = zbl_percent.numpy()

    plot_score_percent(dist_list, zbl_percents, probs, min_dists)


def plot_score_percent(dist_list, zbl_percents, probs, min_dists):
    fig, ax_zbl = plt.subplots(figsize=(8, 5))
    ax_prob = ax_zbl.twinx()

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    lines = []
    for (scale, zbl_percent), color, min_d in zip(zbl_percents.items(), colors, min_dists):
        l, = ax_zbl.plot(dist_list, zbl_percent * 100, label=f"ZBL % (Λ={scale})", color=color)
        lines.append(l)
        if min_d is not None:
            ax_zbl.axvline(min_d, color=color, linestyle="--")

    l_prob, = ax_prob.plot(dist_list, probs, label="Probability (300 K)", color="black", linestyle="--")
    lines.append(l_prob)

    threshold_idx = np.argmax(probs > 1e-5)
    threshold_dist = dist_list[threshold_idx]
    l_vline = ax_prob.axvline(threshold_dist, label="1 in 100 000", color="tab:red", linestyle=":")
    lines.append(l_vline)
    ax_zbl.text(threshold_dist-0.025, 42, f"{threshold_dist:.2f} Å", color="tab:red", ha="right", va="bottom", fontsize=16)

    ax_zbl.set_xlim(dist_list[0], 2.5)
    ax_zbl.set_ylim(0, 100)
    ax_prob.set_xlim(dist_list[0], 2.5)
    ax_prob.set_ylim(0.0, 0.008)
    ax_prob.set_yticks([])

    ax_zbl.set_xlabel("Interatomic distance (Å)", fontsize=16)
    ax_zbl.set_ylabel("ZBL score contribution (%)", fontsize=16)
    ax_prob.set_ylabel("Probability", fontsize=16)
    ax_zbl.tick_params(labelsize=16)
    ax_zbl.set_title("ZBL score contribution vs interatomic distance (r$_{cut}$ = 2.2 Å)", fontsize=16)

    ax_zbl.legend(lines, [l.get_label() for l in lines], loc="upper center", bbox_to_anchor=(0.65, 1.0), fontsize=14)

    plt.tight_layout()
    plt.savefig("newgraph2_scorepercent.pdf")
    plt.clf()


def create_atoms(dist_list, si_struct):
    """For each distance in dist_list, return a copy of the Si structure with
    atom 0 displaced along X toward its nearest neighbor to reach that distance.
    Other atoms are not moved."""
    gs = read(si_struct)

    # Find nearest neighbor of atom 0 (by minimum distance)
    pos0 = gs.positions[0]
    best_idx, best_disp = None, None
    best_dist = np.inf
    for i, pos in enumerate(gs.positions[1:], start=1):
        # Account for periodic boundary conditions
        diff = pos - pos0
        diff -= gs.cell.T @ np.round(np.linalg.solve(gs.cell.T, diff))
        dist = np.linalg.norm(diff)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
            best_disp = diff

    nn_dir = best_disp / np.linalg.norm(best_disp)  # unit vector toward neighbor
    nn_pos = pos0 + best_disp  # neighbor position (with PBC image if needed)

    atoms_list = []
    for d in dist_list:
        atoms = gs.copy()
        # Place atom 0 at distance d from the neighbor, along the bond direction
        atoms.positions[0] = nn_pos - d * nn_dir
        atoms_list.append(atoms)

    return atoms_list



def calc_zbl_forces(atoms_list):
    # Returns full Cartesian forces [nconf, natoms, 3] as a torch tensor
    if os.path.exists(ZBL_CACHE):
        print("Reading ZBL data")
        return torch.tensor(np.load(ZBL_CACHE))

    print("Calculating ZBL data")
    zbl_parameters = ZBLForceParameters(radial_cutoff=2.2, inner_radius_fraction=0.5, element_list=["Si"])
    zbl_force = ZBLForce(zbl_parameters)

    nconf = len(atoms_list)
    natoms = len(atoms_list[0])

    A = torch.zeros(nconf, natoms, dtype=torch.long)
    cartesian_positions = torch.tensor(
        np.stack([a.positions for a in atoms_list]), dtype=torch.float32
    )
    basis_vectors = torch.tensor(
        np.stack([a.cell.array for a in atoms_list]), dtype=torch.float32
    )

    forces = zbl_force.get_cartesian_forces(A, cartesian_positions, basis_vectors).detach()

    np.save(ZBL_CACHE, forces.numpy())
    return forces

def zbl_egnn_fractions(F_cart, atoms_list, t, force_activation_scale):
    # Step 1: Cartesian -> fractional forces: F_rel[b,i] = B[b]^{-1} @ F_cart[b,i]
    B = torch.tensor(
        np.stack([a.cell.array for a in atoms_list]), dtype=F_cart.dtype
    )
    B_inv_T = torch.linalg.inv(B).transpose(-1, -2)          # [batch, 3, 3]
    F_rel = F_cart @ B_inv_T                                   # [batch, N, 3]

    # Step 2: per-configuration norm over all atoms and spatial dims
    F_norm = torch.norm(F_rel.reshape(F_rel.shape[0], -1), dim=-1)  # [batch]

    # Step 3: sigmoid-like gate
    g = F_norm / (F_norm + force_activation_scale)

    # Step 4: time-weighted ZBL fraction
    zbl_percent = (1.0 - t) * g
    egnn_percent = 1.0 - zbl_percent

    return zbl_percent, egnn_percent


def calc_probs(sw_energies, T=300.0):
    kB = 8.617333e-5  # eV/K
    e_shifted = sw_energies - np.nanmin(sw_energies)
    boltzmann = np.exp(-e_shifted / (kB * T))
    boltzmann = np.clip(boltzmann, 0.0, None)
    return boltzmann / boltzmann.sum()


def sw_calculate_traj(atoms_list):
    if os.path.exists(SW_CACHE):
        print("Reading lammps data")
        data = np.load(SW_CACHE, allow_pickle=True).item()
        return data["energies"], data["forces"]

    print("Calculating lammps data")
    energies = []
    forces = []
    for atoms in atoms_list:
        e, f = lammpscalc_from_atoms(atoms)
        energies.append(e)
        forces.append(f)
    energies = np.array(energies)
    forces = np.array(forces)
    np.save(SW_CACHE, {"energies": energies, "forces": forces})
    return energies, forces


def lammpscalc_from_atoms(atoms):
    """Run a LAMMPS SW single-point calculation on an ASE Atoms object.
    Returns (energy, force_on_atom_0) as floats/arrays."""
    from lammps import lammps

    lmp = lammps(cmdargs=["-screen", "none", "-log", "none"])

    # Box: assume orthorhombic cell (diamond Si cubic cell)
    cell = atoms.cell.array
    xlo, xhi = 0.0, cell[0, 0]
    ylo, yhi = 0.0, cell[1, 1]
    zlo, zhi = 0.0, cell[2, 2]

    lmp.commands_string(f"""
units metal
atom_style atomic
boundary p p p
region box block {xlo} {xhi} {ylo} {yhi} {zlo} {zhi}
create_box 1 box
mass 1 28.0855
""")

    # Create atoms
    for pos in atoms.positions:
        lmp.command(f"create_atoms 1 single {pos[0]} {pos[1]} {pos[2]} units box")

    lmp.commands_string(f"""
pair_style sw
pair_coeff * * {SW_POTENTIAL} Si
run 0
""")

    energy = lmp.get_thermo("pe")

    # LAMMPS may reorder atoms; match by position to find the displaced atom
    n = lmp.get_natoms()
    x_ptr = lmp.extract_atom("x", 3)
    f_ptr = lmp.extract_atom("f", 3)
    target = atoms.positions[0]
    idx = min(range(n), key=lambda i: abs(x_ptr[i][0] - target[0])
                                     + abs(x_ptr[i][1] - target[1])
                                     + abs(x_ptr[i][2] - target[2]))
    force0_norm = np.sqrt(f_ptr[idx][0]**2 + f_ptr[idx][1]**2 + f_ptr[idx][2]**2)

    lmp.close()
    return energy, force0_norm

def plot_forces_and_probs(dist_list, sw_forces, zbl_forces, probs):
    fig, ax_force = plt.subplots(figsize=(8, 5))
    ax_prob = ax_force.twinx()

    l1, = ax_force.plot(dist_list, sw_forces,  label="SW force norm (eV/Å)",  color="tab:blue")
    l2, = ax_force.plot(dist_list, zbl_forces, label="ZBL force norm (eV/Å)", color="tab:orange")
    l3, = ax_prob.plot(dist_list, probs,       label="Probability (300 K)", color="black", linestyle="--")
    # Find the smallest distance where prob first exceeds 1e-5 (repulsive side)
    threshold_idx = np.argmax(probs > 1e-5)
    threshold_dist = dist_list[threshold_idx]
    l4 = ax_prob.axvline(threshold_dist, label="1 in 100 000", color="tab:red", linestyle=":")
    
    ax_force.text(threshold_dist, 1e2, f"{threshold_dist:.2f} Å", color="tab:red",
                  ha="right", va="bottom", fontsize=16)
    ax_force.set_yscale("log")
    ax_force.set_xlim(dist_list[0], 2.5)
    ax_force.set_ylim(1e-5, 1e5)
    ax_prob.set_xlim(dist_list[0], 2.5)
    ax_prob.set_ylim(0.0, 0.008)
    ax_prob.set_yticks([])

    ax_force.set_xlabel("Interatomic distance (Å)", fontsize=16)
    ax_force.set_ylabel("Force (eV/Å)", fontsize=16)
    ax_prob.set_ylabel("Probability", fontsize=16)
    ax_force.tick_params(labelsize=16)
    ax_force.set_title("SW and ZBL force norms vs interatomic distance", fontsize=16)

    lines = [l1, l2, l3, l4]
    xmin, xmax = dist_list[0], 2.5
    legend_x = (threshold_dist - xmin) / (xmax - xmin)
    ax_force.legend(lines, [l.get_label() for l in lines],
                    loc="lower left", fontsize=14)

    plt.tight_layout()
    plt.savefig("newgraph1_forceprob.pdf")
    #plt.show()
    plt.clf()


if __name__ == "__main__":
    main()
