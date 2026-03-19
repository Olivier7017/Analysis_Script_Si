from pathlib import Path
import numpy as np
import ase.io
from ase.calculators.singlepoint import SinglePointCalculator


def main():
    structs = ["Si8", "Si64", "Si216", "Si4Ge4",
              "Si32Ge32", "Si108Ge108"]
    nval = 5
    output_traj_folder = "/home/olivi/projects/02-Simodels-ZBL/samples/ref/"
    for struct in structs:
        folder = "SiGe_lammps_ref" if "Ge" in struct else "Si_lammps_ref"
        for ival in range(1, nval+1):
            fn = Path(folder) / f"{struct}_{ival}.dump"
            traj = read_one_dump(fn)
            traj_name = fn.name.split(".")[0] + ".traj"
            traj_fn = output_traj_folder + traj_name
            ase.io.write(traj_fn, traj)

def get_energy(logfn, timestep_spacing=100):
    data = {}
    in_thermo = False
    print("READING ", logfn)
    with open(logfn) as f:
        for line in f:
            if "Step" in line and "E_pair" in line:
                cols = line.split()
                step_col, eng_col = cols.index("Step"), cols.index("E_pair")
                in_thermo = True
                data = {}
                continue
            if in_thermo:
                if line.startswith("Loop"):
                    in_thermo = False
                    continue
                try:
                    vals = line.split()
                    step = int(vals[step_col])
                    if step % timestep_spacing == 0:
                        data[step] = float(vals[eng_col])
                except (ValueError, IndexError):
                    in_thermo = False
    return np.array([data[s] for s in sorted(data)])


def read_one_dump(filename, specorder=None):
    filename = Path(filename)
    if specorder is None:
        specorder = ["Si", "Ge"] if "SiGe" in str(filename) else ["Si"]
    frames = ase.io.read(filename, format="lammps-dump-text", index=":", specorder=specorder)
    energies = get_energy(filename.with_suffix(".log"))
    for atoms, energy in zip(frames, energies):
        forces = atoms.get_forces()
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    return frames


if __name__=="__main__":
    main()
