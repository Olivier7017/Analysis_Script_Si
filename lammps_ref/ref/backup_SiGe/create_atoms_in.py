from pathlib import Path
from ase.io.lammpsdata import write_lammps_data
from ase.io import read


fold = Path("/home/olivi/Data/Structures/")
fns = ["Si8.cif", "Si64.cif", "Si216.cif"]

for fn in fns:
    in_name = fold / fn
    out_name = Path(fn).with_suffix(".in")
    atoms = read(in_name)
    write_lammps_data(out_name, atoms, velocities=True, atom_style="atomic", specorder=["Si"])

                      
