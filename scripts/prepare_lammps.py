import re
import sys
import shutil
from pathlib import Path

from ase.io import read

from energydist_calc import (
    LAMMPS_REF_BASE,
    create_atoms_in,
    create_lammps_input,
    _SUPERCELL_TO_NATOMS,
)

JOB_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=lammps_eval
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=lammps_%j.out

lammps -in lammps_input.in
"""


def _parse_category(fn):
    """Infer category (e.g. Si111, SiGe222) from trajectory filename."""
    stem = Path(fn).stem
    m = re.search(r'(SiGe|Si)(111|222|333)', stem)
    if not m:
        raise ValueError(f"Cannot infer category from filename: {fn!r}")
    return m.group(1) + m.group(2)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <trajfile>")
        sys.exit(1)

    fn = sys.argv[1]
    CALC_FOLDER = Path(Path(fn).stem)
    category = _parse_category(fn)
    is_sige = category.startswith("SiGe")
    elements = ["Si", "Ge"] if is_sige else ["Si"]

    atoms = read(fn, index=":")

    CALC_FOLDER.mkdir(parents=True, exist_ok=True)

    create_atoms_in(atoms, calc_folder=CALC_FOLDER, prefix=category, specorder=elements)

    potential_src = LAMMPS_REF_BASE / (
        "ref/SiGe_lammps_ref/SiGe.sw" if is_sige else "ref/Si_lammps_ref/Si.sw"
    )
    shutil.copy(potential_src, CALC_FOLDER)

    log_file = CALC_FOLDER / "energies.log"
    create_lammps_input(
        len(atoms),
        calc_folder=CALC_FOLDER,
        potential=potential_src,
        atoms_prefix=category,
        log_file=log_file.resolve(),
        elements=elements,
    )

    (CALC_FOLDER / "job.sh").write_text(JOB_TEMPLATE)

    print(f"LAMMPS inputs written to {CALC_FOLDER}/")
    print(f"  {len(atoms)} structures as {category}_1.in ... {category}_{len(atoms)}.in")
    print(f"  lammps_input.in, {potential_src.name}, job.sh")
    print(f"Run: cd {CALC_FOLDER} && sbatch job.sh")


if __name__ == "__main__":
    main()
