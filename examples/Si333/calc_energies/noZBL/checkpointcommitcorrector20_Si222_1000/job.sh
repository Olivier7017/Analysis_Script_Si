#!/bin/bash
#SBATCH --job-name=lammps_eval
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=lammps_%j.out

lammps -in lammps_input.in
