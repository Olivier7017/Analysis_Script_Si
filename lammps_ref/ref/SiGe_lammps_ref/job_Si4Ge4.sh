#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --mem-per-cpu=2048M
#SBATCH --time=00-00:05
#SBATCH --account=def-cotemich
#SBATCH --output=Si4Ge4.log

source /home/pomax/bin/env_gpu
export OMP_NUM_THREADS=1

for i in {1..5}; do
    srun --exclusive -n1 -c1 lmp -in Si4Ge4_validation$i/lammps_input.in &
done

wait
