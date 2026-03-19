#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00-24:00
#SBATCH --account=def-cotemich
#SBATCH --output=%N-%j.out

# Preparing env 
source /home/pomax/bin/env_gpu
export OMP_NUM_THREADS=1

python run_1000.py
