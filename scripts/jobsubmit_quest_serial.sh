#!/bin/bash
#SBATCH -A b1026
#SBATCH -p cosmoscompute
#SBATCH -N 1 # number of nodes
#SBATCH -t 10:00:00 # <hh:mm:ss>
#SBATCH --mem=350G # Reserving a total amount of memory for each job within the array
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

module purge

python scripts/profiles_zbins_serial_m13_nofb.py
