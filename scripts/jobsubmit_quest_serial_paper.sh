#!/bin/bash
#SBATCH -A b1026
#SBATCH -p cosmoscompute
#SBATCH --job-name="processfire3sims"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52
#SBATCH -t 25:00:00 # <hh:mm:ss>
#SBATCH --mem=1400G # Reserving a total amount of memory for each job within the array
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=imransultan2025@u.northwestern.edu

arg1=$1

module purge

date

# cd /home/ias627/projects/massive_halos/notebooks

# python cachesnap.py

cd /home/ias627/projects/massive_halos

python scripts/particle_tracking.py "$arg1"

date