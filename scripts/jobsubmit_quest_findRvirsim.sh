#!/bin/bash
#SBATCH -A b1026
#SBATCH -p cosmoscompute
#SBATCH --job-name="findRvirsim"
#SBATCH -N 1 # number of nodes
#SBATCH -t 20:00:00 # <hh:mm:ss>
#SBATCH --mem=120G # Reserving a total amount of memory for each job within the array
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --mail-type=BEGIN,END,FAIL

module purge

ZMAX=5

cd /home/ias627/projects/massive_halos/

for i in /projects/b1026/isultan/halos/*; do
    date
    echo $i
    time python scripts/find_Rvir_sim.py $i $ZMAX
    date
    echo
done

for i in /projects/b1026/isultan/*_noAGNfb; do
    date
    echo $i
    time python scripts/find_Rvir_sim.py $i $ZMAX
    date
    echo
done