#!/bin/bash
#SBATCH -A b1026
#SBATCH -p cosmoscompute
#SBATCH -N 2 # number of nodes
#SBATCH -n 63 # number of cores
#SBATCH --ntasks-per-node=32
#SBATCH -t 03:00:00 # <hh:mm:ss>
#SBATCH --mem=1400G #500G # Reserving a total amount of memory for each job within the array
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

module purge
# module load mpi/mpich-3.0.4-gcc-4.6.3
# module load mpi/mpich-3.0.4-gcc-6.4.0
# module load mpi/openmpi-4.0.5-gcc-10.2.0
module load mpi/mpich-3.3-gcc-6.4.0

# mpiexec -n $SLURM_NTASKS python scripts/gather_MPI_test.py
mpiexec -n $SLURM_NTASKS python scripts/profiles_zbins_MPI_m13_nofb.py
# srun python scripts/profiles_zbins_MPI_m13_nofb.py