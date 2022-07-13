#!/bin/bash
#----------------------------------------------------
# Example Slurm job script
# for TACC Frontera CLX nodes
#
#   *** Hybrid Job in Normal Queue ***
# 
#       This sample script specifies:
#         10 nodes (capital N)
#         40 total MPI tasks (lower case n); this is 4 tasks/node
#         14 OpenMP threads per MPI task (56 threads per node)
#
# Last revised: 20 May 2019
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch clx.hybrid.slurm" on Frontera login node.
#
#   -- Use ibrun to launch MPI codes on TACC systems.
#      Do NOT use mpirun or mpiexec.
#
#   -- In most cases it's best to keep
#      ( MPI ranks per node ) x ( threads per rank )
#      to a number no more than 56 (total cores).
#
#   -- If you're running out of memory, try running
#      fewer tasks and/or threads per node to give each 
#      process access to more memory.
#
#   -- IMPI does sensible process pinning by default.
#
#----------------------------------------------------

#SBATCH -J findRvirsim      # Job name
#SBATCH -o myjob.o%j        # Name of stdout output file
#SBATCH -e myjob.e%j        # Name of stderr error file
#SBATCH -p development      # Queue (partition) name
#SBATCH -N 40               # Total # of nodes 
#SBATCH --ntasks-per-node=1 # MPI tasks per node
#SBATCH --cpus-per-task=56  # Cores per task
#SBATCH -t 02:00:00         # Run time (hh:mm:ss)
#SBATCH -A AST21010         # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-type=all     # Send email at begin and end of job
#SBATCH --mail-user=imransultan2025@u.northwestern.edu

# Any other commands must follow all #SBATCH directives...
module list
pwd
date

# Set thread count (default value is 1)...
# export OMP_NUM_THREADS=14

# Launch MPI code... 
ibrun python /work2/08044/tg873432/frontera/projects/massive-halos/scripts/process_frontera_sims.py

date