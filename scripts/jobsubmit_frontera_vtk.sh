#!/bin/bash
#----------------------------------------------------
# Example Slurm job script
# for TACC Frontera CLX nodes
#
#   *** Multicore Job in Dev Queue ***
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

#SBATCH -J FIREtoVTK      # Job name
#SBATCH -o myjob.o%j        # Name of stdout output file
#SBATCH -e myjob.e%j        # Name of stderr error file
#SBATCH -p development      # Queue (partition) name
#SBATCH -N 1                # Total # of nodes 
#SBATCH --ntasks-per-node=1 # MPI tasks per node
#SBATCH --cpus-per-task=56  # Cores per task
#SBATCH -t 02:00:00         # Run time (hh:mm:ss)
#SBATCH -A AST21010         # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-type=all     # Send email at begin and end of job
#SBATCH --mail-user=imransultan2025@u.northwestern.edu

# Any other commands must follow all #SBATCH directives...
export PYTHONPATH=${WORK}/tools/itk:${WORK}/projects/massive-halos
module list
pwd
date

# Set thread count (default value is 1)...
# export OMP_NUM_THREADS=14

# Launch multicore code... 
python ${WORK}/projects/massive-halos/scripts/FIREtoVTK_frontera.py "m13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000"
python ${WORK}/projects/massive-halos/scripts/FIREtoVTK_frontera.py "m13h206_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5"
python ${WORK}/projects/massive-halos/scripts/FIREtoVTK_frontera.py "m13h029_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000"

date