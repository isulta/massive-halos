#!/bin/bash
#SBATCH -J processfire3sims # Job name
#SBATCH -o myjob.o%j        # Name of stdout output file
#SBATCH -e myjob.e%j        # Name of stderr error file
#SBATCH -p nvdimm           # Queue (partition) name
#SBATCH -N 1                # Total # of nodes 
#SBATCH --ntasks-per-node=1 # MPI tasks per node
#SBATCH --cpus-per-task=112 # Cores per task
#SBATCH -t 06:00:00         # Run time (hh:mm:ss)
#SBATCH -A AST21010         # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-type=all     # Send email at begin and end of job
#SBATCH --mail-user=imransultan2025@u.northwestern.edu

cd /work2/08044/tg873432/frontera/projects/massive-halos/notebooks

date

python cachesnap.py

date