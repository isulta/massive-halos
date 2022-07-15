zmax = 10
n_jobs = -1
verbose = 1
mempernode = 192 # memory per Frontera node in GB

codedir = '/work2/08044/tg873432/frontera/projects/massive-halos'
import sys
sys.path.append(codedir)

from mpi4py import MPI
from scripts.find_Rvir_sim import main, read_param_file, sim_path, params_from_filename, get_dir_size
import numpy as np
import os.path

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Load paths to all AGN sims on Frontera on rank 0, and broadcast to all ranks
if rank==0:
    allsims = np.genfromtxt(os.path.join(codedir, 'sims_frontera.txt'), dtype=str)
    allsimpaths = np.sort([ sim_path(params_from_filename(sim), 'frontera') for sim in allsims ])

else:
    allsimpaths = None
allsimpaths = comm.bcast(allsimpaths, root=0)

# Divide sims approximately evenly to all ranks (in round-robin ordering to try to balance data read by each rank)
ranksimpaths = list( allsimpaths[np.arange(rank, len(allsimpaths), size)] )
print(f'Rank {rank} of {size} beginning to process {len(ranksimpaths)} simulations.', flush=True)

# Process all simulations, adjusting n_jobs to account for nodes' memory limit
for snapdir in ranksimpaths:
    simsize, maxfilesize = get_dir_size(str(snapdir)) * 1e-9 * 4 # 4 * total size of simulation and max file size in GB
    print(f'\n{rank}: {snapdir} is {round(simsize,2)}GB', flush=True)
    if simsize > mempernode:
        maxsnapsize = maxfilesize * read_param_file(snapdir)['NumFilesPerSnapshot']
        n_jobs = int(mempernode/maxsnapsize)
        print(f'{rank}: using n_jobs={n_jobs} for {snapdir} due to memory exceeding memory/node', flush=True)
    else:
        n_jobs = -1
    main(snapdir, zmax, n_jobs, verbose)

print(f'Rank {rank} of {size} completed processing {len(ranksimpaths)} simulations.', flush=True)