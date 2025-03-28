zmax = 4
n_jobs = -1
verbose = 1
mempernode = 192 # memory per Frontera node in GB
dofire3 = True

from mpi4py import MPI
from scripts.find_Rvir_sim import main, read_param_file, sim_path, params_from_filename, get_dir_size, sim_path_fire3
import numpy as np
import time
from datetime import timedelta

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Load paths to all AGN sims on Frontera on rank 0, and broadcast to all ranks
if rank==0:
    if dofire3:
        allsims = np.genfromtxt('sims_frontera_fire3.txt', dtype=str)
        allsimpaths = np.sort([ sim_path_fire3(sim) for sim in allsims ])
    else:
        allsims = np.genfromtxt('sims_frontera.txt', dtype=str)
        allsimpaths = np.sort([ sim_path(params_from_filename(sim), 'frontera') for sim in allsims ])
else:
    allsimpaths = None
allsimpaths = comm.bcast(allsimpaths, root=0)

# Divide sims approximately evenly to all ranks (in round-robin ordering to try to balance data read by each rank)
ranksimpaths = list( allsimpaths[np.arange(rank, len(allsimpaths), size)] )
print(f'Rank {rank} of {size} beginning to process {len(ranksimpaths)} simulations.', flush=True)

# Process all simulations, adjusting n_jobs to account for nodes' memory limit
start_time = time.time()
for snapdir in ranksimpaths:
    simsize, maxfilesize = get_dir_size(str(snapdir)) * 1e-9 * 4 # 4 * total size of simulation and max file size in GB
    print(f'\n{rank}: {snapdir} is {round(simsize,2)}GB', flush=True)
    if simsize > mempernode:
        maxsnapsize = maxfilesize * read_param_file(snapdir)['NumFilesPerSnapshot']
        n_jobs = int(mempernode/maxsnapsize)
        print(f'{rank}: using n_jobs={n_jobs} for {snapdir} due to memory exceeding memory/node', flush=True)
    else:
        n_jobs = -1
    main(snapdir, zmax, n_jobs, verbose, onlyFindRvir=False)

print(f'Rank {rank} of {size} completed processing {len(ranksimpaths)} simulations in {timedelta(seconds=time.time()-start_time)}', flush=True)