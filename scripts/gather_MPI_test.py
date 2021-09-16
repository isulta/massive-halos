from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
ranks = comm.Get_size()

def flatten(t):
    return [item for sublist in t for item in sublist]

import numpy as np
x = np.random.randint(0,100,rank)
# x = [(1,2),(3,4)]
print('x', rank, x, flush=True)

allx = comm.gather(x, root=0)
if rank == 0:
    print(allx, flush=True)
    allx_flat = flatten(allx)
    print(allx_flat, flush=True)
    print(len(allx_flat), flush=True)