from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
ranks = comm.Get_size()

import time
from datetime import datetime, timedelta
from scripts.halo_analysis_scripts import *

def printr(s, root=0):
    if rank == root:
        print(f'[{datetime.now()}] rank {root}: {s}', flush=True)
    comm.Barrier()

def flatten(t):
    return [item for sublist in t for item in sublist]

def profiles_zbins_MPI(snapdir, redshifts, Rvir_allsnaps, zmin=1, zmax=4, zbinwidth=0.5, outfile=None):
    '''MPI implementation of `profiles_zbins`'''
    if rank == 0:
        allprofiles = {}
    comm.Barrier()
    for z0 in np.arange(zmin,zmax,zbinwidth):
        allprofiles_z0_rank = []
        z1 = z0 + zbinwidth
        
        printr(f'Beginning bin from z={z0} to {z1}.')
        snapnums_bin = np.flatnonzero(inrange(redshifts, (z0,z1), right_bound_inclusive=False))
        snapnum_median = snapnums_bin[len(snapnums_bin)//2]
        Rvir = Rvir_allsnaps[snapnum_median]
        printr(f'Median redshift is {redshifts[snapnum_median]} with snapnum {snapnum_median} and virial radius {Rvir} kpc.')
        
        printr(f'Computing profiles for snapshots {snapnums_bin.min()} to {snapnums_bin.max()}.')
        printr(np.array_split(snapnums_bin, ranks)[rank], rank)
        for snapnum in np.array_split(snapnums_bin, ranks)[rank]:
            p0 = load_p0(snapdir, snapnum, Rvir=Rvir, loud=0)
            allprofiles_z0_rank.append( profiles(p0) )
        
        allprofiles_z0 = comm.gather(allprofiles_z0_rank, root=0)
        if rank == 0:
            allprofiles[z0] = flatten(allprofiles_z0)
        comm.Barrier()
    
    if outfile and (rank==0):
        pickle_save_dict(outfile, {'allprofiles':allprofiles})
    comm.Barrier()

if __name__ == '__main__':
    printr(f'Beginning script...'); start_script = time.time()

    for halo, snapdir in Quest_sims['nofb'].items():
        simname = os.path.basename(snapdir)
        printr(f'Starting {simname}...'); start_sim = time.time()
        
        # Load and save Rvir and redshift data
        printr(f'Loading/saving Rvir and redshift data...'); start = time.time()
        if rank == 0:
            Rvir_allsnaps, redshifts = load_Rvir_allsnaps(snapdir, Quest_nofb_m13_ahf(halo), 'data/', simname)
        else:
            Rvir_allsnaps, redshifts = None, None
        Rvir_allsnaps = comm.bcast(Rvir_allsnaps, root=0)
        redshifts = comm.bcast(redshifts, root=0)
        printr(f'Finished loading/saving Rvir and redshift data in {timedelta(seconds=time.time()-start)}')
        
        # Compute profiles for all snapshots in each redshift bin
        printr(f'Computing profiles for all snapshots in each redshift bin...'); start_profile = time.time()
        profiles_zbins_MPI(snapdir, redshifts, Rvir_allsnaps, outfile=f'data/{simname}_allprofiles_widezbins.pkl')
        printr(f'Finished Computing profiles for all snapshots in each redshift bin in {timedelta(seconds=time.time()-start_profile)}')

        printr(f'Finished {simname} in {timedelta(seconds=time.time()-start_sim)}')
    
    printr(f'Finished script in {timedelta(seconds=time.time()-start_script)}')