zmax = 10
n_jobs = -1
verbose = 1
mempernode = 192 # memory per Frontera node in GB

codedir = '/work2/08044/tg873432/frontera/projects/massive-halos'
import sys
sys.path.append(codedir)

from mpi4py import MPI
from scripts.find_Rvir_sim import main, read_param_file
import numpy as np
import os.path

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

'''Sarah functions'''
# Takes a simulation filename (like the ones written in the .txt files) and converts it into a dictionary of parameters
def params_from_filename(filename):
    params = {}
    plist = filename.split('_')
    params['haloID']              = plist[0]
    params['mass_res']            = plist[1]
    params['fb_method']           = plist[2]
    if plist[2] != 'control':
        params['alpha_disk_factor']   = plist[3]
        params['gravaccretion_model'] = plist[4]
        params['accretion_factor']    = plist[5]
        params['v_wind']              = plist[6]
        params['cr_loading']          = plist[7]
        params['seed_mass']           = plist[8]
        params['stellar_m_per_seed']  = plist[9]
        params['spawned_wind_mass']   = plist[10]
        params['f_accreted']          = plist[11]
        params['t_wind_spawn']        = plist[12]
        params['fluxmom_factor']      = plist[13]

    return params

# Reverse of above
def filename_from_params(params):
    filename = ''
    for param in params.keys():
        filename = filename+params[param]+'_'
    filename = filename[:-1]  # To remove last underscore

    return filename

# Returns the names + paths of the "control" set, plus the snapshot indices in the 600-snapshot runs that correspond to the same redshifts as the 60-snapshot runs
# Note: requires you to have the 60-snapshot timing file, which you can find in any of the AGN feedback simulation folders
def controlsims():
    simnames = ['m10q_m2e2', 'm10v_m2e2',
                'm11a_m2e3', 'm11b_m2e3', 'm11d_m7e3', 'm11e_m7e3', 
                'm11f_m1e4', 'm11h_m7e3', 'm11i_m7e3', 'm11q_m7e3', 
                'm12b_m6e4', 'm12f_m6e4', 'm12i_m6e4', 'm12m_m6e4', 
                'm12q_m6e4', 'm12r_m6e4', 'm12w_m6e4',
                'm13h02_m3e5', 'm13h29_m3e5', 'm13h113_m3e5', 'm13h206_m3e5']
    controlpaths = ['core/m10q_res250', 'CR_suite/m10v/cr_700', 
                    'CR_suite/m11a_res2100/cr_700', 'CR_suite/m11b/cr_700', 'CR_suite/m11d/cr_700', 'CR_suite/m11e_res7100/cr_700', 
                    'CR_suite/m11f/cr_700', 'CR_suite/m11h/cr_700', 'CR_suite/m11i_res7100/cr_700', 'core/m11q_res7100',
                    'CR_suite/m12b_res7100/cr_700', 'CR_suite/m12f_mass56000/cr_700', 'CR_suite/m12i_mass56000/cr_700', 'CR_suite/m12m_mass56000/cr_700', 
                    'CR_suite/m12q_res57000/cr_700', 'CR_suite/m12r_res7100/cr_700', 'CR_suite/m12w_res7100/cr_700',
                    'MassiveFIRE/A8_res33000', 'MassiveFIRE/A2_res33000', 'MassiveFIRE/A4_res33000', 'MassiveFIRE/A1_res33000']
    
    asnaps = np.loadtxt('../data/snapshot_scale-factors.txt')
    snaps_600 = np.loadtxt('../../fire2/core/m12i_res57000/snapshot_times.txt')
    asnaps_600 = snaps_600[:,1]
    isnaps_600 = snaps_600[:,0]
    isnaps = np.interp(asnaps, asnaps_600, isnaps_600).astype(int)
    
    return simnames, controlpaths, isnaps

# Returns the path to a simulation data directory given the parameter dictionary and the machine it lives on
def sim_path(params, machine=None):
    if params['fb_method'] == 'control':
        filebase = '../../fire2/'
        allcontrols = controlsims()
        folder = allcontrols[1][allcontrols[0].index(params['haloID']+'_'+params['mass_res'])]
        return filebase + folder
    else:
        if machine == 'frontera':
            filebase = '/scratch3/01799/phopkins/bhfb_suite_done/'
        elif machine == 'cca':
            filebase = '../../fire2/AGN_suite/'
        elif machine == 'lou':
            filebase = '/u/pfhopki1/agn_fb/'
        else:
            print("Please specify a valid machine, e.g. 'frontera', 'lou', 'cca'")
            return None

        folder = params['haloID']+'_'+params['mass_res']+'/'
        return filebase + folder + filename_from_params(params)
'''End Sarah functions'''

'''Find total size in bytes of (possibly nested) directory.
Adapted from https://stackoverflow.com/a/1392549'''
def get_size(start_path = '.'):
    total_size = 0
    max_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                s = os.path.getsize(fp)
                total_size += s
                max_size = max(max_size, s)

    return np.array([total_size, max_size])

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
    simsize, maxfilesize = get_size(str(snapdir)) * 1e-9 * 4 # 4 * total size of simulation and max file size in GB
    print(f'\n{rank}: {snapdir} is {round(simsize,2)}GB', flush=True)
    if simsize > mempernode:
        maxsnapsize = maxfilesize * read_param_file(snapdir)['NumFilesPerSnapshot']
        n_jobs = int(mempernode/maxsnapsize)
        print(f'{rank}: using n_jobs={n_jobs} for {snapdir} due to memory exceeding memory/node', flush=True)
    else:
        n_jobs = -1
    main(snapdir, zmax, n_jobs, verbose)

print(f'Rank {rank} of {size} completed processing {len(ranksimpaths)} simulations.', flush=True)