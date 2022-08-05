'''Script to run find_Rvir_SO for a given simulation and save output.
'''
from scripts.halo_analysis_scripts import *
from abg_python.system_utils import getfinsnapnum
from joblib import Parallel, delayed

def find_Rvir_halo(snapdir, halo, snapstart, snapend, resume=True):
    res = {'snapnum':[], 'z':[], 'Rvir':[], 'Mvir':[]}

    print(f'{halo}: Starting find_Rvir from snapnum={snapstart} to {snapend} in snapdir={snapdir}')

    if resume and os.path.exists(f'data/Rvir/findRvirSO_{halo}.h5'):
        res = h5todict(f'data/Rvir/findRvirSO_{halo}.h5')
        snapstart = res['snapnum'][-1]+1
        print(f'{halo}: findRvirSO file found. Resuming find_Rvir from snapnum={snapstart} to {snapend}')
    
    for snapnum in range(snapstart, snapend+1):
        part = load_allparticles(snapdir, snapnum, loud=False)
        Rvir, Mvir = find_Rvir_SO(part, halo=halo, snapnum=snapnum)
        
        z = part[0]['Redshift']

        res['snapnum'].append(snapnum)
        res['z'].append(z)
        res['Rvir'].append(Rvir)
        res['Mvir'].append(Mvir)

        print(f'{halo}: snapnum={snapnum}, z={z}, Rvir={Rvir} physical kpc, Mvir={Mvir:e} Msun', flush=True)

        dicttoh5(res, f'data/Rvir/findRvirSO_{halo}.h5', mode='w')
        plot_Rvir(halo)

def find_Rvir_snapnum(snapdir, snapnum):
    try:
        part = load_allparticles(snapdir, snapnum, loud=False)
    except OSError: #snapshot not found or snapshot subfile corrupted
        print(f'{snapdir}: failed to load snapshot {snapnum}', flush=True)
        return -1, -1, -1, -1
    Rvir, Mvir = find_Rvir_SO(part)
    z = part[0]['Redshift']
    return snapnum, z, Rvir, Mvir

def find_Rvir_halo_parallel(snapdir, halo, snapstart, snapend, n_jobs=-4, verbose=10):
    print(f'{halo}: Starting find_Rvir_SO from snapnum={snapstart} to {snapend} in snapdir={snapdir}', flush=True)
    res_par = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(find_Rvir_snapnum)(snapdir, i) for i in range(snapstart, snapend+1))

    res = {
        'snapnum': [ t[0] for t in res_par if t[0] != -1 ], 
        'z': [ t[1] for t in res_par if t[0] != -1 ], 
        'Rvir': [ t[2] for t in res_par if t[0] != -1 ], 
        'Mvir': [ t[3] for t in res_par if t[0] != -1 ]
        }
    
    fname = f'data/Rvir/findRvirSO_{halo}.h5'
    dicttoh5(res, fname, mode='w')
    print(f'Saved to {fname}', flush=True)

def plot_Rvir(halo):
    res = h5todict(f'data/Rvir/findRvirSO_{halo}.h5')
    plt.figure()
    plt.plot(res['z'], res['Rvir'], '-o', ms=4)
    plt.xlabel('z')
    plt.ylabel('Rvir (physical kpc)')
    plt.title(f'find_Rvir_SO\n{halo}', fontsize=7)
    plt.savefig(f'Figures/findRvirSO/findRvirSO_{halo}.png')
    plt.close()

def calculate_profiles_snapnum(snapdir, snapnum):
    try:
        part = load_allparticles(snapdir, snapnum, Rvir='find_Rvir_SO', loud=False)
    except OSError: #snapshot not found or snapshot subfile corrupted
        print(f'{snapdir}: failed to load snapshot {snapnum}', flush=True)
        return None
    except KeyError: #snapshot_utils.py", line 150, in openSnapshot
        print(f'{snapdir}: failed to load snapshot {snapnum} KeyError', flush=True)
        return None
    with np.errstate(divide='ignore', invalid='ignore'): res = profiles(part)
    return res

def calculate_profiles_halo_parallel(snapdir, halo, snapstart, snapend, n_jobs=-4, verbose=10):
    print(f'{halo}: Calculating profiles from snapnum={snapstart} to {snapend} in snapdir={snapdir}', flush=True)
    res_par = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(calculate_profiles_snapnum)(snapdir, i) for i in range(snapstart, snapend+1))

    res = { 'SnapNum' + str(snapnum).zfill(3): res_par[i] for i,snapnum in enumerate(range(snapstart, snapend+1)) if res_par[i] is not None }
    
    fname = f'data/profiles/profiles_{halo}.h5'
    dicttoh5(res, fname, mode='w')
    print(f'Saved to {fname}', flush=True)

def main(snapdir, zmax, n_jobs=-4, verbose=10, onlyFindRvir=True):
    snapdir = snapdir.rstrip('/')
    zmax = float(zmax)

    redshifts = redshifts_snapshots(snapdir)
    snapstart, snapend = np.flatnonzero(redshifts <= zmax).min(), np.flatnonzero(redshifts <= zmax).max()
    try:
        snapend = getfinsnapnum(os.path.join(snapdir, 'output/') if os.path.exists(os.path.join(snapdir, 'output/')) else snapdir)
    except ValueError:#improper snapshot file name
        print(f'{snapdir}: getfinsnapnum error', flush=True)
        return

    halo = os.path.basename(snapdir)
    if onlyFindRvir:
        find_Rvir_halo_parallel(snapdir, halo, snapstart, snapend, n_jobs, verbose)
        plot_Rvir(halo)
    else:
        calculate_profiles_halo_parallel(snapdir, halo, snapstart, snapend, n_jobs, verbose)

if __name__ == '__main__':
    import sys
    main(sys.argv[1], sys.argv[2], onlyFindRvir=False)