'''Script to run find_Rvir_SO for a given simulation and save output.
'''
from scripts.halo_analysis_scripts import *
from abg_python.system_utils import getfinsnapnum
import sys
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
    part = load_allparticles(snapdir, snapnum, loud=False)
    Rvir, Mvir = find_Rvir_SO(part)
    z = part[0]['Redshift']
    return snapnum, z, Rvir, Mvir

def find_Rvir_halo_parallel(snapdir, halo, snapstart, snapend, n_jobs=-4, verbose=10):
    print(f'{halo}: Starting find_Rvir_SO from snapnum={snapstart} to {snapend} in snapdir={snapdir}')
    res_par = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(find_Rvir_snapnum)(snapdir, i) for i in range(snapstart, snapend+1))

    res = {
        'snapnum': [ t[0] for t in res_par ], 
        'z': [ t[1] for t in res_par ], 
        'Rvir': [ t[2] for t in res_par ], 
        'Mvir': [ t[3] for t in res_par ]
        }
    
    fname = f'data/Rvir/findRvirSO_{halo}.h5'
    dicttoh5(res, fname, mode='w')
    print(f'Saved to {fname}')

def plot_Rvir(halo):
    res = h5todict(f'data/Rvir/findRvirSO_{halo}.h5')
    plt.figure()
    plt.plot(res['z'], res['Rvir'])
    plt.xlabel('z')
    plt.ylabel('Rvir (physical kpc)')
    plt.title(f'findRvirSO_{halo}')
    plt.savefig(f'Figures/findRvirSO/findRvirSO_{halo}.png')
    plt.close()

if __name__ == '__main__':
    snapdir = sys.argv[1].rstrip('/')
    zmax = float(sys.argv[2])

    redshifts = redshifts_snapshots(snapdir)
    snapstart, snapend = np.flatnonzero(redshifts <= zmax).min(), np.flatnonzero(redshifts <= zmax).max()
    snapend = getfinsnapnum(os.path.join(snapdir, 'output/') if os.path.exists(os.path.join(snapdir, 'output/')) else snapdir)

    halo = os.path.basename(snapdir)
    find_Rvir_halo_parallel(snapdir, halo, snapstart, snapend)
    plot_Rvir(halo)