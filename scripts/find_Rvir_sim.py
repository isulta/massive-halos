'''Script to run find_Rvir_SO for a given simulation and save output.
'''
from scripts.halo_analysis_scripts import *
import sys

def find_Rvir_halo(snapdir, halo, snapstart, snapend, resume=True):
    res = {'snapnum':[], 'z':[], 'Rvir':[], 'Mvir':[]}

    print(f'{halo}: Starting find_Rvir from snapnum={snapstart} to {snapend} in snapdir={snapdir}')

    if resume and os.path.exists(f'data/Rvir/findRvirSO_{halo}.h5'):
        res = h5todict(f'data/Rvir/findRvirSO_{halo}.h5')
        snapstart = res['snapnum'][-1]+1
        print(f'{halo}: findRvirSO file found. Resuming find_Rvir from snapnum={snapstart} to {snapend}')
    
    for snapnum in range(snapstart, snapend+1):
        try:
            part = load_allparticles(snapdir, snapnum, loud=False)
        except:
            continue
        Rvir, Mvir = find_Rvir_SO(part, halo=halo, snapnum=snapnum)
        
        z = part[0]['Redshift']

        res['snapnum'].append(snapnum)
        res['z'].append(z)
        res['Rvir'].append(Rvir)
        res['Mvir'].append(Mvir)

        print(f'{halo}: snapnum={snapnum}, z={z}, Rvir={Rvir} physical kpc, Mvir={Mvir:e} Msun', flush=True)

        dicttoh5(res, f'data/Rvir/findRvirSO_{halo}.h5', mode='w')
        plot_Rvir(halo)

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

    halo = os.path.basename(snapdir)
    find_Rvir_halo(snapdir, halo, snapstart, snapend)
    plot_Rvir(halo)