'''Script to run find_Rvir for a given simulation and save output.
'''
from scripts.halo_analysis_scripts import *
from itk import h5_write_dict, h5_read_dict
import matplotlib.pyplot as plt
import sys
import os.path

def find_Rvir_halo(snapdir, halo, snapstart, snapend):
    res = {'snapnum':[], 'z':[], 'Rvir':[]}

    print(f'{halo}: Starting find_Rvir from snapnum={snapstart} to {snapend} in snapdir={snapdir}')
    try:
        for snapnum in range(snapstart, snapend+1):
            part = load_allparticles(snapdir, snapnum, loud=False)
            Rvir = find_Rvir(part, halo=halo, snapnum=snapnum)
            
            z = part[0]['Redshift']

            res['snapnum'].append(snapnum)
            res['z'].append(z)
            res['Rvir'].append(Rvir)

            print(f'{halo}: snapnum={snapnum}, z={z}, Rvir={Rvir} physical kpc')
    except:
        pass

    h5_write_dict(f'data/findRvir_{halo}.h5', res, 'res')

def plot_Rvir(halo):
    res = h5_read_dict(f'data/findRvir_{halo}.h5', 'res')
    plt.figure()
    plt.plot(res['z'], res['Rvir'])
    plt.xlabel('z')
    plt.ylabel('Rvir (physical kpc)')
    plt.title(f'findRvir_{halo}')
    plt.savefig(f'Figures/findRvir_{halo}.png')
    plt.close()

if __name__ == '__main__':
    simdir = sys.argv[1]
    redshifts = redshifts_snapshots(simdir)
    zmax = 5
    snapstart, snapend = np.flatnonzero(redshifts <= zmax).min(), np.flatnonzero(redshifts <= zmax).max()

    snapdir = os.path.join(simdir, 'output/')
    snapdir = snapdir if os.path.exists(snapdir) else simdir
    halo = os.path.basename(simdir)
    find_Rvir_halo(snapdir, halo, snapstart, snapend)
    
    plot_Rvir(halo)