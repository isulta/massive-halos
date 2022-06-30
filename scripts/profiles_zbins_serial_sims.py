from scripts.halo_analysis_scripts import *
from itk import h5_read_dict
import sys
import os.path

if __name__ == '__main__':
    simdir = sys.argv[1]
    redshifts = redshifts_snapshots(simdir)

    snapdir = os.path.join(simdir, 'output/')
    halo = os.path.basename(simdir)

    res = h5_read_dict(f'data/findRvir_{halo}.h5', 'res')
    Rvir_allsnaps = dict(zip(res['snapnum'], res['Rvir']))

     # Compute profiles for all snapshots in each redshift bin
    print(f'{halo}: Starting profiles_zbins in snapdir={snapdir}')
    profiles_zbins(snapdir, redshifts, Rvir_allsnaps, outfile=f'data/{halo}_allprofiles_widezbins.h5')