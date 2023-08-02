'''Script to combine existing fire3 cache file (find_Rvir_SO results for FIRE-3 simulations) with new snapshots from `'data/simcache'` and save output.
'''
from scripts.halo_analysis_scripts import *
import re

fire3 = [os.path.join('data/simcache', f) for f in os.listdir('data/simcache') if 'fire3' in f]

fname = f'/home/ias627/fire3_mainsimulationset.h5'
allres = h5todict(fname)

for f in tqdm(fire3):
    simname, snapnum = re.search(r".*simcache_(.*)_\d+\.h5", f).group(1), re.search(r".*_(\d+)\.h5", f).group(1)
    res = h5todict(f)
    if snapnum in allres[simname]:
        assert allres[simname][snapnum]['Mvir'] == res['pro']['Mvir']
        assert allres[simname][snapnum]['Rvir'] == res['pro']['Rvir']
        assert np.array_equal(allres[simname][snapnum]['posC'], res['pro']['posC'])
        assert allres[simname][snapnum]['z'] == res['pro']['Redshift']
    else:
        allres[simname][snapnum] = {
            'Mvir': res['pro']['Mvir'],
            'Rvir': res['pro']['Rvir'],
            'posC': res['pro']['posC'], 
            'z':    res['pro']['Redshift']
        }

fname = f'/home/ias627/fire3_mainsimulationset_new.h5'
dicttoh5(allres, fname, mode='w')
print(f'Saved to {fname}', flush=True)