'''Script to combine all find_Rvir_SO results for FIRE-3 simulations and save output.
'''
from scripts.halo_analysis_scripts import *

fire3 = [os.path.join('data/Rvir/', f) for f in os.listdir('data/Rvir/') if 'fire3' in f]

allres = {}

for f in fire3:
    simname = f.split('findRvirSO_')[1].split('.h5')[0]
    d = h5todict(f)
    # allres[simname] = {d['snapnum'][i] : {k:d[k][i] for k in d.keys()} for i in range(len(d['snapnum']))}
    allres[simname] = {d['snapnum'][i] : {k:d[k][i] for k in ['Mvir', 'Rvir', 'posC', 'z']} for i in range(len(d['snapnum']))}
    for k in ['HubbleParam', 'Omega_Baryon', 'Omega_Lambda', 'Omega_Matter', 'Omega_Radiation']: allres[simname][k] = d[k][-1].item()

fname = f'/work2/08044/tg873432/frontera/HaloCenteringCache/fire3_mainsimulationset.h5'
dicttoh5(allres, fname, mode='w')
print(f'Saved to {fname}', flush=True)