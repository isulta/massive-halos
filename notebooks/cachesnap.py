from scripts.halo_analysis_scripts import *
from scripts.analytic_modeling import Simulation
from astropy import units as un, constants as cons
from joblib import Parallel, delayed

def cachesnap(simdir, snapnum):
    try:
        Simulation(simdir, snapnum, cachesim=True)
    except:
        print(f'Could not find {snapnum} for {simdir}')
    return 0

PaperSimNames = {'m12f_NoBH': 'm12f_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e10_gacc31_fa0.5',
 'm12f_BH': 'm12f_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp2e-4_gacc31_fa0.5',
 'm12f_BHCR': 'm12f_m6e4_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
 'm12q_NoBH': 'm12q_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e10_gacc31_fa0.5',
 'm12q_BH': 'm12q_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp2e-4_gacc31_fa0.5',
 'm12q_BHCR': 'm12q_m6e4_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
 'm13h113_NoBH': 'm13h113_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
 'm13h113_BH': 'm13h113_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e-4_gacc31_fa0.5',
 'm13h113_BHCR': 'm13h113_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
 'm13h206_NoBH': 'm13h206_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
 'm13h206_BH': 'm13h206_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp3e-4_gacc31_fa0.5',
 'm13h206_BHCR': 'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000'}

# d = '/projects/b1026/snapshots/fire3/m12q_m7e3/m12q_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp2e-4_gacc31_fa0.5'
# d = '/projects/b1026/snapshots/fire3/m13h113_m3e4/m13h113_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e-4_gacc31_fa0.5'
# d = '/projects/b1026/snapshots/fire3/m13h206_m3e4/m13h206_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp3e-4_gacc31_fa0.5'

# Simulation(d, 310, cachesim=True)
# Parallel(n_jobs=25, verbose=10)(delayed(cachesnap)(d, snapnum) for snapnum in range(290, 360))
# Parallel(n_jobs=48, verbose=10)(delayed(cachesnap)(d, snapnum) for snapnum in range(282, 330))

# d = sim_path_fire3(PaperSimNames['m12f_BH'])
# Parallel(n_jobs=27, verbose=10)(delayed(Simulation)(d, snapnum, cachesim=True) for snapnum in range(290, 317))

# d = sim_path_fire3(PaperSimNames['m12q_NoBH'])
# Parallel(n_jobs=27, verbose=10)(delayed(Simulation)(d, snapnum, cachesim=True) for snapnum in range(290, 317))

# d = sim_path_fire3(PaperSimNames['m12q_BH'])
# Parallel(n_jobs=20, verbose=10)(delayed(Simulation)(d, snapnum, cachesim=True) for snapnum in range(290, 310))

todo = []
for k,v in PaperSimNames.items():
    snaps = sorted([int(f.split('_')[-1].split('.')[0]) for f in os.listdir('../data/simcache') if v in f])
    if snaps[-1]>100: 
        for snap in list(set(np.arange(258, 304))- set(snaps)):
            todo.append((v, snap))

Parallel(n_jobs=-1, verbose=10)(delayed(Simulation)(sim_path_fire3(v), snapnum, cachesim=True) for (v, snapnum) in todo)

'''#Quest
d1 = '/projects/b1026/snapshots/fire3/m13h113_m3e4/m13h113_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e-4_gacc31_fa0.5'
d2 = '/projects/b1026/snapshots/fire3/m13h206_m3e4/m13h206_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp3e-4_gacc31_fa0.5'
todo = [(d1, snapnum) for snapnum in range(362, 372)] + [(d2, snapnum) for snapnum in range(330, 338)]
Parallel(n_jobs=20, verbose=10)(delayed(Simulation)(d, snapnum, cachesim=True) for (d, snapnum) in todo)
'''