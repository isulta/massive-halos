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

d = '/projects/b1026/snapshots/fire3/m12q_m7e3/m12q_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp2e-4_gacc31_fa0.5'
# d = '/projects/b1026/snapshots/fire3/m13h113_m3e4/m13h113_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e-4_gacc31_fa0.5'
# d = '/projects/b1026/snapshots/fire3/m13h206_m3e4/m13h206_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp3e-4_gacc31_fa0.5'

Simulation(d, 310, cachesim=True)
# Parallel(n_jobs=25, verbose=10)(delayed(cachesnap)(d, snapnum) for snapnum in range(290, 360))
# Parallel(n_jobs=48, verbose=10)(delayed(cachesnap)(d, snapnum) for snapnum in range(282, 330))