from scripts.halo_analysis_scripts import *
from scripts.analytic_modeling import Simulation
from astropy import units as un, constants as cons
from joblib import Parallel, delayed

def cachesnap(simdir, snapnum):
    try:
        Simulation(simdir, snapnum, cachesim=True, satellitecut=True, calculateOutflows=True)
    except Exception as error:
        # print(f'Could not find {snapnum} for {simdir}')
        print(f'An exception occurred on snapnum {snapnum} for {simdir}:', error)
    return 0

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
'''
todo = []
for k,v in PaperSimNames.items():
    snaps = sorted([int(f.split('_')[-1].split('.')[0]) for f in os.listdir('../data/simcachev2') if v in f])
    if snaps[-1]>100: 
        for snap in list(set(np.arange(259, 501))- set(snaps)):
            todo.append((v, snap))
print(todo)
Parallel(n_jobs=-1, verbose=10)(delayed(cachesnap)(sim_path_fire3(v), snapnum) for (v, snapnum) in todo)
'''
'''
#Quest
d1 = '/projects/b1026/snapshots/fire3/m13h113_m3e4/m13h113_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e-4_gacc31_fa0.5'
d2 = '/projects/b1026/snapshots/fire3/m13h206_m3e4/m13h206_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp3e-4_gacc31_fa0.5'
todo = [(d1, snapnum) for snapnum in range(360, 362)] #+ [(d2, snapnum) for snapnum in range(330, 338)]
Parallel(n_jobs=20, verbose=10)(delayed(Simulation)(d, snapnum, cachesim=True) for (d, snapnum) in todo)
'''

d = '/projects/b1026/isultan/fire3'
todo = []
for k,v in PaperSimNames.items():
    print(v)
    simdir = os.path.join(d, v)
    if not os.path.isdir(simdir): continue
    snaps = sorted([int(f.split('_')[-1].split('.')[0]) for f in os.listdir(os.path.join(simdir, 'output')) if 'snapshot_' in f])
    if snaps[-1]<100:
        for snapnum in np.intersect1d(snaps, np.arange(50, 61)): todo.append((simdir, snapnum))
        print(snaps)
    print()
print(todo)
Parallel(n_jobs=-1, verbose=10)(delayed(cachesnap)(simdir, snapnum) for (simdir, snapnum) in todo)

todo = []
for k,v in PaperSimNames.items():
    print(v)
    simdir = os.path.join(d, v)
    if not os.path.isdir(simdir): continue
    snaps = sorted([int(f.split('_')[-1].split('.')[0]) for f in os.listdir(os.path.join(simdir, 'output')) if 'snapshot_' in f])
    if snaps[-1]>100:
        for snapnum in np.intersect1d(snaps, [259, 279, 303, 330, 364, 382, 386, 391, 395, 399, 500, snaps[-1]]): todo.append((simdir, snapnum))
        print(snaps)
    print()
print(todo)
Parallel(n_jobs=10, verbose=10)(delayed(cachesnap)(simdir, snapnum) for (simdir, snapnum) in todo)


# Parallel(n_jobs=-1, verbose=10)(delayed(cachesnap)(sim_path_fire3(PaperSimNames['m13h113_NoBH'], quest=True), snapnum) for snapnum in range(50, 61))
# Parallel(n_jobs=-1, verbose=10)(delayed(cachesnap)(sim_path_fire3(PaperSimNames['m13h206_NoBH'], quest=True), snapnum) for snapnum in range(50, 61))

# Parallel(n_jobs=-1, verbose=10)(delayed(cachesnap)(sim_path_fire3(PaperSimNames['m13h113_NoBH'], quest=True), snapnum) for snapnum in range(50, 61))
# Parallel(n_jobs=-1, verbose=10)(delayed(cachesnap)(sim_path_fire3(PaperSimNames['m13h206_NoBH'], quest=True), snapnum) for snapnum in range(50, 61))

#dont need this Parallel(n_jobs=-1, verbose=10)(delayed(cachesnap)('/projects/b1026/isultan/fire2_core_no_md/'+d, 600) for d in ['m12b_r57000', 'm12i_r57000'])

# Parallel(n_jobs=-1, verbose=10)(delayed(cachesnap)('/projects/b1026/isultan/fire2_core_no_md/m12b_r57000', snapnum) for snapnum in range(381,600))

bd = '/projects/b1026/snapshots/fire3_m12_new/'
Parallel(n_jobs=-1, verbose=10)(delayed(cachesnap)(os.path.join(bd,d), 500) for d in os.listdir(bd))

#FIRE-2 metal_diffusion sims
# sims = [f'/projects/b1026/isultan/metal_diffusion/m12{k}_r7100' for k in 'bcmrw']+['/projects/b1026/isultan/metal_diffusion/m12z_r4200']+[f'/projects/b1026/isultan/metal_diffusion/cr_heating_fix/m12{k}_r7100' for k in 'fi']
sims = [f'/projects/b1026/isultan/metal_diffusion/m12{k}_r7100' for k in 'bcmrwif']+['/projects/b1026/isultan/metal_diffusion/m12z_r4200']+['/projects/b1026/isultan/metal_diffusion/m12i_r57000']
Parallel(n_jobs=-1, verbose=10)(delayed(cachesnap)(sim, 600) for sim in sims)