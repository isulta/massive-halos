from scripts.halo_analysis_scripts import *
from abg_python.system_utils import getfinsnapnum
from joblib import Parallel, delayed
from pyevtk.hl import pointsToVTK
import sys

QuestFlag = False # True is on Quest, else Frontera

def FIREtoVTK(snapdir, snapnum, scale_factor_all, fn_out, find_ids=None, datacols=['Masses', 'Density', 'Temperature', 'Vi']):
    part = load_allparticles(snapdir, snapnum, particle_types=[0,1], keys_to_extract={0:['Coordinates', 'Masses', 'ParticleIDs', 'Density', 'Temperature'],1:['Coordinates', 'Masses']}, Rvir='find_Rvir_SO', loud=False)
    scale_factor = scale_factor_all[snapnum]

    for ptype, p_i in [(0, part[0])]:#part.items():
        # position relative to center in units comoving kpc
        p_i_CoordinatesRelative = (p_i['Coordinates'] - p_i['posC']) / scale_factor
        x, y, z = p_i_CoordinatesRelative[:,0].copy(), p_i_CoordinatesRelative[:,1].copy(), p_i_CoordinatesRelative[:,2].copy()

        # volume in units (comoving kpc)^3
        p_i['Vi'] = p_i['Vi'] / scale_factor**3

        # density in units 1e10 Msun/(comoving kpc)^3
        p_i['Density'] = p_i['Density'] * scale_factor**3
        
        data = {k:p_i[k].copy() for k in datacols}
        pointsToVTK(fn_out + f'_particle{ptype}_snapnum{snapnum:02}', x, y, z, data=data)

        if find_ids is not None:
            _, _, idx2 = find_particles_snapnum(p_i, snapnum, scale_factor, find_ids, fn_out)
            return np.sum(p_i['r_scaled'][idx2] <= 2) / len(idx2) # fraction of found particles within 2 Rvir

def find_particles_snapnum(p0, snapnum, scale_factor, ids, fn_out):
    '''Given set of gas particle ids, find matching particles at snapnum and save their coordinates relative to halo center in units ckpc.'''
    _, idx1, idx2 = np.intersect1d(ids, p0['ParticleIDs'], return_indices=True)
    coordinates_scaled = ((p0['Coordinates'][idx2] - p0['posC'])/ scale_factor).copy()
    pids = p0['ParticleIDs'][idx2].copy()
    
    x, y, z = coordinates_scaled[:,0].copy(), coordinates_scaled[:,1].copy(), coordinates_scaled[:,2].copy()
    pointsToVTK(fn_out + f'_particletrack_snapnum{snapnum:02}', x, y, z, data={'ParticleIDs':pids})
    
    return coordinates_scaled, pids, idx2

def particle_forward_tracking(p0_initial, snapnum, scale_factor_all, searchregion=0.90, searchregioncentral=0.25, numpart=100):
    '''Finds ids of 100 random gas particles within central 25% of the 90% mass region of a halo at the given snapshot.'''
    scale_factor_initial = scale_factor_all[snapnum]

    # Find gas particles at z=z0 that are in 25% central region of 90% distance region
    d0 = p0_initial['r_scaled'] * p0_initial['Rvir'] / scale_factor_initial # distance relative to center in units ckpc
    # coordinates_scaled_initial = ((p0_initial['Coordinates'] - p0_initial['posC'])/ scale_factor_initial) # position relative to center in units comoving kpc
    # dist_cut = np.sort(d0)[int(len(d0)*searchregion)] * searchregioncentral # using sphere that contains 90% of particles

    dist_cut = np.sort(d0)[np.flatnonzero( np.cumsum(p0_initial['Masses'][np.argsort(d0)])/np.sum(p0_initial['Masses']) >= searchregion )[0]] * searchregioncentral # using sphere that contains 90% of mass

    ids_initial = np.random.choice(p0_initial['ParticleIDs'][d0<dist_cut], numpart)

    print(f'Searched within {dist_cut} ckpc ({searchregioncentral*100}% of {searchregion*100}% mass region) and picked {numpart} random particles.')
    return ids_initial

def outershell(pointsarr, numparts=10000, dim=3):
    '''Given an array of points, finds the N points which make up the outer boundary of the volume made up of the points.
    Returns indices of the N boundary points in `pointsarr`.
    '''
    from scipy.spatial import ConvexHull
    
    points = pointsarr.copy()
    idx = np.arange(len(points))
    idx_vert = np.empty(0, dtype=np.int64)

    while len(idx_vert) < numparts:
        hull = ConvexHull(points[:,:dim])
        idx_vert = np.concatenate((idx_vert, idx[hull.vertices]), axis=0)

        print(f'Points found in iterations: {len(idx_vert)}')
        
        points = np.delete(points, hull.vertices, axis=0)
        idx = np.delete(idx, hull.vertices, axis=0)

    return idx_vert

if __name__ == '__main__':
    simname = sys.argv[1] #'m13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000'
    snapdir = os.path.join('/projects/b1026/isultan/fire3', simname) if QuestFlag else sim_path_fire3(simname)
    fn_out = 'data/vtknew/' + simname
    scale_factor_all = np.loadtxt(os.path.join(snapdir, 'snapshot_scale-factors.txt'))

    # Load z=z0
    p0_initial = load_allparticles(snapdir, 0, particle_types=[0,1], keys_to_extract={0:['Coordinates','Masses','ParticleIDs'],1:['Coordinates', 'Masses']}, ptype_centering=1, Rvir='find_Rvir_SO', loud=True)[0]
    
    # Find 100 random central gas particles at snapshot 0 to track
    # find_ids = particle_forward_tracking(p0_initial, 0, scale_factor_all)

    # Find ~1000 outer boundary gas particles at snapshot 0 to track
    find_ids = p0_initial['ParticleIDs'][outershell(p0_initial['Coordinates'])]
    
    fraction = Parallel(n_jobs=-1, verbose=10)(delayed(FIREtoVTK)(snapdir, snapnum, scale_factor_all, fn_out, find_ids, ['Masses', 'Temperature']) for snapnum in range(61))
    print(fraction)