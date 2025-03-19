import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from pyevtk.hl import pointsToVTK
from scripts.halo_analysis_scripts import *
from scripts.analytic_modeling import calculateMr, Potential_FIRE
from firestudio.studios.gas_studio import GasStudio
from firestudio.studios.star_studio import StarStudio
import utilities.coordinate as utc
import iht

from astropy import units as un, constants as cons
from scipy.stats import binned_statistic
import pickle
import sys

keys_to_extract = {
    0:['Coordinates', 'Masses', 'Density', 
    'Temperature', 'InternalEnergy', 'CosmicRayEnergy', 
    'Velocities', 'Metallicity', 'SoundSpeed', 'CoolingRate', 'SmoothingLength', 'MagneticField', 'ParticleIDs'],
    1:['Coordinates', 'Masses', 'Velocities'],
    2:['Coordinates', 'Masses', 'Velocities'],
    4:['Coordinates', 'Masses', 'Velocities', 'ParticleIDs'],
    5:['Coordinates', 'Masses', 'Velocities']
}

def half_mass_radius(radii, masses):
    total_mass = np.sum(masses)
    sorted_indices = np.argsort(radii)
    sorted_radii = radii[sorted_indices]
    cumulative_mass = np.cumsum(masses[sorted_indices])
    
    index = np.searchsorted(cumulative_mass, total_mass / 2.0)
    r1, r2 = sorted_radii[index - 1], sorted_radii[index]
    m1, m2 = cumulative_mass[index - 1], cumulative_mass[index]

    return r1 + (total_mass / 2.0 - m1) * (r2 - r1) / (m2 - m1)

def find_Rcirc(p, halorange=(0.1, 1/3)):
    Mr = calculateMr(p)
    potential = Potential_FIRE(Mr)
    p[0]['vc'] = potential.vc(potential.Rvir*p[0]['r_scaled']).value #save vc for all particles

    idxj = inrange(p[0]['r_scaled'], halorange)
    jzavg = np.sum(p[0]['j'][idxj,2]) / np.sum(p[0]['Masses'][idxj])
    # javg = np.sum(np.linalg.norm(p[0]['j'][idxj], axis=1)) / np.sum(p[0]['Masses'][idxj])
    vc = potential.vc(potential.Rvir*p[0]['r_scaled'][idxj])

    Rcircavg = np.sum(p[0]['j'][idxj,2] / vc.value) / np.sum(p[0]['Masses'][idxj])
    Rcirc = 50
    for i in range(10): Rcirc = jzavg / potential.vc(un.kpc * Rcirc).value
    return Rcirc/p[0]['Rvir'], Rcircavg/p[0]['Rvir']

def weighted_std(values, weights):
    return np.sqrt( np.sum(values**2*weights)/np.sum(weights) - (np.sum(values*weights)/np.sum(weights))**2 )

def MassInflowRates(p0, rbins=np.array([0, 0.05, 0.1, 1/3, 2/3, 1]), Tcut=10**5.5):
    resMdot = {}
    for label,r0,r1 in zip(['galaxy', 'galaxyouter', 'inner', 'middle', 'outer'],rbins[:-1],rbins[1:]): #inner, middle, and outer halo
        idxall = np.flatnonzero(inrange( p0['r_scaled'], (r0, r1) ))
        idxhot = np.flatnonzero(inrange( p0['r_scaled'], (r0, r1) )& (p0['Temperature'] >= Tcut))
        idxcool = np.flatnonzero(inrange( p0['r_scaled'], (r0, r1) )& (p0['Temperature'] < Tcut))
        
        dL = (r1 - r0)*p0['Rvir']
        Mdots_bin = [ np.sum(p0['vrad'][idx] * 1e10 * p0['Masses'][idx] / dL ) * (un.km / un.s * un.Msun / un.kpc).to(un.Msun / un.yr) for idx in (idxall, idxhot, idxcool) ]
        Mdotin_bin = [ np.sum(p0['vrad'][idx][p0['vrad'][idx] > 0] * 1e10 * p0['Masses'][idx][p0['vrad'][idx] > 0] / dL ) * (un.km / un.s * un.Msun / un.kpc).to(un.Msun / un.yr) for idx in (idxall, idxhot, idxcool) ]
        Mdotout_bin = [ np.sum(p0['vrad'][idx][p0['vrad'][idx] < 0] * 1e10 * p0['Masses'][idx][p0['vrad'][idx] < 0] / dL ) * (un.km / un.s * un.Msun / un.kpc).to(un.Msun / un.yr) for idx in (idxall, idxhot, idxcool) ]
        M_bin = [ np.sum(p0['Masses'][idx])*1e10 for idx in (idxall, idxhot, idxcool) ]

        vphi_bin = [ np.sum(p0['vphi'][idx]*p0['Masses'][idx])/np.sum(p0['Masses'][idx]) for idx in (idxall, idxhot, idxcool) ]
        sigmagstd_bin = [ np.std(p0['vphi'][idx]) for idx in (idxall, idxhot, idxcool) ]
        sigmag_bin = [ weighted_std(p0['vphi'][idx], p0['Masses'][idx]) for idx in (idxall, idxhot, idxcool) ]
        resMdot[label] = Mdots_bin + M_bin + Mdotin_bin + Mdotout_bin + vphi_bin + sigmag_bin + sigmagstd_bin
    return resMdot

def load(simdir, snapnum, rot=None, rgal_limit=0.15, returnrot=False):
    p = load_allparticles(simdir, snapnum, [0,1,2,4,5], keys_to_extract=keys_to_extract, loud=0)
    velC = iht.velocity_COM(p)
    for k in p.keys():
        # p[k]['Coordinates0'] = p[k]['Coordinates'].copy()
        p[k]['Coordinates'] -= p[k]['posC']
        p[k]['Velocities'] -= velC
    rotnew = iht.rotate_coords(p, p[4]['Rvir'], rot=rot)
    # with open(f'/projects/b1026/isultan/data/rotcache/rot_{os.path.basename(os.path.normpath(simdir))}_{snapnum}.pkl', 'wb') as f: pickle.dump(rotnew, f) #cache rotation
    p[0]['vrad'], p[0]['vtheta'], p[0]['vphi'] = iht.spherical_velocities(v=p[0]['Velocities'], r=p[0]['Coordinates'])
    p[0]['vrad'] *= -1 # Define vrad as inflow velocity
    # do same for stars
    #p[4]['vrad'], p[4]['vtheta'], p[4]['vphi'] = iht.spherical_velocities(v=p[4]['Velocities'], r=p[4]['Coordinates'])
    #p[4]['vrad'] *= -1 # Define vrad as inflow velocity

    XH = 1 - p[0]['Metallicity'][:,0] - p[0]['Metallicity'][:,1] #hydrogen mass fraction
    p[0]['nH'] = ( XH * (p[0]['Density'] * 1e10 * un.Msun/un.kpc**3) / cons.m_p ).to(un.cm**-3).value #number density of hydrogen atoms
    
    for k in (0,4):
        p[k]['j'] = np.cross(p[k]['Coordinates'], p[k]['Velocities']) * p[k]['Masses'][:, None]
        p[k]['jalign'] = p[k]['j'][:,2] / np.linalg.norm(p[k]['j'], axis=1)
    
    RgalZach = 0#TODO: change to RgalZach #4 * half_mass_radius((p[4]['r_scaled'] * p[4]['Rvir'])[p[4]['r_scaled'] <= rgal_limit], p[4]['Masses'][p[4]['r_scaled'] <= rgal_limit]) / p[4]['Rvir']
    Rcirc, Rcircavg = find_Rcirc(p)
    for k in p.keys():
        p[k]['Rgal'] = 0.05#RgalZach#Rcirc
        p[k]['RgalZach'] = RgalZach 
        p[k]['Rcircavg'] = Rcircavg
        p[k]['Rcirc'] = Rcirc
    
    # p[0]['vrad'], p[0]['vtheta'], p[0]['vphi'] = iht.spherical_velocities(v=p[0]['Velocities'], r=p[0]['Coordinates'])
    # rbins=np.power(10, np.arange(np.log10(0.005258639741921723), np.log10(3), 0.05))
    # rmid = (rbins[:-1]+rbins[1:])/2 #in units of Rvir
    # vphi_avg = np.array([ np.sum((p[0]['vphi'][inrange(p[0]['r_scaled'], (r0,r1))]*p[0]['Masses'][inrange(p[0]['r_scaled'], (r0,r1))])) / np.sum(p[0]['Masses'][inrange(p[0]['r_scaled'], (r0,r1))]) for r0,r1 in zip(rbins[:-1],rbins[1:]) ])
    # Mr['vphi_avg'] = vphi_avg
    # Mr['rmid_vphi_avg'] = rmid
    # p[0]['Mr'] = Mr

    if returnrot: 
        return p, rotnew
    else:
        return p
    
def selectparticles_zach(p_tf, p_ti, aperture=0.05):
    mask_tf_gas = (p_tf[0]['r_scaled'] < p_tf[0]['Rgal']) & (p_tf[0]['nH'] > 0.13)
    mask_tfstars = (p_tf[4]['r_scaled'] < p_tf[0]['Rgal'])
    mask_tf = np.concatenate((mask_tf_gas, mask_tfstars))
    mask_ti = inrange(p_ti[0]['r_scaled'], (0.1, 1))
    
    # Get the unique ParticleIDs in p_tf with counts
    p_tf_04_ParticleIDs = np.concatenate([p_tf[i]['ParticleIDs'] for i in (0,4)])
    unique_ids_tf, counts_tf = np.unique(p_tf_04_ParticleIDs, return_counts=True)

    # Select only the ParticleIDs that appear exactly once
    unique_ids_tf = unique_ids_tf[counts_tf == 1]
    
    # Now perform the intersection with only the unique IDs
    ParticleIDs, _, _ = np.intersect1d(
        p_tf_04_ParticleIDs[mask_tf & np.isin(p_tf_04_ParticleIDs, unique_ids_tf)],
        p_ti[0]['ParticleIDs'][mask_ti],
        return_indices=True
    )    
    return ParticleIDs

def findparticles(p, ParticleIDs):
    p_04_ParticleIDs = np.concatenate([p[i]['ParticleIDs'] for i in (0,4)])
    sorter = np.argsort(p_04_ParticleIDs)
    idx = sorter[np.searchsorted(p_04_ParticleIDs, ParticleIDs, sorter=sorter)]
    
    assert len(idx)==len(ParticleIDs), 'Missing particles!'
    return idx

def trackparticles(p, ParticleIDs, simname, snapnum):
    a = p[0]['ScaleFactor']
    # returns Radius [pkpc], T [K], nH [cm^-3], jalign, t [Gyr], 
    
    idx = findparticles(p, ParticleIDs)
    mask_gas = idx < len(p[0]['ParticleIDs'])
    mask_stars = idx >= len(p[0]['ParticleIDs'])
    idx_gas = idx[mask_gas]
    idx_stars = idx[mask_stars] - len(p[0]['ParticleIDs'])

    res = {k:np.full(len(idx), np.nan) for k in ['r_scaled', 'Temperature', 'nH', 'jalign', 'Masses', 'ParticleIDs', 'vphi', 'vc']}
    for k in ['Coordinates', 'Velocities', 'j']: res[k] = np.full((len(idx), 3), np.nan)

    for k in res.keys():
        res[k][mask_gas] = p[0][k][idx_gas]
        if k in p[4]: res[k][mask_stars] = p[4][k][idx_stars]

    for k in ['TimeGyr', 'Rvir', 'Rgal', 'ScaleFactor', 'RgalZach', 'Rcircavg', 'Rcirc', 'posC']: res[k] = p[0][k]

    fname = f'/projects/b1026/isultan/data/particlescache_Rcirc/particles_{simname}_{snapnum:03}.h5'
    dicttoh5(res, fname, mode='w')
    
    return res

def load_track(simdir, snapnum, p_tf, p_ti, s_tf, s_ti, ParticleIDs, simname, rot):
    try:
        if snapnum==s_tf:
            p = p_tf
        elif snapnum==s_ti:
            p = p_ti
        else:
            p = load(simdir, snapnum, rot=rot)
        trackparticles(p, ParticleIDs, simname, snapnum)
    except Exception as e:
        print('EXCEPTION Could not complete!!!', simdir, snapnum, e)
    
def process_sim(simdir, simname, s_tf=600, s_ti=547, s_tfout=600, s_tiout=547, n_jobs=20):
    '''
    s_ti, s_tf are the first and last snapshots to select particles
    s_tiout, s_tfout are the first and last snapshots to track and output particles
    '''
    print(f'Processing {simname}')
    #p_tf, p_ti = Parallel(n_jobs=2, verbose=10)(delayed(load)(simdir, s) for s in (s_tf, s_ti))
    # ***Use rotation vector of s_tf to rotate ALL snapshots. Cache this rotation vector.***
    p_tf, rot_tf = load(simdir, s_tf, returnrot=True)
    p_ti = load(simdir, s_ti, rot=rot_tf)
    ParticleIDs = selectparticles_zach(p_tf, p_ti)
    
    Parallel(n_jobs=n_jobs, verbose=10)(delayed(load_track)(simdir, s, p_tf, p_ti, s_tf, s_ti, ParticleIDs, simname, rot_tf) for s in range(s_tiout, s_tfout+1))
    print(f'Finished {simname}')

def load_Mdot(simdir, snapnum, simname):
    try:
        p = load(simdir, snapnum)
        resMdot = MassInflowRates(p[0])
        fname = f'/projects/b1026/isultan/data/Mdotcache_AHF/Mdot_{simname}_{snapnum:03}.h5'
        dicttoh5(resMdot, fname, mode='w')
    except Exception as e:
        print('EXCEPTION Could not complete!!!', simdir, snapnum, e)

def selectparticles_allaccrete(p_tf, p_ti, Rgal=0.05):
    mask_tf_gas = (p_tf[0]['r_scaled'] < Rgal)
    mask_tfstars = (p_tf[4]['r_scaled'] < Rgal)
    mask_tf = np.concatenate((mask_tf_gas, mask_tfstars))
    mask_ti = (p_ti[0]['r_scaled'] > Rgal)
    
    # Get the unique ParticleIDs in p_tf with counts
    p_tf_04_ParticleIDs = np.concatenate([p_tf[i]['ParticleIDs'] for i in (0,4)])
    unique_ids_tf, counts_tf = np.unique(p_tf_04_ParticleIDs, return_counts=True)

    # Select only the ParticleIDs that appear exactly once
    unique_ids_tf = unique_ids_tf[counts_tf == 1]
    
    # Now perform the intersection with only the unique IDs
    ParticleIDs, _, _ = np.intersect1d(
        p_tf_04_ParticleIDs[mask_tf & np.isin(p_tf_04_ParticleIDs, unique_ids_tf)],
        p_ti[0]['ParticleIDs'][mask_ti],
        return_indices=True
    )    
    return ParticleIDs

def load_allaccrete(simdir, s0):
    p_t0 = load_allparticles(simdir, s0, [0], keys_to_extract={0:['Coordinates', 'ParticleIDs']}, loud=0)
    p_t1 = load_allparticles(simdir, s0+1, [0,4], keys_to_extract={0:['Coordinates', 'ParticleIDs'], 4:['Coordinates', 'ParticleIDs']}, loud=0)
    ParticleIDs = selectparticles_allaccrete(p_t1, p_t0)
    return ParticleIDs

def finduniqueparticles_allaccrete(p_tf, ParticleIDs):
    # Flatten the list of ParticleIDs from all snapshots
    ParticleIDs = np.concatenate(ParticleIDs)

    # Get the unique ParticleIDs in p_tf with counts
    p_tf_04_ParticleIDs = np.concatenate([p_tf[i]['ParticleIDs'] for i in (0,4)])
    unique_ids_tf, counts_tf = np.unique(p_tf_04_ParticleIDs, return_counts=True)

    # Select only the ParticleIDs that appear exactly once
    unique_ids_tf = unique_ids_tf[counts_tf == 1]

    # Filter ParticleIDs to remove duplicates
    ParticleIDs = np.intersect1d(ParticleIDs, unique_ids_tf)
    
    return ParticleIDs

def process_sim_allaccrete(simdir, simname, s_tf=277, s_ti=40, n_jobs=20):
    '''
    s_ti, s_tf are the first and last snapshots to track accreted particles
    '''
    print(f'Processing {simname}')
    
    ParticleIDs_list = Parallel(n_jobs=n_jobs, verbose=10)(delayed(load_allaccrete)(simdir, s0) for s0 in range(s_ti, s_tf))

    p_tf, rot_tf = load(simdir, s_tf, returnrot=True)
    ParticleIDs = finduniqueparticles_allaccrete(p_tf, ParticleIDs_list)
    
    Parallel(n_jobs=n_jobs, verbose=10)(delayed(load_track)(simdir, s, p_tf, None, s_tf, -100, ParticleIDs, simname, None) for s in range(s_ti, s_tf+1))
    print(f'Finished {simname}')
'''
for sim in ['h206_A1_res33000', 'h113_A4_res33000', 'h29_A2_res33000']:
    process_sim(f'/projects/b1026/snapshots/MassiveFIRE/{sim}', sim, s_tf=142, s_ti=95, s_tfout=142, s_tiout=36, n_jobs=20)

process_sim('/projects/b1026/isultan/metal_diffusion/m12i_r7100/output', 'm12i_r7100', s_tfout=600, s_tiout=504)
for sim in ['h2_A8_res33000']:
    process_sim(f'/projects/b1026/snapshots/MassiveFIRE/{sim}', sim, s_tf=142, s_ti=95, s_tfout=142, s_tiout=36, n_jobs=5)
# process_sim('/projects/b1026/isultan/metal_diffusion/cr_heating_fix/m12i_r7100/output', 'm12i_r7100_crfix')
# process_sim('/projects/b1026/isultan/metal_diffusion/m12c_r7100/output', 'm12c_r7100')
# process_sim('/projects/b1026/isultan/metal_diffusion/m12b_r7100/output', 'm12b_r7100')
# process_sim('/projects/b1026/isultan/metal_diffusion/m12f_r7100/output', 'm12f_r7100')
'''


# Mdot cache
if __name__ == '__main__':
    simname = sys.argv[1]
    print(f'Processing {simname}')
    simdir  = f'/projects/b1026/snapshots/MassiveFIRE/{simname}'
    #snapnums = [np.argmin(np.abs(redshifts_snapshots(os.path.join(simdir, 'output')) - zi)) for zi in np.arange(2, 4.1, 0.5)]
    #Parallel(n_jobs=-1, verbose=10)(delayed(load_Mdot)(simdir, snapnum, simname) for snapnum in snapnums)
    n_jobs = 20 if simname in ['h113_A4_res33000', 'h29_A2_res33000'] else 5 if simname == 'h2_A8_res33000' else -1
    Parallel(n_jobs=n_jobs, verbose=10)(delayed(load_Mdot)(simdir, snapnum, simname) for snapnum in range(40, 278))
    print(f'Finished {simname}')

'''
if __name__ == '__main__':
    for sim in ['h206_A1_res33000', 'h113_A4_res33000', 'h29_A2_res33000']:
        process_sim(f'/projects/b1026/snapshots/MassiveFIRE/{sim}', sim, s_tf=113, s_ti=58, s_tfout=113, s_tiout=58, n_jobs=20)
        process_sim(f'/projects/b1026/snapshots/MassiveFIRE/{sim}', sim, s_tf=159, s_ti=114, s_tfout=159, s_tiout=114, n_jobs=20)
        process_sim(f'/projects/b1026/snapshots/MassiveFIRE/{sim}', sim, s_tf=201, s_ti=160, s_tfout=201, s_tiout=160, n_jobs=20)
        process_sim(f'/projects/b1026/snapshots/MassiveFIRE/{sim}', sim, s_tf=241, s_ti=202, s_tfout=241, s_tiout=202, n_jobs=20)

    for sim in ['h2_A8_res33000']:
        process_sim(f'/projects/b1026/snapshots/MassiveFIRE/{sim}', sim, s_tf=113, s_ti=58, s_tfout=113, s_tiout=58, n_jobs=5)
        process_sim(f'/projects/b1026/snapshots/MassiveFIRE/{sim}', sim, s_tf=159, s_ti=114, s_tfout=159, s_tiout=114, n_jobs=5)
        process_sim(f'/projects/b1026/snapshots/MassiveFIRE/{sim}', sim, s_tf=201, s_ti=160, s_tfout=201, s_tiout=160, n_jobs=5)
        process_sim(f'/projects/b1026/snapshots/MassiveFIRE/{sim}', sim, s_tf=241, s_ti=202, s_tfout=241, s_tiout=202, n_jobs=5)
'''
'''
if __name__ == '__main__':
    sim = sys.argv[1]
    n_jobs = 20 if sim in ['h113_A4_res33000', 'h29_A2_res33000'] else 5 if sim == 'h2_A8_res33000' else -1
    process_sim_allaccrete(f'/projects/b1026/snapshots/MassiveFIRE/{sim}', sim, n_jobs=n_jobs)'
'''