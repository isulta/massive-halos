# good AGN feedback models
CCAbaseDir = '/home/jovyan/fire2/AGN_suite/'
CCA_goodsim_h206 = lambda model : CCAbaseDir + f'm13h206_m3e5/m13h206_m3e5_{model}_alpha10_gacc30_accf1_vw10000_cr1e-2_msd1e-8_sdp3e-3_mw4e-7_fa0.5_tw1e4_fmom1'
CCA_sims = {
    'push':{
        'h206': CCA_goodsim_h206('push'),
        'h29':CCAbaseDir+'m13h29_m3e5/m13h29_m3e5_push_alpha10_gacc30_accf1_vw10000_cr1e-2_msd1e-8_sdp5e-3_mw4e-7_fa0.5_tw1e4_fmom1'
    },
    'jet':{
        'h206': CCA_goodsim_h206('jet')
    },
    'spawn':{
        'h206': CCA_goodsim_h206('spawn')
    }
}

FronterabaseDir = '/scratch3/01799/phopkins/bhfb_suite_done/'
Frontera_h206push_base = FronterabaseDir + 'm13h206_m3e5/m13h206_m3e5_push_'
Frontera_sims = {
    'push':{
        'h206':{
            'radfboff':             Frontera_h206push_base + 'alpha10_gacc30_accf1_vw10000_cr1e-2_msd1e-8_sdp3e-3_mw4e-7_fa0.5_tw1e4_fmom1e-4',
            'good':                 Frontera_h206push_base + 'alpha10_gacc30_accf1_vw10000_cr1e-2_msd1e-8_sdp3e-3_mw4e-7_fa0.5_tw1e4_fmom1',
            'CRsoff_veryhighradfb': Frontera_h206push_base + 'alpha10_gacc30_accf1_vw10000_cr1e-6_msd1e-8_sdp3e-3_mw4e-7_fa0.5_tw1e4_fmom100',
            'CRsoff':               Frontera_h206push_base + 'alpha10_gacc30_accf1_vw10000_cr1e-6_msd1e-8_sdp3e-3_mw4e-7_fa0.5_tw1e4_fmom1',
            'veryslowwinds':        Frontera_h206push_base + 'alpha10_gacc30_accf1_vw100_cr1e-2_msd1e-8_sdp3e-3_mw4e-7_fa0.5_tw1e4_fmom1',
            'CRsoff_veryfastwinds': Frontera_h206push_base + 'alpha10_gacc30_accf1_vw42500_cr1e-6_msd1e-8_sdp3e-3_mw4e-7_fa0.5_tw1e4_fmom1'
        }
    }
}

QuestbaseDir = '/projects/b1026/anglesd/FIRE/'
Quest_nofb_m13_id = '_HR_sn1dy300ro100ss'
Quest_nofb_m13 = lambda halo : QuestbaseDir + halo + Quest_nofb_m13_id
Quest_nofb_m13_ahf = lambda halo : '/projects/b1026/halo_files/anglesd_m13/' + halo + Quest_nofb_m13_id
Quest_sims = {
    'nofb':{
        'h206': Quest_nofb_m13('h206'), #A1
        'h29':  Quest_nofb_m13('h29'),  #A2
        'h113': Quest_nofb_m13('h113'), #A4
        'h2':   Quest_nofb_m13('h2')    #A8
    }
}

COLOR_SCHEME = ['#2402ba','#b400e0','#98c1d9','#ff0000','#292800','#ff9b71']

profilelabels = {
    'rho':r'$\log \left< \rho / \left( \mathrm{M_\odot} / \mathrm{pc}^3 \right) \right>$',
    'T':r'$\left< \log \left( T / \mathrm{K} \right) \right>$',
    'e_CR':r'$\log \left< \epsilon_{CR} / \left( 10^{10} \mathrm{M_\odot} \mathrm{km}^2 / \mathrm{kpc}^3 \mathrm{s}^2 \right) \right>$',
    'P_th':r'$\left< \log \left[ P_{th} / \left(k_B \mathrm{K}/\mathrm{cm}^3\right) \right] \right>$',
    'P_CR':r'$\left< \log \left[ P_{CR} / \left(k_B \mathrm{K}/\mathrm{cm}^3\right) \right] \right>$',
    'T lin':r'$\log \left< T / \mathrm{K} \right>$',
    'P_th lin':r'$\log \left< P_{th} / \left(k_B \mathrm{K}/\mathrm{cm}^3\right) \right>$',
    'P_CR lin':r'$\log \left< P_{CR} / \left(k_B \mathrm{K}/\mathrm{cm}^3\right) \right>$'
}

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from abg_python.snapshot_utils import openSnapshot
from abg_python.cosmo_utils import load_AHF, load_rockstar
from astropy.constants import k_B
from astropy import units
from itk import inrange, loadpickle, pickle_save_dict, n_array_equal, sync_lim
from silx.io.dictdump import dicttoh5, h5todict
from tqdm import tqdm
import os.path

def read_param_file(simdir, params_file='params.txt'):
    '''Reads GIZMO parameter file and returns dictionary of parameters.
    '''
    f = os.path.join(simdir, params_file)
    params = [l.strip().split('%')[0].split() for l in open(f, 'r') if not (l.startswith(('%','\n')) or l.strip()=='')]
    
    p = {}
    for k,v in params:
        if v.isdigit():
            p[k] = int(v)
        else:
            try:
                p[k] = float(v)
            except:
                p[k] = v
    return p

def redshifts_snapshots(simdir, snapshot_scalefactors_file = 'snapshot_scale-factors.txt'):
    '''Loads redshifts of the snapshots of a simulation.
    Returns array `redshifts` where `redshifts[i]` is the redshift of snapshot `i`.
    '''
    scale_factors = np.loadtxt(os.path.join(simdir, snapshot_scalefactors_file))
    redshifts = scale_factor_to_redshift(scale_factors)
    return redshifts

def load_Rvir_allsnaps(snapdir, ahf_path=None, outputFlag=True, simname=None):
    '''Given the snapshot directory and ahf directory for a simulation, loads the virial radius at each snapshot 
    that is available in the AHF data files.
    If `ahf_path` is not given, loads Rockstar halo files from `{snapdir}/halo/rockstar_dm/catalog_hdf5`.
    If `outputFlag` is True, the output is saved as file `'Rvir_{simname}.h5'`.
    
    Returns:
        Dictionary where `Rvir_allsnaps[snapnum]` is the virial radius (in units of physical kpc) for snapshot `snapnum`.
        Array `redshifts` where `redshifts[i]` is the redshift of snapshot `i`.
    '''
    res = {'snapnum':[], 'z':[], 'Rvir':[]}

    # Load redshift of each snapshot
    redshifts = redshifts_snapshots(snapdir)
    totsnaps = len(redshifts)

    # Load little h for simulation
    hubble = read_param_file(snapdir)['HubbleParam']
    print(f'h={hubble}; {totsnaps} snapshots found in scalefactors file.')
    
    if ahf_path is not None:
        # Load array of all snapshot numbers available in AHF file
        ahf_snaps = np.sort(np.genfromtxt(os.path.join(ahf_path, 'halo_00000_smooth.dat'), delimiter='\t', skip_header=True, dtype=int)[:,0])
        
        print(f'Snapshots in AHF file range from {ahf_snaps.min()} to {ahf_snaps.max()} (z={redshifts[ahf_snaps.min()]} to {redshifts[ahf_snaps.max()]}).')
        
        # load Rvir of each snapshot
        for snapnum in ahf_snaps:
            res['Rvir'].append( load_AHF('', snapnum, hubble=hubble, ahf_path=ahf_path, extra_names_to_read=[])[1] )
            res['snapnum'].append(snapnum)
            res['z'].append(redshifts[snapnum])
    else: #use Rockstar halo files
        # Find array of all snapshot numbers available in Rockstar files
        rockstar_snaps = np.sort([int(f.split('_')[1].split('.')[0]) for f in os.listdir(os.path.join(snapdir, 'halo/rockstar_dm/catalog_hdf5')) if 'halo' in f])
        
        print(f'Snapshots in Rockstar files range from {rockstar_snaps.min()} to {rockstar_snaps.max()} (z={redshifts[rockstar_snaps.min()]} to {redshifts[rockstar_snaps.max()]}).')
        
        # load Rvir of each snapshot
        for snapnum in rockstar_snaps:
            res['Rvir'].append( load_rockstar(os.path.join(snapdir, 'output/'), snapnum)[1] )
            res['snapnum'].append(snapnum)
            res['z'].append(redshifts[snapnum])
    
    # save Rvir and redshift data
    if outputFlag: 
        simname = os.path.basename(snapdir.rstrip('/')) if simname is None else simname
        fname = f'data/Rvir/Rvir_{simname}.h5'
        dicttoh5(res, fname, mode='w')
        print(f'Saved res to {fname}.')

    Rvir_allsnaps = dict(zip(res['snapnum'], res['Rvir']))
    return Rvir_allsnaps, redshifts

'''For a given simulation, loads virial radius data from either `Rvir_{simname}.h5` or `Rvir_{halo}_noAGNfb.h5` files.
If the former file does not exist or `useNoAGNFb` is True, the latter file is used to perform a redshift matching 
between the given sim (for which Rvir(z) is not known) and No AGN feedback sim. 
'''
def Rvir_sim(snapdir, simname=None, useNoAGNFb=True):
    # Load redshift of each snapshot
    redshifts = redshifts_snapshots(snapdir)

    simname = os.path.basename(snapdir.rstrip('/')) if simname is None else simname
    fname = f'data/Rvir/Rvir_{simname}.h5'
    if os.path.exists(fname) and (not useNoAGNFb or '_noAGNfb' in snapdir):
        res = h5todict(fname)
        print(f'Found {fname} and loaded res.')
    else:
        print(f'Loading No AGN feedback file instead of {fname}...')
        fname = f'data/Rvir/Rvir_{simname.split("_")[0]}_noAGNfb.h5'
        resNoAGNfb = h5todict(fname)
        print(f'Found {fname} and loaded resNoAGNfb.')

        # Set Rvir_allsnaps
        _, idx1, idx2 = np.intersect1d(redshifts, resNoAGNfb['z'], return_indices=True)
        res = {}
        res['Rvir'] = resNoAGNfb['Rvir'][idx2[::-1]]
        res['snapnum'] = idx1[::-1]
        res['z'] = resNoAGNfb['z'][idx2[::-1]]
        
        # Check if all snapshots for which an Rvir match was found are contiguous
        assert np.array_equal( res['snapnum'], np.arange(idx1.min(), idx1.max()+1) ), 'Redshift match not found for at least one intermediate snapshot.'
    
    return res, redshifts

### Halo centering ###
def center_of_mass(coords, masses):
    return np.array([np.sum((coords[:,i] * masses)) for i in range(3)])/np.sum(masses)

@njit
def dist(r1, r0):
    res = np.zeros_like(r1[:,0])

    rrel = r1 - r0
    for i in range(len(res)):
        for j in range(3):
            res[i] += rrel[i][j]**2
        res[i] = np.sqrt(res[i])
    return res

def halo_center(coords, masses, shrinkpercent=50, minparticles=1000, initialradiusfactor=0.25, verbose=False):
    '''See Power et al. 2003 (their parameter values: shrinkpercent=2.5, minparticles=1000, initialradiusfactor=1)
    '''
    com = center_of_mass(coords, masses)

    r = dist(coords, com)
    
    radius = r.max()*initialradiusfactor

    Nconverge = min(minparticles, len(masses)*0.01)
    iteration = 0

    coords_it = coords.copy()
    masses_it = masses.copy()

    comlist = [com]
    radiuslist = [radius]

    while len(masses_it) > Nconverge:
        radius *= (100-shrinkpercent)/100

        mask = r <= radius
        coords_it = coords_it[mask, :]
        masses_it = masses_it[mask]

        com = center_of_mass(coords_it, masses_it)

        r = dist(coords_it, com)
        
        iteration += 1
        comlist.append(com)
        radiuslist.append(radius)

        if verbose:
            print(iteration, radius, np.format_float_scientific(len(masses_it)), com)
    
    return com, comlist, radiuslist

def halo_center_wrapper(pdata, shrinkpercent=50, minparticles=1000, initialradiusfactor=0.25):
    coords = pdata['Coordinates']
    masses = pdata['Masses']
    return halo_center(coords, masses, shrinkpercent, minparticles, initialradiusfactor)

### TD profiles ###
def load_p0(snapdir, snapnum, ahf_path=None, Rvir=None, loud=1, keys_to_extract=None, calculate_qtys=True):
    '''Loads gas particle snapshot and adds `CoordinatesRelative`, `r`, `r_scaled`, `Vi`, `posC`, and `Rvir` columns to dictionary.
    `Rvir` argument must be in units of physical kpc. If `ahf_path` is passed instead, `Rvir` will be read in with units physical kpc.
    '''
    p0 = openSnapshot(snapdir, snapnum, 0, loud=loud, keys_to_extract=keys_to_extract)
    p1 = openSnapshot(snapdir, snapnum, 1, loud=loud, keys_to_extract=['Coordinates', 'Masses'])
    
    if loud:
        print(f"Loading redshift {p0['Redshift']}")

    posC = halo_center_wrapper(p1)[0]

    if ahf_path:
        _, Rvir = load_AHF('', snapnum, p0['Redshift'], hubble=p0['HubbleParam'], ahf_path=ahf_path, extra_names_to_read=[])

    # position relative to center
    p0['CoordinatesRelative'] = p0['Coordinates'] - posC
    
    if calculate_qtys:
        # distance from halo center
        p0['r'] = np.linalg.norm(p0['CoordinatesRelative'], axis=1)

        # distance from halo center in units of virial radius
        p0['r_scaled'] = p0['r']/Rvir

        # volume of each particle in units (physical kpc)^3
        p0['Vi'] = p0['Masses']/p0['Density']

        p0['posC'] = posC #halo center in units physical kpc
        p0['Rvir'] = Rvir #virial radius in units physical kpc
    
    return p0

'''Finds virial radius (units physical kpc) given particle dict and halo center, using spherical overdensity definition.
If `halo` and `snapnum` are defined, a plot of density vs. distance from halo center is saved.
The virial mass (mass within a sphere of radius Rvir centered at posC) is also returned in units Msun.
'''
def find_Rvir_SO(part, posC=None, halo=None, snapnum=None):
    if posC is None:
        posC = part[0]['posC']

    Masses = []
    r = []
    for ptype, p_i in part.items():
        if not 'Masses' in p_i.keys(): #some simulations don't have ptype 5 (black holes)
            print(f'{halo}: Masses not found for ptype {ptype} at snapshot {snapnum}')
            continue
        
        Masses.append(p_i['Masses'])
        
        # position relative to center
        p_i_CoordinatesRelative = p_i['Coordinates'] - posC

        # distance from halo center
        p_i_r = np.linalg.norm(p_i_CoordinatesRelative, axis=1)

        r.append(p_i_r)
    
    Masses = np.concatenate(Masses)
    r = np.concatenate(r)

    idx = np.argsort(r)
    Masses = Masses[idx]
    r = r[idx]
    Volume = 4/3 * np.pi * r**3 # Volume in units (physical kpc)^3

    Masses = np.cumsum(Masses) * 1.e10 # Total mass in units Msun within sphere of radius r

    with np.errstate(divide='ignore'): Density = Masses/Volume * 1.e9 # Density in units Msun/Mpc^3

    OmegaM0, OmegaL0, hubble, z = part[0]['Omega0'], part[0]['OmegaLambda'], part[0]['HubbleParam'], part[0]['Redshift']
    rhovir = deltavir(OmegaM0, OmegaL0, z) * rhocritz(OmegaM0, OmegaL0, z) * hubble**2 # Virial density in units Msun/Mpc^3

    if halo is not None:
        plt.plot(r, Density)
        plt.yscale('log')
        plt.axhline(rhovir, label=r'$\Delta_{vir} \rho_{crit}$')
        plt.xlim(0,300)
        plt.xlabel('r (pkpc)')
        plt.ylabel('Density (Msun/pMpc^3)')
        plt.legend()
        plt.savefig(f'Figures/density/density_{halo}_snapnum_{snapnum}.png')
        plt.close()
    
    idx_vir = np.flatnonzero(Density <= rhovir)[0]
    return r[idx_vir], Masses[idx_vir] # return Rvir in units physical kpc, and Mvir in units Msun
    # simple linear interpolation with next closest point, and InterpolatedUnivariateSpline.roots() both seem to return approximately same Rvir as the 1 point method above.

def load_allparticles(snapdir, snapnum, particle_types=[0,1,2,4,5], keys_to_extract={0:['Coordinates', 'Masses', 'Density', 'Temperature', 'InternalEnergy'],1:['Coordinates', 'Masses'],2:['Coordinates', 'Masses'],4:['Coordinates', 'Masses'],5:['Coordinates', 'Masses']}, ptype_centering=1, Rvir=None, ahf_path=None, loud=1):
    '''Loads all particle data from simulation directory `snapdir` for a snapshot  `snapnum`, and returns dict of dicts for each particle type.

    Notes:
        - `posC` (array of len 3; halo center in units physical kpc) is added to each particle dicts. 
        - If either `Rvir` or `ahf_path` is defined, `r_scaled` (array; particle distances from halo center in units of Rvir) and `Rvir` (scalar; virial radius in units physical kpc) columns are added to each particle dict.
        - If a particle dict contains `'Masses'` and `'Density'` keys, `Vi` (array; particle volumes in units (physical kpc)^3) column is added to that particle dict.

    Optional parameters:
        `particle_types`: list of particle types to load
        `keys_to_extract`: dict that maps each particle type to a list of keys to read. If dict does not contain a particle type, ALL keys are read in for that particle type.
        `ptype_centering`: particle type used to find halo center (shrinking sphere method)
        `Rvir`: virial radius in units of physical kpc
        `ahf_path`: directory with AHF file. If defined, `Rvir` will be read in with units physical kpc (this will overwrite any value passed for `Rvir`).
        `loud`: verbose output if `True`
    '''
    snapdir = os.path.join(snapdir, 'output/') if os.path.exists(os.path.join(snapdir, 'output/')) else snapdir #some sims have snapshots in output/

    keys_to_extract = {**{ptype:None for ptype in particle_types}, **keys_to_extract} #load all keys for particle types for which keys were not given
    
    part = { ptype : openSnapshot(snapdir, snapnum, ptype, loud=loud, keys_to_extract=keys_to_extract[ptype]) for ptype in particle_types }
    
    if loud:
        print(f"Loading redshift {part[particle_types[0]]['Redshift']}")

    # posC = halo_center_wrapper(part[ptype_centering])[0]
    posC = halo_center_wrapper(part[ptype_centering], shrinkpercent=2.5, minparticles=1000, initialradiusfactor=1)[0] #most accurate parameters
    # posC = halo_center_wrapper(part[ptype_centering], shrinkpercent=10, minparticles=1000, initialradiusfactor=1)[0] #pretty accurate parameters and 2x faster than shrinkpercent=2.5 (some minor problems, e.g. a small dip at z=6 for h29_noAGNfb)

    if ahf_path:
        _, Rvir = load_AHF('', snapnum, part[particle_types[0]]['Redshift'], hubble=part[particle_types[0]]['HubbleParam'], ahf_path=ahf_path, extra_names_to_read=[])

    for ptype, p_i in part.items():
        p_i['posC'] = posC #halo center in units physical kpc
        
        if Rvir is not None:
            p_i['Rvir'] = Rvir #virial radius in units physical kpc

            # position relative to center
            p_i_CoordinatesRelative = p_i['Coordinates'] - posC

            # distance from halo center
            p_i_r = np.linalg.norm(p_i_CoordinatesRelative, axis=1)

            # distance from halo center in units of virial radius
            p_i['r_scaled'] = p_i_r/Rvir

        if ('Masses' in p_i) and ('Density' in p_i):
            # volume of each particle in units (physical kpc)^3
            p_i['Vi'] = p_i['Masses']/p_i['Density']
    
    return part

def load_p0_allsnaps(snapdir, snapnums, Rvir=1, 
                     keys_to_extract=['Coordinates','Masses','Metallicity','SmoothingLength','Temperature'], 
                     calculate_qtys=False):
    p0_allsnaps = []
    for snapnum in tqdm(range(snapnums)):
        p0 = load_p0(snapdir, snapnum, Rvir=Rvir, loud=False, keys_to_extract=keys_to_extract, calculate_qtys=calculate_qtys)
        p0_allsnaps.append(p0)
    return p0_allsnaps

def u_CR(E_CR, M):
    '''Returns the specific CR energy in physical units (km/s)^2, 
    given `'CosmicRayEnergy'` in default units (1e10 Msun (km/s)^2), and mass in units 1e10 Msun.
    '''
    return E_CR/M

def Pressure(u, rho, gamma=None, typeP=None):
    '''Returns pressure in physical units k_B K/cm^3, 
    given specific energy in physical units (km/s)^2 and density in physical units 1e10 Msun/(kpc)^3.
    `gamma` can either be explictly given, or the pressure type (thermal, CR) specified.
    '''
    gammadict = {'CR':4/3, 'thermal':5/3}
    if gamma is None:
        gamma = gammadict[typeP]
    P = (gamma-1) * u * rho #in physical units 1e10 Msun/(kpc)^3 (km/s)^2
    return ((P * 1e10 * units.Msun/units.kpc**3 * (units.km/units.s)**2).to(k_B * units.K/units.cm**3)).value

def profiles( part, Tmask=True, rbins=np.power(10, np.arange(np.log10(0.005258639741921723), np.log10(3), 0.05)), outfile=None ):
    '''
    Default Tmask and rbins chosen to match Stern+20 Fig. 6: `np.power(10, np.arange(np.log10(0.005258639741921723), np.log10(1.9597976388995666), 0.05))`
   
    `part` is output of `load_allparticles`; `part[ptype]` is snapshot dict for particle `pytpe` and must have `Masses` and `r_scaled` columns.
    Input gas particle snapshot dict `part[0]` must have `r_scaled`, `Vi`, `posC`, and `Rvir` columns.
    
    If `outfile` is defined, a pickled dict of the profiles is saved to disk.
    '''
    p0 = part[0]
    rmid = (rbins[:-1]+rbins[1:])/2 #in units of Rvir
    logprofiles = {'T':[], 'rho':[], 'P_th':[], 'e_CR':[], 'P_CR':[], 'T lin':[], 'P_th lin':[], 'P_CR lin':[]}
    Mbins = { f'PartType{ptype}':[] for ptype in part.keys() }

    for r0,r1 in zip(rbins[:-1],rbins[1:]):
        idx = np.flatnonzero(Tmask & inrange( p0['r_scaled'], (r0, r1) ))
        V = np.sum(p0['Vi'][idx]) #volume of shell in physical kpc^3

        # Temperature profile: <log T/K>
        logTavg = np.sum(np.log10(p0['Temperature'][idx]) * p0['Vi'][idx]) / V
        logprofiles['T'].append(logTavg)

        # Density profile: log <rho/(Msun/pc^3)>
        rhoavg = np.sum(p0['Masses'][idx]) / V
        logprofiles['rho'].append(np.log10(rhoavg*10))

        # Thermal pressure profile: <log P_th/(k_B K/cm^3)>
        P_thi = Pressure(p0['InternalEnergy'][idx], p0['Density'][idx], typeP='thermal')
        logPthavg = np.sum(np.log10(P_thi) * p0['Vi'][idx]) / V
        logprofiles['P_th'].append(logPthavg)
        
        if 'CosmicRayEnergy' in p0:
            # CR energy density profile: log <e_CR/(1e10 Msun/kpc^3 (km/s)^2)>
            CReavg = np.sum(p0['CosmicRayEnergy'][idx]) / V
            logprofiles['e_CR'].append(np.log10(CReavg))

            # CR pressure profile: <log P_CR/(k_B K/cm^3)>
            P_CRi = Pressure(u_CR(p0['CosmicRayEnergy'][idx], p0['Masses'][idx]), p0['Density'][idx], typeP='CR')
            logPCRavg = np.sum(np.log10(P_CRi) * p0['Vi'][idx]) / V
            logprofiles['P_CR'].append(logPCRavg)

        # Temperature profile (averaging in linear space): log <T/K>
        Tavg = np.sum(p0['Temperature'][idx] * p0['Vi'][idx]) / V
        logprofiles['T lin'].append(np.log10(Tavg))

        # Thermal pressure profile (averaging in linear space): log <P_th/(k_B K/cm^3)>
        Pthavg = np.sum(P_thi * p0['Vi'][idx]) / V
        logprofiles['P_th lin'].append(np.log10(Pthavg))
        
        if 'CosmicRayEnergy' in p0:
            # CR pressure profile (averaging in linear space): log <P_CR/(k_B K/cm^3)>
            PCRavg = np.sum(P_CRi * p0['Vi'][idx]) / V
            logprofiles['P_CR lin'].append(np.log10(PCRavg))
        
        # Total mass in radial bin for each particle type: 1e10 Msun
        for ptype, p_i in part.items():
            idx_i = np.flatnonzero(inrange( p_i['r_scaled'], (r0, r1) ))
            Mbin_i = np.sum(p_i['Masses'][idx_i])
            Mbins[f'PartType{ptype}'].append(Mbin_i)
    
    resdict = {'rmid':rmid, **logprofiles, **Mbins, 'posC':p0['posC'], 'Rvir':p0['Rvir']}
    if outfile:
        pickle_save_dict(outfile, resdict)
    
    return resdict

def profiles_zbins(snapdir, redshifts=None, Rvir_allsnaps=None, zmin=1, zmax=4, zbinwidth=0.5, outfile=None):
    '''Compute profiles for all snapshots in each redshift bin. In each redshift bin, the virial radius at the median (center) snapshot is used.

    Parameters:
        `snapdir`: directory with snapshots
        `redshifts`: 1d array where `redshifts[i]` is the redshift at snapshot `i`
        `Rvir_allsnaps`: dictionary where `Rvir_allsnaps[i]` is the virial radius (in kpc) at snapshot `i`
        `zmin`, `zmax`, `zbinwidth`: redshift bins will be created with edges `z=[z0,z0+zbinwidth)`, where `z0` is in `np.arange(zmin,zmax,zbinwidth)`
    Returns:
        Dictionary where each key is a redshift bin, and each item is a dict with key snapshot and item output of `profiles` at that snapshot.
        Output will be saved to disk as a nested dict in HDF5 format if `outfile` is passed (output file path/name).
    '''
    # Load Rvir from `Rvir_{halo}_noAGNfb.h5` file if not passed
    if Rvir_allsnaps is None:
        Rvir_allsnaps, redshifts = Rvir_sim(snapdir)
    
    allprofiles = {}
    for z0 in np.arange(zmin,zmax,zbinwidth):
        allprofiles[f'z0_{z0}'] = {}
        z1 = z0 + zbinwidth
        
        print(f'Beginning bin from z={z0} to {z1}.')
        snapnums_bin = np.flatnonzero(inrange(redshifts, (z0,z1), right_bound_inclusive=False))
        snapnum_median = snapnums_bin[len(snapnums_bin)//2]
        Rvir = Rvir_allsnaps[snapnum_median]
        print(f'Median redshift is {redshifts[snapnum_median]} with snapnum {snapnum_median} and virial radius {Rvir} kpc.')
        
        print(f'Computing profiles for snapshots {snapnums_bin.min()} to {snapnums_bin.max()}.')
        for snapnum in tqdm(snapnums_bin):
            part = load_allparticles(snapdir, snapnum, Rvir=Rvir, loud=0)
            allprofiles[f'z0_{z0}']['SnapNum' + str(snapnum).zfill(3)] = profiles(part)
    
    if outfile:
        dicttoh5(allprofiles, outfile, mode='w')
    return allprofiles

def plot_rho_profiles_zbins(allprofiles, zbinwidth=0.5, simname='', outfile='', rbins=np.power(10, np.arange(np.log10(0.005258639741921723), np.log10(1.9597976388995666), 0.05))):
    '''Plots mean and median rho profile for every redshift bin in `allprofiles`.
    
    Parameters:
        `allprofiles`: output of `profiles_zbins`
        `zbinwidth`: width of redshift bins
        `simname`: simulation name for plot title
        `outfile`: output file path/name with extension
    '''
    # For each redshift bin, create 2D array where the rows are rho profiles for every snapshot in bin
    all_rhoavgbins = {k:np.array([logprofiles['rho'] for rmid, logprofiles in profiles_zbin]) for k,profiles_zbin in allprofiles.items()}

    # For each redshift bin, plot mean and median rho profile
    rmid = (rbins[:-1]+rbins[1:])/2

    plt.figure(dpi=120)
    for z0, c in zip(all_rhoavgbins.keys(), COLOR_SCHEME):
        rhoavgbins_mean = np.mean(all_rhoavgbins[z0], axis=0)
        rhoavgbins_median = np.median(all_rhoavgbins[z0], axis=0)
        plt.plot(np.log10(rmid), rhoavgbins_mean, '-', label=f'z=[{z0},{z0+zbinwidth})', c=c)
        plt.plot(np.log10(rmid), rhoavgbins_median, '--', c=c)

    plt.xlabel(r'$\log (r/R_{vir})$')
    plt.ylabel(profilelabels['rho'])
    plt.legend()
    plt.title(simname)

    if outfile:
        plt.savefig(outfile)

def plot_profiles_zbins(allprofiles, ax, profiletype='rho', zbinwidth=0.5, rbins=np.power(10, np.arange(np.log10(0.005258639741921723), np.log10(1.9597976388995666), 0.05)), cmap=plt.cm.Reds, xlabel=False, ylabel=False):
    '''Plots median rho, T, e_CR, P_thermal, or P_CR profile for every redshift bin in `allprofiles`.
    
    Parameters:
        `allprofiles`: output of `profiles_zbins`
        `profiletype`: profile to plot (one of `'rho'`, `'T'`, `'P_th'`, `'e_CR'`, `'P_CR'`)
        `zbinwidth`: width of redshift bins
    '''
    # For each redshift bin, create 2D array where the rows are rho/T/e_CR/P_th/P_CR profiles for every snapshot in bin
    all_profileavgbins = {float(k):np.array([logprofiles[profiletype] for logprofiles in profiles_zbin.values()]) for k,profiles_zbin in allprofiles.items()}

    # For each redshift bin, plot median profile
    rmid = (rbins[:-1]+rbins[1:])/2

    for z0, c in zip( sorted(all_profileavgbins.keys()), cmap(np.linspace(0.1,1,len(all_profileavgbins.keys())))[::-1] ):
        profileavgbins_median = np.median(all_profileavgbins[z0], axis=0)
        ax.plot(np.log10(rmid), profileavgbins_median, '-', label=f'z=[{z0},{z0+zbinwidth})', c=c)
    
    if xlabel:
        ax.set_xlabel(r'$\log (r/R_{vir})$')
    if ylabel:
        ax.set_ylabel( profilelabels[profiletype] )

### COSMOLOGY CODE ###
def scale_factor_to_redshift(a):
    z = 1/a - 1
    return z

def Ez(OmegaM0, OmegaL0, z):
    return ( OmegaM0*(1+z)**3 + OmegaL0 )**0.5

def OmegaM(OmegaM0, OmegaL0, z):
    return OmegaM0 * (1+z)**3 / Ez(OmegaM0, OmegaL0, z)**2

def xz(OmegaM0, OmegaL0, z):
    return OmegaM(OmegaM0, OmegaL0, z) - 1

def deltavir(OmegaM0, OmegaL0, z):
    '''Virial overdensity fitting function from Bryan & Norman (1998)'''
    x = xz(OmegaM0, OmegaL0, z)
    return 18*np.pi**2 + 82*x - 39*x**2

def rhocritz(OmegaM0, OmegaL0, z):
    '''Returns the critical density at z in units h^2 Msun/Mpc^3.'''
    rhocrit0 = 2.77536627e11 #h^2 Msun/Mpc^3
    return rhocrit0 * Ez(OmegaM0, OmegaL0, z)**2

def Rvir(Mvir, OmegaM0, OmegaL0, z):
    '''Returns in units of Mpc/h assuming [Mvir]=Msun/h.'''
    return ( 3/4 * Mvir / (deltavir(OmegaM0, OmegaL0, z) * rhocritz(OmegaM0, OmegaL0, z) * np.pi) )**(1/3)
