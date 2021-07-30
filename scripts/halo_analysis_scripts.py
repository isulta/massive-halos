import numpy as np
from numba import njit

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
def load_p0(snapdir, snapnum, ahf_path=None, Rvir=None, loud=1):
    '''Loads gas particle snapshot and adds `CoordinatesRelative`, `r`, `r_scaled`, `Vi`, `posC`, and `Rvir` columns to dictionary.'''
    p0 = openSnapshot(snapdir, snapnum, 0, loud=loud)
    p1 = openSnapshot(snapdir, snapnum, 1, loud=loud, keys_to_extract=['Coordinates', 'Masses'])
    
    if loud:
        print(f"Loading redshift {p0['Redshift']}")

    posC = halo_center_wrapper(p1)[0]

    if ahf_path:
        _, Rvir = load_AHF('', snapnum, p0['Redshift'], hubble=p0['HubbleParam'], ahf_path=ahf_path, extra_names_to_read=[])

    # position relative to center
    p0['CoordinatesRelative'] = p0['Coordinates'] - posC

    # distance from halo center
    p0['r'] = np.linalg.norm(p0['CoordinatesRelative'], axis=1)

    # distance from halo center in units of virial radius
    p0['r_scaled'] = p0['r']/Rvir

    # volume of each particle
    p0['Vi'] = p0['Masses']/p0['Density']
    
    p0['posC'] = posC
    p0['Rvir'] = Rvir
    
    return p0

def profiles( p0, Tmask=True, rbins=np.power(10, np.arange(np.log10(0.005258639741921723), np.log10(1.9597976388995666), 0.05)), outfile=None ):
    '''
    Default Tmask and rbins chosen to match Stern+20 Fig. 6.
    Input gas particle snapshot dict `p0` must have `r_scaled`, `Vi`, `posC`, and `Rvir` columns.
    If `outfile` is defined, a pickled dict of the profiles is saved to disk.
    '''
    rmid = (rbins[:-1]+rbins[1:])/2
    logTavgbins = []
    rhoavgbins = []

    for r0,r1 in zip(rbins[:-1],rbins[1:]):
        idx = np.flatnonzero(Tmask & inrange( p0['r_scaled'], (r0, r1) ))

        # Temperature profile
        logTavg = np.sum(np.log10(p0['Temperature'][idx]) * p0['Vi'][idx]) / np.sum(p0['Vi'][idx])
        logTavgbins.append(logTavg)

        # Density profile
        rhoavg = np.sum(p0['Masses'][idx]) / np.sum(p0['Vi'][idx])
        rhoavgbins.append(rhoavg)
    
    if outfile:
        pickle_save_dict(outfile, {'rmid':rmid, 'logTavgbins':logTavgbins, 'rhoavgbins':rhoavgbins, 'posC':p0['posC'], 'Rvir':p0['Rvir']})
    
    return rmid, logTavgbins, rhoavgbins

def profiles_zbins(snapdir, redshifts, Rvir_allsnaps, zmin=1, zmax=4, zbinwidth=0.5, outfile=None):
    '''Compute profiles for all snapshots in each redshift bin.

    Parameters:
        `snapdir`: directory with snapshots
        `redshifts`: 1d array where `redshifts[i]` is the redshift at snapshot `i`
        `Rvir_allsnaps`: dictionary where `Rvir_allsnaps[i]` is the virial radius (in kpc) at snapshot `i`
        `zmin`, `zmax`, `zbinwidth`: redshift bins will be created with edges `z=[z0,z0+zbinwidth)`, where `z0` is in `np.arange(zmin,zmax,zbinwidth)`
    Returns:
        Dictionary where each key is a redshift bin, and each item is a list of `(rmid, logTavgbins, rhoavgbins)` calculated for each snapshot in that redshift bin.
        Output will be pickled and saved to disk if `outfile` is passed (output file path/name).
    '''
    allprofiles = {}
    for z0 in np.arange(zmin,zmax,zbinwidth):
        allprofiles[z0] = []
        z1 = z0 + zbinwidth
        
        print(f'Beginning bin from z={z0} to {z1}.')
        snapnums_bin = np.flatnonzero(inrange(redshifts, (z0,z1), right_bound_inclusive=False))
        snapnum_median = snapnums_bin[len(snapnums_bin)//2]
        Rvir = Rvir_allsnaps[snapnum_median]
        print(f'Median redshift is {redshifts[snapnum_median]} with snapnum {snapnum_median} and virial radius {Rvir} kpc.')
        
        print(f'Computing profiles for snapshots {snapnums_bin.min()} to {snapnums_bin.max()}.')
        for snapnum in tqdm(snapnums_bin):
            p0 = load_p0(snapdir, snapnum, Rvir=Rvir, loud=0)
            allprofiles[z0].append( profiles(p0) )
    
    if outfile:
        pickle_save_dict(outfile, {'allprofiles':allprofiles})
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
    all_rhoavgbins = {k:np.array([rhoavgbins for rmid, logTavgbins, rhoavgbins in profiles_zbin]) for k,profiles_zbin in allprofiles.items()}

    # For each redshift bin, plot mean and median rho profile
    rmid = (rbins[:-1]+rbins[1:])/2

    plt.figure(dpi=120)
    for z0, c in zip(all_rhoavgbins.keys(), COLOR_SCHEME):
        rhoavgbins_mean = np.mean(all_rhoavgbins[z0], axis=0)
        rhoavgbins_median = np.median(all_rhoavgbins[z0], axis=0)
        plt.plot(np.log10(rmid), np.log10(rhoavgbins_mean), '-', label=f'z=[{z0},{z0+zbinwidth})', c=c)
        plt.plot(np.log10(rmid), np.log10(rhoavgbins_median), '--', c=c)

    plt.xlabel(r'$\log (r/R_{vir})$')
    plt.ylabel(r'$\log \left<\rho \right>$')
    plt.legend()
    plt.title(simname)

    if outfile:
        plt.savefig(outfile)

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

'''
def Rvir(coords, masses, com):
    # TODO
    r = dist(coords, com)
    # asort = np.argsort(r)
    return np.sum(masses[r <= (358.68*0.697)]) #/ (4/3 * np.pi * 358.68**3) * 1e19
'''

