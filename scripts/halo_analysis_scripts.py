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

def halo_center(coords, masses, shrinkpercent=2.5, minparticles=1000, initialradiusfactor=1, verbose=False):
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
def load_p0(snapdir, snapnum, ahf_path=None, Rvir=None):
    '''Loads gas particle snapshot and adds `CoordinatesRelative`, `r`, `r_scaled`, `Vi`, `posC`, and `Rvir` columns to dictionary.'''
    p0 = openSnapshot(snapdir, snapnum, 0, loud=1)
    p1 = openSnapshot(snapdir, snapnum, 1, loud=1, keys_to_extract=['Coordinates', 'Masses'])
    
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
    
    return rmid, logTavgbins

### COSMOLOGY CODE ###
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

