from astropy import units as un, constants as cons
import numpy as np

import cooling_flow as CF
import HaloPotential as Halo
import WiersmaCooling as Cool
from scripts.precipitation import PrecipitationModel

from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.integrate import quad
from scipy.misc import derivative
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import pickle
from silx.io.dictdump import dicttoh5, h5todict
from scipy.signal import savgol_filter
from tqdm import tqdm
import utilities.coordinate as utc
import iht

from scripts.halo_analysis_scripts import *
import matplotlib.colors
# %matplotlib inline
plt.rcParams['figure.dpi'] = 110

Zsun = 0.0142 #Asplund+09

def calculateMr(p, Rmin=None, Rmax=None, bins=100):
    if Rmin is None:
        Rmin = 0.01 #pkpc
        Rmax = 10 * p[1]['Rvir'] #pkpc
    rbins = np.logspace(np.log10(Rmin), np.log10(Rmax), bins) #pkpc
    # M(r)
    pall = {}
    pall['Coordinates'] = np.concatenate([p[k]['Coordinates'] for k in p.keys()]) #Assume Coordinates already centered
    pall['Masses'] = np.concatenate([p[k]['Masses'] for k in p.keys()])
    pall['r'] = np.linalg.norm(pall['Coordinates'], axis=1)

    Mrbins = np.array([np.sum(pall['Masses'][pall['r']<=r]) for r in rbins]) * 1e10 #M(r) for r in rbins where [M]=Msun and [r]=pkpc
    Mr = {'r':rbins, 'M(r)':Mrbins, 'Rvir':p[1]['Rvir']}
    return Mr

def part_calculations(p, Rmax_Z=1, usetcoolWiersma=False, usetcoolWiersma_Zbins=True, Zbins=1000, useHubbleFlow=False, rotateDisk=True):
    '''Calculate additional fields and add them to particle data dict.'''
    # Shift position and velocity relative to halo center and COM's (r/Rvir<1) velocity
    for i in p.keys(): p[i]['Coordinates'] -= p[i]['posC']
    for repi in range(3):
        velC = iht.velocity_COM(p)
        for i in p.keys(): p[i]['Velocities'] -= velC

    # Rotate coordinates to disk axis
    if rotateDisk: iht.rotate_coords(p, p[4]['Rvir'])
    
    # Calculate spherical velocites
    p[0]['vrad'], p[0]['vtheta'], p[0]['vphi'] = iht.spherical_velocities(v=p[0]['Velocities'], r=p[0]['Coordinates'])
    p[0]['vrad'] *= -1 # Define vrad as inflow velocity
    p[0]['theta'] = np.arctan2( np.sqrt(p[0]['Coordinates'][:,0]**2 + p[0]['Coordinates'][:,1]**2), p[0]['Coordinates'][:,2] )

    # Convert peculiar velocity to total velocity including Hubble flow
    if 'Omega0' in p[0]: #FIRE-2
        OmegaM0, OmegaL0, hubble, z = p[0]['Omega0'], p[0]['OmegaLambda'], p[0]['HubbleParam'], p[0]['Redshift']
    else: #FIRE-3
        OmegaM0, OmegaL0, hubble, z = p[0]['Omega_Matter'], p[0]['Omega_Lambda'], p[0]['HubbleParam'], p[0]['Redshift']
    p[0]['vradpec'] = p[0]['vrad'].copy()
    if useHubbleFlow:
        p[0]['vrad'] = p[0]['vrad'] - Ez(OmegaM0, OmegaL0, z)*100*hubble*p[0]['r_scaled']*p[0]['Rvir']/1000
        print(Ez(OmegaM0, OmegaL0, z)*100*hubble*1*p[0]['Rvir']/1000)

    # Calculate MachNumber
    gamma = 5/3
    if 'SoundSpeed' not in p[0]: p[0]['SoundSpeed'] = np.sqrt( gamma*(gamma-1) * p[0]['InternalEnergy'] ) #pkm/s
    p[0]['MachNumber'] = p[0]['vrad'] / p[0]['SoundSpeed']

    # Calculate mass-weighted average metallicity/Zsun within Rmax_Z*Rvir
    p[0]['Z2Zsun']  = np.sum((p[0]['Metallicity'][:,0] * p[0]['Masses'])[p[0]['r_scaled']<=Rmax_Z]) / np.sum(p[0]['Masses'][p[0]['r_scaled']<=Rmax_Z]) / Zsun
    
    # Calculate metallicity/Zsun
    p[0]['MetallicitySolar'] = p[0]['Metallicity'][:,0]/Zsun

    # Calculate number density of hydrogen atoms
    XH = 1 - p[0]['Metallicity'][:,0] - p[0]['Metallicity'][:,1] #hydrogen mass fraction
    nH = ( XH * (p[0]['Density'] * 1e10 * un.Msun/un.kpc**3) / cons.m_p ).to(un.cm**-3) #number density of hydrogen atoms
    p[0]['nH'] = nH.to(un.cm**-3).value

    # Calculate entropy of all gas particles
    y_helium = p[0]['Metallicity'][:,1] / (4*(1-p[0]['Metallicity'][:,1]))
    mu = (1 + 4*y_helium) / (1+y_helium+p[0]['ElectronAbundance'])
    n = p[0]['Density']*(1e10 * un.Msun/un.kpc**3) / (mu * cons.m_p)
    K = cons.k_B * p[0]['Temperature']*un.K / n**(2/3)
    p[0]['Kn'] = K.to(un.keV * un.cm**2).value

    if 'CoolingRate' in p[0]: p[0]['CoolingRateFIRE'] = p[0]['CoolingRate'].copy()
    if usetcoolWiersma:
        # Calculate cooling time from predicted CoolingRate from cooling flow solutions code (Wiersma et al. 2009 cooling function with Z=mass-weighted avg. Z in halo)
        with np.errstate(divide='ignore', invalid='ignore'):
            cooling = Cool.Wiersma_Cooling(p[0]['Z2Zsun'],p[0]['Redshift'])
        CoolingRate_pred = cooling.LAMBDA(p[0]['Temperature']*un.K, p[0]['nH']*(un.cm**-3)) * (p[0]['nH']*(un.cm**-3))**2
        tcool_pred = ((Pressure(p[0]['InternalEnergy'], p[0]['Density'], typeP='thermal') * cons.k_B * un.K / un.cm**3) / ( CoolingRate_pred * (5/3-1) ))
        p[0]['tcool'] = tcool_pred.to(un.Gyr).value
    elif usetcoolWiersma_Zbins:
        # Calculate cooling time from predicted CoolingRate from cooling flow solutions code (Wiersma et al. 2009 cooling function with Z=mean Z in `Zbins` Z bins)
        Zidxsplit = np.array_split( np.argsort(p[0]['Metallicity'][:,0]/Zsun), Zbins )
        p[0]['tcool'] = np.zeros_like(p[0]['nH'])
        p[0]['CoolingRate'] = np.zeros_like(p[0]['nH'])
        for Zidx in Zidxsplit:
            Z2Zsun = np.mean(p[0]['Metallicity'][Zidx,0]) / Zsun
            cooling = Cool.Wiersma_Cooling(Z2Zsun, p[0]['Redshift'])
            CoolingRate_pred = cooling.LAMBDA(p[0]['Temperature'][Zidx]*un.K, p[0]['nH'][Zidx]*(un.cm**-3)) * (p[0]['nH'][Zidx]*(un.cm**-3))**2
            tcool_pred = ((Pressure(p[0]['InternalEnergy'][Zidx], p[0]['Density'][Zidx], typeP='thermal') * cons.k_B * un.K / un.cm**3) / ( CoolingRate_pred * (5/3-1) ))
            p[0]['tcool'][Zidx] = tcool_pred.to(un.Gyr).value
            p[0]['CoolingRate'][Zidx] = CoolingRate_pred.to(un.erg/un.s*un.cm**-3).value
    else:
        # Calculate tcool from 'CoolingRate'='InternalEnergy'/tcool where tcool is in code units
        p[0]['tcool'] = 1/p[0]['CoolingRate'] * p[0]['InternalEnergy']*(p[0]['HubbleParam']**-1 * un.kpc / (un.km/un.s)).to(un.Gyr).value#0.978

    # Calculate Hubble time
    p[0]['tHubble'] = (1/(100 * p[0]['HubbleParam'] * (un.km/un.s/un.Mpc))).to(un.Gyr)

    # Calculate entropy: K/(k_B K/cm^3/(1e10 Msun/kpc^3))
    p[0]['K'] = Pressure(p[0]['InternalEnergy'], p[0]['Density'], typeP='thermal') / p[0]['Density']**(5/3)

def interp_smooth(xarr, arr, fittingmask, smoothmask):
    xarr = xarr[smoothmask]
    arr = arr[smoothmask]
    fittingmask = fittingmask[smoothmask]
    
    arr_smooth = np.full_like(arr, np.nan)
    arr_smooth[fittingmask] = arr[fittingmask]
    arr_smooth[:np.flatnonzero(fittingmask)[0]] = arr_smooth[np.flatnonzero(fittingmask)[0]]
    arr_smooth[np.flatnonzero(fittingmask)[-1]:] = arr_smooth[np.flatnonzero(fittingmask)[-1]]

    w = np.isnan(arr_smooth)
    arr_smooth[w]=0
    return UnivariateSpline(xarr, arr_smooth, w=~w)

class Potential_FIRE(CF.Potential):
    def __init__(self, Mr, Rmax=3, pro=None, fitrange=(0.04, 1.6), fittingmask=None):
        self.Rvir = Mr['Rvir'][()]*un.kpc
        
        Phir = cumtrapz( Mr['M(r)'] / Mr['r']**2,  Mr['r'], initial=0)
        Rvir3_idx = np.argmin(np.abs(Mr['r'] - Rmax*Mr['Rvir'][()]))
        Phir = Phir - Phir[Rvir3_idx]
        self.Phi_interp = interp1d(Mr['r'], Phir)
        
        vcr = np.sqrt(Mr['M(r)']/Mr['r'])
        vcr = savgol_filter(vcr, 11, 2) #changed from 10 to 11
        self.vc_interp = interp1d(Mr['r'], vcr, fill_value='extrapolate')
        
        lnvcr = np.gradient(np.log(vcr), np.log(Mr['r']))
        self.lnvc_interp = interp1d(Mr['r'], lnvcr)

        if pro is not None: 
            self.Redshift = pro['Redshift'][()]

            idxspl = inrange(pro['rmid'], fitrange)
            Pturbratio = 10**pro['P_turb'] / 10**pro['Pth_Mweighted']
            # self.alpha_interp = interp1d(np.log10(pro['rmid']*pro['Rvir'])[idxspl], savgol_filter(Pturbratio[idxspl], 10, 2), kind='cubic')
            # self.logZ_interp = interp1d(np.log10(pro['rmid']*pro['Rvir'])[idxspl], savgol_filter(pro['Z_Lweighted'][idxspl], 10, 2), kind='cubic')

            self.alpha_interp = interp_smooth(np.log10(pro['rmid']*pro['Rvir']), Pturbratio, fittingmask, idxspl)
            self.logZ_interp = interp_smooth(np.log10(pro['rmid']*pro['Rvir']), pro['Z_Mweighted'], fittingmask, idxspl) #TODO change to Lweighted

    def vc(self, r):
        r = r.to(un.kpc).value
        return ( self.vc_interp(r) * (cons.G*un.Msun/un.kpc)**0.5 ).to(un.km/un.s)
    def Phi(self, r):
        r = r.to(un.kpc).value
        return self.Phi_interp(r) * (cons.G*un.Msun/un.kpc).to((un.km/un.s)**2)
    def dlnvc_dlnR(self, r):
        r = r.to(un.kpc).value
        return self.lnvc_interp(r)
    def get_Rcirc(self):
        # TODO
        # return minimize(lambda x: -self.vc(x*un.kpc), 1, method = 'Nelder-Mead').x[0] * un.kpc
    
        R_min = 0.1*un.kpc
        R_max = 50*un.kpc
        xarr = np.logspace(np.log10(R_min.to(un.kpc).value), np.log10(R_max.to(un.kpc).value), 100) * un.kpc
        return xarr[np.argmax(self.vc(xarr))]
    def potential_test(self, fig=None, axes=None):
        R_min = 0.1*un.kpc
        R_max = 1.5*self.Rvir  
        xarr = np.logspace(np.log10(R_min.to(un.kpc).value), np.log10(R_max.to(un.kpc).value), 100) * un.kpc

        if fig is None: fig, axes = plt.subplots(1, 3, sharex=True, sharey=False, gridspec_kw={'wspace': .3, 'hspace':.04}, figsize=[4.8*3,4.8*1], dpi=150, facecolor='w')

        axes[0].plot(xarr, self.Phi(xarr))
        axes[0].set_xscale('log')
        # axes[0].set_yscale('log')
        axes[0].set_xlabel('$r$/kpc')
        axes[0].set_ylabel('$\Phi$/($km^2$/$s^2$)')


        axes[1].plot(xarr, self.vc(xarr))
        axes[1].axvline(self.get_Rcirc().value, label='$R_{circ}$', c='k')
        axes[1].legend()
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].set_xlabel('$r$/kpc')
        axes[1].set_ylabel('$v_c$/(km/s)')

        axes[2].plot(xarr, self.dlnvc_dlnR(xarr))
        axes[2].set_xscale('log')
        # axes[2].set_yscale('symlog')
        axes[2].set_xlabel('$r$/kpc')
        axes[2].set_ylabel('$\mathrm{d} ln{v_c} /\mathrm{d} ln{r}$')
    def vesc(self, r):
        return ( -2*self.Phi(r) )**0.5 #see Pandya+21

def continuous_mode(data, bins=100, range=None):
    # Create a histogram
    hist, bin_edges = np.histogram(data, bins=bins, range=range)
    
    # Find the bin with the highest frequency (mode)
    mode_index = np.argmax(hist)
    
    # The mode value will be the midpoint of the mode bin
    mode = (bin_edges[mode_index] + bin_edges[mode_index + 1]) / 2
    
    return mode

def profiles( part, Tmask=True, rbins=np.power(10, np.arange(np.log10(0.005258639741921723), np.log10(3), 0.05)), selectMainBranch=True, mainBranchHalfWidth=0.5, potential=None ):
    '''
    Default Tmask and rbins chosen to match Stern+20 Fig. 6: `np.power(10, np.arange(np.log10(0.005258639741921723), np.log10(1.9597976388995666), 0.05))`
   
    `part` is output of `load_allparticles`; `part[ptype]` is snapshot dict for particle `pytpe` and must have `Masses` and `r_scaled` columns.
    Input gas particle snapshot dict `part[0]` must have `r_scaled`, `Vi`, `posC`, and `Rvir` columns.
    '''
    p0 = part[0]
    rmid = (rbins[:-1]+rbins[1:])/2 #in units of Rvir
    logprofiles = { k:[] for k in 
                   ['rho', 'T', 'Z', 'MachNumber', 'vrad', 'vradpec', 'cs', 'tcool', 'nH', 'P_th', 'P_turb', 'P_turb_cart',
                    'e_CR', 'P_CR', 'K', 'K_Mweighted', 'Kn', 'Z_Mweighted', 'T_Mweighted', 'Pth_Mweighted', 
                    'nH_Mweighted', 'P_turb_rad', 'tcoolshell', 'tcool_Mweighted', 'P_mag', 'CoolingRate', 
                    'Z_Lweighted', 'T_mode', 'nH_mode', 'Mgas', 'Mdot', 'MdotIF', 'js'] }
    hists = []
    Mbins = { f'TotalMass:PartType{ptype}':[] for ptype in part.keys() }

    if potential is not None: p0['vc'] = potential.vc(p0['r_scaled']*p0['Rvir'] * un.kpc).value

    for r0,r1 in zip(rbins[:-1],rbins[1:]):
        idx = np.flatnonzero(Tmask & inrange( p0['r_scaled'], (r0, r1) ))
        if selectMainBranch:
            T_mode = continuous_mode(np.log10(p0['Temperature'][idx]), range=(3,8)) #log space
            logprofiles['T_mode'].append((T_mode, T_mode - mainBranchHalfWidth, T_mode + mainBranchHalfWidth))

            nH_mode = continuous_mode(np.log10(p0['nH'][idx]), range=(-7,0)) #log space
            logprofiles['nH_mode'].append((nH_mode, nH_mode - mainBranchHalfWidth, nH_mode + mainBranchHalfWidth))

            idx = np.flatnonzero(inrange( p0['r_scaled'], (r0, r1) )&
                                 inrange( np.log10(p0['Temperature']), (T_mode - mainBranchHalfWidth, T_mode + mainBranchHalfWidth) )&
                                 inrange( np.log10(p0['nH']), (nH_mode - mainBranchHalfWidth, nH_mode + mainBranchHalfWidth) ))
            idxcool = np.flatnonzero(inrange( p0['r_scaled'], (r0, r1) )&
                                     ~(inrange( np.log10(p0['Temperature']), (T_mode - mainBranchHalfWidth, T_mode + mainBranchHalfWidth) )&
                                       inrange( np.log10(p0['nH']), (nH_mode - mainBranchHalfWidth, nH_mode + mainBranchHalfWidth) )))
            logprofiles['Mgas'].append(np.sum(p0['Masses'][idx]))
        V = np.sum(p0['Vi'][idx]) #volume of shell in physical kpc^3
        Mshell = np.sum(p0['Masses'][idx]) #mass of shell

        # Density profile: log <rho/(Msun/pc^3)>
        rhoavg = (Mshell / V)*10
        logprofiles['rho'].append(np.log10(rhoavg))

        # Mdot profile: Msun/yr
        dL = (r1 - r0)*p0['Rvir']
        # vrad = p0['vrad'][idx] - Ez(p0['Omega_Matter'], p0['Omega_Lambda'], p0['Redshift'][()])*100*p0['HubbleParam']*p0['r_scaled'][idx]*p0['Rvir']/1000
        # if 0.45 < r0 and r0 < 0.55: print(r0, Ez(p0['Omega_Matter'], p0['Omega_Lambda'], p0['Redshift'][()]), np.median(p0['vrad'][idx]), np.median(Ez(p0['Omega_Matter'], p0['Omega_Lambda'], p0['Redshift'][()])*100*p0['HubbleParam']*p0['r_scaled'][idx]*p0['Rvir']/1000), np.median(vrad))
        Mdot = np.sum(p0['vrad'][idx] * 1e10 * p0['Masses'][idx] / dL ) * (un.km / un.s * un.Msun / un.kpc).to(un.Msun / un.yr)
        logprofiles['Mdot'].append(Mdot)

        # Inflow Mdot profile: Msun/yr
        dL = (r1 - r0)*p0['Rvir']
        idxIF = idx[ p0['vrad'][idx] > 0 ]
        MdotIF = np.sum(p0['vrad'][idxIF] * 1e10 * p0['Masses'][idxIF] / dL ) * (un.km / un.s * un.Msun / un.kpc).to(un.Msun / un.yr)
        logprofiles['MdotIF'].append(MdotIF)

        # Temperature profile (averaging in linear space): log <T/K>
        Tavg = np.sum(p0['Temperature'][idx] * p0['Vi'][idx]) / V
        logprofiles['T'].append(np.log10(Tavg))
        Tavg_Mweighted = np.sum(p0['Temperature'][idx] * p0['Masses'][idx]) / Mshell
        logprofiles['T_Mweighted'].append(np.log10(Tavg_Mweighted))
        
        # Z/Zsun profile (averaging in linear space): log <Z/Zsun>
        Zavg = np.sum(p0['Metallicity'][:,0][idx]/Zsun * p0['Vi'][idx]) / V
        logprofiles['Z'].append(np.log10(Zavg))
        
        Zavg_Mweighted = np.sum(p0['Metallicity'][:,0][idx]/Zsun * p0['Masses'][idx]) / Mshell
        logprofiles['Z_Mweighted'].append(np.log10(Zavg_Mweighted))
        
        Luminosity = p0['CoolingRate'][idx] * p0['Vi'][idx]
        Zavg_Lweighted = np.sum(p0['Metallicity'][:,0][idx]/Zsun * Luminosity) / np.sum(Luminosity)
        logprofiles['Z_Lweighted'].append(np.log10(Zavg_Lweighted))

        # Mach number profile (averaging in linear space): <M>
        MachNumberavg = np.sum(p0['MachNumber'][idx] * p0['Vi'][idx]) / V
        logprofiles['MachNumber'].append(MachNumberavg)
        
        # Radial velocity profile (averaging in linear space): <vrad/(km/s)>
        vradavg = np.sum(p0['vrad'][idx] * p0['Vi'][idx]) / V
        logprofiles['vrad'].append(vradavg)
        
        vradpecavg = np.sum(p0['vradpec'][idx] * p0['Vi'][idx]) / V
        logprofiles['vradpec'].append(vradpecavg)
        
        # Sound speed profile (averaging in linear space): <cs/(km/s)>
        csavg = np.sum(p0['SoundSpeed'][idx] * p0['Vi'][idx]) / V
        logprofiles['cs'].append(csavg)
        
        # tcool profile (averaging in linear space): log <tcool/Gyr>
        # filter out particles where tcool is infinite, keeping shell volume V the same
        assert not np.any(p0['tcool']==np.inf)
        '''idxcool = np.flatnonzero(Tmask & inrange( p0['r_scaled'], (r0, r1) ) & (p0['tcool']!=np.inf))
        print(len(idxcool)/len(idx)*100)
        tcoolavg = np.sum(p0['tcool'][idxcool] * p0['Vi'][idxcool]) / V
        '''
        tcoolavg = np.sum(p0['tcool'][idx] * p0['Vi'][idx]) / V
        logprofiles['tcool'].append(np.log10(tcoolavg))

        tcoolavg_Mweighted = np.sum(p0['tcool'][idx] * p0['Masses'][idx]) / Mshell
        logprofiles['tcool_Mweighted'].append(np.log10(tcoolavg_Mweighted))

        # tcoolavgshell = (np.sum(p0['InternalEnergy'][idx]*p0['Density'][idx]) / np.sum(p0['CoolingRate'][idx]) * 1e10 * un.Msun/un.kpc**3 * (un.km/un.s)**2 / (un.erg/un.s*un.cm**-3)).to(un.Gyr).value
        tcoolavgshell = (np.sum(p0['InternalEnergy'][idx]*p0['Masses'][idx]) / np.sum(Luminosity) * 1e10 * un.Msun * (un.km/un.s)**2 / (un.erg/un.s*un.cm**-3 * un.kpc**3)).to(un.Gyr).value
        logprofiles['tcoolshell'].append(np.log10(tcoolavgshell))

        CoolingRateavg = np.sum(p0['CoolingRate'][idx] * p0['Vi'][idx]) / V
        logprofiles['CoolingRate'].append(np.log10(CoolingRateavg))

        if 'MagneticField' in p0:
            # Magnetic pressure profile (averaging in linear space): log <P_mag/(k_B K/cm^3)>
            P_magi = (np.linalg.norm(p0['MagneticField'][idx], axis=1)**2 * un.Gauss**2 / (2*cons.mu0)).to(cons.k_B * un.K / un.cm**3).value
            Pmagavg = np.sum(P_magi * p0['Vi'][idx]) / V
            logprofiles['P_mag'].append(np.log10(Pmagavg))

        # nH profile (averaging in linear space): log <nH/cm^-3>
        nHavg = np.sum(p0['nH'][idx] * p0['Vi'][idx]) / V
        logprofiles['nH'].append(np.log10(nHavg))
        nHavg_Mweighted = np.sum(p0['nH'][idx] * p0['Masses'][idx]) / Mshell
        logprofiles['nH_Mweighted'].append(np.log10(nHavg_Mweighted))
        
        # Thermal pressure profile (averaging in linear space): log <P_th/(k_B K/cm^3)>
        P_thi = Pressure(p0['InternalEnergy'][idx], p0['Density'][idx], typeP='thermal')
        Pthavg = np.sum(P_thi * p0['Vi'][idx]) / V
        logprofiles['P_th'].append(np.log10(Pthavg))
        Pthavg_Mweighted = np.sum(P_thi * p0['Masses'][idx]) / Mshell
        logprofiles['Pth_Mweighted'].append(np.log10(Pthavg_Mweighted))
        
        # Entropy profile (averaging in linear space): log <K/(k_B K/cm^3/(1e10 Msun/kpc^3))>
        Ki = P_thi / p0['Density'][idx]**(5/3)
        Kavg = np.sum(Ki * p0['Vi'][idx]) / V
        Kavg_Mweighted = np.sum(Ki * p0['Masses'][idx]) / Mshell
        logprofiles['K'].append(np.log10(Kavg))
        logprofiles['K_Mweighted'].append(np.log10(Kavg_Mweighted))

        # Entropy profile (averaging in linear space): log <K/(keV cm^2))>
        Knavg = np.sum(p0['Kn'][idx] * p0['Vi'][idx]) / V
        logprofiles['Kn'].append(np.log10(Knavg))
        
        # Turbulent pressure profile: log <P_turb/(k_B K/cm^3)>
        var = 1/3 * ( np.var(p0['vrad'][idx]) + np.var(p0['vtheta'][idx]) + np.var(p0['vphi'][idx]) )
        Pturbavg = (rhoavg*un.Msun/un.pc**3 * var*(un.km/un.s)**2).to(cons.k_B * un.K / un.cm**3).value
        logprofiles['P_turb'].append(np.log10(Pturbavg))
        
        var2 = np.var(p0['vrad'][idx])
        Pturbavg2 = (rhoavg*un.Msun/un.pc**3 * var2*(un.km/un.s)**2).to(cons.k_B * un.K / un.cm**3).value
        logprofiles['P_turb_rad'].append(np.log10(Pturbavg2))

        var_cartesian = 1/3 * ( np.var(p0['Velocities'][idx, 0]) + np.var(p0['Velocities'][idx, 1]) + np.var(p0['Velocities'][idx, 2]) )
        Pturbavg_cart = (rhoavg*un.Msun/un.pc**3 * var_cartesian*(un.km/un.s)**2).to(cons.k_B * un.K / un.cm**3).value
        logprofiles['P_turb_cart'].append(np.log10(Pturbavg_cart))

        if 'CosmicRayEnergy' in p0:
            # CR energy density profile: log <e_CR/(1e10 Msun/kpc^3 (km/s)^2)>
            CReavg = np.sum(p0['CosmicRayEnergy'][idx]) / V
            logprofiles['e_CR'].append(np.log10(CReavg))

            # CR pressure profile (averaging in linear space): log <P_CR/(k_B K/cm^3)>
            P_CRi = Pressure(u_CR(p0['CosmicRayEnergy'][idx], p0['Masses'][idx]), p0['Density'][idx], typeP='CR')
            PCRavg = np.sum(P_CRi * p0['Vi'][idx]) / V
            logprofiles['P_CR'].append(np.log10(PCRavg))

        # JS tests
        idxtot = np.flatnonzero(inrange( p0['r_scaled'], (r0, r1) ))
        Ltot = (np.sum(p0['CoolingRate'][idxtot] * p0['Vi'][idxtot]) * un.erg/un.s*un.cm**-3 * un.kpc**3).to(un.erg/un.s).value
        Lvirial = (np.sum(p0['CoolingRate'][idx] * p0['Vi'][idx]) * un.erg/un.s*un.cm**-3 * un.kpc**3).to(un.erg/un.s).value
        Mdottot = np.sum(p0['vrad'][idxtot] * 1e10 * p0['Masses'][idxtot] / dL ) * (un.km / un.s * un.Msun / un.kpc).to(un.Msun / un.yr)
        Mtot = np.sum(p0['Masses'][idxtot])*1e10
        Mvirial = np.sum(p0['Masses'][idx])*1e10

        # tcoolavgshelltot = (np.sum(p0['InternalEnergy'][idxtot]*p0['Density'][idxtot]) / np.sum(p0['CoolingRate'][idxtot]) * 1e10 * un.Msun/un.kpc**3 * (un.km/un.s)**2 / (un.erg/un.s*un.cm**-3)).to(un.yr).value
        tcoolavgshelltot = (np.sum(p0['InternalEnergy'][idxtot]*p0['Masses'][idxtot]) / np.sum(p0['CoolingRate'][idxtot]*p0['Vi'][idxtot]) * 1e10 * un.Msun * (un.km/un.s)**2 / (un.erg/un.s*un.cm**-3 * un.kpc**3)).to(un.yr).value
        tcoolavgshell = (np.sum(p0['InternalEnergy'][idx]*p0['Masses'][idx]) / np.sum(p0['CoolingRate'][idx]*p0['Vi'][idx]) * 1e10 * un.Msun * (un.km/un.s)**2 / (un.erg/un.s*un.cm**-3 * un.kpc**3)).to(un.yr).value
        r = (r0+r1)/2 * p0['Rvir']
        # print(f'{Mtot:e} Msun, {r} kpc, {tcoolavgshelltot:e} yr, {dL} kpc')
        Mdotcool = Mtot * r / (tcoolavgshelltot * dL) #Msun/yr
        Mdotcoolvirial = Mvirial * r / (tcoolavgshell * dL) #Msun/yr

        # fountain flow tests
        idxtotIF = idxtot[ p0['vrad'][idxtot] > 0 ]
        idxtotOF = idxtot[ p0['vrad'][idxtot] < 0 ]
        MdottotIF = np.sum(p0['vrad'][idxtotIF] * 1e10 * p0['Masses'][idxtotIF] / dL ) * (un.km / un.s * un.Msun / un.kpc).to(un.Msun / un.yr)
        MdottotOF = np.sum(-p0['vrad'][idxtotOF] * 1e10 * p0['Masses'][idxtotOF] / dL ) * (un.km / un.s * un.Msun / un.kpc).to(un.Msun / un.yr)

        # cold gas tests
        idxcoolIF = idxcool[ p0['vrad'][idxcool] > 0 ]
        idxcoolOF = idxcool[ p0['vrad'][idxcool] < 0 ]
        MdotcoolIF = np.sum(p0['vrad'][idxcoolIF] * 1e10 * p0['Masses'][idxcoolIF] / dL ) * (un.km / un.s * un.Msun / un.kpc).to(un.Msun / un.yr)
        MdotcoolOF = np.sum(-p0['vrad'][idxcoolOF] * 1e10 * p0['Masses'][idxcoolOF] / dL ) * (un.km / un.s * un.Msun / un.kpc).to(un.Msun / un.yr)

        vradavgvirial = np.sum(p0['vrad'][idx] * p0['Masses'][idx]) / np.sum(p0['Masses'][idx])
        vphiavgvirial = np.sum(p0['vphi'][idx] * p0['Masses'][idx]) / np.sum(p0['Masses'][idx])

        vradavgtot = np.sum(p0['vrad'][idxtot] * p0['Masses'][idxtot]) / np.sum(p0['Masses'][idxtot])
        vphiavgtot = np.sum(p0['vphi'][idxtot] * p0['Masses'][idxtot]) / np.sum(p0['Masses'][idxtot])

        jtot = iht.find_angmom_vector(p0['Coordinates'][idxtot], p0['Velocities'][idxtot], p0['Masses'][idxtot]/np.sum(p0['Masses'][idxtot]), normalize=False)
        jvirial = iht.find_angmom_vector(p0['Coordinates'][idx], p0['Velocities'][idx], p0['Masses'][idx]/np.sum(p0['Masses'][idx]), normalize=False)

        # galactic plane test
        idxtotplane = idxtot[ np.abs(p0['theta'][idxtot] - np.pi/2) < (np.pi/8) ]
        idxplane = idx[ np.abs(p0['theta'][idx] - np.pi/2) < (np.pi/8) ]
        vphiavgtotplane = np.sum(p0['vphi'][idxtotplane] * p0['Masses'][idxtotplane]) / np.sum(p0['Masses'][idxtotplane])
        vphiavgvirialplane = np.sum(p0['vphi'][idxplane] * p0['Masses'][idxplane]) / np.sum(p0['Masses'][idxplane])

        # excluding ISM tests
        Mtotrot = Mvirialrot = []
        if potential is not None:
            frottot = p0['vphi'][idxtot]/p0['vc'][idxtot]
            frot = p0['vphi'][idx]/p0['vc'][idx]
            
            Mtotrot = [ np.sum(p0['Masses'][idxtot][frottot<frotthresh])*1e10 for frotthresh in np.arange(0.5, 1, 0.1) ]
            Mvirialrot = [ np.sum(p0['Masses'][idx][frot<frotthresh])*1e10 for frotthresh in np.arange(0.5, 1, 0.1) ]

        idxhard = np.flatnonzero(inrange( p0['r_scaled'], (r0, r1) )&(p0['Temperature'] > 10**4.5))
        Lhard = (np.sum(p0['CoolingRate'][idxhard] * p0['Vi'][idxhard]) * un.erg/un.s*un.cm**-3 * un.kpc**3).to(un.erg/un.s).value
        
        binedges = np.linspace(-3,1,101)
        histZM_virial = np.histogram(np.log10(p0['Metallicity'][:,0][idx]/Zsun), bins=binedges, weights=p0['Masses'][idx]*1e10)[0]
        histZL_virial = np.histogram(np.log10(p0['Metallicity'][:,0][idx]/Zsun), bins=binedges, weights=(p0['CoolingRate'][idx] * p0['Vi'][idx] * un.erg/un.s*un.cm**-3 * un.kpc**3).to(un.erg/un.s).value)[0]
        histZM_hard = np.histogram(np.log10(p0['Metallicity'][:,0][idxhard]/Zsun), bins=binedges, weights=p0['Masses'][idxhard]*1e10)[0]
        histZL_hard = np.histogram(np.log10(p0['Metallicity'][:,0][idxhard]/Zsun), bins=binedges, weights=(p0['CoolingRate'][idxhard] * p0['Vi'][idxhard] * un.erg/un.s*un.cm**-3 * un.kpc**3).to(un.erg/un.s).value)[0]
        
        logprofiles['js'].append((Ltot, Lvirial, Mdottot, Mdot, Mtot, Mvirial, np.sum(Luminosity), Mdotcool, Mdotcoolvirial, tcoolavgshell/1e9, 
                                  Lhard, MdottotIF, MdottotOF, vradavgvirial, vphiavgvirial, vradavgtot, vphiavgtot,
                                  jtot[0], jtot[1], jtot[2],
                                  jvirial[0], jvirial[1], jvirial[2],
                                  vphiavgtotplane, vphiavgvirialplane,
                                  MdotcoolIF, MdotcoolOF,
                                  *Mtotrot, *Mvirialrot))
        hists.append((histZM_virial, histZL_virial, histZM_hard, histZL_hard))

        # Total mass in radial bin for each particle type: 1e10 Msun
        for ptype, p_i in part.items():
            idx_i = np.flatnonzero(inrange( p_i['r_scaled'], (r0, r1) ))
            Mbin_i = np.sum(p_i['Masses'][idx_i])
            Mbins[f'TotalMass:PartType{ptype}'].append(Mbin_i)
        
    logprofiles = {k:np.array(v) for k,v in logprofiles.items()}
    resdict = {'rmid':rmid, **logprofiles, **Mbins, 'hists':hists, 'posC':p0['posC'], 'Rvir':p0['Rvir'], 'Mvir':p0['Mvir'], 'Redshift':p0['Redshift']}
    
    return resdict

def make_Mdot_profile(vrad, r_mag, Masses, Rmin=1, Rmax=1500, Rmin_Mdotmean = 1e2, Tmask=True):
            
    ''' 
    *** Units ***
    vrad: pkm/s
    r_mag: pkpc
    Masses: 1e10 Msun
    Rmin, Rmax, Rmin_Mdotmean: pkpc
    '''

    rbins = np.logspace(np.log10(Rmin), np.log10(Rmax), 100) #pkpc
    rmid = (rbins[:-1]+rbins[1:])/2 #in units of pkpc

    Mdot_profile = [] #in units of Msun/yr

    for r0,r1 in zip(rbins[:-1],rbins[1:]):
        dL = r1 - r0
        idx = np.flatnonzero(Tmask & inrange( r_mag, (r0, r1) ))

        # Mdot profile: Msun/yr
        Mdot = np.sum(vrad[idx] * 1e10 * Masses[idx] / dL ) * (un.km / un.s * un.Msun / un.kpc).to(un.Msun / un.yr)
        Mdot_profile.append(Mdot)

    Mdot_avg = np.mean(np.array(Mdot_profile)[rmid>Rmin_Mdotmean])
    return {'rmid_Mdot':rmid, 'Mdot':Mdot_profile}, Mdot_avg

def colormap(p, k, krange, f=1, rrange=(0, 3), log=True, bins=100):
    idx = inrange( p['r_scaled']*p['Rvir'], np.power(10, rrange) )
    Mtest = f*p[k][idx]
    if log: Mtest = np.log10(Mtest)
    
    rtest = np.log10(p['r_scaled'][idx]*p['Rvir'])
    hrange = [[rtest.min(), rtest.max()],krange]
    
    H, xedges, yedges = np.histogram2d(rtest, Mtest, bins=bins, range=hrange, density=True, weights=p['Masses'][idx])
    X, Y = np.meshgrid(xedges, yedges)
    H = H.T
    
    for i in range(H.shape[1]):
        H[:,i] = H[:,i] / np.sum(H[:,i])
    
    return {'X':X, 'Y':Y, 'H':H}

def excise_satellites(rhf, gas_tree, star_tree, posC_host, Rvir_host, Rmax_star=0.1, Rmax_gas=0.2, Nmin_star=5, majormerger=0.3):
    r_scaled = np.linalg.norm(rhf['position'] - posC_host, axis=1) / Rvir_host
    rmask = inrange(r_scaled, (0,0.2)) #is subhalo within 0.2 Rvir of host halo center
    if rmask.sum() > 0:
        host_idx = np.flatnonzero(rmask)[rhf['mass.vir'][rmask].argmax()]
        host_id = rhf['id'][host_idx]
        host_mvir = rhf['mass.vir'][host_idx]
    else:
        host_id = rhf['id'][rhf['mass.vir'].argmax()]
        host_mvir = rhf['mass.vir'].max()

    def excise_satellite(hid, mvir, rvir, pos, nearbycentral):
        if hid==host_id: return []
        if (mvir/host_mvir > majormerger) and nearbycentral: 
            print('Major central merger!', mvir/host_mvir)
            return []

        star_indices = star_tree.query_ball_point(pos, Rmax_star*rvir)
        gas_indices = gas_tree.query_ball_point(pos, Rmax_gas*rvir)

        Nstars = len(star_indices)
        if Nstars >= Nmin_star: 
            return gas_indices
        else:
            return []
    
    gas_mask = np.ones(gas_tree.n, dtype=bool)
    for hid, mvir, rvir, pos, nearbycentral in zip(rhf['id'], rhf['mass.vir'], rhf['radius'], rhf['position'], rmask):
        gas_indices = excise_satellite(hid, mvir, rvir, pos, nearbycentral)
        gas_mask[gas_indices] = False
    return gas_mask

class Simulation:
    def __init__(self, simdir, snapnum, cachesim=False, find200c=False, satellitecut=False, calculateOutflows=False, datadir='simcachev2_Khist_outflow_selectMainBranch_0.5dex_T1e5cut_nH_Mdot_modecut_MdotIF_jstests_ZLtcoolv3_rotateDisk_vcom3_coldgas_excludeISM', fbranchcut=0.5, maskToUse=None):
        self.simdir = simdir
        self.snapnum = snapnum
        self.simname = os.path.basename(simdir)
        self.datadir = datadir
        
        if cachesim:
            # Load data
            keys_to_extract = {
                0:['Coordinates', 'Masses', 'Density', 
                'Temperature', 'InternalEnergy', 'CosmicRayEnergy',
                'Velocities', 'Metallicity', 'SoundSpeed', 'CoolingRate', 'SmoothingLength', 'MagneticField', 'ElectronAbundance'],
                1:['Coordinates', 'Masses', 'Velocities'],
                2:['Coordinates', 'Masses', 'Velocities'],
                4:['Coordinates', 'Masses', 'Velocities'],
                5:['Coordinates', 'Masses', 'Velocities']
            }
            # keys_to_extract={0:['Coordinates','Masses','Metallicity','SmoothingLength','Temperature']}

            self.part = load_allparticles(simdir, snapnum, 
                                particle_types=sorted(keys_to_extract.keys()), 
                                keys_to_extract=keys_to_extract, 
                                Rvir='None', loud=0) #find_Rvir_SO',
            # Find 200c SO measurements
            if find200c: self.R200c, self.M200c = find_Rvir_SO(self.part, useM200c=True)
            # Shift gas velocities to COM, calculate Z(<Rvir), and add additional fields to self.part
            part_calculations(self.part)
            self.Z2Zsun = self.part[0]['Z2Zsun']
            self.Redshift = self.part[0]['Redshift']
            self.tHubble = self.part[0]['tHubble'] #Gyr

            # Calculate potential
            self.Mr = calculateMr(self.part)
            self.potential = Potential_FIRE(self.Mr)

            # Calculate profiles and Mdot profile
            mask = True
            if satellitecut: mask = (self.part[0]['Temperature'] > 10**5.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                self.pro = profiles(self.part, Tmask=mask, potential=self.potential)
            self.Mdot_profile, self.Mdot_avg = make_Mdot_profile(self.part[0]['vrad'], self.part[0]['r_scaled']*self.part[0]['Rvir'], self.part[0]['Masses'])
            self.Mdot_avg = self.Mdot_avg*un.Msun/un.yr

            if calculateOutflows: self.Mdot_profile['Mdot_outflows'] = self.outflow_rates()
            self.cacheeverything()
        else:
            res = h5todict(f'../data/{self.datadir}/simcache_{self.simname}_{self.snapnum}.h5')
            self.simdirrun = res['simdir']
            self.Z2Zsun = res['Z2Zsun']
            self.Redshift = res['Redshift']
            self.tHubble = res['tHubble']
            self.pro = res['pro']
            self.Mdot_profile = res['Mdot_profile']
            self.Mr = res['Mr']
            if find200c:
                self.R200c = res['R200c']
                self.M200c = res['M200c']

            [setattr(self, k, v) for k,v in res.items() if 'cmap_' in k];
            # self.cmap_nH = res['cmap_nH']
            # self.cmap_T = res['cmap_T']
            # self.cmap_Z = res['cmap_Z']
            # self.cmap_MachNumber = res['cmap_MachNumber']
            # self.cmap_tcool = res['cmap_tcool']
            # self.cmap_K = res['cmap_K']
            # self.cmap_Kn = res['cmap_Kn']
            # if 'cmap_vrad' in res:
            #     self.cmap_vrad = res['cmap_vrad']
            #     self.cmap_vradpec = res['cmap_vradpec']

            if maskToUse is not None: self.pro = res['pro'][maskToUse]
            self.Rmin = 0.1 * self.pro['Rvir'] #minimum radius for Mdot, Z avg in units pkpc
            self.Rcool = 1 * self.pro['Rvir'] #maximum radius, hardcoded and use 'all' mask for averaging
            
            # self.Rcool = self.pro['rmid'][np.flatnonzero(self.pro['tcool'] > np.log10(self.tHubble))][0] * self.pro['Rvir'] #cooling radius in units pkpc 
            # self.Rcool = self.pro['rmid'][self.pro['rmid']*self.pro['Rvir']>self.Rmin][np.flatnonzero(self.pro['tcoolshell'][self.pro['rmid']*self.pro['Rvir']>self.Rmin] > np.log10(self.tHubble))][0] * self.pro['Rvir'] #cooling radius in units pkpc 
            # if self.Rcool < self.Rmin: self.Rcool = 0.5 * self.pro['Rvir'] #raise Exception('Rcool not defined', self.simname)
            
            self.Mdot_avg = np.mean(self.Mdot_profile['Mdot'][(self.Rmin < self.Mdot_profile['rmid_Mdot'])&(self.Mdot_profile['rmid_Mdot'] < self.Rcool)])*un.Msun/un.yr
            # self.Mdot_avg = np.mean(self.Mdot_profile['Mdot'][self.Mdot_profile['rmid_Mdot']>1e2])
            # self.Mdot_avg = self.Mdot_avg*un.Msun/un.yr

            fbranch = self.pro['Mgas'] / self.pro['TotalMass:PartType0']
            idxZavg= (self.Rmin <= self.pro['rmid']*self.pro['Rvir'])&(self.pro['rmid']*self.pro['Rvir'] <= self.Rcool)&(fbranch >= fbranchcut)
            self.Z2ZsunRvir = self.Z2Zsun
            self.Z2Zsun = np.sum(10**self.pro['Z_Mweighted'][idxZavg] * self.pro['Mgas'][idxZavg]) / np.sum(self.pro['Mgas'][idxZavg]) #TODO look at TotalMass
            alpha = 10**self.pro['P_turb'][idxZavg] / 10**self.pro['Pth_Mweighted'][idxZavg]
            # self.alpha_avg = np.mean(alpha)
            self.alpha_avg = np.sum(alpha * self.pro['Mgas'][idxZavg]) / np.sum(self.pro['Mgas'][idxZavg]) #Mass-weighted average
            self.fittingidx = idxZavg

            self.Mdot_avg2 = np.mean(self.pro['Mdot'][idxZavg])*un.Msun/un.yr
            self.Mdot_avg3 = np.sum(self.pro['Mdot'][idxZavg] * self.pro['Mgas'][idxZavg]) / np.sum(self.pro['Mgas'][idxZavg])*un.Msun/un.yr #Mass-weighted average
            
            self.Mdot_avg4 = np.mean(self.pro['MdotIF'][idxZavg])*un.Msun/un.yr
            self.Mdot_avg5 = np.sum(self.pro['MdotIF'][idxZavg] * self.pro['Mgas'][idxZavg]) / np.sum(self.pro['Mgas'][idxZavg])*un.Msun/un.yr #Mass-weighted average
        
            self.potential = Potential_FIRE(self.Mr, pro=self.pro, fittingmask=self.fittingidx)

            self.Z2Zsun_M = np.sum(10**self.pro['Z_Mweighted'][idxZavg] * self.pro['Mgas'][idxZavg]) / np.sum(self.pro['Mgas'][idxZavg])
            self.Z2Zsun_L = np.sum(10**self.pro['Z_Lweighted'][idxZavg] * self.pro['Mgas'][idxZavg]) / np.sum(self.pro['Mgas'][idxZavg])
            self.Z2Zsun_LL= np.sum(10**self.pro['Z_Lweighted'][idxZavg] * self.pro['js'][idxZavg,1]) / np.sum(self.pro['js'][idxZavg,1])

            self.Z2Zsun = self.Z2Zsun_M
        return
        self.potential.potential_test()

        # Integrate cooling flow solution
        self.integrate_cooling_flow()

        # PrecipitationModel(self.part[0]['Redshift'], self.potential, self.part[0]['Z2Zsun'], self.pro, CF.mu)

    def cacheeverything(self):
        colormaps = {
            'cmap_nH':          colormap( self.part[0], 'nH', (-7,1) ),
            'cmap_T':           colormap( self.part[0], 'Temperature', (3,8) ),
            'cmap_Z':           colormap( self.part[0], 'MetallicitySolar', (-2.5,0) ),
            'cmap_MachNumber':  colormap( self.part[0], 'MachNumber', (-1,1), log=False ),
            'cmap_tcool':       colormap( self.part[0], 'tcool', (-2,4) ),
            'cmap_K':           colormap( self.part[0], 'K', (12,16) ),
            'cmap_Kn':          colormap( self.part[0], 'Kn', (0, 3) ),
            'cmap_vrad':        colormap( self.part[0], 'vrad', (-200,200), log=False ),
            'cmap_vradpec':     colormap( self.part[0], 'vradpec', (-200,200), log=False )
        }
        res = {
            'simdir':       self.simdir,
            'snapnum':      self.snapnum,
            'simname':      self.simname,
            'Z2Zsun':       self.Z2Zsun,
            'Redshift':     self.Redshift,
            'tHubble':      self.tHubble,
            'pro':          self.pro,
            'Mdot_profile': self.Mdot_profile,
            'Mr':           self.Mr,
            #'R200c':        self.R200c,
            #'M200c':        self.M200c,
            **colormaps
        }
        fname = f'../data/{self.datadir}/simcache_{self.simname}_{self.snapnum}.h5'
        dicttoh5(res, fname, mode='w')
    
    def outflow_rates(self, r0=0.2, r1=0.3): 
        ''' 
        *** Units ***
        vrad: pkm/s
        r_mag: pkpc
        Masses: Msun
        Rmin, Rmax, Rmin_Mdotmean: pkpc
        '''
        vrad = -self.part[0]['vrad'] #positive for outflows
        r_mag = self.part[0]['r_scaled']*self.part[0]['Rvir']
        Masses = self.part[0]['Masses'] * 1e10

        # Get velocity dispersion of all particles within r/Rvir<1 (all particle velocities should already be adjusted for COM)
        # pall = {}
        # pall['Velocities'] = np.concatenate([self.part[k]['Velocities'] for k in self.part.keys()])
        # pall['r_scaled'] = np.concatenate([self.part[k]['r_scaled'] for k in self.part.keys()])
        # sigmav1d = np.linalg.norm(np.std( pall['Velocities'][pall['r_scaled']<1], axis=0 )) / 3**1/2 #pkm/s
        sigmav1d = np.linalg.norm(np.std( self.part[1]['Velocities'][self.part[1]['r_scaled']<1], axis=0 )) / 3**1/2 #pkm/s only use HRDM
        print('sigmav1d',sigmav1d)

        idx_r = inrange( self.part[0]['r_scaled'], (r0, r1) ) #net flow rate
        idx_vcut0 = idx_r & (vrad>0) #only select outflowing particles (Muratov+15)
        idx_vcutsigma = idx_r & (vrad>sigmav1d) #only select outflowing particles with v above 1D velocity dispersion of the halo (Muratov+15)
        
        # Pandya+21 inflow rate (ISM boundary shell)
        r0, r1 = 0.1, 0.2
        dL = (r1-r0)*self.part[0]['Rvir'] #pkpc
        r2 = 0.5 * self.part[0]['Rvir']*un.kpc #pkpc
        idx = inrange( self.part[0]['r_scaled'], (r0, r1) ) & (vrad>0)
        
        self.potential = Potential_FIRE(self.Mr)
        vsec_r2 = self.potential.vesc(r2)

        vB2 = 1/2*vrad[idx]**2 *(un.km/un.s)**2 + self.part[0]['SoundSpeed'][idx]**2/(5/3-1) *(un.km/un.s)**2 - (1/2*self.potential.vesc(r_mag[idx]*un.kpc)**2)

        idx_pandya = np.flatnonzero(idx)[vB2 > (-1/2*vsec_r2**2)]

        Mdot_outflows = []
        dL = (r1-r0)*self.part[0]['Rvir'] #pkpc
        for idx in (idx_r, idx_vcut0, idx_vcutsigma, idx_pandya):
            # Mdot profile: Msun/yr
            Mdot = np.sum(vrad[idx] * Masses[idx] / dL ) * (un.km / un.s * un.Msun / un.kpc).to(un.Msun / un.yr)
            Mdot_outflows.append(Mdot)
        return Mdot_outflows

    def binary_search_R_sonic(self, R_sonic_low, R_sonic_high, R_max=1.5, R_min=1, tol=1e-1, verbose=True, alpha=0.):
        cooling = Cool.Wiersma_Cooling(self.Z2Zsun, self.Redshift)
        R_sonic_mid = (R_sonic_low + R_sonic_high)/2
        
        def solution(R_sonic):
            return CF.shoot_from_sonic_point(self.potential, cooling, R_sonic=R_sonic, R_max=R_max*self.potential.Rvir, R_min=R_min*un.kpc, alpha=alpha)
        
        def error(solution):
            return np.abs(self.Mdot_avg - solution.Mdot).value
        
        if verbose: print(self.Mdot_avg)
        if verbose: print(R_sonic_low)
        solution_low = solution(R_sonic_low)
        if verbose: print(solution_low.Mdot)

        if R_sonic_low==R_sonic_high: return solution_low, R_sonic_low
        
        if verbose: print(R_sonic_mid)
        solution_mid = solution(R_sonic_mid)
        if verbose: print(solution_mid.Mdot)
        
        if verbose: print(R_sonic_high)
        solution_high = solution(R_sonic_high)
        if verbose: print(solution_high.Mdot)
        
        error_low = error(solution_low)
        error_mid = error(solution_mid)
        error_high = error(solution_high)
        
        Mdot_monotonically_increasing = solution_high.Mdot > solution_low.Mdot
        
        assert (Mdot_monotonically_increasing and solution_low.Mdot < self.Mdot_avg and self.Mdot_avg < solution_high.Mdot) or\
        (not Mdot_monotonically_increasing and solution_low.Mdot > self.Mdot_avg and self.Mdot_avg > solution_high.Mdot)
        
        while error_mid > tol:
            if ((solution_mid.Mdot > self.Mdot_avg) and Mdot_monotonically_increasing) or ((solution_mid.Mdot < self.Mdot_avg) and not Mdot_monotonically_increasing):
                R_sonic_high = R_sonic_mid
                solution_high = solution_mid
                error_high = error_mid
            else:
                R_sonic_low = R_sonic_mid
                solution_low = solution_mid
                error_low = error_mid
            
            R_sonic_mid = (R_sonic_low + R_sonic_high)/2
            solution_mid = solution(R_sonic_mid)
            error_mid = error(solution_mid)
            if verbose: print(R_sonic_mid, error_mid, solution_mid.Mdot)

        return solution_mid, R_sonic_mid
    
    def integrate_cooling_flow(self, R_max=None, R_circ=None, Mdot=None, T_low=None, T_high=None, max_step=0.1):
        if R_max is None: R_max = 1.5*self.potential.Rvir
        if R_circ is None: R_circ = self.potential.get_Rcirc()
        if Mdot is None: Mdot = self.Mdot_avg
        if T_low is None: T_low = 1e4*un.K
        if T_high is None: T_high = 1e8*un.K

        # Define cooling function
        cooling = Cool.Wiersma_Cooling(self.Z2Zsun, self.Redshift)
        
        self.stalled_solution = CF.shoot_from_R_circ(
            self.potential,
            cooling,
            R_circ,
            Mdot,
            R_max,
            max_step=max_step,
            T_low=T_low,
            T_high=T_high,
            pr=True)