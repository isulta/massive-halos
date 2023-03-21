from astropy import units as un, constants as cons
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.misc import derivative
import pickle
from silx.io.dictdump import dicttoh5, h5todict
from scipy.signal import savgol_filter
from tqdm import tqdm

import WiersmaCooling as Cool

class PrecipitationModel:
    def __init__(self, z, potential, Z2Zsun, pro, mu, savgolZ=(55,2)):
        self.z = z
        self.potential = potential
        self.Z2Zsun = Z2Zsun
        self.pro = pro
        self.mu = mu
        
        # dlnZ_dlnr calculation
        self.rmid = pro['rmid']*pro['Rvir'] #units pkpc
        self.Zsmoothed = savgol_filter(pro['Z'], savgolZ[0], savgolZ[1])
        self.f_Z_Zsun = interp1d( self.rmid, 10**self.Zsmoothed ) #Z/Zsun as a fn of r in pkpc
        self.f_dlnZ_dlnr = interp1d( self.rmid, np.gradient(self.Zsmoothed, np.log(pro['rmid'])) ) #fn of r in pkpc

    def f_dlntff_dlnr(self, r):
        return 1 - self.potential.dlnvc_dlnR(r)

    def f_tff(self, r):
        return (np.sqrt(2)*r / self.potential.vc(r)).to('Gyr')

    def f_dlnLambda_dlnZ(self, Z=None, T=None, nH=None, dZ=0.01, fittingfn=False):
        '''Computes partial derivative of lnLambda wrt lnZ at metallicty Z and redshift z, and at constant T and nH.
        Uses a finite different method with metallicity width 2*dZ.'''
        if fittingfn:
            return 0.9 #using Lambda fitting function from Stern+2020
        else:
            Zi = Z-dZ
            Zf = Z+dZ
            coolingi = Cool.Wiersma_Cooling(Zi,self.z)
            coolingf = Cool.Wiersma_Cooling(Zf,self.z)
            dlnLambda_dlnZ = Z/ Cool.Wiersma_Cooling(Z,self.z).LAMBDA(T, nH) * (coolingf.LAMBDA(T, nH) - coolingi.LAMBDA(T, nH))/ (Zf-Zi)
            # np.log((coolingf.LAMBDA(T, nH) - coolingi.LAMBDA(T, nH)).value)/ np.log((Zf-Zi)*Zsun) above works better than this finite difference
            return dlnLambda_dlnZ

    def precipintegrand(self, dlnLambda_dlnZ, dlnZ_dlnr, dlntff_dlnr, dlnLambda_dlnrho, dlnLambda_dlnT, r, tff, T):
        dlnLambda_dlnnH = dlnLambda_dlnrho
        dlnLambda_dlntau = dlnLambda_dlnT
        
        s = (2*self.mu*cons.m_p*r**2 / (tff**2 * T * cons.k_B)).si
        return ((dlnLambda_dlnZ*dlnZ_dlnr+dlntff_dlnr)/(1+dlnLambda_dlnnH) - s) / (1+(1-dlnLambda_dlntau)/(1+dlnLambda_dlnnH))

    def precip_nH(self, T, Lambda, tff, xi, gamma=5/3, X=0.75):
        tau = cons.k_B * T
        return ( tau / ((gamma-1)*X*self.mu*Lambda*xi*tff) ).to(un.cm**-3)

    def integrate_precip(self, rvir, Tvir, nHvir, xi, dlnr=-0.1, R_min=3*un.kpc, constantZ=False):#if True, use Z2Zsun
        rarray = []
        Tarray = []
        nHarray = []
        dlnLambda_dlnZarray = []
        
        ri = rvir
        Ti = Tvir
        nHi = nHvir

        while ri > R_min:
            Tarray.append(Ti.to('K').value)
            nHarray.append(nHi.to(un.cm**-3).value)
            rarray.append(ri.to('kpc').value)

            Zi = self.f_Z_Zsun(ri.to('kpc').value) #Z2Zun at ri
            cooling = Cool.Wiersma_Cooling(self.Z2Zsun if constantZ else Zi, self.z)
            dlnLambda_dlnZ = self.f_dlnLambda_dlnZ(Zi, Ti, nHi)
            dlnLambda_dlnZarray.append(dlnLambda_dlnZ)

            pi = self.precipintegrand(
                dlnLambda_dlnZ=dlnLambda_dlnZ, 
                dlnZ_dlnr=0 if constantZ else self.f_dlnZ_dlnr(ri.to('kpc').value),
                dlntff_dlnr=self.f_dlntff_dlnr(ri),
                dlnLambda_dlnrho=cooling.f_dlnLambda_dlnrho(Ti, nHi), 
                dlnLambda_dlnT=cooling.f_dlnLambda_dlnT(Ti, nHi),
                r=ri, 
                tff=self.f_tff(ri), 
                T=Ti)

            ri = np.exp(np.log(ri.value)+dlnr)*un.kpc
            Ti = np.exp( pi*dlnr + np.log(Ti.to('K').value) ) * un.K
            Lambdai = cooling.LAMBDA(Ti, nHi)
            tffi = self.f_tff(ri)
            nHi = self.precip_nH(Ti, Lambdai, tffi, xi)
        
        return rarray*un.kpc, Tarray*un.K, nHarray*un.cm**-3, dlnLambda_dlnZarray

    def precip_test(self, rarray, dlnLambda_dlnZarray):
        fig, axes = plt.subplots(1, 3, sharex=False, sharey=False, gridspec_kw={'wspace': .3, 'hspace':.04}, figsize=[4.8*3,4.8*1], dpi=150, facecolor='w')

        axes[0].plot(np.log10(self.rmid), self.pro['Z'])
        axes[0].plot(np.log10(self.rmid), self.Zsmoothed, label='smoothed')
        axes[0].set_xlabel('$\log (r/\mathrm{pkpc})$')
        axes[0].set_ylabel('$\log (Z/Z_\odot)$')
        axes[0].legend()

        dln = np.gradient(self.pro['Z'], np.log(self.pro['rmid']))
        dln2 = np.gradient( self.Zsmoothed, np.log(self.pro['rmid']) )
        axes[1].plot(self.rmid,dln)
        axes[1].plot(self.rmid,dln2)
        axes[1].set_xlabel('$r/\mathrm{pkpc}$')
        axes[1].set_ylabel('$\mathrm{d}\ln{Z}/\mathrm{d}\ln{r}$')

        axes[2].plot(np.log10(rarray), dlnLambda_dlnZarray, c='gray')
        axes[2].set_xlabel('$\log (r/\mathrm{pkpc})$')
        axes[2].set_ylabel('$\mathrm{d}\ln{\Lambda}/\mathrm{d}\ln{Z}$')
    
    def integrate_precip_wrapper(self, xi=10, testplots=True, RBC=1):
        '''Integrates precipitation model, using the point in pro closest to RBC*Rvir as the boundary condition for T and nH.'''
        idx_Rvir = np.argmin(np.abs(self.pro['rmid'] - RBC))
        rvir = self.rmid[idx_Rvir]*un.kpc
        Tvir = 10**self.pro['T'][idx_Rvir]*un.K
        nHvir = 10**self.pro['nH'][idx_Rvir]*un.cm**-3

        rarray, Tarray, nHarray, dlnLambda_dlnZarray = self.integrate_precip(rvir, Tvir, nHvir, xi)
        if testplots: self.precip_test(rarray.value, dlnLambda_dlnZarray)

        self.rarray = rarray
        self.Tarray = Tarray
        self.nHarray = nHarray
        self.xi = xi
        return rarray, Tarray, nHarray
    
    def plot(self):
        '''Plot T curve using cached integration.'''
        plt.plot(np.log10(self.rarray.value), np.log10(self.Tarray.value), label='1D precip. model')
        plt.plot(np.log10(self.rmid), self.pro['T'], label='FIRE-3 h206 (No BHs)')

        plt.xlabel('$\log (r/\mathrm{pkpc})$')
        # plt.ylabel(profilelabels['T lin'])

        plt.legend()