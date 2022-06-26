'''Script to test find_Rvir function for m13 no AGN feedback simulations.
'''
from scripts.halo_analysis_scripts import *
from itk import h5_write_dict, h5_read_dict
import matplotlib.pyplot as plt

def find_Rvir_test(halo = 'h206'):
    res = {'snapnum':[], 'z':[], 'Rvir':[], 'Rvir_ahf':[]}

    for snapnum in range(87, 278): # redshift 4.03999996 to 1.0 inclusive
        part = load_allparticles(Quest_sims['nofb'][halo], snapnum, ahf_path=Quest_nofb_m13_ahf(halo), loud=False)
        Rvir = find_Rvir(part)

        Rvir_ahf = part[0]['Rvir']
        z = part[0]['Redshift']

        res['snapnum'].append(snapnum)
        res['z'].append(z)
        res['Rvir'].append(Rvir)
        res['Rvir_ahf'].append(Rvir_ahf)

        print(snapnum, z, Rvir, Rvir_ahf)

    h5_write_dict(f'data/findRvirtests_{halo}.h5', res, 'res')

def plot_Rvir_test(halo = 'h206'):
    res = h5_read_dict(f'data/findRvirtests_{halo}.h5', 'res')
    # plt.plot(res['z'], res['Rvir'])
    # plt.plot(res['z'], res['Rvir_ahf'])
    plt.plot( res['z'], (res['Rvir'] - res['Rvir_ahf'])/res['Rvir_ahf']*100 )
    plt.xlabel('z')
    plt.ylabel('% error Rvir')
    plt.savefig(f'Figures/findRvirtests_{halo}.png')