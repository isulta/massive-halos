'''Simulation paths (AGN and no-AGN) on Quest, CCA, and Frontera
'''
import numpy as np
import os.path

PaperSimNames = {
    'm12b_NoBH': 'm12b_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e10_gacc31_fa0.5',
    'm12i_NoBH': 'm12i_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e10_gacc31_fa0.5',
    'm12f_NoBH': 'm12f_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e10_gacc31_fa0.5',
    'm12f_BH': 'm12f_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp2e-4_gacc31_fa0.5',
    'm12f_BHCR': 'm12f_m6e4_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
    'm12q_NoBH': 'm12q_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e10_gacc31_fa0.5',
    'm12q_BH': 'm12q_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp2e-4_gacc31_fa0.5',
    'm12q_BHCR': 'm12q_m6e4_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
    'm13h113_NoBH': 'm13h113_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
    'm13h113_BH': 'm13h113_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e-4_gacc31_fa0.5',
    'm13h113_BHCR': 'm13h113_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
    'm13h206_NoBH': 'm13h206_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
    'm13h206_BH': 'm13h206_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp3e-4_gacc31_fa0.5',
    'm13h206_BHCR': 'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000'
}

'''Sarah functions'''
# Takes a simulation filename (like the ones written in the .txt files) and converts it into a dictionary of parameters
def params_from_filename(filename):
    params = {}
    plist = filename.split('_')
    params['haloID']              = plist[0]
    params['mass_res']            = plist[1]
    params['fb_method']           = plist[2]
    if plist[2] != 'control':
        params['alpha_disk_factor']   = plist[3]
        params['gravaccretion_model'] = plist[4]
        params['accretion_factor']    = plist[5]
        params['v_wind']              = plist[6]
        params['cr_loading']          = plist[7]
        params['seed_mass']           = plist[8]
        params['stellar_m_per_seed']  = plist[9]
        params['spawned_wind_mass']   = plist[10]
        params['f_accreted']          = plist[11]
        params['t_wind_spawn']        = plist[12]
        params['fluxmom_factor']      = plist[13]

    return params

# Reverse of above
def filename_from_params(params):
    filename = ''
    for param in params.keys():
        filename = filename+params[param]+'_'
    filename = filename[:-1]  # To remove last underscore

    return filename

# Returns the names + paths of the "control" set, plus the snapshot indices in the 600-snapshot runs that correspond to the same redshifts as the 60-snapshot runs
# Note: requires you to have the 60-snapshot timing file, which you can find in any of the AGN feedback simulation folders
def controlsims(create_symlinks=False):
    simnames = ['m10q_m2e2', 'm10v_m2e2',
                'm11a_m2e3', 'm11b_m2e3', 'm11d_m7e3', 'm11e_m7e3', 
                'm11f_m1e4', 'm11h_m7e3', 'm11i_m7e3', 'm11q_m7e3', 
                'm12b_m6e4', 'm12f_m6e4', 'm12i_m6e4', 'm12m_m6e4', 
                'm12q_m6e4', 'm12r_m6e4', 'm12w_m6e4',
                'm13h02_m3e5', 'm13h29_m3e5', 'm13h113_m3e5', 'm13h206_m3e5']
    controlpaths = ['core/m10q_res250', 'CR_suite/m10v/cr_700', 
                    'CR_suite/m11a_res2100/cr_700', 'CR_suite/m11b/cr_700', 'CR_suite/m11d/cr_700', 'CR_suite/m11e_res7100/cr_700', 
                    'CR_suite/m11f/cr_700', 'CR_suite/m11h/cr_700', 'CR_suite/m11i_res7100/cr_700', 'core/m11q_res7100',
                    'CR_suite/m12b_res7100/cr_700', 'CR_suite/m12f_mass56000/cr_700', 'CR_suite/m12i_mass56000/cr_700', 'CR_suite/m12m_mass56000/cr_700', 
                    'CR_suite/m12q_res57000/cr_700', 'CR_suite/m12r_res7100/cr_700', 'CR_suite/m12w_res7100/cr_700',
                    'MassiveFIRE/A8_res33000', 'MassiveFIRE/A2_res33000', 'MassiveFIRE/A4_res33000', 'MassiveFIRE/A1_res33000']
    if create_symlinks:
        for s,c in zip(simnames, controlpaths):
            src = os.path.join('/home/jovyan/fire2/', c)
            dst = os.path.join('/home/jovyan/home/data', s.split('_')[0]+'_noAGNfb')
            if os.path.exists(src): os.symlink(src, dst)
    else:
        asnaps = np.loadtxt('../data/snapshot_scale-factors.txt')
        snaps_600 = np.loadtxt('../../fire2/core/m12i_res57000/snapshot_times.txt')
        asnaps_600 = snaps_600[:,1]
        isnaps_600 = snaps_600[:,0]
        isnaps = np.interp(asnaps, asnaps_600, isnaps_600).astype(int)
        
        return simnames, controlpaths, isnaps

# Returns the path to a simulation data directory given the parameter dictionary and the machine it lives on
def sim_path(params, machine=None):
    if params['fb_method'] == 'control':
        filebase = '../../fire2/'
        allcontrols = controlsims()
        folder = allcontrols[1][allcontrols[0].index(params['haloID']+'_'+params['mass_res'])]
        return filebase + folder
    else:
        if machine == 'frontera':
            filebase = '/scratch3/01799/phopkins/bhfb_suite_done/'
        elif machine == 'cca':
            filebase = '../../fire2/AGN_suite/'
        elif machine == 'lou':
            filebase = '/u/pfhopki1/agn_fb/'
        else:
            print("Please specify a valid machine, e.g. 'frontera', 'lou', 'cca'")
            return None

        folder = params['haloID']+'_'+params['mass_res']+'/'
        return filebase + folder + filename_from_params(params)
'''End Sarah functions'''

# Returns path of FIRE-3 sims on Frontera
def sim_path_fire3(filename, quest=False):
    if quest: return os.path.join( '/projects/b1026/isultan/fire3', filename )
    filebase = '/scratch3/01799/phopkins/fire3_suite_done'
    if filename == 'm13h02_m3e5_MHD_fire3_fireBH_Sep052021_crdiffc690_sdp1e-4_gacc31_fa0.5':
        simbase = 'm13h002_m3e5'
    elif filename == 'm13h29_m3e5_MHD_fire3_fireBH_Sep052021_crdiffc690_sdp1e-4_gacc31_fa0.5':
        simbase = 'm13h029_m3e5'
    else:
        simbase = '_'.join( filename.split('_')[:2] )
    return os.path.join(filebase, simbase,  filename)

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