import time
from datetime import datetime, timedelta
from scripts.halo_analysis_scripts import *

def printr(s):
    print(f'[{datetime.now()}]: {s}', flush=True)

if __name__ == '__main__':
    printr(f'Beginning script...'); start_script = time.time()

    for halo, snapdir in Quest_sims['nofb'].items():
        if os.path.exists(halo):
            continue
        else:
            open(halo, 'a').close()
        simname = os.path.basename(snapdir)
        printr(f'Starting {simname}...'); start_sim = time.time()
        
        # Load and save Rvir and redshift data
        printr(f'Loading/saving Rvir and redshift data...'); start = time.time()
        Rvir_allsnaps, redshifts = load_Rvir_allsnaps(snapdir, Quest_nofb_m13_ahf(halo), 'data/', simname)
        printr(f'Finished loading/saving Rvir and redshift data in {timedelta(seconds=time.time()-start)}')
        
        # Compute profiles for all snapshots in each redshift bin
        printr(f'Computing profiles for all snapshots in each redshift bin...'); start_profile = time.time()
        profiles_zbins(snapdir, redshifts, Rvir_allsnaps, outfile=f'data/{simname}_allprofiles_widezbins.pkl')
        printr(f'Finished Computing profiles for all snapshots in each redshift bin in {timedelta(seconds=time.time()-start_profile)}')

        printr(f'Finished {simname} in {timedelta(seconds=time.time()-start_sim)}')
        break
    printr(f'Finished script in {timedelta(seconds=time.time()-start_script)}')