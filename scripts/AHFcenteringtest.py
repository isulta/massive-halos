import numpy as np
import matplotlib.pyplot as plt

plt.figure(dpi=150)

for sim in ['h206_A1_res33000', 'h113_A4_res33000', 'h29_A2_res33000', 'h2_A8_res33000']:
    simdir = f'/projects/b1026/snapshots/MassiveFIRE/{sim}/halo/ahf/halo_00000_smooth.dat'
    ahf = np.genfromtxt(simdir, delimiter='\t', skip_header=True)
    
    sort_idx = np.argsort(ahf[:,0]) #sort AHF output by snapshot number
    snapshots = ahf[:,0][sort_idx]
    centers = ahf[:,7:10][sort_idx] #center positions in comoving kpc/h
    Delta_xcenter = np.linalg.norm(centers[1:]-centers[:-1], axis=1) #difference in center positions between snapshots
    
    plt.plot(snapshots[1:], Delta_xcenter, label=sim)

plt.yscale('log')
plt.legend()
plt.xlabel('snapshot')
plt.ylabel(r'$\Delta (\vec{x}_{\mathrm{center}})$ [comoving kpc/h]')
plt.show()