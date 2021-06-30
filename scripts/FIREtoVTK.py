import sys
sys.path.insert(1, '/home/ias627/tools')
from abg_python.snapshot_utils import openSnapshot
from abg_python.cosmo_utils import load_AHF
import numpy as np
from pyevtk.hl import pointsToVTK

def Points1d(x, y, z, data, fn_out):
    print(fn_out+'.vtu')
    pointsToVTK(fn_out, x, y, z, data=data)

def FIREtoVTK(snapdir, snapnum, ptype, datacols, fn_out):
    pdata = openSnapshot(snapdir, snapnum, ptype, loud=1)
    print(f"Loaded redshift {pdata['Redshift']}")

    x, y, z = pdata['Coordinates'][:,0].copy(), pdata['Coordinates'][:,1].copy(), pdata['Coordinates'][:,2].copy()
    data = {k:pdata[k].copy() for k in datacols}
    Points1d(x, y, z, data, fn_out)

simname = 'h2_HR_sn1dy300ro100ss'
snapdir = '/projects/b1026/anglesd/FIRE/' + simname
# gas type=0
# FIREtoVTK(snapdir, 277, 0, ['Temperature', 'Density', 'Masses', 'Potential'], f'/projects/b1026/isultan/{simname}_p0')

# DM type=1
# mass 1.709241e-05
# FIREtoVTK(snapdir, 277, 1, ['Potential'], f'/projects/b1026/isultan/{simname}_p1')

# stars type=4
# FIREtoVTK(snapdir, 277, 4, ['Masses', 'Potential'], f'/projects/b1026/isultan/{simname}_p4')