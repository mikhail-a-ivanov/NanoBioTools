import numpy as np
import mdtraj as md
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from NanoBioTools import readXTC, getSurfaceDistanceGeneral, normalizeSphere, plotDensityHistogram

trajname = '/home/misha/Documents/AAMD/anatase-NP-POPE/traj-whole-skip100-nowater-500ns.xtc'
topname = '/home/misha/Documents/AAMD/anatase-NP-POPE/confin-whole-nowater.gro'

# Call readXTC function
traj = readXTC(trajname, topname, stride=1)

# Call getSurfaceDistance function
distances = getSurfaceDistanceGeneral(traj, topname, resname='H144', atomname='N')

# Normalize density
density = normalizeSphere(traj, distances, topname, atomname='N', binWidth=0.01, outname='anatase-NP-2nm-POPE')

# Plot the profile
plotDensityHistogram(density, filename='anatase-NP-2nm-POPE-N-NumberDensity.png', color='navy', label='N(PE)-TiO$_2$', width=0.01, x_min=0, x_max=1.5)
#plotDensityHistogram(density, filename='anatase-NP-2nm-POPE-N-NumberDensity-long.png', color='navy', label='N(PE)-TiO$_2$', width=0.01, x_min=0, x_max=4.5)