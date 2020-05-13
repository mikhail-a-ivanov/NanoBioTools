import numpy as np
import mdtraj as md
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from NumberDensityProfiles import readXTC, getSurfaceDistanceGeneral, getSurfaceDistanceSlab, normalizeSlab, normalizeSphere, plotProfile

#trajname = '/home/misha/Documents/AAMD/anatase-101-POPE-2/traj-whole-skip100.xtc'
#topname = '/home/misha/Documents/AAMD/anatase-101-POPE-2/anatase-101-POPE-2-confin-whole.gro'

trajname = '/home/misha/Documents/AAMD/anatase-NP-POPE/traj-whole-skip100-nowater-500ns.xtc'
topname = '/home/misha/Documents/AAMD/anatase-NP-POPE/confin-whole-nowater.gro'

# Call readXTC function
traj = readXTC(trajname, topname, stride=1)

# Call getSurfaceDistance function
#distances = getSurfaceDistanceSlab(traj, topname, resname='H151', atomname='N', cutoffBulk=1.25)
distances = getSurfaceDistanceGeneral(traj, topname, resname='H144', atomname='N')

# Normalize density
#density = normalizeSlab(traj, distances, topname, atomname='N', binWidth=0.01, outname='anatase-101-POPE-2')
density = normalizeSphere(traj, distances, topname, atomname='N', binWidth=0.005, outname='anatase-NP-2nm-POPE')

# Plot the profile
plotProfile(density, filename='anatase-NP-2nm-POPE-N-NumberDensity.png', color='navy', label='N(PE)-TiO$_2$', x_min=0, x_max=1.5)
plotProfile(density, filename='anatase-NP-2nm-POPE-N-NumberDensity-long.png', color='navy', label='N(PE)-TiO$_2$', x_min=0, x_max=4.5)