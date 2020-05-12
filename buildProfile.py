import numpy as np
import mdtraj as md
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from NumberDensityProfiles import readXTC, removeOH, getSurfaceDistance, normalize, plotProfile

trajname = '/home/misha/Documents/AAMD/anatase-101-POPE-2/traj-whole-skip100.xtc'
topname = '/home/misha/Documents/AAMD/anatase-101-POPE-2/anatase-101-POPE-2-confin-whole.gro'

# Call readXTC function
traj = readXTC(trajname, topname, stride=1)

# Call getSurfaceDistance function
distances = getSurfaceDistance(traj, topname, resname='H151', atomname='N', isSlab=False, cutoffBulk=0.)

# Normalize density
density = normalize(traj, distances, topname, atomname='N', binWidth=0.01, outname='anatase-101-POPE-2')

# Plot the profile
plotProfile(density, filename='anatase-101-POPE-2-N-NumberDensity.png', color='navy', label='N(PE)-TiO$_2$', x_min=0, x_max=1.5)
plotProfile(density, filename='anatase-101-POPE-2-N-NumberDensity-long.png', color='navy', label='N(PE)-TiO$_2$', x_min=0, x_max=4.5)