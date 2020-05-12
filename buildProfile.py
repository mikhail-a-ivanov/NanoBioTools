import numpy as np
import mdtraj as md
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from NumberDensityProfiles import readXTC, getSurfaceDistance, normalize, plotProfile

trajname = '/home/misha/Documents/AAMD/anatase-101-POPE-2/traj-whole-skip100.xtc'
topname = '/home/misha/Documents/AAMD/anatase-101-POPE-2/anatase-101-POPE-2-confin-whole.gro'

# Call readXTC function
traj = readXTC(trajname, topname, stride=1)

# Call getSurfaceDistance function
distances = getSurfaceDistance(traj, topname, 'H151', 'N')

# Normalize density
density = normalize(traj, distances, topname, 'N', 0.01, 'anatase-101-POPE-2')

# Plot the profile
plotProfile(density, 'anatase-101-POPE-2-N-NumberDensity.png', 'navy', 'N(PE)-TiO$_2$', 0, 1.5)
plotProfile(density, 'anatase-101-POPE-2-N-NumberDensity-long.png', 'navy', 'N(PE)-TiO$_2$', 0, 4.5)