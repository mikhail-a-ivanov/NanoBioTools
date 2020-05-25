import numpy as np
import mdtraj as md
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from NanoBioTools import readXTC, getSurfaceDistanceGeneral, plotResidenceHistogram, getResidenceTime, plotResidenceHistogram

trajname = '/home/misha/Documents/AAMD/anatase-NP-POPE/traj-whole-skip100-nowater-500ns.xtc'
topname = '/home/misha/Documents/AAMD/anatase-NP-POPE/confin-whole-nowater.gro'

# Call readXTC function
traj = readXTC(trajname, topname, stride=1)

# Call getSurfaceDistance function
distances = getSurfaceDistanceGeneral(traj, topname, resname='H144', atomname='N', resname_molecule='POPE')

# Get residence times
residence_time = getResidenceTime(distances, atomname='N', resname='H144', outname='anatase-NP-2nm-POPE', distance_threshold=0.35, timestep=0.5, Nbins=50)

# Plot the residence time histogram
plotResidenceHistogram(residence_time, filename='anatase-NP-2nm-POPE-N-ResidenceTime.png', color='navy', label='N(PE)-TiO$_2$', width=5)