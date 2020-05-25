import numpy as np
import mdtraj as md
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from NanoBioTools import readXTC, getSurfaceDistanceSlab, plotResidenceHistogram, getResidenceTime, plotResidenceHistogram

trajname = '/home/misha/Documents/AAMD/anatase-101-POPE-2/traj-whole-skip100.xtc'
topname = '/home/misha/Documents/AAMD/anatase-101-POPE-2/anatase-101-POPE-2-confin-whole.gro'

# Call readXTC function
traj = readXTC(trajname, topname, stride=1)

# Call getSurfaceDistance function
distances = getSurfaceDistanceSlab(traj, topname, resname='H151', atomname='N', resname_molecule='POPE', cutoffBulk=1.25)

# Get residence times
residence_time = getResidenceTime(distances, atomname='N', resname='H151', outname='anatase-101-POPE-2', distance_threshold=0.35, timestep=0.5, Nbins=100)

# Plot the residence time histogram
plotResidenceHistogram(residence_time, filename='anatase-101-POPE-2-N-ResidenceTime.png', color='navy', label='N(PE)-TiO$_2$', width=5)