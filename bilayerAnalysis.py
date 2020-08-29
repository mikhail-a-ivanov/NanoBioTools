import numpy as np
import mdtraj as md
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from NanoBioTools import getBilayerLeaflets

topname = '/mnt/c/Users/mikha/Work/anatase-101-120POPE-2/confin-whole.gro'

# Call getBilayerLeaflets function
getBilayerLeaflets(topname, resname='H151', resname_molecule='POPE', cutoffOH=0.12)