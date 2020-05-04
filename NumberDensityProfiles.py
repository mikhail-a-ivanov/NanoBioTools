import numpy as np
import mdtraj as md
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

# A function that reads XTC trajectory file and returns MDtraj trajectory object
def readXTC(trajname, topname):
    start = time.time()
    print(f'Opening {trajname} with {topname} as a topology...')
    traj = md.load_xtc(trajname, top=topname)
    end = time.time()
    print(f'Reading trajectory file took {round(end - start, 4)} seconds.')
    return traj


# This function reads a topology file (for example, a single GRO or PDB file), finds OH groups
# within a certain residue (for example, TiO2 slab or NP) and returns numpy array with the atom indices
# of OH groups. First, the hydrogen atoms are found, then the function looks for neighbors of hydrogen atoms
# within certain cutoff distance (0.12 nm by default). It assumes that ALL hydrogen atoms within a residue
# belong to OH groups (no bonded water!). 
def selectOH(topname, resname, cutoff = 0.12):
    coordinates = md.load(topname)
    top = md.load(topname).topology
    selectH = f'resname == {resname} and symbol == H'
    print(f'Selecting hydrogens using "{selectH}" keywords...\nLooking for oxygen atoms within {cutoff} nm from the hydrogen atoms...')
    hydrogens = top.select(selectH)
    assert hydrogens.size != 0, ('Hydrogen atoms selection is empty. Please check the name of the residue or the topology file.')
    print(f'There are {len(hydrogens)} hydrogen atoms in the OH groups of the residue.')
    oxygens = md.compute_neighbors(coordinates, cutoff, hydrogens)[0]
    assert len(hydrogens) == len(oxygens), (
        'Number of hydrogen atoms does not match the number of oxygen atoms. Please check the cut-off distance \
(default value is 0.12 nm), selections or the topology file.')
    print(f'There are {len(oxygens)} oxygen atoms in the OH groups of the residue.')
    print(f'Number of oxygen atoms matches the number of hydrogen atoms.')
    return np.concatenate((oxygens, hydrogens))


# This function reads a topology file (for example, a single GRO or PDB file), selects
# a certain residue (for example TiO2 slab or NP) then calls selectOH function to find OH groups
# and returns a numpy array that contains indices of the original residue without OH groups
def removeOH(topname, resname, cutoff = 0.12):
    print(f'Selecting residue "{resname}"...')
    top = md.load(topname).topology
    selectResidue = f'resname == {resname}'
    residue_with_OH = top.select(selectResidue)
    assert residue_with_OH.size != 0, ('Residue atoms selection is empty. Please check the name of the residue or the topology file.')
    print(f'Residue "{resname}" contains {len(residue_with_OH)} atoms.')
    print(f'Finding and deleting OH groups...\n')
    OH = selectOH(topname, resname, cutoff)
    residue_without_OH = np.array([atom for atom in residue_with_OH if atom not in OH])
    assert residue_without_OH.size != 0, ('No atoms were found in the residue after deleting OH atoms. Please check the name of the residue or the topology file.')
    print(f'Residue "{resname}" contains {len(residue_without_OH)} atoms after deleting OH groups.')
    return residue_without_OH


"""
# Call readXTC function
traj = readXTC('../traj-whole-skip100.xtc', '../confin-whole.gro')
print(traj)

# Call selectOH function
OH = selectOH('../confin-whole.gro', 'H151')
print(OH.shape)
print(OH)
"""

# Get indices of TiO2 residue without OH groups
H151 = removeOH('../confin-whole.gro', 'H151', cutoff=0.12)
print(H151.shape)
print(H151)

# Get indices of N atoms
top = md.load('../confin-whole.gro').topology
N = top.select("name == N")
print(N.shape)
print(N)

# Get array of pairs N atom - all TiO2 atoms
N1 = np.repeat(N[0], len(H151), axis=0)
pairs = np.stack((N1, H151), axis=0).T
print(pairs.shape)
print(pairs)

# Load coordinates and compute distances between one N atom and all TiO2 atoms
coordinates = md.load('../confin-whole.gro')
start = time.time()
distances = md.compute_distances(coordinates, pairs)
end = time.time()
print(f'Distance calculation took {round(end - start, 8)} seconds.')
print(distances)
print(distances.shape)

# Load coordinates and compute distances between all N atoms and all TiO2 atoms (one frame)
N_frame = np.repeat(N, len(H151), axis=0)
pairs_frame = np.stack((N_frame, np.tile(H151, len(N)))).T
print(pairs_frame.shape)
print(pairs_frame)

coordinates = md.load('../confin-whole.gro')
start = time.time()
distances = md.compute_distances(coordinates, pairs_frame)
end = time.time()
print(f'Distance calculation took {round(end - start, 8)} seconds.')
print(distances)
print(distances.shape)









