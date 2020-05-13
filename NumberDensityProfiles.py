import numpy as np
import mdtraj as md
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

# A function that reads XTC trajectory file and returns MDtraj trajectory object
def readXTC(trajname, topname, stride=1):
    start = time.time()
    print(f'Opening {trajname} trajectory file (stride = {stride}) with {topname} as a topology...')
    traj = md.load_xtc(trajname, top=topname, stride=stride)
    end = time.time()
    print(f'Reading trajectory file took {round(end - start, 4)} seconds.')
    print(f'{traj}\n')
    return traj


# This function removes bulk atoms from a slab based on their distance along Z axis to COM.
def removeSlabBulk(topname, resname, cutoffBulk=0):
    print(f'Selecting residue "{resname}"...')
    print(f'Using {cutoffBulk} nm as a minimum distance from COM along Z axis to remove bulk slab atoms.\n')
    
    # Get indices of the residue
    top = md.load(topname).topology
    selectResidue = f'resname == {resname}'
    residue = top.select(selectResidue)
    assert residue.size != 0, ('Residue atoms selection is empty. Please check the name of the residue or the topology file.')

    # Load coordinates of the residue atoms
    coordinates_all = md.load(topname)
    coordinates_residue = coordinates_all.atom_slice(residue)

    # Calculate COM of the residue
    com = md.compute_center_of_geometry(coordinates_residue)

    # Calculate the distance from every atom to COM along Z axis
    Z_to_COM = abs(coordinates_residue.xyz[0].T[2] - com[0][2])

    # Return indices of atoms that are separated from COM
    # by more than cutoffBulk (nm) along Z axis:
    residue_modified = np.argwhere(Z_to_COM > cutoffBulk).T[0]
    assert residue_modified.size != 0, ('No atoms were found in the residue after deleting bulk atoms. Please change the cutoffBulk (nm) parameter.')
    print(f'Residue "{resname}" contains {len(residue_modified)} atoms after deleting bulk atoms.\n')

    return residue_modified


# This function reads a topology file (for example, a single GRO or PDB file), finds OH groups
# within a certain residue (for example, TiO2 slab or NP) and returns numpy array with the atom indices
# of OH groups. First, the hydrogen atoms are found, then the function looks for neighbors of hydrogen atoms
# within certain cutoff distance (0.12 nm by default). It assumes that ALL hydrogen atoms within a residue
# belong to OH groups (no bonded water!). 
def selectOH(topname, resname, cutoffOH = 0.12):
    print(f'Selecting residue "{resname}"...')
    coordinates = md.load(topname)
    top = md.load(topname).topology
    selectH = f'resname == {resname} and symbol == H'
    print(f'Selecting hydrogens using "{selectH}" keywords...\nLooking for oxygen atoms within {cutoffOH} nm from the hydrogen atoms...')
    hydrogens = top.select(selectH)
    #assert hydrogens.size != 0, ('Hydrogen atoms selection is empty. Please check the name of the residue or the topology file.')
    print(f'There are {len(hydrogens)} hydrogen atoms in the OH groups of the residue.')
    oxygens = md.compute_neighbors(coordinates, cutoffOH, hydrogens)[0]
    assert len(hydrogens) == len(oxygens), (
        'Number of hydrogen atoms does not match the number of oxygen atoms. Please check the cut-off distance \
(default value is 0.12 nm), selections or the topology file.')
    print(f'There are {len(oxygens)} oxygen atoms in the OH groups of the residue.')
    print(f'Number of oxygen atoms matches the number of hydrogen atoms.')
    return np.concatenate((oxygens, hydrogens))


# This function takes an array of atom indices corresponding to
# a certain residue (for example TiO2 slab or NP) then calls selectOH function to find OH groups
# and returns a numpy array that contains indices of the original residue without OH groups
def removeOH(topname, resname, residue, cutoffOH = 0.12):
    assert len(residue) != 0, ('Input residue selection is empty!')
    print(f'Input residue contains {len(residue)} atoms.')
    print(f'Finding and deleting OH groups...\n')
    OH = selectOH(topname, resname, cutoffOH=cutoffOH)
    residue_without_OH = np.array([atom for atom in residue if atom not in OH])
    assert residue_without_OH.size != 0, ('No atoms were found in the residue after deleting OH atoms. Please check the name of the residue or the topology file.')
    print(f'Residue "{resname}" contains {len(residue_without_OH)} atoms after deleting OH groups.\n')
    return residue_without_OH


# Select atoms from a topology and return list of indices of the atoms
def selectAtoms(top, atomname):
    selectAtoms = f'name == {atomname}'
    atoms = top.select(selectAtoms)
    print(f'Selecting {len(atoms)} "{atomname}" atoms...')
    return atoms


# This function takes trajectory file that is read by readXTC function, indices of atoms
# from matching topology, the name of the residue and cutoff distance
# for finding oxygens of OH groups in the residue.
# The distances between all the atoms and residue atoms are calculated and the minimum
# distance between each atom and every atom of the residue is taken
# and outputed as an array
def getSurfaceDistance(traj, topname, resname, atomname, cutoffOH=0.12, isSlab=False, cutoffBulk=0):

    if not isSlab:
        # Get indices of the residue without OH groups
        top = md.load(topname).topology
        selectResidue = f'resname == {resname}'
        residue_full = top.select(selectResidue)
        residue = removeOH(topname, resname, residue_full, cutoffOH=cutoffOH)
    else:
        # Get indices of the residue without OH groups and without bulk atoms
        top = md.load(topname).topology
        residue_modified = removeSlabBulk(topname, resname, cutoffBulk=cutoffBulk)
        residue = removeOH(topname, resname, residue_modified, cutoffOH=cutoffOH)

    # Get indices of the atoms
    atoms = selectAtoms(top, atomname)

    # Get pairs
    atomsRepeated = np.repeat(atoms, len(residue), axis=0)
    pairs = np.stack((atomsRepeated, np.tile(residue, len(atoms)))).T

    # Compute distances
    print(f'Computing the closest distances between "{resname}" residue and selected atoms...')
    start = time.time()
    distances = md.compute_distances(traj, pairs)
    end = time.time()
    print(f'Distance calculation took {round(end - start, 8)} seconds.\n')
    distances_reshaped = np.reshape(distances, (traj.n_frames, len(atoms), len(residue)))
 
    # distance_reshaped is an array with (N_frames * N_atoms * N_atoms_residue) shape.
    # minimum element of the array along axis=2 means taking the distance from every frame and every atom
    # to the nearest residue atom (hence the surface distance)  

    return np.amin(distances_reshaped, axis=2).flatten()


# This function builds histogram for the atom - residue distances. It takes trajectory file 
# that is read by readXTC function, indices of atoms
# from matching topology, name of the atoms, bin width for histogram (nm) and the name
# of the system for the output file name.
def normalize(traj, distances, topname, atomname, binWidth, outname):
    # Load the topology
    top = md.load(topname).topology

    # Number of atoms
    atoms = selectAtoms(top, atomname)
    Natoms = len(atoms) 

    # Normalization properties
    box = traj.unitcell_lengths[0]
    Nframes = traj.n_frames
    boxVolume = box[0] * box[1] * box[2]
    averageNumberDensity = Natoms / boxVolume
    slabSurfaceArea = box[0] * box[1]
    binVolume = binWidth * slabSurfaceArea
    Nbins = int((max(distances) - min(distances)) / binWidth)

    print(f'Building histogram...')
    print(f'Number of frames: {Nframes}')
    print(f'Average number density: {round(averageNumberDensity, 8)}')
    print(f'Bin volume: {round(binVolume, 4)} nm3')
    print(f'Number of bins: {Nbins}')

    # Build histogram
    hist = np.histogram(distances, bins=Nbins, density=False)
    density = np.array((hist[1][1:], hist[0]/(Nframes*binVolume)))

    # Write the histogram to file
    header = f'{atomname} number density ({outname}) \nDistance, nm; Number density, nm^-3'
    filename = f'{outname}-{atomname}-NumberDensity.dat'
    np.savetxt(filename, density.T, fmt='%.6f', header=header)

    return density

# Plots the number density profile
def plotProfile(density, filename, color, label, x_min, x_max):
    plt.rcParams.update({'font.size': 14})

    fig, ax = plt.subplots(figsize=(12,7))

    ax.plot(density[0], density[1], color=color, label=label, lw=2)
    ax.set(xlabel='Distance (nm)', ylabel='Number density $nm^{-3}$', title='')
    plt.xlim(x_min, x_max)
    ax.legend()
    ax.grid()

    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    plt.show()

    return