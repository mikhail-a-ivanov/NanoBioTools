import numpy as np
import mdtraj as md
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
import time

def readXTC(trajname, topname, stride=1):
    """A function that reads XTC trajectory file and returns MDtraj trajectory object."""

    start = time.time()
    print(f'Opening {trajname} trajectory file (stride = {stride}) with {topname} as a topology...')
    traj = md.load_xtc(trajname, top=topname, stride=stride)
    end = time.time()
    print(f'Reading trajectory file took {round(end - start, 4)} seconds.')
    print(f'{traj}\n')
    return traj

def removeSlabBulk(topname, resname, cutoffBulk=0):
    """This function removes bulk atoms from a slab based on their distance along Z axis to COM."""

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

def selectOH(topname, resname, cutoffOH = 0.12):
    """This function reads a topology file (for example, a single GRO or PDB file), finds OH groups
    within a certain residue (for example, TiO2 slab or NP) and returns numpy array with the atom indices
    of OH groups. First, the hydrogen atoms are found, then the function looks for neighbors of hydrogen atoms
    within certain cutoff distance (0.12 nm by default). It assumes that ALL hydrogen atoms within a residue
    belong to OH groups (no bonded water!). """

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

def removeOH(topname, resname, residue, cutoffOH = 0.12):
    """This function takes an array of atom indices corresponding to
    a certain residue (for example TiO2 slab or NP) then calls selectOH function to find OH groups
    and returns a numpy array that contains indices of the original residue without OH groups"""

    assert len(residue) != 0, ('Input residue selection is empty!')
    print(f'Input residue contains {len(residue)} atoms.')
    print(f'Finding and deleting OH groups...\n')
    OH = selectOH(topname, resname, cutoffOH=cutoffOH)
    residue_without_OH = np.array([atom for atom in residue if atom not in OH])
    assert residue_without_OH.size != 0, ('No atoms were found in the residue after deleting OH atoms. Please check the name of the residue or the topology file.')
    print(f'Residue "{resname}" contains {len(residue_without_OH)} atoms after deleting OH groups.\n')
    return residue_without_OH

def selectAtoms(top, resname, atomname):
    """Select atoms from a topology and return list of indices of the atoms"""

    selectAtoms = f'resname == {resname} and name == {atomname}'
    atoms = top.select(selectAtoms)
    print(f'Selecting {len(atoms)} "{atomname}" atoms from resname "{resname}"')
    return atoms

def getBilayerLeaflets(topname, resname, lipid_resname, cutoffOH=0.12):
    """This function calculates distance from COM of every lipid molecule to the slab surface
    along the normal (difference in Z coordinates of lipid COM and the slab surface atom)
    By looking at lipid COM - time plot one can judge if there are any lipid flip flops (lipid diffusion along the bilayer normal)
    The function also outputs indeces of lipids, belonging to the bottom or the upper bilayer leaflet"""

    # Get indices of the residue without OH groups
    top = md.load(topname).topology
    selectResidue = f'resname == {resname}'
    residue = top.select(selectResidue)
    assert residue.size != 0, ('Residue atoms selection is empty. Please check the name of the residue or the topology file.')
    residue_without_OH = removeOH(topname, resname, residue, cutoffOH=cutoffOH)

    # Load coordinates of the residue atoms
    coordinates_all = md.load(topname)
    coordinates_residue = coordinates_all.atom_slice(residue_without_OH)

    # Compute slab COM, Z_max and Z_min
    slab_com = md.compute_center_of_geometry(coordinates_residue)
    slab_Z_com = slab_com[0][2]
    slab_Z_max = np.amax(coordinates_residue.xyz[0].T[2])
    slab_Z_min = np.amin(coordinates_residue.xyz[0].T[2])

    print(f'Slab COM = {slab_Z_com:.4f} nm')
    print(f'Slab max Z coordinate = {slab_Z_max:.4f} nm')
    print(f'Slab min Z coordinate = {slab_Z_min:.4f} nm')

    # Pick lipid residues
    # The list will contain indices of lipid residues (molecules)
    lipids = []
    
    # Loop over all residues and save indices
    # of residues that contain 'lipid_resname' in their names
    for residue in top.residues:
        if lipid_resname in residue.name:
            lipids.append(residue.index)
    
    print(lipids)

    # To select a residue based on its index
    # use 'resid' == [index] 
    selectLipid = f'resid == 1'
    lipid_test = top.select(selectLipid)
    print(lipid_test)

    return

def getSurfaceDistanceSlab(traj, topname, resname, atomname, resname_molecule, cutoffOH=0.12, cutoffBulk=0):
    """This function takes trajectory file that is read by readXTC function, indices of atoms
    from matching topology, the name of the residue and cutoff distance
    for finding oxygens of OH groups in the residue.
    The distances between all the atoms and residue atoms are calculated and the minimum
    distance between each atom and every atom of the residue is taken
    and outputed as an array"""

    # A suggestion: analyze one frame at a time?

    # Get indices of the residue without OH groups and without bulk atoms
    top = md.load(topname).topology
    residue_modified = removeSlabBulk(topname, resname, cutoffBulk=cutoffBulk)
    residue = removeOH(topname, resname, residue_modified, cutoffOH=cutoffOH)

    # Get indices of the atoms
    atoms = selectAtoms(top, resname_molecule, atomname)

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

    # Returns an array of atom-to-surface distances with (N_atoms, N_frames) shape  

    return np.amin(distances_reshaped, axis=2).T

def getNanoparticleRadius(traj, topname, resname, cutoffOH=0.12):
    """Estimate radius of a nanoparticle by calculating distances
    between all NP atoms (without OH groups) and its COM. Radius
    is obtained by taking maximum NP atom - NP COM distance at 
    each frame and averaging it over the whole trajectory.
    
    The function returns (mean radius, standard deviation, max radius, min radius)"""

    print(f"Estimating radius of a nanoparticle (resname == {resname})... \n")

    # Get indices of the residue without OH groups
    top = md.load(topname).topology
    selectResidue = f'resname == {resname}'
    residue_full = top.select(selectResidue)
    residue = removeOH(topname, resname, residue_full, cutoffOH=cutoffOH)

    # Get coordinates of the residue (without OH groups) at each frame
    residue_coordinates = traj.atom_slice(residue)

    # Compute COM at each frame
    coms = md.compute_center_of_geometry(residue_coordinates)

    # Compute max NP atom - NP COM distances (radii) for each frame
    max_radii = [np.amax(np.linalg.norm(residue_coordinates.xyz[i] - coms[i], axis=1)) for i in range(len(residue_coordinates.xyz))]

    print(f'Nanoparticle radius estimation: \nR_mean = {np.mean(max_radii)} nm \nR_std = {np.std(max_radii)} nm \
    \nR_max = {np.amax(max_radii)} nm \nR_min = {np.amin(max_radii)} nm \n')

    return np.mean(max_radii), np.std(max_radii), np.amax(max_radii), np.amin(max_radii)

def getSurfaceDistanceGeneral(traj, topname, resname, atomname, resname_molecule, cutoffOH=0.12):
    """This function takes trajectory file that is read by readXTC function, indices of atoms
    from matching topology, the name of the residue and cutoff distance
    for finding oxygens of OH groups in the residue.
    The distances between all the atoms and residue atoms are calculated and the minimum
    distance between each atom and every atom of the residue is taken
    and outputed as an array"""

    # Get indices of the residue without OH groups
    top = md.load(topname).topology
    selectResidue = f'resname == {resname}'
    residue_full = top.select(selectResidue)
    residue = removeOH(topname, resname, residue_full, cutoffOH=cutoffOH)
   
    # Get indices of the atoms
    atoms = selectAtoms(top, resname_molecule, atomname)

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
    
    # Returns an array of atom-to-surface distances with (N_atoms, N_frames) shape   

    return np.amin(distances_reshaped, axis=2).T

def normalizeSlab(traj, distances, topname, outname, atomname, resname_molecule, binWidth=0.005, r_max=4.5):
    """This function builds histogram for the atom - residue distances. It takes trajectory file 
    that is read by readXTC function, indices of atoms
    from matching topology, name of the atoms, bin width for histogram (nm) and the name
    of the system for the output file name.
    
    The density is normalized and converted to nm^-3."""

    # Load the topology
    top = md.load(topname).topology

    # Number of atoms
    atoms = selectAtoms(top, resname_molecule, atomname)
    Natoms = len(atoms) 

    # Normalization properties
    box = traj.unitcell_lengths[0]
    Nframes = traj.n_frames
    boxVolume = box[0] * box[1] * box[2]
    averageNumberDensity = Natoms / boxVolume
    slabSurfaceArea = box[0] * box[1]
    binVolume = binWidth * slabSurfaceArea
    # Flexible bin values
    #Nbins = int((np.amax(distances.flatten()) - np.amin(distances.flatten())) / binWidth)
    # Fixed bin values
    Nbins = int(r_max / binWidth)

    print(f'Building histogram...')
    print(f'Number of frames: {Nframes}')
    print(f'Average number density: {round(averageNumberDensity, 8)}')
    print(f'Bin volume: {round(binVolume, 4)} nm3')
    print(f'Number of bins: {Nbins}')

    # Build histogram
    hist = np.histogram(distances.flatten(), bins=Nbins, range=(0, r_max), density=False)
    density = np.array((hist[1][1:], hist[0]/(Nframes*binVolume)))

    # Write the histogram to file
    header = f'{atomname} number density ({outname}) \nDistance, nm; Number density, nm^-3'
    filename = f'{outname}-{atomname}-NumberDensity.dat'
    np.savetxt(filename, density.T, fmt='%.6f', header=header)

    return density

def normalizeSphere(traj, distances, topname, outname, atomname, resname_NP, resname_molecule, binWidth=0.005, r_max=4.5, normalize=True):
    """This function builds histogram for the atom - residue distances. It takes trajectory file 
    that is read by readXTC function, indices of atoms
    from matching topology, name of the atoms, bin width for histogram (nm) and the name
    of the system for the output file name."""

    # Load the topology
    top = md.load(topname).topology

    # Number of atoms
    atoms = selectAtoms(top, resname_molecule, atomname)
    Natoms = len(atoms) 

    # Normalization properties
    box = traj.unitcell_lengths[0]
    Nframes = traj.n_frames
    boxVolume = box[0] * box[1] * box[2]
    averageNumberDensity = Natoms / boxVolume

    # Fixed bin values
    Nbins = int(r_max / binWidth)

    # Obtain radius of the spherical NP
    R, Rstd, Rmax, Rmin = getNanoparticleRadius(traj, topname, resname=resname_NP)

    sphereRadii = np.linspace(R, (R + r_max), Nbins)
    binVolumes = 4 * np.pi * sphereRadii**2 * binWidth

    
    print(f'Building histogram...')
    print(f'Number of frames: {Nframes}')
    print(f'Average number density: {round(averageNumberDensity, 8)}')
    print(f'Min bin volume: {np.amin(binVolumes)} nm3')
    print(f'Max bin volume: {np.amax(binVolumes)} nm3')
    print(f'Number of bins: {Nbins}')

    # Build histogram (use distances to COM instead, then subtract radius again to get the distance to the surface)
    hist = np.histogram(distances.flatten() + R, bins=Nbins, range=(R, (R + r_max)), density=False)
    # Normalize to nm^-3 by dividing with the bin volumes:
    if normalize:
        density = np.array((hist[1][1:], hist[0]/(Nframes*binVolumes)))
    else:
        density = np.array((hist[1][1:], Nbins * hist[0]/Nframes))

    density[0] -= R

    # Instead, I will do a general normalization where I divide the occurenece by the number of frames and multiply by the number of bins
    #density = np.array((hist[1][1:], Nbins * hist[0]/Nframes))

    # Write the histogram to file
    if normalize:
        header = f'{atomname} number density ({outname}) \nDistance, nm; Number density, nm^-3 \n \
        Nanoparticle radius estimation: \nR_mean = {R} nm \nR_std = {Rstd} nm \nR_max = {Rmax} nm \nR_min = {Rmin} nm'
        filename = f'{outname}-{atomname}-NumberDensity.dat'
        np.savetxt(filename, density.T, fmt='%.6f', header=header)
    else:
        header = f'{atomname} occurrence ({outname}) \nDistance, nm; Occurrence \n \
        Nanoparticle radius estimation: \nR_mean = {R} nm \nR_std = {Rstd} nm \nR_max = {Rmax} nm \nR_min = {Rmin} nm'
        filename = f'{outname}-{atomname}-occurrence.dat'
        np.savetxt(filename, density.T, fmt='%.6f', header=header)

    return density

def plotDensityProfile(density, filename, color, label, x_min, x_max):
    """Plots the number density profile"""

    plt.rcParams.update({'font.size': 14})

    fig, ax = plt.subplots(figsize=(12,7))

    ax.plot(density[0], density[1], color=color, label=label, lw=2)
    ax.set(xlabel='Distance (nm)', ylabel='Number density ($nm^{-3}$)', title='')
    plt.xlim(x_min, x_max)
    ax.legend()
    ax.grid()

    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    plt.show()

    return

def plotDensityHistogram(density, filename, color, label, x_min, x_max):
    """Plots the density histogram (for spheres)"""

    plt.rcParams.update({'font.size': 14})

    fig, ax = plt.subplots(figsize=(12,7))

    ax.plot(density[0], density[1], color=color, label=label, lw=2)
    ax.set(xlabel='Distance (nm)', ylabel='Occurence', title='')
    plt.xlim(x_min, x_max)
    ax.legend()
    ax.grid()

    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    plt.show()

    return

def runsOfOnes(bits):
    """Auxiliary function to calculate lengths of sequences of ones for 0,1 arrays"""
    return [sum(g) for b, g in itertools.groupby(bits) if b]

def getResidenceTime(distances, atomname, resname, outname, distance_min=0.25, distance_max=0.35, timestep=0.5, Nbins=2000):
    """This function builds a histogram of lengths of binding events and returns a weighted average of the mean residence time
    Distances array should have a shape of (N_atoms, N_frames) as getSurfaceDistances function returns
    The function takes two distance threshold values - distance_min and distance_max. That allows
    to get residence times for separate atom density peaks """

    # Total simulation time is the number of frames multiplied by the timestep minus the first frame of the simulation
    total_simulation_time = (len(distances.T) - 1) * timestep
    #Nbins = int(total_simulation_time / timestep)

    print(f'Calculating residence time for {resname}-{atomname} pair with the min distance of {distance_min} nm and max distance of {distance_max} nm.\n\
Total simulation time = {total_simulation_time} ns, time step = {timestep} ns.\n\
Using {Nbins} bins for building the histogram.\n')

    # Get 0,1 array where 1 corresponds to bonded state and 0 to non-bonded state (max distance threshold only)
    #bonded_states = (distances < distance_threshold).astype(np.int)
     
    # Get 0,1 array where 1 corresponds to bonded state and 0 to non-bonded state
    bonded_states = np.logical_and(distances < distance_max, distances > distance_min).astype(int)

    # Loop over all atoms and binding events
    residence_times = []
    for i in range(len(bonded_states)):
        residence_times += runsOfOnes(bonded_states[i])

    # Build the histogram
    hist = np.histogram(np.array(residence_times) * timestep, bins=Nbins, density=False)
    bins = hist[1][1:]
    occurrence= hist[0]
    # Normalize the occurrence in such a way that it is related to the total possible number of such events
    # e.g. with 1000 ns total simulation time the occurrence of 10 ns binding events is multiplied by 1000 / (1000 - 10), etc.
    #occurrence_normalized = hist[0]*(bins/total_simulation_time) # old incorrect normalization
    occurrence_normalized = hist[0] * (total_simulation_time/(total_simulation_time - bins + 1))

    residence_time_data = np.array((bins, occurrence))

    residence_time_data_normalized = np.array((bins, occurrence_normalized))
        
    # Write the histogram to file
    header = f'{outname}-{atomname}-ResidenceTime \nDistance min = {distance_min} nm; Distance max = {distance_max} nm; \
 Total simulation time = {total_simulation_time};\
 Time step = {timestep} ns; Number of bins = {Nbins} \nResidence time, ns; Occurrence'
    filename = f'{outname}-{atomname}-ResidenceTime.dat'
    np.savetxt(filename, residence_time_data.T, fmt='%.6f', header=header)
    print(f'Residence time data is written to {filename}.')

     # Write the normalized histogram to file
    header_normalized = f'{outname}-{atomname}-ResidenceTime-normalized \nDistance min = {distance_min} nm; Distance max = {distance_max} nm; \
 Total simulation time = {total_simulation_time};\
 Time step = {timestep} ns; Number of bins = {Nbins} \nResidence time, ns; Occurrence'
    filename_normalized = f'{outname}-{atomname}-ResidenceTime-normalized.dat'
    np.savetxt(filename_normalized, residence_time_data_normalized.T, fmt='%.6f', header=header_normalized)
    print(f'Residence time data is written to {filename_normalized}.')

    # Estimate mean (lower bound) for the residence time
    mean_residence_time = round(np.average(bins, weights=occurrence), 6)
    print(f'Mean residence time = {mean_residence_time} ns')

    # Estimate mean (lower bound) for the residence time (normalized)
    mean_residence_time_normalized = round(np.average(bins, weights=occurrence_normalized), 6)
    print(f'Mean residence time (normalized) = {mean_residence_time_normalized} ns')

    return residence_time_data, residence_time_data_normalized, mean_residence_time, mean_residence_time_normalized

def plotResidenceHistogram(residence_time_data, filename, color, label, width=5):
    """Plots the residence time histogram"""

    plt.rcParams.update({'font.size': 14})

    fig, ax = plt.subplots(figsize=(12,7))

    ax.bar(residence_time_data[0], residence_time_data[1], color=color, label=label, width=width, alpha=0.75)
    ax.set(xlabel='Residence time (ns)', ylabel='Occurrence', title='')
    ax.legend()
    ax.grid()

    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    plt.show()

    return