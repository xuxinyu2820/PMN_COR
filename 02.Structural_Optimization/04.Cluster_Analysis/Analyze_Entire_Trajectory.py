import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import ase, ase.io
from utilities import *


latt_size = [12, 12, 12]
l1, l2, l3 = latt_size
ncells = l1 * l2 * l3
traj_folder = '../03.12X12X12/600K/trajs'

## load the file
if os.path.exists('buffer/traj.pkl'):
    with open('buffer/traj.pkl', 'rb') as f:
        traj = pickle.load(f)
else:
    traj = []
    for frame_idx in range(1,4004):
        file_path = os.path.join(traj_folder, f'f{frame_idx}.lmp')
        atoms = ase.io.read(file_path, format='lammps-data', atom_style='atomic')
        traj.append(atoms)
    
    ## Reset the chemical symbols of B-site atoms. Lammps data file uses 'H', 'Be', 'He', 'Li' to represent Mg, Pb, Nb, O, respectively.
    for atoms in traj:
        syms_lmp = atoms.get_chemical_symbols()
        syms = syms_lmp.copy()
        for idx, s in enumerate(syms_lmp):
            if s == 'H':
                syms[idx] = 'Mg'
            elif s == 'Be':
                syms[idx] = 'Pb'
            elif s == 'He':
                syms[idx] = 'Nb'
            elif s == 'Li':
                syms[idx] = 'O'
            else:
                raise ValueError('Invalid symbol')
        atoms.set_chemical_symbols(syms)
    os.makedirs('buffer', exist_ok=True)
    with open('buffer/traj.pkl', 'wb') as f:
        pickle.dump(traj, f)

## get the effective lattice of B-site atoms from the first frame. The indices are the same for all frames.
lattice_idx = get_effective_lattice(traj[0], latt_size, ['Mg', 'Nb'])
print(lattice_idx.shape)

## define the variables to store the results
Nb_in_cluster_list = []
Nb_in_CRO_list = []
Mg_NN_pair_list = []
largest_Nb_cluster_list = []

for idx, atoms in enumerate(traj):
    ## initialize the lattice values (0 means Mg, 1 means Nb)
    lattice_values = np.zeros_like(lattice_idx, dtype=int)
    syms = atoms.get_chemical_symbols()
    for i in range(l1):
        for j in range(l2):
            for k in range(l3):
                Bsite_sym = syms[lattice_idx[i,j,k]]
                lattice_values[i,j,k] = 0 if Bsite_sym == 'Mg' else 1

    ## find those Nb sites that are embedded in Nb clusters (neighboring at most one Mg), and those Nb sites that are in 1:1 chemically ordered regions (neighboring at least five Mg)
    ## also find those Mg sites that have neighboring Mg atoms
    CRO_flag = np.zeros_like(lattice_values, dtype=int)
    cluster_flag = np.zeros_like(lattice_values, dtype=int)
    Mgpair_flag = np.zeros_like(lattice_values, dtype=int)
    for i in range(l1):
        for j in range(l2):
            for k in range(l3):
                if lattice_values[i,j,k] == 0:
                    neighbor_values = find_neighbor_values(lattice_values, i, j, k)
                    if neighbor_values.count(0) > 0:
                        Mgpair_flag[i,j,k] = 1
                else:
                    neighbor_values = find_neighbor_values(lattice_values, i, j, k)
                    if neighbor_values.count(0) <= 1:
                        cluster_flag[i,j,k] = 1
                    if neighbor_values.count(0) >= 5:
                        CRO_flag[i,j,k] = 1
    Nb_in_cluster_list.append(cluster_flag.sum())
    Nb_in_CRO_list.append(CRO_flag.sum())
    Mg_NN_pair_list.append(Mgpair_flag.sum())
    ## get the clusters
    all_clusters = BFS_cluster_analysis(cluster_flag)
    largest_Nb_cluster = all_clusters[np.argmax([len(cluster) for cluster in all_clusters])]
    largest_Nb_cluster_list.append(len(largest_Nb_cluster))
    
    ## Calculate the surface-volume ratio of the largest Nb cluster
    largest_Nb_cluster_volume = len(largest_Nb_cluster)
    largest_cluster_flag = np.zeros_like(lattice_values, dtype=int)
    for i, j, k in largest_Nb_cluster:
        largest_cluster_flag[i,j,k] = 1
    surface_size = calculate_cluster_surface_size(largest_cluster_flag)
    surface_volume_ratio = surface_size / largest_Nb_cluster_volume
    
    ## print the results
    if (idx-1) % 100 == 0:
        print(f'Frame {idx} done')
        print(f'Nb in cluster: {Nb_in_cluster_list[-1]}, Nb in CRO: {Nb_in_CRO_list[-1]}, Mg NN pair: {Mg_NN_pair_list[-1]}')
        print(f'Largest Nb cluster: {largest_Nb_cluster_list[-1]}')

        print(f'Surface-volume ratio of the largest Nb cluster: {surface_volume_ratio}')


plt.rcParams.update({
    'axes.linewidth': 2,   # ④
    'axes.labelsize': 14,  # ⑤
    'xtick.labelsize': 12, # ⑤
    'ytick.labelsize': 12  # ⑤
})

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(Nb_in_cluster_list,        label='All Nb clusters',     color='gray',    alpha=0.6)
ax.plot(largest_Nb_cluster_list,   label='Largest Nb cluster',  color='#206FB6')
ax.set_xlabel('Accepted Swaps')
ax.set_ylabel('Number of Nb sites')

plt.tight_layout()
plt.savefig('Traj_analysis.png', dpi=300)
plt.close()