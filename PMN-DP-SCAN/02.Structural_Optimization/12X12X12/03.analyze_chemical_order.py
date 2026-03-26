import os, glob
import numpy as np
from ase.io import read
import matplotlib.pyplot as plt
from perostruc import update_element, FIRESwapOptimizer, find_clusters


plt.rcParams.update({
    'axes.linewidth': 2,     
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

simulation = FIRESwapOptimizer(
    temperature = 300.0,
    A_site_elements = ['Pb'],
    B_site_elements = ['Mg', 'Nb'],
    element_specorder = ['Mg','Nb','O','Pb'],
    neighbor_cutoff = 6.0
)
# simulation.initialize(supercell_size = [12, 12, 12], calculator = 1, restart = True, build_neighborlist = False)
atoms = read('trajs/f1.lmp', format='lammps-data', atom_style='atomic')
update_element(atoms, ['Mg','Nb','O','Pb'])
simulation.atoms = atoms
simulation.ncell = int(np.prod([12, 12, 12]))


if os.path.exists('A_neighborlist.pkl'):
    simulation.load_neighborlist()
else:
    simulation._build_swap_site_list(simulation.atoms)
    simulation.save_neighborlist()
lattice_map = simulation.get_lattice_map('B', [12, 12, 12])

# path to the directory containing the trajectory files 
traj_dir = 'trajs'
lmp_pattern = os.path.join(traj_dir, 'f*.lmp')
files = sorted(glob.glob(lmp_pattern),
               key=lambda x: int(os.path.splitext(os.path.basename(x))[0][1:]))

checkboard_order = []
Nb_all_cluster_sum = []
Nb_largest_cluster_size= []
Nb_surface_volume_ratio = []
Mg_all_cluster_sum = []
Mg_largest_cluster_size= []
Mg_surface_volume_ratio = []
for path in files:
    pmn = read(path, format='lammps-data', atom_style='atomic')
    update_element(pmn, ['Mg','Nb','O','Pb'])
    order_parameter = simulation.get_pair_product(pmn, score_map = {'Mg': -1, 'Nb': +1}, type='B')
    checkboard_order.append(order_parameter)
    ## analyze the cluster structure of Sr
    _all_cluster_sum, _largest_cluster_size, _surface_volume_ratio = find_clusters(pmn, lattice_map, 'Nb', tolerance=1)
    Nb_all_cluster_sum.append(_all_cluster_sum)
    Nb_largest_cluster_size.append(_largest_cluster_size)
    Nb_surface_volume_ratio.append(_surface_volume_ratio)
    ## analyze the cluster structure of Mg
    _all_cluster_sum, _largest_cluster_size, _surface_volume_ratio = find_clusters(pmn, lattice_map, 'Mg', tolerance=3)
    Mg_all_cluster_sum.append(_all_cluster_sum)
    Mg_largest_cluster_size.append(_largest_cluster_size)
    Mg_surface_volume_ratio.append(_surface_volume_ratio)
    print(f'Frame {path} done')
    print(f'Nb all cluster sum: {Nb_all_cluster_sum[-1]}, Nb largest cluster size: {Nb_largest_cluster_size[-1]}, Nb surface-volume ratio: {Nb_surface_volume_ratio[-1]}')
    print(f'Mg all cluster sum: {Mg_all_cluster_sum[-1]}, Mg largest cluster size: {Mg_largest_cluster_size[-1]}, Mg surface-volume ratio: {Mg_surface_volume_ratio[-1]}')


## Plot the order parameter
plt.figure(figsize=(6,4))
plt.plot(checkboard_order, linestyle='-', linewidth=2, color='black', alpha=1.0, label=None)
plt.ylabel(r'$O(\mathbf{s})$')
plt.xlabel('Accepted Steps')
plt.tight_layout()
plt.savefig('order_parameter.png', dpi=300)

## Plot the Nb cluster analysis
plt.figure(figsize=(6,4))
plt.plot(Nb_all_cluster_sum, linestyle='-', linewidth=2, color='gray', alpha=1.0, label=None)
plt.plot(Nb_largest_cluster_size, linestyle='-', linewidth=2, color='#206FB6', alpha=1.0, label=None)
plt.ylabel('Number of Nb sites')
plt.xlabel('Accepted Steps')
plt.tight_layout()
plt.savefig('Nb_cluster_analysis.png', dpi=300)

## Plot the Mg cluster analysis
plt.figure(figsize=(6,4))
plt.plot(Mg_all_cluster_sum, linestyle='-', linewidth=2, color='gray', alpha=1.0, label=None)
plt.plot(Mg_largest_cluster_size, linestyle='-', linewidth=2, color='#206FB6', alpha=1.0, label=None)
plt.ylabel('Number of Mg sites')
plt.xlabel('Accepted Steps')
plt.tight_layout()
plt.savefig('Mg_cluster_analysis.png', dpi=300)

## Plot the surface-volume ratio
plt.figure(figsize=(6,4))
plt.plot(Nb_surface_volume_ratio, linestyle='-', linewidth=2, color='gray', alpha=1.0, label='Nb')
plt.plot(Mg_surface_volume_ratio, linestyle='-', linewidth=2, color='#206FB6', alpha=1.0, label='Mg')
plt.legend()
plt.ylabel(r'Surface-volume ratio')
plt.xlabel('Accepted Steps')
plt.tight_layout()
plt.savefig('surface_volume_ratio.png', dpi=200)

steps = np.arange(len(Nb_all_cluster_sum))
data = np.column_stack([steps, Nb_all_cluster_sum, Nb_largest_cluster_size])
np.savetxt('cluster.csv', data, delimiter=',', header='accepted_step,Nb_all_cluster_sum,Nb_largest_cluster_size', comments='')