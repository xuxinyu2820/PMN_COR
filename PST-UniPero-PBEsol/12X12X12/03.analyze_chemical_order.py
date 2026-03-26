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
    A_site_elements = ['Pb', 'Sr'],
    B_site_elements = ['Ti'],
    element_specorder = ['Pb','Sr','O','Ti'],
    neighbor_cutoff = 6.0
)
simulation.initialize(supercell_size = [12, 12, 12], calculator = 1, restart = True, build_neighborlist = False)

if os.path.exists('A_neighborlist.pkl'):
    simulation.load_neighborlist()
else:
    simulation._build_swap_site_list(simulation.atoms)
    simulation.save_neighborlist()
lattice_map = simulation.get_lattice_map('A', [12, 12, 12])

# path to the directory containing the trajectory files 
traj_dir = 'trajs'
lmp_pattern = os.path.join(traj_dir, 'f*.lmp')
files = sorted(glob.glob(lmp_pattern),
               key=lambda x: int(os.path.splitext(os.path.basename(x))[0][1:]))

checkboard_order = []
Pb_all_cluster_sum = []
Pb_largest_cluster_size= []
Pb_surface_volume_ratio = []
Sr_all_cluster_sum = []
Sr_largest_cluster_size= []
Sr_surface_volume_ratio = []
for path in files:
    pmn = read(path, format='lammps-data', atom_style='atomic')
    update_element(pmn, ['Pb','Sr','O','Ti'])
    order_parameter = simulation.get_pair_product(pmn, score_map = {'Pb': -1, 'Sr': +1}, type='A')
    checkboard_order.append(order_parameter)
    ## analyze the cluster structure of Sr
    _all_cluster_sum, _largest_cluster_size, _surface_volume_ratio = find_clusters(pmn, lattice_map, 'Sr', tolerance=1)
    Sr_all_cluster_sum.append(_all_cluster_sum)
    Sr_largest_cluster_size.append(_largest_cluster_size)
    Sr_surface_volume_ratio.append(_surface_volume_ratio)
    ## analyze the cluster structure of Pb
    _all_cluster_sum, _largest_cluster_size, _surface_volume_ratio = find_clusters(pmn, lattice_map, 'Pb', tolerance=3)
    Pb_all_cluster_sum.append(_all_cluster_sum)
    Pb_largest_cluster_size.append(_largest_cluster_size)
    Pb_surface_volume_ratio.append(_surface_volume_ratio)
    print(f'Frame {path} done')
    print(f'Pb all cluster sum: {Pb_all_cluster_sum[-1]}, Pb largest cluster size: {Pb_largest_cluster_size[-1]}, Pb surface-volume ratio: {Pb_surface_volume_ratio[-1]}')
    print(f'Sr all cluster sum: {Sr_all_cluster_sum[-1]}, Sr largest cluster size: {Sr_largest_cluster_size[-1]}, Sr surface-volume ratio: {Sr_surface_volume_ratio[-1]}')


## Plot the order parameter
plt.figure(figsize=(6,4))
plt.plot(checkboard_order, linestyle='-', linewidth=2, color='black', alpha=1.0, label=None)
plt.ylabel(r'$O(\mathbf{s})$')
plt.xlabel('Accepted Steps')
plt.tight_layout()
plt.savefig('order_parameter.png', dpi=300)

## Plot the Sr cluster analysis
plt.figure(figsize=(6,4))
plt.plot(Sr_all_cluster_sum, linestyle='-', linewidth=2, color='gray', alpha=1.0, label=None)
plt.plot(Sr_largest_cluster_size, linestyle='-', linewidth=2, color='#206FB6', alpha=1.0, label=None)
plt.ylabel(r'Number of Sr sites')
plt.xlabel('Accepted Steps')
plt.tight_layout()
plt.savefig('Sr_cluster_analysis.png', dpi=300)

## Plot the Pb cluster analysis
plt.figure(figsize=(6,4))
plt.plot(Pb_all_cluster_sum, linestyle='-', linewidth=2, color='gray', alpha=1.0, label=None)
plt.plot(Pb_largest_cluster_size, linestyle='-', linewidth=2, color='#206FB6', alpha=1.0, label=None)
plt.ylabel(r'Number of Pb sites')
plt.xlabel('Accepted Steps')
plt.tight_layout()
plt.savefig('Pb_cluster_analysis.png', dpi=300)

## Plot the surface-volume ratio
plt.figure(figsize=(6,4))
plt.plot(Sr_surface_volume_ratio, linestyle='-', linewidth=2, color='gray', alpha=1.0, label='Sr')
plt.plot(Pb_surface_volume_ratio, linestyle='-', linewidth=2, color='#206FB6', alpha=1.0, label='Pb')
plt.legend()
plt.ylabel(r'Surface-volume ratio')
plt.xlabel('Accepted Steps')
plt.tight_layout()
plt.savefig('surface_volume_ratio.png', dpi=200)

