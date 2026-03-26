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
    B_site_elements = ['Zr', 'Ti'],
    element_specorder = ['Zr','Ti','O','Pb'],
    neighbor_cutoff = 6.0
)
simulation.initialize(supercell_size = [12, 12, 12], calculator = 1, restart = True, build_neighborlist = False)

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
Ti_all_cluster_sum = []
Ti_largest_cluster_size= []
Ti_surface_volume_ratio = []
Zr_all_cluster_sum = []
Zr_largest_cluster_size= []
Zr_surface_volume_ratio = []
for path in files:
    pmn = read(path, format='lammps-data', atom_style='atomic')
    update_element(pmn, ['Zr','Ti','O','Pb'])
    order_parameter = simulation.get_pair_product(pmn, score_map = {'Zr': -1, 'Ti': +1}, type='B')
    checkboard_order.append(order_parameter)
    ## analyze the cluster structure of Sr
    _all_cluster_sum, _largest_cluster_size, _surface_volume_ratio = find_clusters(pmn, lattice_map, 'Ti', tolerance=1)
    Ti_all_cluster_sum.append(_all_cluster_sum)
    Ti_largest_cluster_size.append(_largest_cluster_size)
    Ti_surface_volume_ratio.append(_surface_volume_ratio)
    ## analyze the cluster structure of Zr
    _all_cluster_sum, _largest_cluster_size, _surface_volume_ratio = find_clusters(pmn, lattice_map, 'Zr', tolerance=3)
    Zr_all_cluster_sum.append(_all_cluster_sum)
    Zr_largest_cluster_size.append(_largest_cluster_size)
    Zr_surface_volume_ratio.append(_surface_volume_ratio)
    print(f'Frame {path} done')
    print(f'Ti all cluster sum: {Ti_all_cluster_sum[-1]}, Ti largest cluster size: {Ti_largest_cluster_size[-1]}, Ti surface-volume ratio: {Ti_surface_volume_ratio[-1]}')
    print(f'Zr all cluster sum: {Zr_all_cluster_sum[-1]}, Zr largest cluster size: {Zr_largest_cluster_size[-1]}, Zr surface-volume ratio: {Zr_surface_volume_ratio[-1]}')


## Plot the order parameter
plt.figure(figsize=(6,4))
plt.plot(checkboard_order, linestyle='-', linewidth=2, color='black', alpha=1.0, label=None)
plt.ylabel(r'$O(\mathbf{s})$')
plt.xlabel('Accepted Steps')
plt.tight_layout()
plt.savefig('order_parameter.png', dpi=300)

## Plot the Sr cluster analysis
plt.figure(figsize=(6,4))
plt.plot(Ti_all_cluster_sum, linestyle='-', linewidth=2, color='gray', alpha=1.0, label=None)
plt.plot(Ti_largest_cluster_size, linestyle='-', linewidth=2, color='#206FB6', alpha=1.0, label=None)
plt.ylabel(r'Number of Ti sites')
plt.xlabel('Accepted Steps')
plt.tight_layout()
plt.savefig('Ti_cluster_analysis.png', dpi=300)

## Plot the Pb cluster analysis
plt.figure(figsize=(6,4))
plt.plot(Zr_all_cluster_sum, linestyle='-', linewidth=2, color='gray', alpha=1.0, label=None)
plt.plot(Zr_largest_cluster_size, linestyle='-', linewidth=2, color='#206FB6', alpha=1.0, label=None)
plt.ylabel(r'Number of Zr sites')
plt.xlabel('Accepted Steps')
plt.tight_layout()
plt.savefig('Zr_cluster_analysis.png', dpi=300)

## Plot the surface-volume ratio
plt.figure(figsize=(6,4))
plt.plot(Ti_surface_volume_ratio, linestyle='-', linewidth=2, color='gray', alpha=1.0, label='Ti')
plt.plot(Zr_surface_volume_ratio, linestyle='-', linewidth=2, color='#206FB6', alpha=1.0, label='Zr')
plt.legend()
plt.ylabel(r'Surface-volume ratio')
plt.xlabel('Accepted Steps')
plt.tight_layout()
plt.savefig('surface_volume_ratio.png', dpi=200)

