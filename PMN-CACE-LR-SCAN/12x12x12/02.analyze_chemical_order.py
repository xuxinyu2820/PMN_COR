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

init_atoms = read('traj.xyz', index=0)

simulation = FIRESwapOptimizer(
    temperature = 450.0,
    A_site_elements = ['Pb'],
    B_site_elements = ['Mg', 'Nb'],
    element_specorder = ['Mg','Nb','O','Pb'],
    neighbor_cutoff = 6.0
)

simulation.initialize(
    supercell_size=[12, 12, 12],
    init_config=init_atoms,
    calculator=1,
    restart=False,
    build_neighborlist=False
)

if os.path.exists('A_neighborlist.pkl'):
    simulation.load_neighborlist()
else:
    simulation._build_swap_site_list(simulation.atoms)
    simulation.save_neighborlist()
lattice_map = simulation.get_lattice_map('B', [12, 12, 12])

# path to the directory containing the trajectory files 
traj_file = 'traj.xyz'
frames = read(traj_file, index=':')

checkboard_order = []
Nb_all_cluster_sum = []
Nb_largest_cluster_size = []
Nb_surface_volume_ratio = []

for i, pmn in enumerate(frames):

    # order parameter
    order_parameter = simulation.get_pair_product(
        pmn,
        score_map={'Mg': -1, 'Nb': +1},
        type='B'
    )
    checkboard_order.append(order_parameter)

    # Nb cluster analysis
    _all_cluster_sum, _largest_cluster_size, _surface_volume_ratio = find_clusters(
        pmn, lattice_map, 'Nb', tolerance=1
    )

    Nb_all_cluster_sum.append(_all_cluster_sum)
    Nb_largest_cluster_size.append(_largest_cluster_size)
    Nb_surface_volume_ratio.append(_surface_volume_ratio)

    print(f'Frame {i} done')
    print(f'Nb all cluster sum: {_all_cluster_sum}, '
          f'Nb largest cluster size: {_largest_cluster_size}, '
          f'Nb surface-volume ratio: {_surface_volume_ratio}')


## Plot the order parameter
plt.figure(figsize=(6,4))
plt.plot(checkboard_order, linestyle='-', linewidth=2, color='black', alpha=1.0, label=None)
plt.ylabel(r'$O(\mathbf{s})$')
plt.xlabel('Accepted Steps')
plt.tight_layout()
plt.savefig('order_parameter.png', dpi=300)

## Plot the Sr cluster analysis
plt.figure(figsize=(6,4))
plt.plot(Nb_all_cluster_sum, linestyle='-', linewidth=2, color='gray', alpha=1.0, label=None)
plt.plot(Nb_largest_cluster_size, linestyle='-', linewidth=2, color='#206FB6', alpha=1.0, label=None)
plt.ylabel(r'Number of Nb sites')
plt.xlabel('Accepted Steps')
plt.tight_layout()
plt.savefig('Nb_cluster_analysis.png', dpi=300)

## Plot the surface-volume ratio
plt.figure(figsize=(6,4))
plt.plot(Nb_surface_volume_ratio, linestyle='-', linewidth=2, color='gray', alpha=1.0, label='Nb')
plt.legend()
plt.ylabel(r'Surface-volume ratio')
plt.xlabel('Accepted Steps')
plt.tight_layout()
plt.savefig('surface_volume_ratio.png', dpi=200)

steps = np.arange(len(Nb_largest_cluster_size))

data = np.column_stack([
    steps,
    Nb_all_cluster_sum,
    Nb_largest_cluster_size,
    Nb_surface_volume_ratio
])

np.savetxt(
    'cluster.csv',
    data,
    delimiter=',',
    header='accepted_step,all_cluster_sum,largest_cluster_size,surface_volume_ratio',
    comments=''
)

