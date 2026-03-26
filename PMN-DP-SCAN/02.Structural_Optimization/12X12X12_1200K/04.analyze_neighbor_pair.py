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
ncells = 12 * 12 * 12
simulation.ncell = ncells


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
elements = simulation.B_site_elements
site_indices = simulation.B_neighborlist['indices']
site_neighborlist = simulation.B_neighborlist['neighborlist']

Nb_Nb_pair = []
Nb_Mg_pair = []
Mg_Mg_pair = []
for path in files:
    atoms = read(path, format='lammps-data', atom_style='atomic')
    update_element(atoms, ['Mg','Nb','O','Pb'])
    syms = atoms.get_chemical_symbols()
    n_NbNb = 0
    n_NbMg = 0
    n_MgMg = 0
    for site_idx, nb_indices in zip(site_indices, site_neighborlist):
        sym_center = syms[site_idx]
        if len(nb_indices) < 6:
            raise ValueError('Each site should have equal to or more than 6 nearest neighbors. Here it has {}'.format(len(nb_indices)))
        else:
            nnb_indices = nb_indices[:6]  ## note that the indices are already ordered by distance from low to high
        for nb_idx in nnb_indices:
            sym_nb = syms[nb_idx]
            if sym_center == 'Nb' and sym_nb == 'Nb':
                n_NbNb += 1
            elif sym_center == 'Nb' and sym_nb == 'Mg':
                n_NbMg += 1
            elif sym_center == 'Mg' and sym_nb == 'Nb':
                n_NbMg += 1
            elif sym_center == 'Mg' and sym_nb == 'Mg':
                n_MgMg += 1
    print(f'Frame {path} done')
    print(f'Nb-Nb: {n_NbNb // 2}, Nb-Mg: {n_NbMg // 2}, Mg-Mg: {n_MgMg // 2}')
    assert n_NbNb % 2 == 0, 'n_NbNb is not even'
    assert n_NbMg % 2 == 0, 'n_NbMg is not even'
    assert n_MgMg % 2 == 0, 'n_MgMg is not even'
    assert n_NbNb + n_NbMg + n_MgMg == ncells * 6, 'The number of pairs is not correct'
    Nb_Nb_pair.append(n_NbNb // 2)
    Nb_Mg_pair.append(n_NbMg // 2)
    Mg_Mg_pair.append(n_MgMg // 2)
 

 
## Plot the surface-volume ratio
plt.figure(figsize=(6,4))
plt.plot(np.array(Nb_Nb_pair) / (ncells*3), linestyle='-', linewidth=2, color='gray', alpha=1.0, label='Nb-Nb')
plt.plot(np.array(Nb_Mg_pair) / (ncells*3), linestyle='-', linewidth=2, color='blue', alpha=1.0, label='Nb-Mg')
plt.plot(np.array(Mg_Mg_pair) / (ncells*3), linestyle='-', linewidth=2, color='green', alpha=1.0, label='Mg-Mg')
plt.legend()
plt.ylabel(r'Proportion of pairs')
plt.xlabel('Accepted Steps')
plt.tight_layout()
plt.savefig('neighbor_pair.png', dpi=200)

