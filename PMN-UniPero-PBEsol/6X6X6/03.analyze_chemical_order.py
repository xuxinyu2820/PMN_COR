import os, glob
import numpy as np
from ase.io import read
import matplotlib.pyplot as plt
from perostruc import update_element, FIRESwapOptimizer

plt.rcParams.update({
    'axes.linewidth': 2,     
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

mc_log = np.loadtxt('logs/mc.log', skiprows=2, delimiter=',')
iters = mc_log[:, 0]  ## the iteration index of the accepted trial
accept = mc_log[:, 3]  ## the acceptance of the accepted trial
iters = iters[accept > 0.5]

simulation = FIRESwapOptimizer(
    temperature = 600.0,
    A_site_elements = ['Pb'],
    B_site_elements = ['Mg', 'Nb'],
    element_specorder = ['Mg','Nb','O','Pb'],
    neighbor_cutoff = 6
)
simulation.initialize(supercell_size = [6, 6, 6], calculator = 1, restart = True)


# path to the directory containing the trajectory files 
traj_dir = 'trajs'
lmp_pattern = os.path.join(traj_dir, 'f*.lmp')
files = sorted(glob.glob(lmp_pattern),
               key=lambda x: int(os.path.splitext(os.path.basename(x))[0][1:]))
global_params = []
for path in files:
    pmn = read(path, format='lammps-data', atom_style='atomic')
    update_element(pmn, ['Mg','Nb','O','Pb'])
    order_parameter = simulation.get_pair_product(pmn, score_map = {'Nb': +1, 'Mg': -1}, type='B')
    global_params.append(order_parameter)

## Plot the order parameter
plt.figure(figsize=(6,4))
plt.plot(iters, global_params, linestyle='-', linewidth=2, color='black', alpha=1.0, label=None)
plt.ylabel(r'$O(\mathbf{s})$')
plt.xlabel('Iterations')
plt.tight_layout()
plt.savefig('order_parameter.png', dpi=300)

## 
save_data = np.array([iters, global_params])
np.save('order_parameter.npy', save_data)
