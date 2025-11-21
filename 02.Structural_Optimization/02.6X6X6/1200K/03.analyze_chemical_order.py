import os, glob
import numpy as np
from ase.io import read
import matplotlib.pyplot as plt
from utility import update_element

plt.rcParams.update({
    'axes.linewidth': 2,     
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

iters = np.loadtxt('logs/energy_accepted.csv', skiprows=1, delimiter=',')[1:, 0]   ## the iteration index of the accepted trial

# path to the directory containing the trajectory files 
traj_dir = 'trajs'
a        = 4.05
dims     = np.array([6, 6, 6])
score_map = {'Nb': +1, 'Mg': -1}
neighbor_offsets = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]


lmp_pattern = os.path.join(traj_dir, 'f*.lmp')
files = sorted(glob.glob(lmp_pattern),
               key=lambda x: int(os.path.splitext(os.path.basename(x))[0][1:]))

global_params = []

for path in files:
    pmn = read(path, format='lammps-data', style='atomic')
    update_element(pmn, ['Mg','Nb','O','Pb'])
    symbols   = np.array(pmn.get_chemical_symbols())
    positions = pmn.get_positions(wrap=True)

    mapping = {}
    for idx, (s, pos) in enumerate(zip(symbols, positions)):
        if s not in score_map:
            continue
        i, j, k = (np.floor(pos / a).astype(int) % dims)
        mapping[(i, j, k)] = idx

    corrected_scores = []
    for (i, j, k), idx_center in mapping.items():
        center_val = score_map[symbols[idx_center]]
        neigh_vals = []
        for di, dj, dk in neighbor_offsets:
            ni, nj, nk = (i+di) % dims[0], (j+dj) % dims[1], (k+dk) % dims[2]
            nb_key = (ni, nj, nk)
            if nb_key in mapping:
                neigh_vals.append(score_map[symbols[mapping[nb_key]]])
        if len(neigh_vals) >= 6:
            avg_pair = sum(neigh_vals[:6]) / 6.0
            corrected_scores.append(avg_pair * center_val)

    global_param = float(np.mean(corrected_scores)) if corrected_scores else np.nan
    global_params.append(global_param)
    print(f"[Processed] {os.path.basename(path):12s} → global_param = {global_param:.4f}")


## Plot the order parameter
plt.figure(figsize=(6,4))
plt.plot(iters, global_params,
         linestyle='-',
         linewidth=2,
         color='black',
         alpha=1.0,
         label=None)

plt.ylabel(r'$O(\mathbf{s})$')
plt.xlabel('Iterations')

plt.tight_layout()
plt.savefig('order_parameter_L6.png', dpi=300)
plt.close()
