import numpy as np
from ase.io import read
import matplotlib.pyplot as plt

plt.rcParams.update({
    'axes.linewidth': 2,     
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

traj_path = "./fireswap_anneal.xyz" 
out_npy   = "order_parameter.npy"

a = 4.05
dims = np.array([6, 6, 6], dtype=int)
score = {"Nb": 1, "Mg": -1}
nbrs  = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

def order_param(atoms):
    # atoms.set_cell(dims * a)
    cell = atoms.get_cell()
    cell = cell.diagonal() / 6.0
    atoms.set_pbc(True)

    syms = np.array(atoms.get_chemical_symbols())
    pos  = atoms.get_positions(wrap=True)

    m = {}
    for idx, (s, p) in enumerate(zip(syms, pos)):
        if s in score:
            i, j, k = (np.floor(p / cell).astype(int) % dims)
            m[(int(i), int(j), int(k))] = idx

    vals = []
    for (i, j, k), ic in m.items():
        si = score[syms[ic]]
        nn = []
        for di, dj, dk in nbrs:
            key = ((i + di) % dims[0], (j + dj) % dims[1], (k + dk) % dims[2])
            if key in m:
                nn.append(score[syms[m[key]]])
        if len(nn) == 6:
            vals.append(si * (sum(nn) / 6.0))

    return float(np.mean(vals)) if vals else np.nan

traj = read(traj_path, index=":")
n = len(traj)

iters = (np.arange(n) * 100).astype(int)  
orders = np.array([order_param(traj[i]) for i in range(n)], dtype=float)
print(orders)
## Plot the order parameter
plt.figure(figsize=(6,4))
plt.plot(iters, orders, linestyle='-', linewidth=2, color='black', alpha=1.0, label=None)
plt.ylabel(r'$O(\mathbf{s})$')
plt.xlabel('Iterations')
plt.tight_layout()
plt.savefig('order_parameter.png', dpi=300)

arr = np.array([iters, orders])   
np.save(out_npy, arr)

print("saved:", out_npy, "shape =", arr.shape)