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

sim_folders = ['PMN-DP-SCAN/02.Structural_Optimization/6X6X6_600K', 'PMN-UniPero-PBEsol/6X6X6', 'PMN-CACE-LR-PBEsol/6X6X6', 'PMN-CACE-LR-SCAN/6X6X6', 'PZT-UniPero-PBEsol/6X6X6', 'PST-UniPero-PBEsol/6X6X6',]
labels = ['PMN-DP-SCAN', 'PMN-UniPero-PBEsol', 'PMN-CACE-PBEsol', 'PMN-CACE-SCAN', 'PZT-UniPero-PBEsol', 'PST-UniPero-PBEsol']
colors = ['blue', 'purple', 'tab:purple', 'tab:blue', 'tab:orange', 'tab:green']
fig, ax = plt.subplots(figsize=(4,4))
iters_max = 10000

for folder, label, color in zip(sim_folders, labels, colors):
    data = np.load(os.path.join(folder, 'order_parameter.npy'))
    iters = data[0]
    global_params = data[1]
    ax.step(iters[iters <= iters_max], global_params[iters <= iters_max], linestyle='-', linewidth=1.5, color=color, alpha=1.0, label=label)

ax.axhline(y=-1/3, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax.axhline(y=1/9, color='black', linestyle='--', linewidth=2, alpha=0.5)

ax.set_ylabel(r'$O(\mathbf{s})$')
ax.set_xlabel('Iterations')
# ax.set_ylim(-0.35, 0.4)
# ax.legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig('order_parameter.png', dpi=300)
