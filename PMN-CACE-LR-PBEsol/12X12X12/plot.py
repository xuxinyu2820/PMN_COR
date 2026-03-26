import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'axes.linewidth': 2,     
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

data = np.loadtxt('cluster.dat', skiprows=1)
iters = data[:, 0]
largest_cluster = data[:, 1]
all_clusters = data[:, 2]

## Plot the order parameter
plt.figure(figsize=(6,4))
plt.plot(iters, largest_cluster, linestyle='-', linewidth=2, color='#206FB6', alpha=1.0, label=None)
plt.plot(iters, all_clusters, linestyle='-', linewidth=2, color='gray', alpha=1.0, label=None)
plt.ylabel(r'Number of Nb sites')
plt.xlabel('Iterations')
plt.tight_layout()
plt.savefig('cluster.png', dpi=300)
