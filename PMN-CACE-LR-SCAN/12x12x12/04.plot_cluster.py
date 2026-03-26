import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    'axes.linewidth': 2,     
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

cluster1 = '/global/homes/x/xinyuxu/m5025/Ferroic/PMN_revision/PMN-DP-SCAN/02.Structural_Optimization/12X12X12/cluster.csv'
energy_file = '/global/homes/x/xinyuxu/m5025/Ferroic/PMN_revision/PMN-DP-SCAN/02.Structural_Optimization/12X12X12/logs/energy_accepted.csv'

cluster2 = '/global/homes/x/xinyuxu/m5025/Ferroic/PMN_revision/PMN-CACE-LR-SCAN/12x12x12/cluster.csv'

data1 = np.loadtxt(cluster1, delimiter=',', skiprows=1)
energy_data = np.loadtxt(energy_file, delimiter=',', skiprows=2)

x1 = energy_data[:, 0]

all1 = data1[:, 1]
largest1 = data1[:, 2]
if len(x1) != len(all1):
    raise ValueError("DP-SCAN: energy_accepted.csv not as the same length as cluster.csv. Please check the files.")


data2 = np.loadtxt(cluster2, delimiter=',', skiprows=1)

x2 = data2[:, 0] * 100

all2 = data2[:, 1]
largest2 = data2[:, 2]


# plot
plt.figure(figsize=(6,4))

# all
plt.plot(x1, all1, color='gray', linewidth=2)
plt.plot(x2, all2, color='lightgray', linestyle='--', linewidth=2)

# largest
plt.plot(x1, largest1, color='#206FB6', linewidth=2)
plt.plot(x2, largest2, color='#5DADE2', linestyle='--', linewidth=2)

plt.xlabel('Iterations')
plt.ylabel('Number of Nb sites')
plt.tight_layout()
plt.savefig('compare_cluster.png', dpi=300)
plt.show()