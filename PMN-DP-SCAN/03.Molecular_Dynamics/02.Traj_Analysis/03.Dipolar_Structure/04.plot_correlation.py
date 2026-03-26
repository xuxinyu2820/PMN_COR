import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['axes.linewidth'] = 3      
mpl.rcParams['axes.titlesize'] = 30     
mpl.rcParams['axes.labelsize'] = 30     
mpl.rcParams['xtick.labelsize'] = 30    
mpl.rcParams['ytick.labelsize'] = 30    
filename = 'summary_correlations.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

temp = data[:, 0]
nb = data[:, 1]
mg = data[:, 2]

plt.figure(figsize=(8, 5))
plt.plot(temp, nb, label='Nb-Nb', color='#206FB6', marker='o', markersize=8, linewidth=4)
plt.plot(temp, mg, label='Nb-Mg', color='#C00000', marker='o', markersize=8, linewidth=4)
plt.xlabel('T (K)', fontsize=30)
plt.ylabel('$C_{ij}$', fontsize=30)
plt.legend(frameon=False, fontsize=24)
plt.tight_layout()
# plt.grid()

#save figure
plt.savefig('correlation_plot.png', dpi=300)
