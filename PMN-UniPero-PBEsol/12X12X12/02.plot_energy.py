import numpy as np
import os
import matplotlib.pyplot as plt


# data = np.loadtxt('logs/energy_accepted.csv', skiprows=1, delimiter=',')
mc_log_file = os.path.join('logs', 'mc.log')
mc_log = np.loadtxt(mc_log_file, skiprows=2, delimiter=',')
iters = mc_log[:, 0]  ## the iteration index of the accepted trial
energy = mc_log[:, 2]  ## the energy of the accepted trial


fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(iters, energy)
ax.set_xlabel('Iteration')
ax.set_ylabel('Energy [meV/atom]')
plt.savefig('energy_vs_iter.png')