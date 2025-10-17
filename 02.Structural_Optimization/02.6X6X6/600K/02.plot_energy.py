import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt('logs/energy_accepted.csv', skiprows=1, delimiter=',')

fig, ax = plt.subplots(figsize=(10, 6))
iters = data[:, 0]  ## the iteration index of the accepted trial
energy = data[:, 1]  ## the energy of the accepted trial
ax.plot(iters, energy)
ax.set_xlabel('Iteration')
ax.set_ylabel('Energy [meV/atom]')
plt.savefig('energy_vs_iter.png')