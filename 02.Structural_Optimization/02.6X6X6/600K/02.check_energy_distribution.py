import numpy as np
import matplotlib.pyplot as plt


natoms = 36 * 6 * 6 * 5
energy = np.loadtxt('logs/swap_energies.csv', skiprows=1, delimiter=',')


energy_system = energy[:, 0] - energy[0, 0]
energy_trial = energy[:, 1] - energy[0, 0]
energy_diff = energy_trial - energy_system


fig, ax = plt.subplots(1,3,figsize=(12, 3))
# ax.plot(energy_change)
ax[0].hist(energy_diff, bins=100, density=True)
ax[0].set_xlabel('dE [eV]')
ax[0].set_ylabel('Probability density')
ax[0].set_title('Energy change distribution (all)')


ax[1].hist(energy_diff[energy_diff < 0], bins=100, density=True)
ax[1].set_xlabel('dE [eV]')
ax[1].set_ylabel('Probability density')
ax[1].set_title('Energy change distribution (accepted)')

ax[2].plot(energy_system)
ax[2].set_xlabel('Iteration')
ax[2].set_ylabel('Energy [eV]')
ax[2].set_title('Energy vs Iteration')
 
plt.tight_layout()
fig.savefig('energy_change.png')



