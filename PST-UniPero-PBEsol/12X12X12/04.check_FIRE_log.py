import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
import os
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['font.size'] = 12


with open("logs/fire.log", "r") as f:
    lines = f.readlines()

header = "      Step     Time          Energy          fmax"
indices = [i for i, line in enumerate(lines) if line.strip() == header.strip()]

energy_before_FIRE_list = []
energy_after_FIRE_list = []
for idx in indices:
    # Get the last line above the header (if exists)
    if idx > 0:
        last_above = lines[idx - 1].rstrip('\n').split()
        last_above = float(last_above[-2])
    else:
        last_above = None
    # Get the first line below the header (if exists)
    if idx + 1 < len(lines):
        first_below = lines[idx + 1].rstrip('\n').split()
        first_below = float(first_below[-2])
    else:
        first_below = None
    if first_below is not None:
        energy_before_FIRE_list.append(first_below)
    if last_above is not None:
        energy_after_FIRE_list.append(last_above)

## Align
energy_before_FIRE_list = energy_before_FIRE_list[:-1]
ntrials = len(energy_before_FIRE_list)
assert ntrials == len(energy_after_FIRE_list)
print(f"Number of trials: {ntrials}")

## Remove the first trial
energy_before_FIRE_list = np.array(energy_before_FIRE_list[1:])
energy_after_FIRE_list = np.array(energy_after_FIRE_list[1:])

## Analyze dE distribution
FIRE_energy_change = energy_after_FIRE_list - energy_before_FIRE_list
Total_energy_change = energy_after_FIRE_list[1:] - energy_after_FIRE_list[:-1]
fig, ax = plt.subplots(figsize=(4, 3))
ax.hist(FIRE_energy_change, bins=50, range=(-8, 2), density=True, alpha=1, label=r'$\Delta E_{\text{FIRE}}$')
ax.hist(Total_energy_change, bins=50, range=(-8, 2), density=True, alpha=1, label=r'$\Delta E_{\text{Total}}$')
ax.set_xlim(-8, 2)
ax.set_xlabel('Energy Difference (eV)')
ax.set_ylabel('Probability Density')
ax.legend(frameon=False, handlelength=0.7, fontsize=11, loc='upper center') ## make the dash smaller
plt.tight_layout()
plt.savefig('dE_distribution.png', dpi=300)