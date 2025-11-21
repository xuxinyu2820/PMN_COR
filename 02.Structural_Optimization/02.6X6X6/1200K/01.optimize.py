import os
from time import time
import numpy as np
import csv
from ase.optimize import FIRE
from ase.io import write
from deepmd.calculator import DP
from utility import *


"""
Simulation Parameters
"""
nepoch = 10000
temperature = 1200  ## temperature in K
kb = 8.617e-5  ## Boltzmann constant in eV/K
kbT = kb * temperature
fmax = 0.02  ## force threshold for the FIRE optimization, previously 0.01, seems to be too strict
fire_max_steps = 200  ## maximum number of steps for the FIRE optimization, previously unlimited
# neighbor_cutoff = 5 ## allow only nearest neighbor swap. we should try this, checking if it causes any issue.
neighbor_cutoff = 6 ## allow nearest and next-nearest neighbor swap.
cell_size = np.array([6, 6, 6])
ncell = np.prod(cell_size)

"""
Preprocessing
"""
## create directories for saving the figures and trajectories
traj_dir = 'trajs'
logging_dir = 'logs'
os.makedirs(traj_dir, exist_ok=True)
os.makedirs(logging_dir, exist_ok=True)

# Load initial config
pmn = ase.io.read('../../01.initial_configs/disordered_L6X6X6.lmp', format='lammps-data', atom_style='atomic')
update_element(pmn,['Mg', 'Nb','O','Pb'])
natoms = len(pmn)
print(pmn)

## get the filter for B-site atoms (Mg or Nb). Although Nb and Mg will be swapped, the filter for B-site won't change.
sym = pmn.get_chemical_symbols()
Mg_filter = np.array(sym) == 'Mg'
Nb_filter = np.array(sym) == 'Nb'
B_filter = Mg_filter | Nb_filter
Bsites_indices = np.arange(natoms)[B_filter]   # indices of all B-site
if B_filter.astype(int).sum() != ncell:
    raise ValueError('There are {} B-sites, but the number of B-sites should be {}'.format(B_filter.astype(int).sum(), ncell))

## for each B-site, we will find the index of its 6 nearest neighbors. Again, after swapping, the neighboring relation won't change. Only the element type of B-site atoms will be changed.
Bsites_neighborlist = []
for bsite_idx in Bsites_indices:
    nb_idx, nb_sym, nb_dist = get_neighbor(pmn, bsite_idx, cutoff=neighbor_cutoff)  ## sorted by distance from low to high.  
    nb_Mg_filter = np.array(nb_sym) == 'Mg'
    nb_Nb_filter = np.array(nb_sym) == 'Nb'
    nb_B_filter = nb_Mg_filter | nb_Nb_filter
    nb_idx = nb_idx[nb_B_filter]
    nb_sym = nb_sym[nb_B_filter]
    nb_dist = nb_dist[nb_B_filter]
    if nb_idx.size <6:
        raise ValueError('Each B-site should have equal to or more than 6 nearest neighbors. Here it has {}'.format(nb_idx.size))
    Bsites_neighborlist.append(nb_idx)

# Describe the interatomic interactions with DP model
dpmodel = DP(model="model-compress.pb")
pmn.calc = dpmodel
print("initial E={}eV/atom".format(pmn.get_potential_energy()/natoms))
 
# Set the optimizer
dyn = FIRE(pmn, logfile='{}/fire.log'.format(logging_dir)  )
dyn.run(fmax=fmax)  ## initial relaxation, do not limit the number of steps
penergy = [ pmn.get_potential_energy() ]

"""
MC-MD simulation
"""
# create list to save the energy
nswap = 0
iters   = [0]
before_energies = []
after_energies = []
for i in range(nepoch):
    ## randomly choose one Mg/Nb atom and get its neighborlist
    choose_bsite_seed = np.random.choice(np.arange(ncell))
    atom1_idx = Bsites_indices[choose_bsite_seed]
    atom2_idx = np.random.choice(Bsites_neighborlist[choose_bsite_seed])
    if sym[atom1_idx] == sym[atom2_idx]:
        continue
    else:
        t0 = time()
        before_energies.append(penergy[-1])
        sym = pmn.get_chemical_symbols()
        sym_new =  sym.copy()
        sym_new[atom1_idx] = sym[atom2_idx]
        sym_new[atom2_idx] = sym[atom1_idx]
        pmn_new = pmn.copy()
        pmn_new.set_chemical_symbols(sym_new)
        pmn_new.calc = dpmodel
        dyn = FIRE(pmn_new, logfile='{}/fire.log'.format(logging_dir) )
        dyn.run(fmax=fmax, steps=fire_max_steps)   ## limit the number of steps for the FIRE optimization
        penergy_new = pmn_new.get_potential_energy()
        after_energies.append(penergy_new)
        energy_diff = penergy_new - penergy[-1]
        if energy_diff < 0 or (np.random.rand() < np.exp(-energy_diff/kbT)):
            pmn = pmn_new
            penergy.append(penergy_new)
            nswap += 1
            iters.append(i)
            ase.io.write('./{}/f{}.lmp'.format(traj_dir, nswap), pmn, format='lammps-data')
            ## logging
            t1 = time()
            print('========== epoch={},  swap-{},  time cost={:.3f}s =========='.format(i, nswap, t1-t0))
            print('attempt to swap {} and {} succeeded, dE={:.3f}eV, '.format(
                sym[atom1_idx], sym[atom2_idx],energy_diff  ))
        else:
            t1 = time()
            print('========== epoch={},  failed swap,  time cost={:.3f}s =========='.format(i, t1-t0))
            print('attempt to swap {} and {} failed, dE={:.3f}eV, '.format(
                sym[atom1_idx], sym[atom2_idx],energy_diff  ))

"""
Postprocessing
"""
iters   = np.array(iters)
penergy = np.array(penergy) * 1000 / natoms # meV/atom
penergy -= penergy[0]

# save all the energy and swap iter
with open('{}/energy_of_all_trials.csv'.format(logging_dir), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['last_energy', 'proposed_energy'])
    for b, a in zip(before_energies, after_energies):
        writer.writerow([b, a])

with open('{}/energy_accepted.csv'.format(logging_dir), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['iter', 'delta_E_meV_per_atom'])
    for it, E in zip(iters, penergy):
        writer.writerow([it, E])