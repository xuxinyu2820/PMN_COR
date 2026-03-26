'''
Test this on 80GB GPU. start an interactive session on Perlmutter:
    salloc -N1 -n32 -t 04:00:00 -C "gpu" -q shared_interactive --gres=gpu:1 -A m5025
    salloc -N1 -n32 -t 04:00:00 -C "gpu&hbm80g" -q shared_interactive --gres=gpu:1 -A m5025
Then activate the Conda environment:
    conda activate /global/cfs/cdirs/m5025/pinchenx/conda_envs/dp2211
'''
import ase
from utility import *
from deepmd.calculator import DP
from ase.md.langevin import Langevin
from ase.md.logger import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
import numpy as np
from ase.io import Trajectory, write
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=int, default=300)
parser.add_argument('--steps', type=int, default=10000)
args = parser.parse_args()
temperature = args.temperature
steps = args.steps
if os.path.exists(f'{temperature}K'):
    pass
else:
    os.makedirs(f'{temperature}K')
traj_name = f'{temperature}K/pmn_{temperature}K_{steps}steps.traj'
log_name = f'{temperature}K/pmn_{temperature}K_{steps}steps.log'


# Load initial config
pmn = ase.io.read('/global/cfs/projectdirs/m5025/Ferroic/PMN_paper_repo/02.Structural_Optimization/03.12X12X12/600K/trajs/f4003.lmp', format='lammps-data', atom_style='atomic')
update_element(pmn,['Mg', 'Nb','O','Pb'])
natoms = len(pmn)

# Calculator
dpmodel = DP(model="/global/cfs/projectdirs/m5025/Ferroic/PMN_paper_repo/01.DP_Model_Training/Production_Model/model-compress.pb")
pmn.calc = dpmodel

# Assign initial velocities corresponding to temperature T
MaxwellBoltzmannDistribution(pmn, temperature_K=temperature)

# Set up NVT dynamics
dt = 2.0 * units.fs
dyn = Langevin(pmn, timestep=dt, temperature_K=temperature, friction= 1/ (100 * dt)) ## typically we take relaxation time for Langevin dynamics to be 100 times the timestep

# setup the directory to save the data
os.makedirs(f'{temperature}K', exist_ok=True)

traj = Trajectory(traj_name, 'w', pmn)
dyn.attach(traj.write, interval=500)
logger = MDLogger(dyn, pmn, log_name, header=True, stress=True, peratom=False)
dyn.attach(logger, interval=500)

def print_status(a=pmn):
    epot = a.get_potential_energy() / natoms
    ekin = a.get_kinetic_energy() / natoms
    print(f'[{dyn.get_number_of_steps():06d}] Epot = {epot:.3f}  Ekin = {ekin:.3f}  Etot = {epot+ekin:.3f}')

dyn.attach(print_status, interval=500) ## change interval

print('Starting NVT simulation...')
dyn.run(steps) 


