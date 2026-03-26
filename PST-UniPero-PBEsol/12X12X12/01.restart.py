import numpy as np
import ase
import ase.io
from perostruc import FIRESwapOptimizer, update_element
from deepmd.calculator import DP

element_specorder = ['Pb','Sr','O','Ti']
dpmodel = DP(model="../unipero.pb")

simulation = FIRESwapOptimizer(
    temperature = 300.0,
    A_site_elements = ['Pb', 'Sr'],
    B_site_elements = ['Ti'],
    element_specorder = element_specorder,
    neighbor_cutoff = 6.0,
    fmax = 0.03,
    fire_max_steps = 100,
    nepoch = 300000,
)

## restart from the last epoch
simulation.initialize(supercell_size = [12, 12, 12], calculator = dpmodel, restart = True)
simulation.optimize(swap_type = 'A', flexible_cell = True, profile = True)