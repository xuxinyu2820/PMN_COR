import numpy as np
import ase
import ase.io
from perostruc import FIRESwapOptimizer, update_element
from deepmd.calculator import DP

element_specorder = ['Mg','Nb','O','Pb']
dpmodel = DP(model="../unipero.pb")

simulation = FIRESwapOptimizer(
    temperature = 600.0,
    A_site_elements = ['Pb'],
    B_site_elements = ['Mg', 'Nb'],
    element_specorder = element_specorder,
    neighbor_cutoff = 6,
    fmax = 0.03,
    fire_max_steps = 100,
    nepoch = 300000,
)

## restart from the last epoch
simulation.initialize(supercell_size = [6, 6, 6], calculator = dpmodel, restart = True)
simulation.optimize(swap_type = 'B', flexible_cell = False, profile = True)