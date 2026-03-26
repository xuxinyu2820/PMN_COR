import numpy as np
import ase
import ase.io
from perostruc import FIRESwapOptimizer, update_element
from deepmd.calculator import DP

init_config = ase.io.read('disordered_L6X6X6.lmp', format='lammps-data', atom_style='atomic')
element_specorder = ['Zr','Ti','O','Pb']
update_element(init_config,element_specorder)
dpmodel = DP(model="../unipero.pb")
init_config.calc = dpmodel

simulation = FIRESwapOptimizer(
    temperature = 300.0,
    A_site_elements = ['Pb'],
    B_site_elements = ['Zr', 'Ti'],
    element_specorder = element_specorder,
    neighbor_cutoff = 6.0,
    fmax = 0.02,
    fire_max_steps = 100,
    nepoch = 10000,
)

## initialize from scratch
simulation.initialize(supercell_size = [6, 6, 6], init_config = init_config)
simulation.run_FIRE(fmax = 0.02, steps = 1000, flexible_cell = True)
simulation.optimize(swap_type = 'B', flexible_cell = True, profile = True)