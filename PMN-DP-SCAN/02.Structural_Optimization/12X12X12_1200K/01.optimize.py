import numpy as np
import ase
import ase.io
from perostruc import FIRESwapOptimizer, update_element
from deepmd.calculator import DP

init_config = ase.io.read('sublattice_L12X12X12.lmp', format='lammps-data', atom_style='atomic')
element_specorder = ['Mg','Nb','O','Pb']
update_element(init_config,element_specorder)
dpmodel = DP(model="/global/cfs/projectdirs/m5025/Ferroic/PMN_paper_repo/01.DP_Model_Training/Production_Model/model-compress.pb")
init_config.calc = dpmodel

simulation = FIRESwapOptimizer(
    temperature = 1200.0,
    A_site_elements = ['Pb'],
    B_site_elements = ['Mg', 'Nb'],
    element_specorder = element_specorder,
    neighbor_cutoff = 6,
    fmax = 0.02,
    fire_max_steps = 100,
    nepoch = 30000,
)

## initialize from scratch
simulation.initialize(supercell_size = [12, 12, 12], init_config = init_config)
simulation.run_FIRE(fmax = 0.02, steps = 1000)
simulation.optimize(swap_type = 'B', flexible_cell = False, profile = True)