import ase
import ase.io
from ase import units
import numpy as np


def update_element(atoms, element_list):
    # Lammps configuration does not specify chmical elements. Update the chemical element manually 
    # mass            1 24.305 -> Mg
    # mass            2 92.906 -> Nb
    # mass            3 15.999 -> O
    # mass            4 207.2  -> Pb
    sym = atoms.get_atomic_numbers()
    sym_new = []
    for s in sym:
        sym_new.append(element_list[s-1])
    atoms.set_chemical_symbols(sym_new)
    return

 