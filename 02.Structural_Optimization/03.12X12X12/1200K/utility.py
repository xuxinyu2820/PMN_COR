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


def printenergy(a):
    """Function to print the potential, kinetic and total energy"""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))


def get_neighbor(atoms, idx, cutoff=6):
    d = atoms.get_distances(idx, np.arange(len(atoms)), mic=True, vector=False)
    neighbor_filter = d < cutoff
    neighbor_idx = np.arange(len(atoms))[neighbor_filter]
    neighbor_sym = np.array(atoms.get_chemical_symbols())[neighbor_idx]
    neighbor_dist = d[neighbor_filter]
    neighbor_dist_sort = np.argsort(neighbor_dist)

    ## sort by distance from low to high. Then remove the first, which is itself
    neighbor_idx = neighbor_idx[neighbor_dist_sort][1:]
    neighbor_sym = neighbor_sym[neighbor_dist_sort][1:]
    neighbor_dist = neighbor_dist[neighbor_dist_sort][1:]
    return  neighbor_idx, neighbor_sym, neighbor_dist