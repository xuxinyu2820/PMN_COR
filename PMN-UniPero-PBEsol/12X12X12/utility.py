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


def get_effective_lattice(atoms:ase.Atoms, supercell:list, central_element:list):
    '''
    Parameters:
    atoms: ase.Atoms
        the perovskite configuration
    supercell: list[int]
        the size of the supercell, e.g. [12, 12, 12]
    central_element: str 
        the chemical symbols of the central atom the lattice associated to, e.g. ['Mg', 'Nb']
    Returns:
    lattice_idx: np.ndarray[int] with shape specified by 'supercell'
    '''
    ## sanity check
    natoms = atoms.get_global_number_of_atoms()
    ncells = natoms // 5
    assert ncells * 5 == natoms, 'We are dealing with perovskites, the number of atoms should be divisible by 5'
    if len(supercell) == 3:
        assert supercell[0] > 1, 'supercell should be 3D'
        assert supercell[1] > 1, 'supercell should be 3D'
        assert supercell[2] > 1, 'supercell should be 3D'
        ncells = supercell[0]*supercell[1]*supercell[2]
    else:
        raise ValueError('supercell should be a list of three integers.')
    assert len(central_element) > 0, 'central_element should be a list of chemical symbols'

    ## get the central atoms
    catoms_filter = np.array(atoms.get_chemical_symbols()) == central_element[0]
    if len(central_element) > 1:
        for element in central_element[1:]:
            catoms_filter = catoms_filter | (np.array(atoms.get_chemical_symbols()) == element)
    assert catoms_filter.sum() * 5 == natoms, 'number of central atoms should be equal to number of primitive perovskite cells'
    catoms_idx = np.arange(natoms)[catoms_filter]
    catoms_scaled_pos = atoms.get_scaled_positions()[catoms_filter]
    if catoms_scaled_pos.min() < 0 or catoms_scaled_pos.max() > 1:
        raise ValueError('the scaled positions of the central atoms are not in the unit cell')
    origin_idx = catoms_idx[np.argmin(catoms_scaled_pos.sum(-1))]
    origin_scaled_pos = atoms.get_scaled_positions()[origin_idx]
    catoms_rel_pos = catoms_scaled_pos - origin_scaled_pos
    _catoms_sites = catoms_rel_pos * np.array(supercell)
    catoms_sites = np.round(_catoms_sites,0).astype(int)
    if np.abs(_catoms_sites - catoms_sites).max() > 0.3:
        raise ValueError('incapable to match the lattice, maybe due to the presence of defects')
    lattice_idx = np.zeros(supercell,dtype=int) - 1 
    lattice_idx[catoms_sites[:,0],catoms_sites[:,1],catoms_sites[:,2]] = catoms_idx
    # for idx, site in zip(catoms_idx, catoms_sites):
    #     lattice_idx[site[0], site[1], site[2]] = idx
    if lattice_idx.min() == -1:
        raise ValueError('some lattice sites missing')

    return lattice_idx




def get_neighbor_indices(i, j, k, l1, l2, l3):
    return [( (i+1) % l1, j, k), ( (i-1) % l1, j, k), (i, (j+1) % l2, k), (i, (j-1) % l2, k), (i, j, (k+1) % l3), (i, j, (k-1) % l3)]

def find_neighbor_values(lattice_values, i, j, k):
    l1, l2, l3 = lattice_values.shape
    neighbors = get_neighbor_indices(i, j, k, l1, l2, l3)
    neighbor_values = []
    for ni, nj, nk in neighbors:
        neighbor_values.append(lattice_values[ni, nj, nk])
    return neighbor_values



def BFS_cluster_analysis(cluster_flag):
    l1, l2, l3 = cluster_flag.shape
    visited = np.zeros_like(cluster_flag, dtype=bool)
    all_clusters = []

    for i in range(l1):
        for j in range(l2):
            for k in range(l3):
                if cluster_flag[i][j][k] == 1 and visited[i][j][k] == False:
                    # New cluster found, start BFS
                    current_cluster = []
                    queue = [(i, j, k)]
                    visited[i][j][k] = True

                    while len(queue) > 0:
                        ci, cj, ck = queue.pop(0)
                        current_cluster.append((ci, cj, ck))

                        # Define neighbors again for traversal
                        neighbors = get_neighbor_indices(ci, cj, ck, l1, l2, l3)

                        for ni, nj, nk in neighbors:
                            if cluster_flag[ni][nj][nk] == 1 and visited[ni][nj][nk] == False:
                                visited[ni][nj][nk] = True
                                queue.append((ni, nj, nk))
                    
                    all_clusters.append(current_cluster)

    return all_clusters


def calculate_cluster_surface_size(cluster_flag):
    '''
    Calculate the surface size of a cluster
    Parameters:
    cluster_flag: np.ndarray[int] with shape (l1, l2, l3). 1 for in the cluster, 0 for out of the cluster.  
        the flag of the cluster
    Returns:
    surface_size: int
        the surface size of the cluster
    '''
    surface_size = 0
    for direction in [0,1,2]:
        flag_shifted = np.roll(cluster_flag, shift=1, axis=direction)
        surface_size += np.sum((cluster_flag != flag_shifted).astype(int))
    return surface_size

