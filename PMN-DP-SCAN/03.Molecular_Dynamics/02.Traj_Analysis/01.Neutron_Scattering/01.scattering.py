import numpy as np
import torch as th
import matplotlib.pyplot as plt
from ase.io import Trajectory
import os
import matplotlib as mpl
# --- Coherent Neutron Scattering Lengths (in femtometers, fm) ---
# A dictionary containing values for some common elements.
# Source: NIST, https://www.ncnr.nist.gov/resources/n-lengths/

mpl.rcParams['axes.linewidth'] = 3   
mpl.rcParams['axes.titlesize'] = 24     
mpl.rcParams['axes.labelsize'] = 20    
mpl.rcParams['xtick.labelsize'] = 20   
mpl.rcParams['ytick.labelsize'] = 20    

SCATTERING_LENGTHS = {
    'O': 5.803,
    'Pb': 9.405,
    'Mg': -3.73,
    'Nb': 7.054,
}

device = 'cuda' if th.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cpu':
    print('Using CPU will be extremely slow for large systems')

def calculate_scattering_function(positions, atom_types, box_dims, q_range):
    """
    Calculates the total coherent neutron scattering function I(Q) for a set of atoms.

    This function uses the Debye scattering equation and accounts for periodic
    boundary conditions using the minimum image convention.

    Args:
        positions (np.ndarray): A NumPy array of atomic positions (shape: [N, 3]),
                                where N is the number of atoms. Units are Angstroms.
        atom_types (list[str]): A list of strings specifying the type of each atom
                                (e.g., ['O', 'H', 'H']). Must match keys in
                                SCATTERING_LENGTHS.
        box_dims (np.ndarray): A NumPy array of the periodic box dimensions
                               [Lx, Ly, Lz] in Angstroms.
        q_range (np.ndarray): A NumPy array of Q values (in Angstrom^-1) for which
                              to calculate the scattering function.

    Returns:
        np.ndarray: The calculated total scattering function I(Q) in fm^2.
    """
    num_atoms = len(atom_types)

    # Get the coherent scattering length for each atom in the system
    try:
        b_values = np.array([SCATTERING_LENGTHS[atom] for atom in atom_types])
    except KeyError as e:
        print(f"Error: Atom type '{e.args[0]}' not found in SCATTERING_LENGTHS dictionary.")
        raise

    # --- Calculate all pairwise distances using Minimum Image Convention ---
    # This is the most computationally intensive part.
    # We create a matrix of all pair separations and apply MIC.
    
    # Create an array of indices for efficient pair selection
    indices = np.arange(num_atoms)
    # Get indices for the upper triangle of the distance matrix to avoid double counting
    i_indices, j_indices = np.triu_indices(num_atoms, k=1)

    # Calculate vector separations
    deltas = positions[i_indices] - positions[j_indices]
    
    # Apply minimum image convention
    # This wraps the distances to find the shortest path in a periodic box
    deltas -= box_dims * np.rint(deltas / box_dims)

    ## All to torch
    pair_distances = th.norm(th.tensor(deltas, dtype=th.float64, device=device), dim=1)
    b_values = th.tensor(b_values, dtype=th.float64, device=device)
    q_range = th.tensor(q_range, dtype=th.float64, device=device)

    # Calculate scalar distances
    # pair_distances = np.linalg.norm(deltas, axis=1)

    # Get the product of scattering lengths for each pair
    b_products = b_values[i_indices] * b_values[j_indices]

    # --- Calculate the scattering function I(Q) using the Debye equation ---
    # The equation is: I(Q) = sum_i(b_i^2) + 2 * sum_{j>i}(b_i * b_j * sin(Q*r_ij)/(Q*r_ij))
    # S(Q) = I(Q) / sum_i(b_i^2) -1
    
    # The "self" part of the scattering (from each atom's interference with itself)
    self_term = (b_values**2).sum()

    total_i_q_list = []
    for q_value in q_range:
        qr = q_value * pair_distances
        sinc_qr = th.sin(qr) / (qr)
        weighted_sinc = sinc_qr * b_products
        distinct_term = 2 * weighted_sinc.sum()
        total_i_q = distinct_term / self_term
        total_i_q_list.append(total_i_q)
    total_i_q_list = th.stack(total_i_q_list).cpu().numpy()
    return total_i_q_list

if __name__ == '__main__':
    temp = 300
    ## Q-grid
    dq = 0.02
    cutoff_q = int(1.8 / dq)
    q_vector = np.arange(1000) * dq + dq       # Q range in Angstrom^-1
    Sq_vector = np.zeros_like(q_vector)
    if os.path.exists('pmn_{}K_scattering_function.npy'.format(temp)):
        data = np.load('pmn_{}K_scattering_function.npy'.format(temp))
        q_vector = data[0]
        scattering_function_avg = data[1]
    else:
        ## Trajectory
        traj_path = "../../01.MD_Simulation/NVT/{:d}K/pmn_{:.1f}K_5000000steps.traj".format(temp, temp)
        traj = Trajectory(traj_path)
        nframes = len(traj)
        scattering_function_list = []
        atom_types = traj[0].get_chemical_symbols()
        simulation_box = traj[0].get_cell().lengths()

        for frame_idx, atoms in enumerate(traj[100::2000]):
            print("Processing frame {} of {}".format(frame_idx, nframes))
            positions = atoms.get_positions()
            scattering_function = calculate_scattering_function(
                positions, 
                atom_types, 
                simulation_box, 
                q_vector
            )
            scattering_function_list.append(scattering_function)
        scattering_function_avg = np.array(scattering_function_list).mean(axis=0)
        scattering_function_avg[:cutoff_q] = scattering_function_avg[:cutoff_q] * 0 + scattering_function_avg[cutoff_q]
        data = np.stack([q_vector, scattering_function_avg], axis=0)
        np.save('pmn_{}K_scattering_function.npy'.format(temp), data)

    # Calculate Radial Distribution Function
    r_vector = np.linspace(1.6, 7, 400)
    rdf_vector = np.zeros_like(r_vector)
    for idx, r in enumerate(r_vector):
        rdf = scattering_function_avg * q_vector * np.sin(q_vector * r)
        rdf = rdf.sum() * dq / 2 / np.pi**2 / r
        rdf_vector[idx] = rdf
    rdf_vector = rdf_vector * 100


    # --- Plotting the results ---
    experimental_data = np.loadtxt('SM.csv', delimiter=',')

    # match the variance
    scale_factor = np.std(rdf_vector) / np.std(experimental_data[:,1])
    experimental_data[:,1] = experimental_data[:,1] * scale_factor

    # This plots the figure used in the paper
    plt.figure(figsize=(8, 2.5))
    plt.plot(r_vector, rdf_vector, label='Simulation'.format(temp), color='purple', linewidth=4)
    plt.plot(experimental_data[:,0], experimental_data[:,1], label='Experiment', color='black', linewidth=4)
    # plt.xlabel('Radius, r (Å)')
    plt.xlim(1.6, 7)
    plt.xticks([])
    plt.ylabel('H(r) (arb. unit)')
    plt.legend(frameon=False, fontsize=14)
    plt.tight_layout()
    plt.savefig('pmn_{}K_scattering_with_SM.png'.format(temp), dpi=300, transparent=True)
    plt.close()

    # This plots a figure not used in the paper. We include also the scattering function.
    fig, ax = plt.subplots(2, 1,figsize=(10, 10))
    ax[0].plot(q_vector[cutoff_q:], scattering_function_avg[cutoff_q:], label='T={}K'.format(temp), color='royalblue')
    ax[0].set_xlabel('Scattering Vector, Q (Å⁻¹)', fontsize=12)
    ax[0].set_ylabel('S(Q)-1', fontsize=12)
    ax[0].legend()
    ax[0].grid(True)
    ax[1].plot(r_vector, rdf_vector, label='T={}K'.format(temp), color='royalblue')
    ax[1].set_xlabel('Radius, r (Å)', fontsize=12)
    ax[1].set_ylabel('Radial Correlation Function, h(r)', fontsize=12)
    ax[1].legend()
    ax[1].grid(True)
    # Set plot limits for clarity
    ax[0].set_xlim(q_vector[cutoff_q], q_vector[-1])
    plt.tight_layout()
    plt.savefig('pmn_{}K_scattering_function.png'.format(temp))

