# conda activate /global/cfs/cdirs/m5025/pinchenx/conda_envs/dp2211
# salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu --account m5025
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ase.io import Trajectory
from ase.neighborlist import NewPrimitiveNeighborList

# load trajectory
traj_path = "/global/homes/x/xinyuxu/m5025/Ferroic/PMN/MD/traj_12/pmn_100.0K_5000000steps.traj"

# A-O cutoff (Pb-O); adjust if needed
AO_cutoff = 3.6   # Angstrom, used to find O neighbors around A-site
A_Z = 82          # Pb
O_Z = 8

mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['axes.titlesize'] = 45
mpl.rcParams['axes.labelsize'] = 36
mpl.rcParams['xtick.labelsize'] = 30
mpl.rcParams['ytick.labelsize'] = 30

def get_A_to_O12(atoms, cutoff=3.6, A_Z=82, O_Z=8):
    """
    Build mapping A -> nearest 12 O indices (based on one atoms snapshot).
    """
    Z = atoms.get_atomic_numbers()
    pos = atoms.get_positions()
    atoms.set_pbc(True)

    nl = NewPrimitiveNeighborList(
        cutoff * np.ones(len(Z)),
        skin=0.1,
        self_interaction=False,
        bothways=True
    )
    nl.update(pbc=atoms.get_pbc(), cell=atoms.get_cell(), positions=pos)

    A_idx = np.where(Z == A_Z)[0]
    A_to_O12 = {}

    for a in A_idx:
        neigh = nl.get_neighbors(a)[0]
        neigh_O = neigh[Z[neigh] == O_Z]

        if neigh_O.size == 0:
            A_to_O12[a] = np.array([], dtype=int)
            continue

        d = atoms.get_distances(a, neigh_O, mic=True)
        neigh_O = neigh_O[np.argsort(d)[:12]]  # nearest 12 O
        A_to_O12[a] = neigh_O

    return A_idx, A_to_O12

def compute_AO12_displacement(atoms, A_idx, A_to_O12):
    """
    For each A-site, compute displacement vector:
    u_A = r_A - mean(r_O12) with MIC under PBC.
    """
    pos = atoms.get_positions()
    box = atoms.get_cell().lengths()  # assumes near-orthorhombic cell; consistent with your original code
    disp = np.zeros((len(A_idx), 3))

    for i, a in enumerate(A_idx):
        O12 = A_to_O12[a]
        if len(O12) < 12:
            # if not enough O neighbors found, mark as NaN
            disp[i, :] = np.nan
            continue

        d = pos[a] - pos[O12]                 # 12 vectors A - O
        d -= box * np.round(d / box)          # MIC wrap
        disp[i, :] = np.mean(d, axis=0)       # A - O_center

    return disp

if __name__ == "__main__":
    traj = Trajectory(traj_path)
    print(f"Loaded trajectory with {len(traj)} frames")

    # Build A->O12 map on frame 0 (same pattern as your old code, cheaper than per-frame rebuild)
    atoms0 = traj[0].copy()
    A_idx, A_to_O12 = get_A_to_O12(atoms0, cutoff=AO_cutoff, A_Z=A_Z, O_Z=O_Z)
    print(f"Found {len(A_idx)} A-site atoms (Z={A_Z}) in frame 0")

    # Collect data rows: (frame, A_index, dx, dy, dz)
    rows = []
    for fi, atoms in enumerate(traj):
        if fi % 10 == 0:
            print(f"Processing frame {fi+1}/{len(traj)}...")

        disp = compute_AO12_displacement(atoms, A_idx, A_to_O12)
        for i, a in enumerate(A_idx):
            dx, dy, dz = disp[i]
            rows.append([fi, int(a), dx, dy, dz])

    # Save to one CSV
    csv_name = "Asite_displacement_AO12_100K.csv"
    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "A_index", "dx", "dy", "dz"])
        writer.writerows(rows)

    print(f"Saved A-site AO12 displacement data to {csv_name}")