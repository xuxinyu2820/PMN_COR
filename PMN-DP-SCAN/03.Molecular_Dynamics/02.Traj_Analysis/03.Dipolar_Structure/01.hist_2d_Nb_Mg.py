# conda activate /global/cfs/cdirs/m5025/pinchenx/conda_envs/dp2211
# salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu --account m5025
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ase.io import Trajectory
from ase.neighborlist import NewPrimitiveNeighborList

# load trajectory
# traj_path = "/global/homes/x/xinyuxu/m5025/Ferroic/PMN/MD/traj_12/pmn_300K_10000steps.traj"
traj_path = "/global/homes/x/xinyuxu/m5025/Ferroic/PMN/MD/traj_12/pmn_700.0K_5000000steps.traj"
BO_cutoff = 2.3   # B–O cutoff for octahedron
BB_cutoff = 3.0   # B–B cutoff for neighbor classification

mpl.rcParams['axes.linewidth'] = 2     
mpl.rcParams['axes.titlesize'] = 45
mpl.rcParams['axes.labelsize'] = 36
mpl.rcParams['xtick.labelsize'] = 30
mpl.rcParams['ytick.labelsize'] = 30

# tools
def get_B_to_O6(atoms, cutoff=2.3):
    Z = atoms.get_atomic_numbers()
    pos = atoms.get_positions()
    atoms.set_pbc(True)

    nl = NewPrimitiveNeighborList(cutoff*np.ones(len(Z)), skin=0.1, self_interaction=False, bothways=True)
    nl.update(pbc=atoms.get_pbc(), cell=atoms.get_cell(), positions=pos)

    B_idx = np.where(np.isin(Z, [12, 41]))[0]  # Mg=12, Nb=41
    B_to_O6 = {}
    for b in B_idx:
        neigh = nl.get_neighbors(b)[0]
        neigh_O = neigh[Z[neigh] == 8]
        d = atoms.get_distances(b, neigh_O, mic=True)
        neigh_O = neigh_O[np.argsort(d)[:6]]
        B_to_O6[b] = neigh_O
    return B_idx, B_to_O6

def get_B_to_B6(atoms, cutoff=3.0):
    """构建 B→最近6个B 邻居索引"""
    Z = atoms.get_atomic_numbers()
    pos = atoms.get_positions()
    atoms.set_pbc(True)

    nl = NewPrimitiveNeighborList(cutoff*np.ones(len(Z)), skin=0.1, self_interaction=False, bothways=True)
    nl.update(pbc=atoms.get_pbc(), cell=atoms.get_cell(), positions=pos)

    B_idx = np.where(np.isin(Z, [12, 41]))[0]
    B_set = set(B_idx.tolist())
    B_to_B6 = {}
    for b in B_idx:
        neigh = nl.get_neighbors(b)[0]
        neigh_B = np.array([j for j in neigh if j in B_set and j != b])
        if neigh_B.size > 0:
            d = atoms.get_distances(b, neigh_B, mic=True)
            neigh_B = neigh_B[np.argsort(d)[:6]]
        B_to_B6[b] = neigh_B
    return B_to_B6

def compute_BO6_displacement(atoms, B_idx, B_to_O6):
    pos = atoms.get_positions()
    box = atoms.get_cell().lengths()
    disp = np.zeros((len(B_idx), 3))

    for i, b in enumerate(B_idx):
        O6 = B_to_O6[b]
        d = pos[b] - pos[O6]
        d -= box * np.round(d / box)
        disp[i] = np.mean(d, axis=0)
    return disp

def save_2d_hist(data, fname, xlabel, ylabel, vmin=0, vmax=25, cmap='Blues'):

    pts = np.array(data)
    lim = 0.4

    plt.figure(figsize=(8, 6))
    h, xedges, yedges, im = plt.hist2d(
        pts[:, 0], pts[:, 1],
        range=[[-lim, lim], [-lim, lim]],
        bins=100,
        density=True,
        cmap=cmap
    )

    # normalization
    h_norm = h / vmax  

    plt.clf()  
    plt.imshow(
        h_norm.T, origin='lower',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap=cmap, vmin=0, vmax=1
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    ax.set_xticks([-0.2, 0.0, 0.2])
    ax.set_xticklabels(['-0.2', '0.0', '0.2'])
    cbar = plt.colorbar(label="")      # colorbar
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["0.0", "0.5", "1.0"])
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved 2D hist (normalized by vmax={vmax}): {fname}")

# main function
if __name__ == "__main__":
    # trajs path
    traj = Trajectory(traj_path)
    print(f"Loaded trajectory with {len(traj)} frames")

    # neighbor lists
    atoms0 = traj[0].copy()
    B_idx, B_to_O6 = get_B_to_O6(atoms0, BO_cutoff)
    B_to_B6 = get_B_to_B6(atoms0, BB_cutoff)
    Z = atoms0.get_atomic_numbers()

    # empty lists for data
    xy_nb_6mg, yz_nb_6mg = [], []
    xy_nb_5to6nb, yz_nb_5to6nb = [], []
    xy_mg_nb6mg_unique, yz_mg_nb6mg_unique = [], []

    for fi, atoms in enumerate(traj):
        if fi % 10 == 0:
            print(f"Processing frame {fi+1}/{len(traj)}...")

        # displacement vectors
        disp = compute_BO6_displacement(atoms, B_idx, B_to_O6)
        Z_frame = atoms.get_atomic_numbers()

        # Mg unique set
        mg_used = set()

        for i_b, b in enumerate(B_idx):
            if Z_frame[b] != 41:
                continue

            v = disp[i_b]          # this Nb's (dx, dy, dz)
            neigh = B_to_B6[b]     # this Nb's 6 B neighbors defined in frame 0

            if len(neigh) < 6:
                continue

            neigh_Z = Z_frame[neigh]
            num_mg = np.sum(neigh_Z == 12)
            num_nb = np.sum(neigh_Z == 41)

            # Nb_II
            if num_mg == 6:
                # Nb
                xy_nb_6mg.append([v[0], v[1]])  # (dx, dy)
                yz_nb_6mg.append([v[1], v[2]])  # (dy, dz)

                # Mg
                for mg in neigh[neigh_Z == 12]:
                    if mg not in mg_used:
                        mg_used.add(mg)
                        idx_local_mg = np.where(B_idx == mg)[0][0]
                        vmg = disp[idx_local_mg]
                        xy_mg_nb6mg_unique.append([vmg[0], vmg[1]])
                        yz_mg_nb6mg_unique.append([vmg[1], vmg[2]])

            # Nb_I  
            if num_nb >= 5:
                xy_nb_5to6nb.append([v[0], v[1]])
                yz_nb_5to6nb.append([v[1], v[2]])
    # save to csv
    csv_name = "displacement_classification_700K.csv"
    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "comp1", "comp2"])

        for x, y in xy_nb_6mg:
            writer.writerow(["nb_6mg_xy", x, y])

        for y, z in yz_nb_6mg:
            writer.writerow(["nb_6mg_yz", y, z])

        for x, y in xy_nb_5to6nb:
            writer.writerow(["nb_5to6nb_xy", x, y])

        for y, z in yz_nb_5to6nb:
            writer.writerow(["nb_5to6nb_yz", y, z])

        for x, y in xy_mg_nb6mg_unique:
            writer.writerow(["mg_nb6mg_xy", x, y])

        for y, z in yz_mg_nb6mg_unique:
            writer.writerow(["mg_nb6mg_yz", y, z])

    print(f"Saved displacement data to {csv_name}")
