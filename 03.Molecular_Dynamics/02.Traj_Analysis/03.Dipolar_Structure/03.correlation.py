import os
import numpy as np
import ase.io
from ase.io import Trajectory
from ase.neighborlist import NewPrimitiveNeighborList

# functions to build neighbor lists and compute order parameters
def get_neigh_list_from_atoms(atoms, bo_cutoff=1.25):
    atoms.set_pbc([True, True, True])
    Z = np.array(atoms.get_atomic_numbers())
    pos = np.array(atoms.get_positions())

    nl = NewPrimitiveNeighborList(
        bo_cutoff * np.ones(Z.shape[0]),
        skin=0.1, self_interaction=False, bothways=True
    )
    nl.update(pbc=atoms.get_pbc(), cell=atoms.get_cell(), positions=pos)

    # B-site indices
    B_sites = np.argwhere(np.isin(Z, [12, 41])).reshape(-1,)
    B_sites_O_nlidx_6 = []
    B_sites_O_nlidx_8 = []

    for i in range(B_sites.shape[0]):
        b = B_sites[i]
        neigh_all = nl.get_neighbors(b)[0]
        neigh_O = neigh_all[Z[neigh_all] == 8]
        d = atoms.get_distances(b, neigh_O, mic=True)
        order = np.argsort(d)
        neigh_O_sorted = neigh_O[order]
        B_sites_O_nlidx_6.append(neigh_O_sorted[:6])
        B_sites_O_nlidx_8.append(neigh_O_sorted[:8])

    B_sites_O_nlidx_6 = np.array(B_sites_O_nlidx_6, dtype=object)
    B_sites_O_nlidx_8 = np.array(B_sites_O_nlidx_8, dtype=object)
    return B_sites, B_sites_O_nlidx_6, B_sites_O_nlidx_8

def compute_BO6_uc_order_parameter(box, atom_pos, B_sites, B_sites_O_nlidx_6):
    B_pos = atom_pos[B_sites, :]
    O6_idx_rows = [np.asarray(row, dtype=int) for row in B_sites_O_nlidx_6]
    O6_pos = np.array([atom_pos[idx_row, :] for idx_row in O6_idx_rows], dtype=float)

    delta = B_pos[:, None, :] - O6_pos
    delta -= box[None, None, :] * np.round(delta / box[None, None, :])
    uc_order_parameter = np.mean(delta, axis=1)
    return uc_order_parameter

def build_B_to_B6_from_atoms(atoms, bb_cutoff=3.0):
    atoms.set_pbc([True, True, True])
    Z = np.array(atoms.get_atomic_numbers())
    pos = np.array(atoms.get_positions())

    B_sites = np.argwhere(np.isin(Z, [12, 41])).reshape(-1,)
    B_set = set(B_sites.tolist())

    nlB = NewPrimitiveNeighborList(
        bb_cutoff * np.ones(Z.shape[0]),
        skin=0.1, self_interaction=False, bothways=True
    )
    nlB.update(pbc=atoms.get_pbc(), cell=atoms.get_cell(), positions=pos)

    B_to_B6 = {}
    for b in B_sites:
        neigh_all = nlB.get_neighbors(b)[0]
        neigh_B = np.array([j for j in neigh_all if (j in B_set and j != b)], dtype=int)
        if neigh_B.size > 0:
            d = atoms.get_distances(b, neigh_B, mic=True)
            neigh_B = neigh_B[np.argsort(d)[:6]]
        B_to_B6[b] = neigh_B
    return B_sites, B_to_B6

# parameters
root_dir = "/global/homes/x/xinyuxu/m5025/Ferroic/PMN/MD/traj_12"
temps = [600,700]  # K
# temps = [100]
BO_cutoff = 2.3
BB_cutoff = 3.0
Z_Mg, Z_Nb = 12, 41
# storage for summary
summary_rows = []

for T in temps:
    traj_path = os.path.join(root_dir, f"pmn_{T}.0K_5000000steps.traj")

    traj = Trajectory(traj_path)
    atoms0 = traj[0].copy()
    Z0 = np.array(atoms0.get_atomic_numbers())

    B_sites_indices, B_to_O6_list, _ = get_neigh_list_from_atoms(atoms0, bo_cutoff=BO_cutoff)
    _, B_to_B6 = build_B_to_B6_from_atoms(atoms0, bb_cutoff=BB_cutoff)
    bidx_to_local = {int(b): i for i, b in enumerate(B_sites_indices)}

    # accumulators
    mg_shifts_sq = [] 
    nb_shifts_sq = [] 
    nbnb_dots = []
    nbmg_dots = []

    for atoms in traj:
        positions = atoms.get_positions()
        box = atoms.get_cell().lengths()
        BO6_u = compute_BO6_uc_order_parameter(box, positions, B_sites_indices, B_to_O6_list)

        seen_nb_nb = set()
        seen_nb_mg = set()

        for idx_local, b_idx in enumerate(B_sites_indices):
            bZ = Z0[b_idx]
            v_b = BO6_u[idx_local]
            mag_b_sq = float(np.dot(v_b, v_b)) # |v|^2 = v·v

            if bZ == Z_Mg:
                mg_shifts_sq.append(mag_b_sq)
            elif bZ == Z_Nb:
                nb_shifts_sq.append(mag_b_sq)

        for idx_local, b_idx in enumerate(B_sites_indices):
            if Z0[b_idx] != Z_Nb:
                continue

            v_b = BO6_u[idx_local]
            neighB = B_to_B6[b_idx]
            for j_idx in neighB:
                jZ = Z0[j_idx]
                j_local = bidx_to_local[j_idx]
                v_j = BO6_u[j_local]
                dotp = float(np.dot(v_b, v_j))

                if jZ == Z_Nb:
                    key = (min(b_idx, j_idx), max(b_idx, j_idx))
                    if key not in seen_nb_nb:
                        nbnb_dots.append(dotp)
                        seen_nb_nb.add(key)
                elif jZ == Z_Mg:
                    key = (b_idx, j_idx)
                    if key not in seen_nb_mg:
                        nbmg_dots.append(dotp)
                        seen_nb_mg.add(key)


    mean_dot_NbNb = float(np.mean(nbnb_dots))
    mean_dot_NbMg = float(np.mean(nbmg_dots))

    mean_sq_Mg_shift = float(np.mean(mg_shifts_sq))
    mean_sq_Nb_shift = float(np.mean(nb_shifts_sq))


    corr_NbNb = mean_dot_NbNb / mean_sq_Nb_shift if mean_sq_Nb_shift != 0 else 0
    denom_NbMg = np.sqrt(mean_sq_Nb_shift * mean_sq_Mg_shift)
    corr_NbMg = mean_dot_NbMg / denom_NbMg if denom_NbMg != 0 else 0
    
    summary_rows.append((T, corr_NbNb, corr_NbMg))
    print(f"T={T} K  |  Corr_NbNb={corr_NbNb:.6e}  Corr_NbMg={corr_NbMg:.6e}")

# write summary to CSV
out_csv = os.path.join(root_dir, "summary_correlations.csv")
header = "T_K,corr_NbNb,corr_NbMg"
with open(out_csv, "w") as f:
    f.write(header + "\n")
    for (T, a, b) in summary_rows:
        f.write(f"{T},{a:.8e},{b:.8e}\n")

print(f"\nSaved summary to: {out_csv}")